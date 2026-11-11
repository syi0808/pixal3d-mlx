"""Replay real Pixal3D sampling with frozen inputs; no production defaults change."""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import mlx.core as mx
import mlx.utils
import numpy as np

from generate_pixal3d import _denormalize_slat, _normalize_slat, extract_proj_features, TEX_SLAT_MEAN, TEX_SLAT_STD
from scripts.download_models import DINO_REVISION, PIXAL_REVISION
from pixalmlx.checkpoint import load_checkpoint
from pixalmlx.conditioning import ZeroConditioning, prepare_sampling
from pixalmlx.models.pixal3d_flow import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel
from pixalmlx.models.sparse_structure_flow import _layernorm_noaffine
from pixalmlx.runtime import InferenceRuntime
from pixalmlx.samplers import flow_euler_sample
from pixalmlx.transformer_experiments import VARIANTS, configure_transformer
from pixalmlx.transformer_runtime import configure_selective_compile
from pixalmlx.weight_loader import load_weights


WEIGHTS = Path.home() / ".cache/huggingface/hub/models--TencentARC--Pixal3D/snapshots" / PIXAL_REVISION / "ckpts"


def prepare(arguments):
    from pixalmlx.models.dinov3 import DINOv3ViT, load_dinov3_weights
    from pixalmlx.models.naf import NAF
    from pixalmlx.models.naf_loader import load_naf_weights

    if arguments.inputs.exists():
        raise ValueError("Refusing to overwrite frozen sampling inputs")
    backbone = DINOv3ViT()
    dino_path = Path.home() / ".cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots" / DINO_REVISION
    load_dinov3_weights(backbone, str(dino_path))
    camera = load_checkpoint(str(arguments.checkpoints), "camera")
    payload = {}
    if arguments.stage == "ss":
        cond, negative = extract_proj_features(arguments.image, backbone, 16, 512, camera, no_rembg=True)
        payload["noise"] = np.array(mx.random.normal((1, 8, 16, 16, 16), key=mx.random.key(arguments.seed)))
    else:
        checkpoint = load_checkpoint(str(arguments.checkpoints), "shape_latent")
        naf = NAF()
        load_naf_weights(naf, "weights/naf_release.safetensors", verbose=False)
        coordinates = checkpoint["quant_coords"]
        cond, negative = extract_proj_features(arguments.image, backbone, int(checkpoint["hr_resolution"]) // 16,
                                               1024, camera, no_rembg=True, use_naf_upsample=True,
                                               upsample_target_size=512, naf_model=naf, coords=coordinates)
        payload["coords"] = coordinates[:, 1:]
        payload["noise"] = np.array(mx.random.normal((len(coordinates), 32),
                                                       key=mx.random.key(arguments.seed + (3 if arguments.stage == "texture" else 2))))
        if arguments.stage == "texture":
            payload["concat_cond"] = np.array(_normalize_slat(mx.array(checkpoint["hr_slat"])))
    payload.update({name: np.array(value) for name, value in cond.items()})
    arguments.inputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.inputs, **payload)
    metadata = {"stage": arguments.stage, "seed": arguments.seed, "image": str(Path(arguments.image).resolve()),
                "image_sha256": hashlib.sha256(Path(arguments.image).read_bytes()).hexdigest(),
                "checkpoints": str(arguments.checkpoints.resolve()),
                "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in payload.items()}}
    arguments.inputs.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def load_model(stage):
    if stage == "ss":
        model = Pixal3DSparseStructureFlowModel()
        name = "ss_flow_img_dit_1_3B_64_bf16"
    else:
        model = Pixal3DSLatFlowModel(in_channels=64 if stage == "texture" else 32,
                                    out_channels=32, proj_in_channels=2048)
        name = "slat_flow_imgshape2tex_dit_1_3B_1024_bf16" if stage == "texture" else "slat_flow_img2shape_dit_1_3B_1024_bf16"
    load_weights(model, str(WEIGHTS / (name + ".safetensors")), streaming=True, verbose=False,
                 target_dtype=mx.float32 if stage == "ss" else None)
    return model


def errors(actual, expected):
    finite = bool(np.isfinite(actual).all())
    if not finite:
        return {"all_finite": False, "max_absolute_error": None, "root_mean_squared_error": None}
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {"all_finite": True, "array_equal": bool(np.array_equal(actual, expected)),
            "max_absolute_error": float(np.max(np.abs(difference))),
            "root_mean_squared_error": float(np.sqrt(np.mean(difference ** 2))),
            "relative_l2_error": float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-30))}


def decode_structure(latent, checkpoints):
    from pixalmlx.models.sparse_structure_decoder import SparseStructureDecoder

    decoder = SparseStructureDecoder()
    load_weights(decoder, str(WEIGHTS / "ss_dec_conv3d_16l8_fp16.safetensors"), streaming=True, verbose=False)
    logits = decoder(latent.astype(mx.float32))
    decoded = np.array(logits[0, 0] > 0)
    ratio = decoded.shape[0] // 32
    actual = decoded.reshape(32, ratio, 32, ratio, 32, ratio).any(axis=(1, 3, 5))
    coordinates = np.argwhere(actual)
    archived = load_checkpoint(str(checkpoints), "ss")["lr_coords"]
    expected = np.zeros((32, 32, 32), dtype=bool)
    expected[tuple(archived.T)] = True
    union = np.logical_or(actual, expected).sum()
    return {"active_voxels": len(coordinates), "archived_active_voxels": len(archived),
            "coordinates_equal": bool(np.array_equal(coordinates, archived)),
            "changed_voxels": int(np.count_nonzero(actual != expected)),
            "intersection_over_union": float(np.logical_and(actual, expected).sum() / max(1, union))}, coordinates


def profile_first_block(model, noise, cond, negative, options, repeats):
    positive, negative, kwargs, contexts = prepare_sampling(model, noise, cond, negative, {}, options.cache_bytes)
    block = model.blocks[0]
    features = model.input_layer(noise.reshape(1, 8, -1).transpose(0, 2, 1).reshape(-1, 8))
    base_modulation = model.adaLN_modulation(model.t_embedder(mx.array([1000.0], dtype=mx.float32)))[0]
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mx.split(base_modulation + block.modulation, 6)
    attention_input = _layernorm_noaffine(features) * (1 + scale_attn) + shift_attn
    attention_output = block.self_attn(attention_input, rope_phases=kwargs["rope_phases"])
    residual = features + attention_output * gate_attn
    cross_output = block.cross_attn(block.norm2(residual), positive)
    cross_residual = residual + cross_output
    ffn_input = _layernorm_noaffine(cross_residual) * (1 + scale_mlp) + shift_mlp
    mx.eval(features, base_modulation, attention_input, attention_output, residual, cross_output, cross_residual, ffn_input)
    functions = {
        "ada_ln": lambda: _layernorm_noaffine(features) * (1 + scale_attn) + shift_attn,
        "self_attention": lambda: block.self_attn(attention_input, rope_phases=kwargs["rope_phases"]),
        "attention_gated_residual": lambda: features + attention_output * gate_attn,
        "cross_attention_cached_projection": lambda: block.cross_attn(block.norm2(residual), positive),
        "ffn": lambda: block.mlp(ffn_input),
        "whole_block": lambda: block(features, base_modulation, positive, rope_phases=kwargs["rope_phases"]),
    }
    measurements = {}
    try:
        for name, function in functions.items():
            mx.eval(function())
            timings = []
            for iteration in range(repeats):
                started = time.perf_counter()
                output = function()
                mx.eval(output)
                mx.synchronize()
                timings.append(time.perf_counter() - started)
            measurements[name] = {"median_seconds": statistics.median(timings), "seconds": timings}
    finally:
        for context in contexts:
            context.clear_invariants()
    return output, {"component_measurements": measurements,
                    "scope": "first conditional SS block; warm fixed inputs and projection cache; separate evaluation boundaries"}


def run(arguments):
    provenance = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                  "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                    for name in ("scripts/investigate_sampling.py", "pixalmlx/transformer_experiments.py",
                                                 "pixalmlx/transformer_runtime.py", "pixalmlx/samplers.py",
                                                 "pixalmlx/models/sparse_structure_flow.py", "pixalmlx/models/pixal3d_flow.py")}}
    production = arguments.variant.startswith("production-")
    selective_compile = arguments.variant == "production-selective"
    options = VARIANTS["reference" if production else arguments.variant]
    metadata = json.loads(arguments.inputs.with_suffix(".json").read_text())
    if metadata["stage"] != arguments.stage:
        raise ValueError("Frozen input stage does not match the requested model")
    with np.load(arguments.inputs) as frozen:
        values = {name: mx.array(frozen[name]) for name in frozen.files}
    cond = {name: values[name] for name in ("global", "proj")}
    negative = ZeroConditioning({name: mx.zeros_like(value) for name, value in cond.items()})
    kwargs = {name: values[name] for name in ("coords", "concat_cond") if name in values}
    noise = values["noise"]
    model = load_model(arguments.stage)
    if production:
        for block in model.blocks:
            block.mlp.chunk_size = options.ffn_chunk_size
        configure_selective_compile(model, selective_compile)
    else:
        model = configure_transformer(model, options)
    mx.eval(model.parameters(), values)
    parameter_bytes = sum(value.nbytes for _, value in mlx.utils.tree_flatten(model.parameters()))
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    if arguments.mode == "profile":
        output, timing_info = profile_first_block(model, noise, cond, negative, options, arguments.repeats)
    elif arguments.mode == "screen":
        positive, negative, kwargs, contexts = prepare_sampling(model, noise, cond, negative, kwargs, options.cache_bytes)
        timestep = mx.array([1000.0], dtype=mx.float32)
        modulation = model.adaLN_modulation(model.t_embedder(timestep))
        kwargs["modulation"] = modulation
        mx.eval(modulation)
        timings = []
        try:
            for iteration in range(arguments.repeats + 1):
                started = time.perf_counter()
                output = model(noise, timestep, positive, **kwargs)
                mx.eval(output)
                mx.synchronize()
                timings.append(time.perf_counter() - started)
            cache_info = [{"used_bytes": context.used_bytes, "hits": context.hits} for context in contexts]
        finally:
            for context in contexts:
                context.clear_invariants()
        elapsed = statistics.median(timings[1:])
        timing_info = {"cold_forward_seconds": timings[0], "warm_forward_seconds": timings[1:],
                       "median_forward_seconds": elapsed, "scope": "single conditional forward, not a full sampler",
                       "conditioning_cache": cache_info}
    else:
        parameters = (dict(guidance_strength=7.5, guidance_rescale=0.7, rescale_t=5.0) if arguments.stage == "ss" else
                      dict(guidance_strength=1.0, guidance_rescale=0.0, rescale_t=3.0) if arguments.stage == "texture" else
                      dict(guidance_strength=7.5, guidance_rescale=0.5, rescale_t=3.0))
        started = time.perf_counter()
        output = flow_euler_sample(model, noise, cond, negative, steps=arguments.steps, verbose=False,
                                   guidance_interval=(0.6, 0.9) if arguments.stage == "texture" else (0.6, 1.0),
                                   conditioning_cache_bytes=options.cache_bytes, ffn_chunk_size=options.ffn_chunk_size,
                                   selective_compile=selective_compile,
                                   **parameters, **kwargs)
        mx.eval(output)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        timing_info = {"sampling_seconds": elapsed, "steps": arguments.steps, "parameters": parameters}
    peak = mx.get_peak_memory()
    host = np.array(output)
    report = {"stage": arguments.stage, "variant": arguments.variant, "mode": arguments.mode,
              "selective_compile": selective_compile,
              "options": asdict(options), **timing_info, "peak_mlx_bytes": peak,
              "parameter_bytes": parameter_bytes, "output_shape": list(host.shape), "all_finite": bool(np.isfinite(host).all()),
              "output_sha256": hashlib.sha256(np.ascontiguousarray(host)).hexdigest(),
              "inputs_sha256": hashlib.sha256(arguments.inputs.read_bytes()).hexdigest(),
              "mlx_version": mx.__version__, "device": mx.device_info(),
              **provenance}
    if arguments.reference:
        with np.load(arguments.reference) as reference:
            report["comparison_to_reference"] = errors(host, reference["latent"])
    if arguments.mode == "sample":
        del model
        gc.collect()
        mx.clear_cache()
        if not report["all_finite"]:
            report["decode_skipped"] = "nonfinite_sampler_output"
        elif arguments.stage == "ss":
            report["structure"], coordinates = decode_structure(output, arguments.checkpoints)
        else:
            checkpoint = load_checkpoint(str(arguments.checkpoints), "shape_latent")
            if arguments.stage == "hr":
                report["comparison_to_archived_shape_latent"] = errors(np.array(_denormalize_slat(output)), checkpoint["hr_slat"])
            else:
                report["texture_latent_finite"] = bool(np.isfinite(np.array(_denormalize_slat(output, mean=TEX_SLAT_MEAN, std=TEX_SLAT_STD))).all())
    if arguments.save_latent:
        arguments.save_latent.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arguments.save_latent, latent=host)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "screen", "sample", "profile", "time"])
    parser.add_argument("--stage", choices=["ss", "hr", "texture"], default="ss")
    parser.add_argument("--variant", choices=[*VARIANTS, "production-reference", "production-selective"], default="reference")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--report", type=Path, default=Path("outputs/runtime/sampling-investigation/report.json"))
    parser.add_argument("--checkpoints", type=Path, default=Path("outputs/turnip512/checkpoints"))
    parser.add_argument("--image", default="outputs/turnip512/turnip_preprocessed.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--save-latent", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12, help="Research only; changing steps changes the sampling algorithm")
    arguments = parser.parse_args()
    if arguments.inputs is None:
        arguments.inputs = Path("outputs/runtime/sampling-investigation") / (arguments.stage + "-inputs.npz")
    if arguments.repeats < 1 or arguments.steps < 1:
        parser.error("Screen repeats and sampling steps must be positive")
    if arguments.mode == "profile" and arguments.stage != "ss":
        parser.error("Component profiling currently uses the dense SS block")
    if arguments.mode != "prepare" and (arguments.report.exists() or arguments.save_latent and arguments.save_latent.exists()):
        parser.error("Refusing to overwrite sampling evidence")
    with InferenceRuntime(model_cache_bytes=0, feature_cache_bytes=0):
        return prepare(arguments) if arguments.mode == "prepare" else run(arguments)


if __name__ == "__main__":
    main()
