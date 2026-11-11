"""Opt-in checkpoint replay for decoder memory research; production paths stay unchanged."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

from generate_pixal3d import TEX_SLAT_MEAN, TEX_SLAT_STD, _denormalize_slat, _normalize_slat, extract_proj_features
from scripts.download_models import DINO_REVISION, PIXAL_REVISION
from pixalmlx.checkpoint import load_checkpoint
from pixalmlx.models.shape_slat_decoder import SLatDecoder, SparseConvNeXtBlock3d, SparseResBlockC2S3d
from pixalmlx.decoder_experiments import DecoderExperiment, ResearchDecoder
from pixalmlx.modules.sparse_conv import build_neighbor_map
from pixalmlx.runtime import InferenceRuntime
from pixalmlx.samplers import flow_euler_sample
from pixalmlx.weight_loader import load_weights


CHECKPOINT_ROOT = Path.home() / ".cache/huggingface/hub/models--TencentARC--Pixal3D/snapshots" / PIXAL_REVISION / "ckpts"


@contextmanager
def preserve_rope_dtype(enabled):
    from pixalmlx.models import sparse_structure_flow

    original = sparse_structure_flow.apply_rope
    if enabled:
        def cast_rotated_query(features, phases):
            return original(features, phases).astype(features.dtype)
        sparse_structure_flow.apply_rope = cast_rotated_query
    try:
        yield
    finally:
        sparse_structure_flow.apply_rope = original


class Recorder:
    def __init__(self):
        self.events = []
        self.started = time.perf_counter()

    def record(self, name, **metadata):
        self.events.append({"name": name, "elapsed_seconds": time.perf_counter() - self.started,
                            "active_mlx_bytes": mx.get_active_memory(), "peak_mlx_bytes": mx.get_peak_memory(),
                            **metadata})


class TracedDecoder(ResearchDecoder):
    def __init__(self, recorder, **kwargs):
        super().__init__(record=recorder.record, **kwargs)
        object.__setattr__(self, "recorder", recorder)

    def _forward_levels(self, feats, coords, stop_after=None, guide_subs=None, return_subs=False):
        upsample_count = 0
        subdivisions = [] if return_subs else None
        neighbors = build_neighbor_map(coords)
        for level_index, level in enumerate(self.blocks):
            if stop_after is not None and upsample_count >= stop_after:
                break
            for block_index, block in enumerate(level):
                if isinstance(block, SparseConvNeXtBlock3d):
                    feats = block(feats, neighbors)
                    mx.eval(feats)
                    self.recorder.record(f"level_{level_index}_block_{block_index}", rows=feats.shape[0], channels=feats.shape[1])
                elif isinstance(block, SparseResBlockC2S3d):
                    parent_rows = feats.shape[0]
                    guide = guide_subs[upsample_count] if guide_subs is not None else None
                    feats, coords, logits, neighbors = block(feats, coords, neighbors, subdiv=guide, return_neighbor_map=True)
                    if return_subs and logits is not None:
                        subdivisions.append(logits)
                    self.recorder.record(f"level_{level_index}_upsample_graph_built",
                                         parent_rows=parent_rows, child_rows=feats.shape[0],
                                         parent_channels=block.channels, child_channels=block.out_channels,
                                         dense_parent_projection_bytes=parent_rows * block.out_channels * 8 * 4,
                                         child_feature_bytes=feats.nbytes,
                                         neighbor_edge_count=neighbors[0].size,
                                         neighbor_index_bytes=sum(value.nbytes for value in neighbors))
                    upsample_count += 1
        if return_subs:
            return feats, coords, subdivisions
        return feats, coords


def prepare_latent(arguments):
    from pixalmlx.models.dinov3 import DINOv3ViT, load_dinov3_weights
    from pixalmlx.models.naf import NAF
    from pixalmlx.models.naf_loader import load_naf_weights
    from pixalmlx.models.pixal3d_flow import Pixal3DSLatFlowModel

    if arguments.latent.exists():
        raise ValueError("Refusing to overwrite the frozen texture latent")
    checkpoint = load_checkpoint(str(arguments.checkpoints), "shape_latent")
    camera = load_checkpoint(str(arguments.checkpoints), "camera")
    shape = mx.array(checkpoint["hr_slat"])
    coordinates = checkpoint["quant_coords"]
    resolution = int(checkpoint["hr_resolution"])
    dinov3 = DINOv3ViT()
    dino_path = Path.home() / ".cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots" / DINO_REVISION
    load_dinov3_weights(dinov3, str(dino_path))
    naf = NAF()
    load_naf_weights(naf, "weights/naf_release.safetensors", verbose=False)
    cond, negative = extract_proj_features(arguments.image, dinov3, resolution // 16, 1024, camera,
                                           no_rembg=True, use_naf_upsample=True, upsample_target_size=512,
                                           naf_model=naf, coords=coordinates)
    del dinov3, naf
    gc.collect()
    mx.clear_cache()
    model = Pixal3DSLatFlowModel(in_channels=64, out_channels=32, proj_in_channels=2048)
    load_weights(model, str(CHECKPOINT_ROOT / "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors"),
                 verbose=False, streaming=True)
    if arguments.flow_mixed:
        model.compute_dtype = mx.float16
    noise = mx.random.normal((len(coordinates), 32), key=mx.random.key(arguments.seed + 3))
    started = time.perf_counter()
    with preserve_rope_dtype(arguments.preserve_rope_dtype):
        latent = flow_euler_sample(model, noise, cond, negative, steps=12, guidance_strength=1.0,
                                   guidance_rescale=0.0, guidance_interval=(0.6, 0.9), rescale_t=3.0,
                                   verbose=False, coords=mx.array(coordinates[:, 1:]), concat_cond=_normalize_slat(shape),
                                   conditioning_cache_bytes=int(arguments.conditioning_cache_mb * 1024 ** 2))
    latent = _denormalize_slat(latent, mean=TEX_SLAT_MEAN, std=TEX_SLAT_STD)
    mx.eval(latent)
    host_latent = np.array(latent)
    arguments.latent.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.latent, latent=host_latent, coordinates=coordinates)
    print(json.dumps({"saved": str(arguments.latent), "sampling_seconds": time.perf_counter() - started,
                      "peak_mlx_bytes": mx.get_peak_memory(), "seed": arguments.seed,
                      "resolution": resolution, "flow_mixed": arguments.flow_mixed,
                      "preserve_rope_dtype": arguments.preserve_rope_dtype,
                      "all_finite": bool(np.isfinite(host_latent).all()),
                      "conditioning_cache_mb": arguments.conditioning_cache_mb}, indent=2), flush=True)


def decode(arguments):
    recorder = Recorder()
    with np.load(arguments.latent) as checkpoint:
        latent = mx.array(checkpoint["latent"])
        coordinates = mx.array(checkpoint["coordinates"])
    guides = [mx.array(value) for value in load_checkpoint(str(arguments.checkpoints), "shape_subs")["subs"]]
    resident_shape = None
    if arguments.resident_shape:
        resident_shape = SLatDecoder(out_channels=7, pred_subdiv=True, spatial_order="morton")
        load_weights(resident_shape, str(CHECKPOINT_ROOT / "shape_dec_next_dc_f16c32_fp16.safetensors"),
                     verbose=False, streaming=True)
        mx.eval(resident_shape.parameters())
    options = {
        "reference": DecoderExperiment(),
        "phased": DecoderExperiment(phase_barriers=True),
        "early-topology": DecoderExperiment(early_topology=True),
        "fused-skip": DecoderExperiment(fused_skip=True),
        "terminal-chunked": DecoderExperiment(terminal_chunk_size=arguments.chunk_size),
        "bounded-conv": DecoderExperiment(convolution_eval_interval=arguments.convolution_eval_interval),
        "bounded-phased": DecoderExperiment(convolution_eval_interval=arguments.convolution_eval_interval,
                                            phase_barriers=True),
        "bounded-final": DecoderExperiment(convolution_eval_interval=arguments.convolution_eval_interval,
                                           convolution_scope="final-projection"),
        "bounded-upsample": DecoderExperiment(convolution_eval_interval=arguments.convolution_eval_interval,
                                              convolution_scope="final-upsample"),
        "combined": DecoderExperiment(phase_barriers=True, early_topology=True, fused_skip=True,
                                       terminal_chunk_size=arguments.chunk_size),
    }[arguments.variant]
    model = TracedDecoder(recorder, experiment=options, out_channels=6, pred_subdiv=False, spatial_order="morton")
    load_weights(model, str(CHECKPOINT_ROOT / "tex_dec_next_dc_f16c32_fp16.safetensors"),
                 verbose=False, streaming=True)
    mx.eval(model.parameters(), latent, coordinates, guides)
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    recorder.started = time.perf_counter()
    recorder.record("decode_start")
    output, output_coordinates = model(latent, coordinates, guide_subs=guides)
    mx.eval(output, output_coordinates)
    mx.synchronize()
    elapsed = time.perf_counter() - recorder.started
    recorder.record("decode_complete", rows=output.shape[0])
    report = {"variant": arguments.variant, "resident_shape": arguments.resident_shape,
              "convolution_eval_interval": options.convolution_eval_interval,
              "experiment": asdict(options), "frozen_latent": str(arguments.latent),
              "frozen_latent_sha256": hashlib.sha256(arguments.latent.read_bytes()).hexdigest(),
              "decoder_seconds": elapsed, "peak_mlx_bytes": mx.get_peak_memory(),
              "parameter_bytes": sum(value.nbytes for _, value in mlx.utils.tree_flatten(model.parameters())),
              "resident_shape_parameter_bytes": 0 if resident_shape is None else sum(value.nbytes for _, value in mlx.utils.tree_flatten(resident_shape.parameters())),
              "events": recorder.events, "mlx_version": mx.__version__, "device": mx.device_info(),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())}
    transformed = np.array(output * 0.5 + 0.5)
    actual_coordinates = np.array(output_coordinates)
    with np.load(arguments.checkpoints / "texture.npz") as expected:
        difference = transformed - expected["tex_out"]
        all_finite = bool(np.isfinite(transformed).all())
        report["comparison_to_archived_output"] = {
            "coordinates_equal": bool(np.array_equal(actual_coordinates, expected["tex_coords"])),
            "max_absolute_error": float(np.max(np.abs(difference))) if all_finite else None,
            "mean_absolute_error": float(np.mean(np.abs(difference))) if all_finite else None,
            "root_mean_squared_error": float(np.sqrt(np.mean(difference ** 2))) if all_finite else None,
            "all_finite": all_finite,
            "per_channel_mean_absolute_error": np.mean(np.abs(difference), axis=0).tolist() if all_finite else None,
            "per_channel_root_mean_squared_error": np.sqrt(np.mean(difference ** 2, axis=0)).tolist() if all_finite else None,
        }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: value for name, value in report.items() if name != "events"}, indent=2), flush=True)


def audit_dtypes(arguments):
    from pixalmlx.models.pixal3d_flow import ProjModulatedBlock
    from pixalmlx.modules.rope import apply_rope

    block = ProjModulatedBlock(24, 1, 8, 32, 8)
    block.apply(lambda value: value.astype(mx.float16))
    features = mx.zeros((4, 24), dtype=mx.float16)
    phases = mx.zeros((4, 12, 2), dtype=mx.float32)
    query = mx.zeros((4, 1, 24), dtype=mx.float16)
    context = {"global": mx.zeros((1, 5, 8), dtype=mx.float16),
               "proj": mx.zeros((4, 8), dtype=mx.float16)}
    output = block(features, mx.zeros((144,), dtype=mx.float16), context, rope_phases=phases)
    report = {"mlx_version": mx.__version__, "audit": "symbolic dtype propagation; not a latency benchmark",
              "activation_dtype": str(features.dtype), "phase_dtype": str(phases.dtype),
              "rotated_query_dtype": str(apply_rope(query, phases).dtype),
              "self_attention_without_rope_dtype": str(block.self_attn(features).dtype),
              "self_attention_with_rope_dtype": str(block.self_attn(features, rope_phases=phases).dtype),
              "block_output_dtype": str(output.dtype)}
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "decode", "audit"])
    parser.add_argument("--checkpoints", type=Path, default=Path("outputs/runtime/e2e1024-native/checkpoints"))
    parser.add_argument("--image", default="outputs/runtime/e2e1024-native/shoe_preprocessed.png")
    parser.add_argument("--latent", type=Path, default=Path("outputs/runtime/decoder-investigation/texture-latent.npz"))
    parser.add_argument("--report", type=Path, default=Path("outputs/runtime/decoder-investigation/reference.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=["reference", "phased", "early-topology", "fused-skip", "terminal-chunked", "combined", "bounded-conv", "bounded-phased", "bounded-final", "bounded-upsample"], default="reference")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--convolution-eval-interval", type=int, default=1)
    parser.add_argument("--resident-shape", action="store_true")
    parser.add_argument("--flow-mixed", action="store_true")
    parser.add_argument("--preserve-rope-dtype", action="store_true",
                        help="Research only: round FP32 rotary output back to the incoming activation dtype")
    parser.add_argument("--conditioning-cache-mb", type=float, default=128.0)
    arguments = parser.parse_args()
    if arguments.preserve_rope_dtype and not arguments.flow_mixed:
        parser.error("--preserve-rope-dtype requires --flow-mixed")
    with InferenceRuntime(model_cache_bytes=0, feature_cache_bytes=0):
        if arguments.mode == "prepare":
            prepare_latent(arguments)
        elif arguments.mode == "decode":
            decode(arguments)
        else:
            audit_dtypes(arguments)


if __name__ == "__main__":
    main()
