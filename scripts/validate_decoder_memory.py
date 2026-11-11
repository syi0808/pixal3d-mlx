"""Replay production shape/texture decoders against frozen 1024 checkpoints."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import mlx.core as mx
import mlx.utils
import numpy as np

from scripts.download_models import PIXAL_REVISION
from pixalmlx.checkpoint import load_checkpoint
from pixalmlx.models.shape_slat_decoder import SLatDecoder
from pixalmlx.runtime import InferenceRuntime
from pixalmlx.weight_loader import load_weights


def array_signature(value):
    host = np.array(value)
    return {"shape": list(host.shape), "dtype": str(host.dtype), "bytes": host.nbytes,
            "sha256": hashlib.sha256(host.reshape(-1)).hexdigest(),
            "all_finite": bool(np.isfinite(host).all())}


def replay(arguments):
    checkpoint = load_checkpoint(str(arguments.checkpoints), "shape_latent")
    archived_subdivisions = load_checkpoint(str(arguments.checkpoints), "shape_subs")["subs"]
    if arguments.kind == "shape":
        latent = mx.array(checkpoint["hr_slat"])
        coordinates = mx.array(checkpoint["quant_coords"])
        source_path = arguments.checkpoints / "shape_latent.npz"
    else:
        with np.load(arguments.texture_latent) as frozen:
            latent = mx.array(frozen["latent"])
            coordinates = mx.array(frozen["coordinates"])
        source_path = arguments.texture_latent
    input_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    guides = [mx.array(value) for value in archived_subdivisions] if arguments.kind == "texture" else None
    weight_name = "shape_dec_next_dc_f16c32_fp16" if arguments.kind == "shape" else "tex_dec_next_dc_f16c32_fp16"
    weight_path = (Path.home() / ".cache/huggingface/hub/models--TencentARC--Pixal3D/snapshots" /
                   PIXAL_REVISION / "ckpts" / (weight_name + ".safetensors"))
    decoder = SLatDecoder(out_channels=7 if arguments.kind == "shape" else 6,
                          pred_subdiv=arguments.kind == "shape", spatial_order="morton",
                          final_upsample_eval_interval=arguments.eval_interval)
    load_weights(decoder, str(weight_path), streaming=True, verbose=False)
    mx.eval(decoder.parameters(), latent, coordinates, guides)
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()
    if arguments.kind == "shape":
        output, output_coordinates, subdivisions = decoder(latent, coordinates, return_subs=True)
    else:
        output, output_coordinates = decoder(latent, coordinates, guide_subs=guides)
        subdivisions = []
    mx.eval(output, output_coordinates, subdivisions)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    peak = mx.get_peak_memory()
    signatures = {"output": array_signature(output), "coordinates": array_signature(output_coordinates),
                  "subdivisions": [array_signature(value) for value in subdivisions]}
    with np.load(arguments.checkpoints / "texture.npz") as archive:
        archived_coordinates_equal = bool(np.array_equal(np.array(output_coordinates), archive["tex_coords"]))
        archived_texture_equal = (bool(np.array_equal(np.array(output * 0.5 + 0.5), archive["tex_out"]))
                                  if arguments.kind == "texture" else None)
    subdivisions_equal = (len(subdivisions) == len(archived_subdivisions) and
                           all(np.array_equal(np.array(actual), expected)
                               for actual, expected in zip(subdivisions, archived_subdivisions))) if arguments.kind == "shape" else None
    reference_match = None
    if arguments.reference_report:
        reference = json.loads(arguments.reference_report.read_text())
        if reference["input_sha256"] != input_sha256 or reference["kind"] != arguments.kind:
            raise ValueError("Reference report must use the same frozen input and decoder kind")
        reference_match = signatures == reference["array_signatures"]
    success = (signatures["output"]["all_finite"] and archived_coordinates_equal and
               subdivisions_equal is not False and archived_texture_equal is not False and reference_match is not False)
    report = {"status": "complete" if success else "mismatch", "kind": arguments.kind,
              "eval_interval": arguments.eval_interval, "decoder_seconds": elapsed,
              "peak_mlx_bytes": peak, "resolution": int(checkpoint["hr_resolution"]),
              "input_sha256": input_sha256, "array_signatures": signatures,
              "archived_coordinates_equal": archived_coordinates_equal,
              "archived_subdivisions_equal": subdivisions_equal, "archived_texture_equal": archived_texture_equal,
              "reference_arrays_bitwise_equal": reference_match,
              "parameter_bytes": sum(value.nbytes for _, value in mlx.utils.tree_flatten(decoder.parameters())),
              "mlx_version": mx.__version__, "device": mx.device_info(), "weights": str(weight_path),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())}
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["shape", "texture"], default="shape")
    parser.add_argument("--checkpoints", type=Path, default=Path("outputs/runtime/e2e1024-native/checkpoints"))
    parser.add_argument("--texture-latent", type=Path, default=Path("outputs/runtime/decoder-investigation/texture-latent.npz"))
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.eval_interval < 0:
        parser.error("Evaluation interval cannot be negative")
    if arguments.report.exists():
        parser.error("Refusing to overwrite a decoder validation report")
    with InferenceRuntime(model_cache_bytes=0, feature_cache_bytes=0):
        return replay(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
