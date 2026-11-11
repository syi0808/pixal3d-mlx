"""Opt-in frozen-geometry cleanup profiling and polygon/UV budget experiments."""

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import subprocess
import time

import numpy as np

from generate import _cleanup_and_simplify_mesh
from pixalmlx.checkpoint import load_checkpoint
from pixalmlx.mesh_budget import enforce_face_limit
from pixalmlx.mesh_experiments import experimental_cleanup, reference_cleanup
from pixalmlx.texture_bake import uv_unwrap


def signature(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(array)).hexdigest()}


def prepare(arguments):
    import mlx.core as mx

    from scripts.download_models import PIXAL_REVISION
    from pixalmlx.models.shape_slat_decoder import SLatDecoder
    from pixalmlx.mesh_extract import decoder_output_to_mesh
    from pixalmlx.runtime import InferenceRuntime
    from pixalmlx.weight_loader import load_weights

    if arguments.raw_mesh.exists():
        raise ValueError("Refusing to overwrite raw geometry")
    checkpoint = load_checkpoint(str(arguments.checkpoints), "shape_latent")
    weights = (Path.home() / ".cache/huggingface/hub/models--TencentARC--Pixal3D/snapshots" /
               PIXAL_REVISION / "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors")
    with InferenceRuntime(model_cache_bytes=0, feature_cache_bytes=0):
        decoder = SLatDecoder(out_channels=7, pred_subdiv=True, spatial_order="morton", final_upsample_eval_interval=1)
        load_weights(decoder, str(weights), streaming=True, verbose=False)
        output, coordinates = decoder(mx.array(checkpoint["hr_slat"]), mx.array(checkpoint["quant_coords"]))
        mx.eval(output, coordinates)
        vertices, faces = decoder_output_to_mesh(np.array(output), np.array(coordinates),
                                                  resolution=int(checkpoint["hr_resolution"]))
    arguments.raw_mesh.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.raw_mesh, vertices=vertices, faces=faces)
    print(json.dumps({"raw_mesh": str(arguments.raw_mesh), "vertices": len(vertices), "faces": len(faces)}), flush=True)


def cleanup(arguments):
    with np.load(arguments.raw_mesh) as raw:
        vertices, faces = raw["vertices"], raw["faces"]
    profiler = cProfile.Profile() if arguments.profile else None
    started = time.perf_counter()
    with (reference_cleanup() if arguments.reference else experimental_cleanup(arguments.experimental)):
        if profiler:
            profiler.enable()
        vertices, faces = _cleanup_and_simplify_mesh(vertices, faces, target_faces=200000, no_cleanup=False)
        if profiler:
            profiler.disable()
    elapsed = time.perf_counter() - started
    archived = load_checkpoint(str(arguments.checkpoints), "mesh")
    report = {"experimental": arguments.experimental, "profiled": arguments.profile, "cleanup_seconds": elapsed,
              "backend": "reference" if arguments.reference else "vectorized",
              "vertices": len(vertices), "faces": len(faces),
              "archived_vertices_equal": bool(np.array_equal(vertices, archived["vertices"])),
              "archived_faces_equal": bool(np.array_equal(faces, archived["faces"])),
              "vertex_signature": signature(vertices), "face_signature": signature(faces)}
    if profiler:
        profiler.dump_stats(str(arguments.report.with_suffix(".prof")))
        stats = pstats.Stats(profiler)
        report["cleanup_profile"] = [{"function": key[2], "calls": value[1], "self_seconds": value[2], "cumulative_seconds": value[3]}
                                      for key, value in sorted(stats.stats.items(), key=lambda item: -item[1][3])
                                      if key[0].endswith("mesh_cleanup.py")][:15]
    return report


def polygon_budgets(arguments):
    checkpoint = load_checkpoint(str(arguments.checkpoints), "mesh")
    source_vertices, source_faces = checkpoint["vertices"], checkpoint["faces"]
    results = []
    for maximum in [None] + arguments.limits:
        started = time.perf_counter()
        diagnostics = {}
        vertices, faces = (source_vertices, source_faces) if maximum is None else enforce_face_limit(
            source_vertices, source_faces, maximum, diagnostics=diagnostics)
        simplify_seconds = time.perf_counter() - started
        started = time.perf_counter()
        uv_vertices, uv_faces, uvs, mapping = uv_unwrap(vertices, faces)
        uv_seconds = time.perf_counter() - started
        result = {"max_faces": maximum, "faces": len(faces), "uv_faces": len(uv_faces),
                  "vertices": len(vertices), "uv_vertices": len(uv_vertices), "simplify_seconds": simplify_seconds,
                  "uv_seconds": uv_seconds, "budget_diagnostics": diagnostics,
                  "finite_vertices_uv": bool(np.isfinite(uv_vertices).all() and np.isfinite(uvs).all())}
        if maximum is not None:
            started = time.perf_counter()
            soft_vertices, soft_faces = _cleanup_and_simplify_mesh(
                source_vertices, source_faces, target_faces=maximum, no_cleanup=False, log=lambda *args, **kwargs: None)
            result["soft_target_faces"] = len(soft_faces)
            result["soft_target_seconds"] = time.perf_counter() - started
        results.append(result)
        print(json.dumps(result), flush=True)
    return {"source_faces": len(source_faces), "results": results,
            "scope": "post-generation LOD from an already-cleaned mesh; not full generation timings"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "cleanup", "budgets"])
    parser.add_argument("--checkpoints", type=Path, default=Path("outputs/turnip512/checkpoints"))
    parser.add_argument("--raw-mesh", type=Path, default=Path("outputs/runtime/postprocess-investigation/raw-mesh.npz"))
    parser.add_argument("--report", type=Path, default=Path("outputs/runtime/postprocess-investigation/report.json"))
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--experimental", action="store_true")
    backend.add_argument("--reference", action="store_true", help="Replay the original Python topology implementation")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--limits", type=int, nargs="+", default=[5000, 10000, 20000, 50000])
    arguments = parser.parse_args()
    if arguments.mode == "prepare":
        return prepare(arguments)
    if arguments.report.exists():
        parser.error("Refusing to overwrite a postprocess report")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    report = cleanup(arguments) if arguments.mode == "cleanup" else polygon_budgets(arguments)
    report.update({"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())})
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
