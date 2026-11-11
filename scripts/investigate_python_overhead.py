"""Frozen-input Python/CPU attribution without changing production inference."""

import argparse
from contextlib import contextmanager
import cProfile
from datetime import datetime, timezone
from functools import wraps
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pstats
import statistics
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def other_inference_processes():
    processes = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True).splitlines()
    active = []
    for process in processes:
        identifier, command = process.strip().split(maxsplit=1)
        executable = Path(command.split()[0]).name.lower()
        if int(identifier) != os.getpid() and executable.startswith("python") and any(
            name in command for name in ("generate_pixal3d.py", "scripts/investigate_", "scripts/benchmark_")
        ) and "scripts/measure_generation.py" not in command:
            active.append({"pid": int(identifier), "command": command})
    return active


def source_snapshot():
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"],
        cwd=ROOT, text=True,
    ).splitlines()
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in paths if (ROOT / path).is_file()}


def timed_call(function):
    wall_started = time.perf_counter()
    thread_started = time.thread_time()
    process_started = time.process_time()
    result = function()
    return result, {"wall_seconds": time.perf_counter() - wall_started,
                    "main_thread_cpu_seconds": time.thread_time() - thread_started,
                    "process_cpu_seconds": time.process_time() - process_started}


def summarize_profile(profiler):
    stats = pstats.Stats(profiler)
    rows = [{"file": key[0], "line": key[1], "function": key[2],
             "primitive_calls": value[0], "calls": value[1],
             "self_seconds": value[2], "cumulative_seconds": value[3],
             "kind": "native_or_builtin" if key[0] == "~" else "python_frame"}
            for key, value in stats.stats.items()]
    return {"total_seconds": stats.total_tt, "total_calls": stats.total_calls,
            "top_self": sorted(rows, key=lambda row: -row["self_seconds"])[:40],
            "top_cumulative": sorted(rows, key=lambda row: -row["cumulative_seconds"])[:40],
            "caution": "Diagnostic, not a timing baseline. Python-frame self time can include ndarray operators; "
                       "Python filenames can also identify compiled Cython functions. Native/builtin time "
                       "can include GPU waits. Cumulative times overlap."}


def array_signature(array):
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array).hexdigest()}


def cpu_replays(function, repeats, profile_path):
    expected = function()
    samples = []
    for _ in range(repeats):
        result, timing = timed_call(function)
        timing["warmup_arrays_equal"] = all(np.array_equal(left, right)
                                            for left, right in zip(expected, result, strict=True))
        samples.append(timing)
    profiler = cProfile.Profile()
    result = profiler.runcall(function)
    profiler.dump_stats(str(profile_path))
    return {"samples": samples, "median_seconds": statistics.median(row["wall_seconds"] for row in samples),
            "output_signatures": [array_signature(array) for array in result],
            "profile_warmup_arrays_equal": all(np.array_equal(left, right)
                                              for left, right in zip(expected, result, strict=True)),
            "profile": summarize_profile(profiler)}


@contextmanager
def cleanup_events(events):
    from pixalmlx import mesh_cleanup

    names = ("remove_duplicate_faces", "remove_small_components", "keep_largest_component",
             "repair_non_manifold_edges", "fill_small_holes", "fix_normals",
             "orient_faces_by_adjacency", "_face_connected_components")
    originals = {name: getattr(mesh_cleanup, name) for name in names}
    stack = []

    def instrument(name, function):
        @wraps(function)
        def wrapped(*arguments, **keywords):
            event = {"function": name, "depth": len(stack)}
            if name == "_face_connected_components":
                event["input_faces"] = len(arguments[0])
            else:
                event.update(input_vertices=len(arguments[0]), input_faces=len(arguments[1]))
            events.append(event)
            stack.append(event)
            try:
                result, timing = timed_call(lambda: function(*arguments, **keywords))
                event.update(timing)
                return result
            finally:
                stack.pop()

        return wrapped

    try:
        for name, original in originals.items():
            setattr(mesh_cleanup, name, instrument(name, original))
        yield
    finally:
        for name, original in originals.items():
            setattr(mesh_cleanup, name, original)


def cleanup_profile(arguments):
    from generate import _cleanup_and_simplify_mesh
    from pixalmlx.checkpoint import load_checkpoint
    from pixalmlx.mesh_experiments import reference_cleanup

    with np.load(arguments.raw_mesh) as source:
        vertices, faces = source["vertices"], source["faces"]
    events = []
    event_samples = []

    def function():
        events.clear()
        with cleanup_events(events):
            result = _cleanup_and_simplify_mesh(vertices, faces, target_faces=200000, no_cleanup=False,
                                                log=lambda *values, **keywords: None)
        event_samples.append(list(events))
        return result

    if arguments.reference:
        with reference_cleanup():
            report = cpu_replays(function, arguments.repeats, arguments.report.with_suffix(".prof"))
    else:
        report = cpu_replays(function, arguments.repeats, arguments.report.with_suffix(".prof"))
    archived = load_checkpoint(str(arguments.checkpoints), "mesh")
    report.update(backend="reference" if arguments.reference else "current",
                  input_vertices=len(vertices), input_faces=len(faces),
                  input_sha256=hashlib.sha256(arguments.raw_mesh.read_bytes()).hexdigest(),
                  stage_samples=event_samples[1:-1],
                  archived_arrays_equal=report["output_signatures"] == [
                      array_signature(archived[name]) for name in ("vertices", "faces")])
    return report


@contextmanager
def evaluation_events(mx, events):
    originals = {name: getattr(mx, name) for name in ("eval", "async_eval", "synchronize")}

    def instrument(name, function):
        @wraps(function)
        def wrapped(*arguments, **keywords):
            result, timing = timed_call(lambda: function(*arguments, **keywords))
            events.append({"function": name, **timing})
            return result

        return wrapped

    try:
        for name, original in originals.items():
            setattr(mx, name, instrument(name, original))
        yield
    finally:
        for name, original in originals.items():
            setattr(mx, name, original)


def lazy_replays(function, repeats, profile_path):
    import mlx.core as mx

    expected = function()
    mx.eval(expected)
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        events = []
        with evaluation_events(mx, events):
            result, call_timing = timed_call(function)
        _, evaluation_timing = timed_call(lambda: (mx.eval(result), mx.synchronize()))
        samples.append({"call": call_timing, "evaluate_and_sync": evaluation_timing,
                        "explicit_evaluations_inside_call": events})
    profiler = cProfile.Profile()
    result = profiler.runcall(function)
    profiler.dump_stats(str(profile_path))
    mx.eval(result)
    mx.synchronize()
    return {"samples": samples,
            "median_call_seconds": statistics.median(row["call"]["wall_seconds"] for row in samples),
            "median_evaluate_seconds": statistics.median(row["evaluate_and_sync"]["wall_seconds"] for row in samples),
            "median_total_seconds": statistics.median(
                row["call"]["wall_seconds"] + row["evaluate_and_sync"]["wall_seconds"] for row in samples),
            "max_abs_difference_from_warmup": float(mx.max(mx.abs(result - expected)).item()),
            "profile_call_only": summarize_profile(profiler)}


def graph_profile(arguments):
    import mlx.core as mx

    from pixalmlx.transformer_runtime import configure_selective_compile
    from scripts.investigate_hardware import load_first_block

    if json.loads(arguments.inputs.with_suffix(".json").read_text())["stage"] != arguments.stage:
        raise ValueError("Frozen input stage does not match requested model")
    model, features, modulation, positive, phases, contexts = load_first_block(arguments.stage, arguments.inputs)
    block = model.blocks[0]
    measurements = {}
    for name, chunk, compiled in (("reference_chunked", 2048, False), ("current_chunked", 2048, True),
                                   ("reference_lazy", 0, False), ("current_lazy", 0, True)):
        block.mlp.chunk_size = chunk
        configure_selective_compile(model, compiled)
        measurements[name] = lazy_replays(
            lambda: block(features, modulation, positive, rope_phases=phases), arguments.repeats,
            arguments.report.with_name(f"{arguments.report.stem}-{name}.prof"))
    for context in contexts:
        context.clear_invariants()
    return {"measurements": measurements, "tokens": features.shape[0], "channels": features.shape[1],
            "stage": arguments.stage, "mlx_version": mx.__version__,
            "input_sha256": hashlib.sha256(arguments.inputs.read_bytes()).hexdigest(),
            "scope": "One warm pretrained conditional block. Lazy call time includes Python AND native graph "
                     "construction, not pure interpreter time. Chunked calls include internal GPU evaluation. "
                     "No sampler/end-to-end extrapolation; no full-model/cache contention."}


def texture_profile(arguments):
    from pixalmlx.checkpoint import load_checkpoint
    from pixalmlx.texture_bake import bake_texture, uv_unwrap

    mesh = load_checkpoint(str(arguments.checkpoints), "mesh")
    texture = load_checkpoint(str(arguments.checkpoints), "texture")
    profiler = cProfile.Profile()
    uv, uv_timing = timed_call(lambda: profiler.runcall(uv_unwrap, mesh["vertices"], mesh["faces"]))
    profiler.dump_stats(str(arguments.report.with_name(f"{arguments.report.stem}-uv.prof")))
    uv_profile = summarize_profile(profiler)
    uv_timing["cprofile_accounted_wall_fraction"] = uv_profile["total_seconds"] / uv_timing["wall_seconds"]
    vertices, faces, uvs, mapping = uv

    def function():
        return bake_texture(vertices, faces, uvs, mapping, texture["tex_coords"][:, 1:4], texture["tex_out"],
                            int(mesh["hr_resolution"]), texture_size=arguments.texture_size, backend="gpu")[:2]

    report = cpu_replays(function, arguments.repeats, arguments.report.with_suffix(".prof"))
    report.update(uv_unwrap={**uv_timing, "profiled": True, "profile": uv_profile},
                  faces=len(faces), texture_size=arguments.texture_size,
                  input_signatures={name: array_signature(array) for name, array in {
                      "vertices": mesh["vertices"], "faces": mesh["faces"],
                      "tex_coords": texture["tex_coords"], "tex_out": texture["tex_out"]}.items()},
                  scope="Frozen generated mesh/voxel attributes. UV unwrap excluded from bake replay timing. "
                        "Bake includes GPU execution and native NumPy/OpenCV; not all time is Python.")
    return report


def extraction_profile(arguments):
    import gc

    import mlx.core as mx

    from pixalmlx.checkpoint import load_checkpoint
    from pixalmlx.mesh_extract import decoder_output_to_mesh
    from pixalmlx.models.shape_slat_decoder import SLatDecoder
    from pixalmlx.runtime import InferenceRuntime
    from pixalmlx.weight_loader import load_weights
    from scripts.download_models import PIXAL_REVISION

    checkpoint = load_checkpoint(str(arguments.checkpoints), "shape_latent")
    weights = (Path.home() / ".cache/huggingface/hub/models--TencentARC--Pixal3D/snapshots" /
               PIXAL_REVISION / "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors")
    started = time.perf_counter()
    with InferenceRuntime(model_cache_bytes=0, feature_cache_bytes=0):
        decoder = SLatDecoder(out_channels=7, pred_subdiv=True, spatial_order="morton",
                              final_upsample_eval_interval=1)
        load_weights(decoder, str(weights), streaming=True, verbose=False)
        output, coordinates = decoder(mx.array(checkpoint["hr_slat"]), mx.array(checkpoint["quant_coords"]))
        mx.eval(output, coordinates)
        features, coordinates_np = np.array(output), np.array(coordinates)
        del decoder, output, coordinates
        gc.collect()
        mx.clear_cache()
    preparation_seconds = time.perf_counter() - started
    report = cpu_replays(lambda: decoder_output_to_mesh(features, coordinates_np,
                                                       resolution=int(checkpoint["hr_resolution"])),
                         arguments.repeats, arguments.report.with_suffix(".prof"))
    with np.load(arguments.raw_mesh) as archived:
        report["archived_arrays_equal"] = report["output_signatures"] == [
            array_signature(archived[name]) for name in ("vertices", "faces")]
    report.update(decoder_preparation_seconds_excluded=preparation_seconds, voxels=len(features),
                  input_signatures=[array_signature(features), array_signature(coordinates_np)],
                  scope="Actual frozen shape-latent replay. Decoder loading/execution excluded from CPU "
                        "extraction time. Dict lookups and vectorized NumPy are profiled together.")
    return report


def normals_profile(arguments):
    from pixalmlx.glb_normals import accessor_array, read_glb
    from pixalmlx.mesh_normals import smooth_corner_normals

    document, binary = read_glb(arguments.glb)
    primitives = [primitive for mesh in document["meshes"] for primitive in mesh["primitives"]]
    if len(primitives) != 1:
        raise ValueError("Normal replay expects exactly one mesh primitive")
    primitive = primitives[0]
    if primitive.get("mode", 4) != 4:
        raise ValueError("Normal replay expects a triangle primitive")
    vertices = accessor_array(document, binary, primitive["attributes"]["POSITION"])
    faces = accessor_array(document, binary, primitive["indices"]).reshape(-1, 3)
    report = cpu_replays(lambda: (smooth_corner_normals(vertices, faces)[0],),
                         arguments.repeats, arguments.report.with_suffix(".prof"))
    report.update(vertices=len(vertices), faces=len(faces),
                  input_sha256=hashlib.sha256(arguments.glb.read_bytes()).hexdigest(),
                  scope="Corner-normal calculation on an existing generated GLB. GLB parsing, attribute "
                        "repacking, image checks and file writing excluded. Geometry is not modified.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cleanup", "graph", "texture", "extract", "normals"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--raw-mesh", type=Path, default=Path("outputs/runtime/postprocess-investigation/raw-mesh.npz"))
    parser.add_argument("--checkpoints", type=Path, default=Path("outputs/turnip512/checkpoints"))
    parser.add_argument("--inputs", type=Path, default=Path("outputs/runtime/sampling-investigation/ss-inputs.npz"))
    parser.add_argument("--stage", choices=("ss", "hr"), default="ss")
    parser.add_argument("--glb", type=Path, default=Path("outputs/turnip512/turnip.glb"))
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--allow-concurrent", action="store_true",
                        help="Allow diagnostic runs alongside detected inference processes; timings may be contaminated")
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.texture_size < 1:
        parser.error("Repeats and texture size must be positive")
    if arguments.report.exists() or list(arguments.report.parent.glob(f"{arguments.report.stem}*.prof")):
        parser.error("Refusing to overwrite investigation evidence")
    concurrent = other_inference_processes()
    if concurrent and not arguments.allow_concurrent:
        parser.error(f"Other inference processes are active: {[row['pid'] for row in concurrent]}")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    before = source_snapshot()
    report = {"mode": arguments.mode, "started_utc": datetime.now(timezone.utc).isoformat(),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "working_tree_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
              "other_inference_processes_before": concurrent,
              "source_sha256": before,
              "versions": {name: importlib.metadata.version(name) for name in ("mlx", "numpy", "scipy", "trimesh", "networkx")}}
    modes = {"cleanup": cleanup_profile, "graph": graph_profile,
             "texture": texture_profile, "extract": extraction_profile, "normals": normals_profile}
    report.update(modes[arguments.mode](arguments))
    after = source_snapshot()
    report["changed_source_paths_during_run"] = sorted(path for path in before.keys() | after.keys()
                                                       if before.get(path) != after.get(path))
    report["other_inference_processes_after"] = other_inference_processes()
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(arguments.report), "mode": arguments.mode,
                      "median_seconds": report.get("median_seconds"),
                      "changed_source_paths_during_run": report["changed_source_paths_during_run"]}), flush=True)


if __name__ == "__main__":
    main()
