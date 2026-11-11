import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import threading

from .store import JobStore
from ..cgal_holes import DEFAULT_HOLE_FILL_BACKEND


def generation_arguments(job, directory):
    options = job["options"]
    arguments = ["--image", str(directory / "input.png"), "--output", str(directory / "model.glb"),
                 "--seed", str(options["seed"]), "--resolution", str(options["resolution"]),
                 "--target-faces", str(options["target_faces"]), "--texture-size", str(options["texture_size"]),
                 "--hole-fill-backend", options.get("hole_fill_backend", DEFAULT_HOLE_FILL_BACKEND),
                 "--runtime-report", str(directory / "runtime.json")]
    for name in ("ss_steps", "lr_steps", "hr_steps", "texture_steps"):
        arguments.extend(["--" + name.replace("_", "-"), str(options.get(name, 12))])
    if options["max_faces"]:
        arguments.extend(["--max-faces", str(options["max_faces"])])
    if not options["remove_background"]:
        arguments.append("--no-rembg")
    return arguments


def execute_job(store, job, runtime, generate=None):
    if generate is None:
        from generate_pixal3d import main as generate

    directory = store.directory / "jobs" / job["id"]
    started = time.perf_counter()
    with (directory / "generation.log").open("a", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                generate(generation_arguments(job, directory), runtime=runtime)
                store.stage(job["id"], "validating")
                model = directory / "model.glb"
                normal_report = json.loads((directory / "model.normals.json").read_text())
                digest = hashlib.sha256(model.read_bytes()).hexdigest()
                if normal_report["status"] != "complete" or normal_report["triangles"] < 1:
                    raise RuntimeError("Normal correction did not complete")
                if digest != normal_report["output_sha256"]:
                    raise RuntimeError("Corrected GLB hash does not match its audit report")
                summary = {"triangles": normal_report["triangles"], "bytes": model.stat().st_size,
                           "sha256": digest, "normal_correction": True,
                           "elapsed_seconds": time.perf_counter() - started}
                runtime_path = directory / "runtime.json"
                runtime_report = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
                if runtime_report:
                    summary["peak_mlx_bytes"] = runtime_report["peak_mlx_bytes"]
                (directory / "report.json").write_text(json.dumps(
                    {"summary": summary, "normals": normal_report, "runtime": runtime_report}, indent=2) + "\n")
                store.finish(job["id"], summary=summary)
            except (Exception, SystemExit) as error:
                traceback.print_exc()
                store.finish(job["id"], error=f"{type(error).__name__}: {error}"[:2000])
                if runtime is not None:
                    runtime.trim()


def main():
    parser = argparse.ArgumentParser(description="Single persistent MLX queue worker")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--idle-cache-seconds", type=float, default=300)
    arguments = parser.parse_args()

    def watch_parent():
        while os.getppid() == arguments.parent_pid:
            time.sleep(1)
        os._exit(143)

    threading.Thread(target=watch_parent, name="pixal3d-parent-watchdog", daemon=True).start()
    store = JobStore(arguments.data_dir)
    runtime = None
    active_job = None
    last_active = time.monotonic()
    try:
        while os.getppid() == arguments.parent_pid:
            job = store.claim()
            if job is None:
                if runtime is not None and time.monotonic() - last_active > arguments.idle_cache_seconds:
                    runtime.close()
                    runtime = None
                time.sleep(0.5)
                continue
            active_job = job["id"]
            if runtime is None:
                from pixalmlx.runtime import InferenceRuntime

                class ReportingRuntime(InferenceRuntime):
                    def advance(self, name):
                        store.stage(active_job, name)
                        super().advance(name)

                runtime = ReportingRuntime().__enter__()
            execute_job(store, job, runtime)
            active_job = None
            last_active = time.monotonic()
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
