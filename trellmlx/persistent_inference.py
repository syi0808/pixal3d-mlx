"""Serial in-process Pixal3D batches with bounded cross-job reuse."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import time

from .runtime import InferenceRuntime


def run_persistent_batch(jobs, options):
    import generate_pixal3d
    from .batch_inference import BatchJobResult, BatchRunReport, SCHEMA

    root = Path(options.repo_root).resolve()
    if root != Path(generate_pixal3d.__file__).resolve().parent:
        raise ValueError("Persistent batches must target this installed Pixal3D checkout")
    if any(len(job.images) != 1 for job in jobs):
        raise ValueError("Pixal3D persistent batches require exactly one image per job")
    results = []
    diagnostics = ["pixal3d_persistent:one_active_gpu_job"]

    def snapshot():
        report = BatchRunReport(schema=SCHEMA, requested_concurrency=options.max_concurrent,
                                effective_concurrency=1 if jobs else 0, diagnostics=diagnostics,
                                repo_root=str(root), results=list(results))
        if options.report_path is not None:
            path = Path(options.report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
            temporary.replace(path)
        return report

    with InferenceRuntime(memory_budget_bytes=int(options.memory_budget_gb * 1e9),
                          model_cache_bytes=int(options.model_cache_gb * 1e9),
                          feature_cache_bytes=int(options.feature_cache_mb * 1024 ** 2)) as runtime:
        for job in jobs:
            output = job.output_path if job.output_path.is_absolute() else root / job.output_path
            output.parent.mkdir(parents=True, exist_ok=True)
            image = Path(job.images[0])
            image = image if image.is_absolute() else root / image
            arguments = ["--image", str(image), "--seed", str(job.seed), "--output", str(output),
                         "--resolution", str(job.resolution), "--max-tokens", str(job.max_tokens),
                         "--target-faces", str(job.target_faces), "--quantize", str(job.quantize),
                         "--memory-budget-gb", str(options.memory_budget_gb),
                         "--model-cache-gb", str(options.model_cache_gb),
                         "--feature-cache-mb", str(options.feature_cache_mb),
                         "--runtime-report", str(output.with_suffix(".runtime.json"))]
            if job.compile:
                arguments.append("--compile-step")
            if job.no_rembg:
                arguments.append("--no-rembg")
            if job.no_cleanup:
                arguments.append("--no-cleanup")
            stdout, stderr = io.StringIO(), io.StringIO()
            before = output.stat().st_mtime_ns if output.exists() else None
            started = time.perf_counter()
            returncode = 0
            failure_phase = None
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    generate_pixal3d.main(arguments, runtime=runtime)
                except (Exception, SystemExit) as error:
                    returncode = 1
                    failure_phase = "generation"
                    print(f"{type(error).__name__}: {error}", file=stderr)
            exists = output.is_file()
            size = output.stat().st_size if exists else None
            if not returncode and (not exists or not size or output.stat().st_mtime_ns == before):
                returncode = 1
                failure_phase = "output_validation"
            results.append(BatchJobResult(seed=job.seed, images=job.images, output_path=str(output),
                                           command=("in-process:generate_pixal3d", *arguments),
                                           returncode=returncode, elapsed_seconds=time.perf_counter() - started,
                                           output_exists=exists, output_size_bytes=size,
                                           stdout=stdout.getvalue(), stderr=stderr.getvalue(),
                                           failure_phase=failure_phase))
            snapshot()
    return snapshot()
