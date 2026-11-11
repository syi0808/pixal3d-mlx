from pathlib import Path
import weakref

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from pixalmlx.runtime import BoundedCache, InferenceRuntime, Stage, StageGraph


def test_bounded_lru_cache_and_oversized_bypass():
    cache = BoundedCache(32, max_entries=2)
    cache["first"] = mx.ones((4,))
    cache["second"] = mx.ones((4,))
    cache["first"]
    cache["third"] = mx.ones((4,))
    assert "second" not in cache
    cache["large"] = mx.ones((100,))
    assert "large" not in cache
    assert cache.used_bytes == 32
    cache.clear()
    assert cache.used_bytes == 0


def test_stage_graph_prunes_unused_work_orders_dependencies_and_releases_intermediates():
    class Payload:
        pass

    references = []

    def create():
        payload = Payload()
        references.append(weakref.ref(payload))
        return payload

    def finish(value):
        assert references[0]() is None
        return value + 1

    graph = StageGraph([
        Stage("finish", ("middle",), ("output",), finish),
        Stage("unused", (), ("discarded",), lambda: pytest.fail("Dead graph node ran")),
        Stage("consume", ("payload",), ("middle",), lambda payload: 4),
        Stage("create", (), ("payload",), create),
    ], outputs=("output",))
    assert graph.execute() == (5,)


def test_graph_rejects_cycles_and_missing_inputs():
    graph = StageGraph([Stage("first", ("second",), ("first",), lambda value: value),
                        Stage("second", ("first",), ("second",), lambda value: value)], ("first",))
    with pytest.raises(ValueError, match="Cyclic"):
        graph.compile()
    with pytest.raises(ValueError, match="Missing"):
        StageGraph([], ("missing",)).compile()


def test_graph_materializes_nested_array_outputs_before_releasing_inputs():
    graph = StageGraph([Stage("nested", (), ("nested",), lambda: {"features": mx.ones((2, 4)) * 3})], ("nested",))
    result, = graph.execute()
    np.testing.assert_array_equal(np.array(result["features"]), np.full((2, 4), 3))


def test_resume_forwards_runtime_options_in_process(monkeypatch):
    import generate_pixal3d
    import resume_pixal3d

    captured = []
    monkeypatch.setattr(resume_pixal3d, "main", lambda argv, runtime=None: captured.append((argv, runtime)))
    generate_pixal3d.main(["--image", "image.png", "--resume", "checkpoint", "--naf-threads", "64",
                           "--memory-budget-gb", "8", "--compile-step", "--dense-conditioning"])
    arguments, runtime = captured[0]
    assert arguments[arguments.index("--naf-threads") + 1] == "64"
    assert arguments[arguments.index("--memory-budget-gb") + 1] == "8.0"
    assert "--compile-step" in arguments
    assert "--dense-conditioning" in arguments
    assert runtime is None


def test_runtime_reuses_models_and_restores_global_limits_after_failure(tmp_path):
    original = mx.set_memory_limit(8_000_000_000)
    loads = []
    try:
        with pytest.raises(RuntimeError, match="test failure"):
            with InferenceRuntime(memory_budget_bytes=6_000_000_000, model_cache_bytes=1024) as runtime:
                first = runtime.load_model("tiny", lambda: nn.Linear(4, 4), lambda model: loads.append(1))
                second = runtime.load_model("tiny", lambda: pytest.fail("Unexpected reload"), lambda model: None)
                assert first is second
                with runtime.job(tmp_path / "failed.json"):
                    runtime.advance("tiny")
                    raise RuntimeError("test failure")
        assert loads == [1]
        assert runtime.closed
        assert not runtime.models
        assert (tmp_path / "failed.json").is_file()
        observed = mx.set_memory_limit(8_000_000_000)
        assert observed == 8_000_000_000
    finally:
        mx.set_memory_limit(original)


def test_runtime_reports_memory_guideline_overshoot_without_claiming_a_hard_cap(monkeypatch):
    with InferenceRuntime(memory_budget_bytes=6_000_000_000, model_cache_bytes=0, feature_cache_bytes=0) as runtime:
        monkeypatch.setattr(mx, "get_peak_memory", lambda: 6_500_000_000)
        report = runtime.report()
        assert report["memory_budget_is_hard_limit"] is False
        assert report["memory_budget_exceeded"] is True
        assert report["memory_budget_excess_bytes"] == 500_000_000


def test_persistent_batch_reuses_one_runtime_and_checks_outputs(tmp_path, monkeypatch):
    import generate_pixal3d
    from pixalmlx.batch_inference import BatchJob, BatchRunOptions, run_batch

    seen = []

    def generate(argv, runtime=None):
        seen.append(id(runtime))
        output = Path(argv[argv.index("--output") + 1])
        if argv[argv.index("--seed") + 1] != "2":
            output.write_bytes(b"glb")

    monkeypatch.setattr(generate_pixal3d, "main", generate)
    jobs = [BatchJob(("image.png",), seed, tmp_path / f"{seed}.glb") for seed in (1, 2, 3)]
    report = run_batch(jobs, BatchRunOptions(max_concurrent=3,
                                             repo_root=Path(generate_pixal3d.__file__).parent,
                                             report_path=tmp_path / "batch.json", engine="pixal3d-persistent"))
    assert len(set(seen)) == 1
    assert report.effective_concurrency == 1
    assert [result.returncode for result in report.results] == [0, 1, 0]
    assert report.results[1].failure_phase == "output_validation"
