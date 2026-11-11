import cProfile
import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from scripts.investigate_python_overhead import (
    cleanup_events,
    cpu_replays,
    evaluation_events,
    lazy_replays,
    other_inference_processes,
    summarize_profile,
    timed_call,
    texture_profile,
)


def test_timed_call_preserves_result_and_separates_cpu_clocks():
    result, timing = timed_call(lambda: "result")
    assert result == "result"
    assert set(timing) == {"wall_seconds", "main_thread_cpu_seconds", "process_cpu_seconds"}
    assert all(value >= 0 for value in timing.values())


def test_profile_distinguishes_self_and_overlapping_cumulative_times():
    profiler = cProfile.Profile()
    profiler.runcall(lambda: sum(range(100)))
    report = summarize_profile(profiler)
    assert report["total_calls"] > 0
    assert {row["kind"] for row in report["top_self"]} == {"native_or_builtin", "python_frame"}
    assert "Cumulative times overlap" in report["caution"]


def test_cpu_replays_separates_warmup_timed_samples_and_profile(tmp_path):
    calls = []

    def function():
        calls.append(True)
        return (np.arange(4),)

    report = cpu_replays(function, 2, tmp_path / "cpu.prof")
    assert len(calls) == 4
    assert len(report["samples"]) == 2
    assert all(sample["warmup_arrays_equal"] for sample in report["samples"])
    assert report["profile_warmup_arrays_equal"]
    assert report["output_signatures"][0]["shape"] == [4]
    assert (tmp_path / "cpu.prof").exists()


def test_cleanup_events_restore_functions_on_exception():
    from pixalmlx import mesh_cleanup

    original = mesh_cleanup.fill_small_holes
    events = []
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    with pytest.raises(RuntimeError):
        with cleanup_events(events):
            mesh_cleanup.fill_small_holes(vertices, faces, max_hole_perimeter=0)
            raise RuntimeError("test")
    assert mesh_cleanup.fill_small_holes is original
    assert events[0]["function"] == "fill_small_holes"
    assert events[0]["input_faces"] == 1
    assert events[0]["wall_seconds"] >= 0


def test_evaluation_events_restore_functions_and_count_only_scoped_calls():
    original = mx.eval
    events = []
    with pytest.raises(RuntimeError):
        with evaluation_events(mx, events):
            mx.eval(mx.array([1]))
            raise RuntimeError("test")
    assert mx.eval is original
    mx.eval(mx.array([2]))
    assert [event["function"] for event in events] == ["eval"]


def test_lazy_replays_distinguishes_internal_evaluation_from_graph_build(tmp_path):
    source = mx.array([1.0, 2.0])
    mx.eval(source)
    lazy = lazy_replays(lambda: source + 1, 2, tmp_path / "lazy.prof")
    assert all(not sample["explicit_evaluations_inside_call"] for sample in lazy["samples"])
    assert lazy["max_abs_difference_from_warmup"] == 0

    def eager():
        result = source + 1
        mx.eval(result)
        return result

    evaluated = lazy_replays(eager, 2, tmp_path / "evaluated.prof")
    assert all(len(sample["explicit_evaluations_inside_call"]) == 1 for sample in evaluated["samples"])
    assert evaluated["median_total_seconds"] > 0


def test_texture_replay_uses_checkpoint_geometry_resolution(tmp_path, monkeypatch):
    import pixalmlx.checkpoint as checkpoint
    import pixalmlx.texture_bake as texture_bake
    import scripts.investigate_python_overhead as investigation

    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    mesh = {"vertices": vertices, "faces": faces, "hr_resolution": 512}
    texture = {"tex_coords": np.zeros((1, 4), dtype=np.int32), "tex_out": np.zeros((1, 6))}
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda path, stage: mesh if stage == "mesh" else texture)
    monkeypatch.setattr(texture_bake, "uv_unwrap", lambda *arrays: (vertices, faces, np.zeros((3, 2)), np.arange(3)))
    observed = []

    def bake(*arrays, **keywords):
        observed.append((arrays[-1], keywords))
        return np.zeros((2, 2, 4)), np.zeros((2, 2, 3)), "OPAQUE"

    monkeypatch.setattr(texture_bake, "bake_texture", bake)
    monkeypatch.setattr(investigation, "cpu_replays", lambda function, *args: {"outputs": function()})
    report = texture_profile(SimpleNamespace(checkpoints=tmp_path, report=tmp_path / "texture.json",
                                             texture_size=1024, repeats=1))
    assert observed == [(512, {"texture_size": 1024, "backend": "gpu"})]
    assert len(report["outputs"]) == 2
    assert report["uv_unwrap"]["profiled"]


def test_other_inference_processes_excludes_self_monitors_and_shell_commands(monkeypatch):
    import scripts.investigate_python_overhead as investigation

    listing = f"""{os.getpid()} /usr/bin/Python scripts/investigate_python_overhead.py graph
101 /usr/bin/Python -u generate_pixal3d.py --image input.png
102 /usr/bin/Python scripts/measure_generation.py -- .venv/bin/python generate_pixal3d.py
103 /bin/zsh -c python scripts/investigate_sampling.py
104 /usr/bin/Python scripts/investigate_sampling.py time --stage ss
105 /usr/bin/Python -m pixalmlx.server
"""
    monkeypatch.setattr(investigation.subprocess, "check_output", lambda *args, **kwargs: listing)
    assert [process["pid"] for process in other_inference_processes()] == [101, 104]


def test_cli_refuses_overlapping_inference_before_creating_evidence(tmp_path, monkeypatch):
    import sys

    import scripts.investigate_python_overhead as investigation

    report = tmp_path / "new" / "report.json"
    monkeypatch.setattr(sys, "argv", ["investigation", "cleanup", "--report", str(report)])
    monkeypatch.setattr(investigation, "other_inference_processes", lambda: [{"pid": 101}])
    with pytest.raises(SystemExit) as stopped:
        investigation.main()
    assert stopped.value.code == 2
    assert not report.parent.exists()
