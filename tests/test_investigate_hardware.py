import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from scripts.investigate_hardware import full_forward, gemm_work, interval_union, measure_case, trace_profile, vm_delta


def test_gemm_work_counts_multiply_and_add_and_minimum_tensor_bytes():
    result = gemm_work(4096, 8192, 1536, 4)
    assert result["flops"] == 103079215104
    assert result["minimum_tensor_bytes"] == 209715200
    assert result["ideal_arithmetic_intensity"] == pytest.approx(491.52)


def test_gemm_work_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError):
        gemm_work(0, 8192, 1536, 4)


def test_vm_delta_distinguishes_swap_usage_from_swap_io():
    counters = dict.fromkeys(("Swapins", "Swapouts", "Compressions", "Decompressions", "Pageins", "Pageouts"), 10)
    before = {"swap_used_bytes": 1024, "vm_stat": {"page_size_bytes": 16384, **counters}}
    after = {"swap_used_bytes": 1024, "vm_stat": {"page_size_bytes": 16384, **counters, "Swapins": 15}}
    result = vm_delta(before, after)
    assert result["swap_used_change_bytes"] == 0
    assert result["system_counter_delta"]["Swapins"] == 5
    assert result["page_size_bytes"] == 16384


def test_measure_case_evaluates_fresh_arrays_and_counts_repeats():
    source = mx.array(np.arange(8, dtype=np.float32))
    calls = []

    def function():
        calls.append(True)
        return source + 1

    report = measure_case(function, 3, {"flops": 8, "streamed_bytes": 64})
    assert len(calls) == 4
    assert len(report["samples"]) == 3
    assert report["median_seconds"] > 0
    assert report["effective_tflops"] > 0
    assert report["effective_stream_gbytes_per_second"] > 0


def test_gpu_intervals_are_clipped_and_overlaps_not_double_counted():
    assert interval_union([(-5, 2), (1, 4), (5, 7), (6, 20), (30, 40)], 0, 10) == 9


def test_gpu_interval_window_must_be_positive():
    with pytest.raises(ValueError):
        interval_union([], 1, 1)


def test_trace_profile_resolves_xml_references_and_reports_execution_not_occupancy(tmp_path):
    path = tmp_path / "intervals.xml"
    path.write_text('''<trace-query-result><node><schema>
        <col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col>
        <col><mnemonic>process</mnemonic></col></schema>
        <row><start-time>0</start-time><duration>2000000000</duration>
        <process id="1" fmt="python (42)"/></row>
        <row><start-time>1000000000</start-time><duration>2000000000</duration><process ref="1"/></row>
        <row><start-time>3000000000</start-time><duration>1000000000</duration>
        <process fmt="WindowServer (1)"/></row>
        <row><start-time>4000000000</start-time><duration>1000000000</duration><process ref="1"/></row>
        </node></trace-query-result>''')
    report = trace_profile(SimpleNamespace(gpu_intervals=path, trace_process="python", window_seconds=5))
    assert report["target_gpu_active_seconds"] == 4
    assert report["target_gpu_active_fraction"] == pytest.approx(0.8)
    assert report["system_gpu_active_fraction"] == 1
    assert "not ALU utilization" in report["scope"]
    with pytest.raises(ValueError, match="Specify one trace process"):
        trace_profile(SimpleNamespace(gpu_intervals=path, trace_process="missing", window_seconds=5))


def test_full_forward_scopes_loading_and_each_forward_without_changing_default_models(tmp_path, monkeypatch):
    from pixalmlx.models.pixal3d_flow import Pixal3DSparseStructureFlowModel
    import scripts.investigate_hardware as hardware
    import scripts.investigate_sampling as sampling

    model = Pixal3DSparseStructureFlowModel(model_channels=32, num_heads=4, num_blocks=1,
                                           mlp_hidden=64, context_channels=16, proj_in_channels=12, resolution=2)
    path = tmp_path / "inputs.npz"
    np.savez(path, **{"noise": np.ones((1, 8, 2, 2, 2), dtype=np.float32),
                     "global": np.zeros((1, 5, 16), dtype=np.float32),
                     "proj": np.zeros((1, 8, 12), dtype=np.float32)})
    path.with_suffix(".json").write_text(json.dumps({"stage": "ss"}))
    counters = dict.fromkeys(("Swapins", "Swapouts", "Compressions", "Decompressions", "Pageins", "Pageouts"), 0)
    monkeypatch.setattr(sampling, "load_model", lambda stage: model)
    monkeypatch.setattr(hardware, "system_usage", lambda: {
        "swap_used_bytes": 0, "vm_stat": {"page_size_bytes": 16384, **counters}})
    monkeypatch.setattr(hardware, "MacProcessUsage", lambda: SimpleNamespace(read=lambda pid: {}))
    report = full_forward(SimpleNamespace(inputs=path, stage="ss", repeats=2))
    assert len(report["forwards"]) == 3
    assert [entry["cold"] for entry in report["forwards"]] == [True, False, False]
    assert report["all_finite"]
    assert report["median_warm_seconds"] > 0
    assert report["model_load_system_delta"]["system_counter_delta"]["Swapins"] == 0
