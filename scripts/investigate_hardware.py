"""Empirical roofline and frozen-input operator diagnostics; no runtime changes."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np

from pixalmlx.conditioning import ZeroConditioning, prepare_sampling
from pixalmlx.models.pixal3d_flow import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel
from pixalmlx.models.sparse_structure_flow import _layernorm_noaffine
from pixalmlx.modules.attention import scaled_dot_product_attention
from pixalmlx.modules.rope import apply_rope
from pixalmlx.runtime import InferenceRuntime
from pixalmlx.weight_loader import load_weights
from scripts.download_models import PIXAL_REVISION
from scripts.measure_generation import MacProcessUsage, system_usage


def gemm_work(rows, columns, inner, itemsize):
    if min(rows, columns, inner, itemsize) <= 0:
        raise ValueError("GEMM dimensions and item size must be positive")
    operations = 2 * rows * columns * inner
    minimum_bytes = (rows * inner + inner * columns + rows * columns) * itemsize
    return {"flops": operations, "minimum_tensor_bytes": minimum_bytes,
            "ideal_arithmetic_intensity": operations / minimum_bytes}


def vm_delta(before, after):
    page_size = after["vm_stat"]["page_size_bytes"]
    return {"swap_used_change_bytes": after["swap_used_bytes"] - before["swap_used_bytes"],
            "page_size_bytes": page_size,
            "system_counter_delta": {
                name: after["vm_stat"][name] - before["vm_stat"][name]
                for name in ("Swapins", "Swapouts", "Compressions", "Decompressions", "Pageins", "Pageouts")}}


def interval_union(intervals, start, end):
    if end <= start:
        raise ValueError("Interval window must be positive")
    clipped = sorted((max(left, start), min(right, end)) for left, right in intervals if right > start and left < end)
    total = 0
    previous_end = start
    for left, right in clipped:
        total += max(0, right - max(left, previous_end))
        previous_end = max(previous_end, right)
    return total


def trace_profile(arguments):
    if arguments.gpu_intervals is None:
        raise ValueError("Trace analysis requires --gpu-intervals exported by xctrace")
    root = ET.parse(arguments.gpu_intervals).getroot()
    references = {element.attrib["id"]: element for element in root.iter() if "id" in element.attrib}

    def resolve(element):
        return references[element.attrib["ref"]] if "ref" in element.attrib else element

    columns = [column.findtext("mnemonic") for column in root.findall(".//schema/col")]
    intervals = []
    for row in root.findall(".//row"):
        fields = dict(zip(columns, [resolve(element) for element in row]))
        start = int(fields["start"].text)
        end = start + int(fields["duration"].text)
        intervals.append((start, end, fields["process"].attrib.get("fmt", "")))
    matches = {process for _, _, process in intervals if arguments.trace_process in process}
    if len(matches) != 1:
        raise ValueError(f"Specify one trace process, found {sorted(matches)}")
    process = matches.pop()
    target = [(start, end) for start, end, name in intervals if name == process]
    end = max(right for _, right in target)
    start = max(min(left for left, _ in target), end - int(arguments.window_seconds * 1e9))
    duration = end - start
    target_active = interval_union(target, start, end)
    system_active = interval_union([(left, right) for left, right, _ in intervals], start, end)
    return {"measurements": {}, "trace_process": process, "window_start_seconds": start / 1e9,
            "window_end_seconds": end / 1e9, "target_gpu_active_seconds": target_active / 1e9,
            "target_gpu_active_fraction": target_active / duration, "system_gpu_active_seconds": system_active / 1e9,
            "system_gpu_active_fraction": system_active / duration,
            "xml_sha256": hashlib.sha256(arguments.gpu_intervals.read_bytes()).hexdigest(),
            "scope": "Union of GPU execution intervals, not ALU utilization, occupancy or DRAM bandwidth"}


def measure_case(function, repeats, work=None):
    mx.eval(function())
    mx.synchronize()
    mx.reset_peak_memory()
    timings = []
    for iteration in range(repeats):
        started = time.perf_counter()
        cpu_started = time.process_time()
        output = function()
        built = time.perf_counter()
        mx.eval(output)
        mx.synchronize()
        timings.append({"wall_seconds": time.perf_counter() - started,
                        "call_seconds": built - started,
                        "process_cpu_seconds": time.process_time() - cpu_started})
        del output
    median = statistics.median(item["wall_seconds"] for item in timings)
    result = {"median_seconds": median, "samples": timings, "peak_mlx_bytes": mx.get_peak_memory(),
              "call_timing_note": "Includes internal evaluations, if any; not a pure graph-construction timer"}
    if work:
        result.update(work)
        if "flops" in work:
            result["effective_tflops"] = work["flops"] / median / 1e12
        if "minimum_tensor_bytes" in work:
            result["minimum_tensor_gbytes_per_second"] = work["minimum_tensor_bytes"] / median / 1e9
        if "streamed_bytes" in work:
            result["effective_stream_gbytes_per_second"] = work["streamed_bytes"] / median / 1e9
    return result


def roofline(arguments):
    measurements = {}
    copy_kernel = mx.fast.metal_kernel(
        name="pixal_diagnostic_copy", input_names=["source"], output_names=["destination"],
        source="uint index = thread_position_in_grid.x; destination[index] = source[index];")
    for mebibytes in (64, 256):
        elements = mebibytes * 1024 ** 2 // 4
        source = mx.random.uniform(shape=(elements,))
        other = mx.random.uniform(shape=(elements,))
        mx.eval(source, other)

        def copy():
            return copy_kernel(inputs=[source], grid=(elements, 1, 1), threadgroup=(256, 1, 1),
                               output_shapes=[source.shape], output_dtypes=[source.dtype])[0]

        measurements[f"copy_{mebibytes}mib"] = measure_case(
            copy, arguments.repeats, {"streamed_bytes": 2 * source.nbytes})
        measurements[f"add_{mebibytes}mib"] = measure_case(
            lambda: source + other, arguments.repeats, {"streamed_bytes": 3 * source.nbytes})
        del source, other
        gc.collect()
        mx.clear_cache()
    for dtype in (mx.float32, mx.float16):
        for rows in (1, 128, 1024, 2048, 4096):
            features = mx.random.normal((rows, 1536)).astype(dtype)
            weight = mx.random.normal((8192, 1536)).astype(dtype)
            mx.eval(features, weight)
            work = {**gemm_work(rows, 8192, 1536, features.itemsize),
                    "shape_mnk": [rows, 8192, 1536], "dtype": str(dtype)}
            measurements[f"ffn_up_{str(dtype).split('.')[-1]}_m{rows}"] = measure_case(
                lambda: features @ weight.T, arguments.repeats, work)
            del features, weight
            gc.collect()
            mx.clear_cache()
    for dtype in (mx.float32, mx.float16):
        for tokens in (1024, 2185, 4096, 10784):
            queries = mx.random.normal((1, 12, tokens, 128)).astype(dtype)
            keys = mx.random.normal(queries.shape).astype(dtype)
            values = mx.random.normal(queries.shape).astype(dtype)
            mx.eval(queries, keys, values)
            measurements[f"sdpa_{str(dtype).split('.')[-1]}_n{tokens}"] = measure_case(
                lambda: scaled_dot_product_attention(queries, keys, values), arguments.repeats,
                {"flops": 4 * tokens ** 2 * 1536, "dtype": str(dtype), "tokens": tokens,
                 "flops_note": "QK and PV only; excludes softmax and layout copies"})
            del queries, keys, values
            gc.collect()
            mx.clear_cache()
    tiny = mx.array([1.0])
    mx.eval(tiny)
    measurements["one_element_add_and_sync"] = measure_case(lambda: tiny + 1, max(100, arguments.repeats))
    measurements["already_evaluated_array_and_sync"] = measure_case(lambda: tiny, max(100, arguments.repeats))
    return {"measurements": measurements,
            "scope": "Synthetic warm resident tensors; read+write stream bytes and GEMM minimum tensor bytes, not DRAM counters"}


def load_first_block(stage, inputs):
    if stage == "ss":
        model = Pixal3DSparseStructureFlowModel(num_blocks=1)
        checkpoint_name = "ss_flow_img_dit_1_3B_64_bf16"
    else:
        model = Pixal3DSLatFlowModel(num_blocks=1, proj_in_channels=2048)
        checkpoint_name = "slat_flow_img2shape_dit_1_3B_1024_bf16"
    checkpoint = (Path.home() / ".cache/huggingface/hub/models--TencentARC--Pixal3D/snapshots" /
                  PIXAL_REVISION / "ckpts" / (checkpoint_name + ".safetensors"))
    load_weights(model, str(checkpoint), streaming=True, verbose=False,
                 target_dtype=mx.float32 if stage == "ss" else None)
    with np.load(inputs) as frozen:
        values = {name: mx.array(frozen[name]) for name in frozen.files}
    context = {name: values[name] for name in ("global", "proj")}
    negative = ZeroConditioning({name: mx.zeros_like(value) for name, value in context.items()})
    kwargs = {"coords": values["coords"]} if "coords" in values else {}
    positive, negative, kwargs, contexts = prepare_sampling(model, values["noise"], context, negative, kwargs,
                                                           128 * 1024 ** 2)
    noise = values["noise"]
    features = model.input_layer(noise.reshape(1, 8, -1).transpose(0, 2, 1).reshape(-1, 8)
                                 if stage == "ss" else noise)
    modulation = model.adaLN_modulation(model.t_embedder(mx.array([1000.0])))[0]
    mx.eval(model.parameters(), values, features, modulation, kwargs)
    return model, features, modulation, positive, kwargs["rope_phases"], contexts


def block_profile(arguments):
    metadata = json.loads(arguments.inputs.with_suffix(".json").read_text())
    if metadata["stage"] != arguments.stage:
        raise ValueError("Frozen input stage does not match requested model")
    model, features, modulation, positive, phases, contexts = load_first_block(arguments.stage, arguments.inputs)
    block = model.blocks[0]
    attention = block.self_attn
    tokens, channels = features.shape
    heads, head_dim = attention.num_heads, attention.head_dim
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mx.split(modulation + block.modulation, 6)
    attention_input = _layernorm_noaffine(features) * (1 + scale_attn) + shift_attn
    projected = attention.to_qkv(attention_input).reshape(tokens, 3, heads, head_dim)
    queries, keys, values = projected[:, 0], projected[:, 1], projected[:, 2]
    normalized_queries, normalized_keys = attention.q_rms_norm(queries), attention.k_rms_norm(keys)
    rotated_queries, rotated_keys = apply_rope(normalized_queries, phases), apply_rope(normalized_keys, phases)
    query_layout = rotated_queries[None].transpose(0, 2, 1, 3)
    key_layout = rotated_keys[None].transpose(0, 2, 1, 3)
    value_layout = values[None].transpose(0, 2, 1, 3)
    attended = scaled_dot_product_attention(query_layout, key_layout, value_layout)
    merged = attended.transpose(0, 2, 1, 3).reshape(tokens, channels)
    attention_output = attention.to_out(merged)
    residual = features + attention_output * gate_attn
    cross_output = block.cross_attn(block.norm2(residual), positive)
    cross_residual = residual + cross_output
    ffn_input = _layernorm_noaffine(cross_residual) * (1 + scale_mlp) + shift_mlp
    ffn_hidden = block.mlp.mlp_0(ffn_input)
    ffn_activated = nn.gelu_approx(ffn_hidden)
    mx.eval(attention_input, projected, normalized_queries, normalized_keys, rotated_queries, rotated_keys,
            attended, merged, attention_output, residual, cross_output, cross_residual, ffn_input,
            ffn_hidden, ffn_activated)

    def prepare_qk(query_input, key_input, rope_phases):
        return (apply_rope(attention.q_rms_norm(query_input), rope_phases),
                apply_rope(attention.k_rms_norm(key_input), rope_phases))

    compiled_prepare_qk = mx.compile(prepare_qk)
    compiled_queries, compiled_keys = compiled_prepare_qk(queries, keys, phases)
    preparation_error = float(mx.maximum(mx.max(mx.abs(compiled_queries - rotated_queries)),
                                         mx.max(mx.abs(compiled_keys - rotated_keys))).item())
    functions = {
        "qkv_linear": lambda: attention.to_qkv(attention_input),
        "qk_norm": lambda: (attention.q_rms_norm(queries), attention.k_rms_norm(keys)),
        "qk_rope": lambda: (apply_rope(normalized_queries, phases), apply_rope(normalized_keys, phases)),
        "qk_norm_rope": lambda: prepare_qk(queries, keys, phases),
        "qk_norm_rope_compiled": lambda: compiled_prepare_qk(queries, keys, phases),
        "sdpa": lambda: scaled_dot_product_attention(query_layout, key_layout, value_layout),
        "attention_out_linear": lambda: attention.to_out(merged),
        "self_attention": lambda: attention(attention_input, rope_phases=phases),
        "cross_attention_cached": lambda: block.cross_attn(block.norm2(residual), positive),
        "ffn_up_linear": lambda: block.mlp.mlp_0(ffn_input),
        "ffn_gelu": lambda: nn.gelu_approx(ffn_hidden),
        "ffn_down_linear": lambda: block.mlp.mlp_2(ffn_activated),
        "ffn_unchunked": lambda: block.mlp(ffn_input),
        "whole_block_unchunked": lambda: block(features, modulation, positive, rope_phases=phases),
    }
    work = {
        "qkv_linear": gemm_work(tokens, 3 * channels, channels, attention_input.itemsize),
        "attention_out_linear": gemm_work(tokens, channels, channels, merged.itemsize),
        "ffn_up_linear": gemm_work(tokens, ffn_hidden.shape[-1], channels, ffn_input.itemsize),
        "ffn_down_linear": gemm_work(tokens, channels, ffn_hidden.shape[-1], ffn_activated.itemsize),
        "sdpa": {"flops": 4 * tokens ** 2 * channels,
                 "flops_note": "QK and PV only; excludes softmax, tiling rereads and temporary tensors"},
    }
    measurements = {name: measure_case(function, arguments.repeats, work.get(name))
                    for name, function in functions.items()}
    block.mlp.chunk_size = 2048
    measurements["ffn_chunked_2048"] = measure_case(lambda: block.mlp(ffn_input), arguments.repeats)
    measurements["whole_block_chunked_2048"] = measure_case(
        lambda: block(features, modulation, positive, rope_phases=phases), arguments.repeats)
    output = block(features, modulation, positive, rope_phases=phases)
    finite = bool(mx.all(mx.isfinite(output)).item())
    if arguments.capture:
        mx.synchronize()
        mx.metal.start_capture(str(arguments.capture.resolve()))
        try:
            mx.eval(block(features, modulation, positive, rope_phases=phases))
            mx.synchronize()
        finally:
            mx.metal.stop_capture()
    if arguments.trace_seconds:
        print("Running warmed block for external trace", flush=True)
        deadline = time.monotonic() + arguments.trace_seconds
        while time.monotonic() < deadline:
            mx.eval(block(features, modulation, positive, rope_phases=phases))
            mx.synchronize()
    for context in contexts:
        context.clear_invariants()
    return {"measurements": measurements, "stage": arguments.stage, "tokens": tokens, "channels": channels,
            "heads": heads, "head_dim": head_dim, "all_finite": finite,
            "activation_dtype": str(features.dtype), "qkv_dtype": str(projected.dtype),
            "sdpa_input_dtypes": [str(value.dtype) for value in (query_layout, key_layout, value_layout)],
            "ffn_weight_dtype": str(block.mlp.mlp_0.weight.dtype), "ffn_hidden_dtype": str(ffn_hidden.dtype),
            "compiled_qk_max_absolute_error": preparation_error,
            "parameter_bytes": sum(value.nbytes for _, value in mlx.utils.tree_flatten(model.parameters())),
            "inputs_sha256": hashlib.sha256(arguments.inputs.read_bytes()).hexdigest(),
            "scope": "Real first conditional block only; other 29 blocks not loaded; prepared fixed inputs; separate evaluation boundaries",
            "capture": str(arguments.capture) if arguments.capture else None}


def full_forward(arguments):
    from pixalmlx.transformer_experiments import VARIANTS, configure_transformer
    from scripts.investigate_sampling import load_model

    metadata = json.loads(arguments.inputs.with_suffix(".json").read_text())
    if metadata["stage"] != arguments.stage:
        raise ValueError("Frozen input stage does not match requested model")
    usage_reader = MacProcessUsage()
    before_load = system_usage()
    started = time.perf_counter()
    model = configure_transformer(load_model(arguments.stage), VARIANTS["reference"])
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - started
    after_load = system_usage()
    with np.load(arguments.inputs) as frozen:
        values = {name: mx.array(frozen[name]) for name in frozen.files}
    context = {name: values[name] for name in ("global", "proj")}
    negative = ZeroConditioning({name: mx.zeros_like(value) for name, value in context.items()})
    kwargs = {"coords": values["coords"]} if "coords" in values else {}
    positive, negative, kwargs, contexts = prepare_sampling(model, values["noise"], context, negative, kwargs,
                                                           128 * 1024 ** 2)
    timestep = mx.array([1000.0])
    kwargs["modulation"] = model.adaLN_modulation(model.t_embedder(timestep))
    mx.eval(values, kwargs)
    mx.clear_cache()
    mx.reset_peak_memory()
    forwards = []
    for iteration in range(arguments.repeats + 1):
        before = system_usage()
        process_before = usage_reader.read(os.getpid())
        started = time.perf_counter()
        output = model(values["noise"], timestep, positive, **kwargs)
        mx.eval(output)
        mx.synchronize()
        seconds = time.perf_counter() - started
        after = system_usage()
        process_after = usage_reader.read(os.getpid())
        forwards.append({"seconds": seconds, "cold": iteration == 0, "system_before": before,
                         "system_after": after, "system_delta": vm_delta(before, after),
                         "process_before": process_before, "process_after": process_after})
        print(f"Forward {iteration}: {seconds:.3f} s; {forwards[-1]['system_delta']}", flush=True)
    for context in contexts:
        context.clear_invariants()
    return {"measurements": {}, "stage": arguments.stage, "model_load_seconds": load_seconds,
            "model_load_system_delta": vm_delta(before_load, after_load),
            "forwards": forwards, "median_warm_seconds": statistics.median(item["seconds"] for item in forwards[1:]),
            "peak_mlx_bytes": mx.get_peak_memory(),
            "parameter_bytes": sum(value.nbytes for _, value in mlx.utils.tree_flatten(model.parameters())),
            "inputs_sha256": hashlib.sha256(arguments.inputs.read_bytes()).hexdigest(),
            "all_finite": bool(mx.all(mx.isfinite(output)).item()),
            "scope": "Full resident model; fixed first conditional timestep; not full CFG sampling or image-to-3D"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("roofline", "block", "forward", "trace"))
    parser.add_argument("--stage", choices=("ss", "hr"), default="ss")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--trace-seconds", type=float, default=0)
    parser.add_argument("--gpu-intervals", type=Path)
    parser.add_argument("--trace-process", default="python")
    parser.add_argument("--window-seconds", type=float, default=5)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.trace_seconds < 0 or arguments.window_seconds <= 0:
        parser.error("Repeats/window must be positive and trace duration nonnegative")
    if arguments.report.exists() or arguments.capture and arguments.capture.exists():
        parser.error("Refusing to overwrite measurement evidence")
    if arguments.mode != "block" and (arguments.capture or arguments.trace_seconds):
        parser.error("GPU capture/trace loop requires block mode")
    if arguments.inputs is None:
        arguments.inputs = Path("outputs/runtime/sampling-investigation") / (arguments.stage + "-inputs.npz")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    if arguments.capture:
        arguments.capture.parent.mkdir(parents=True, exist_ok=True)
    provenance = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                  "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                    (Path(__file__), Path("pixalmlx/models/sparse_structure_flow.py"),
                                     Path("pixalmlx/modules/attention.py"), Path("pixalmlx/modules/rope.py"))}}
    before = system_usage()
    started = time.perf_counter()
    with InferenceRuntime(model_cache_bytes=0, feature_cache_bytes=0):
        function = {"roofline": roofline, "block": block_profile, "forward": full_forward,
                    "trace": trace_profile}[arguments.mode]
        result = function(arguments)
        result["active_mlx_bytes_at_end"] = mx.get_active_memory()
    after = system_usage()
    report = {"schema": "pixalmlx.hardware_investigation.v1", "mode": arguments.mode,
              "measured_utc": datetime.now(timezone.utc).isoformat(), "device": mx.device_info(),
              "mlx_version": mx.__version__, "repeats": arguments.repeats,
              "elapsed_seconds": time.perf_counter() - started, "system_before": before, "system_after": after,
              "system_delta": vm_delta(before, after), **provenance, **result}
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: value for name, value in report.items()
                      if name not in ("system_before", "system_after", "measurements")}, indent=2), flush=True)
    for name, measurement in report["measurements"].items():
        print(f"{name}: {measurement['median_seconds'] * 1000:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
