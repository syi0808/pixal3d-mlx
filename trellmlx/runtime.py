"""Bounded, single-GPU execution with stage liveness and persistent caches."""

from collections import OrderedDict
from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import threading
import time

import mlx.core as mx
import mlx.utils

from .conditioning import array_bytes


_RUNTIME_LOCK = threading.RLock()


def image_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_fingerprint(path):
    path = Path(path).expanduser().resolve()
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    return (str(path), tuple((str(item), item.stat().st_size, item.stat().st_mtime_ns)
                             for item in files if item.is_file()))


class BoundedCache(MutableMapping):
    def __init__(self, max_bytes, max_entries=64):
        if max_bytes < 0 or max_entries < 1:
            raise ValueError("Cache limits must be nonnegative with at least one entry")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.used_bytes = 0
        self.entries = OrderedDict()
        self.hits = 0

    def __getitem__(self, key):
        value, size = self.entries[key]
        self.entries.move_to_end(key)
        self.hits += 1
        return value

    def __setitem__(self, key, value):
        if key in self.entries:
            del self[key]
        size = array_bytes(value.parameters()) if hasattr(value, "parameters") else array_bytes(value)
        if size > self.max_bytes or self.max_bytes == 0:
            return
        while self.entries and (self.used_bytes + size > self.max_bytes or len(self.entries) >= self.max_entries):
            oldest = next(iter(self.entries))
            del self[oldest]
        self.entries[key] = (value, size)
        self.used_bytes += size

    def __delitem__(self, key):
        _, size = self.entries.pop(key)
        self.used_bytes -= size

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def clear(self):
        self.entries.clear()
        self.used_bytes = 0


@dataclass(frozen=True)
class Stage:
    name: str
    inputs: tuple
    outputs: tuple
    run: object


class StageGraph:
    def __init__(self, stages, outputs):
        self.stages = tuple(stages)
        self.outputs = tuple(outputs)
        self.producers = {}
        names = set()
        for stage in self.stages:
            if stage.name in names:
                raise ValueError(f"Duplicate stage name: {stage.name}")
            names.add(stage.name)
            for output in stage.outputs:
                if output in self.producers:
                    raise ValueError(f"Duplicate resource producer: {output}")
                self.producers[output] = stage

    def compile(self, initial_names=()):
        initial = set(initial_names)
        ordered = []
        visiting = set()
        visited = set()

        def require(resource_name):
            if resource_name in initial:
                return
            if resource_name not in self.producers:
                raise ValueError(f"Missing graph input: {resource_name}")
            stage = self.producers[resource_name]
            if stage.name in visiting:
                raise ValueError(f"Cyclic stage graph: {stage.name}")
            if stage.name in visited:
                return
            visiting.add(stage.name)
            for input_name in stage.inputs:
                require(input_name)
            visiting.remove(stage.name)
            visited.add(stage.name)
            ordered.append(stage)

        for output in self.outputs:
            require(output)
        last_use = {}
        for index, stage in enumerate(ordered):
            for name in (*stage.inputs, *stage.outputs):
                last_use[name] = index
        for output in self.outputs:
            last_use[output] = len(ordered)
        return tuple(ordered), last_use

    def execute(self, initial=None, runtime=None):
        values = dict(initial or {})
        ordered, last_use = self.compile(values)
        try:
            for index, stage in enumerate(ordered):
                arguments = [values[name] for name in stage.inputs]
                if runtime is not None:
                    runtime.advance(stage.name)
                result = stage.run(*arguments)
                results = (result,) if len(stage.outputs) == 1 else tuple(result)
                if len(results) != len(stage.outputs):
                    raise ValueError(f"Wrong output count from stage {stage.name}")
                mx.eval(*[value for _, value in mlx.utils.tree_flatten(results) if isinstance(value, mx.array)])
                values.update(zip(stage.outputs, results))
                del arguments, result, results
                for name in tuple(values):
                    if last_use.get(name, -1) <= index:
                        del values[name]
            return tuple(values[name] for name in self.outputs)
        finally:
            values.clear()


class InferenceRuntime:
    def __init__(self, memory_budget_bytes=10_000_000_000, model_cache_bytes=2 * 1024 ** 3,
                 feature_cache_bytes=256 * 1024 ** 2, allocator_cache_bytes=128 * 1024 ** 2):
        recommended = mx.device_info().get("max_recommended_working_set_size", memory_budget_bytes)
        if memory_budget_bytes <= 0 or memory_budget_bytes > recommended:
            raise ValueError("Memory budget must be positive and not exceed the recommended Metal working set")
        if model_cache_bytes + feature_cache_bytes + allocator_cache_bytes >= memory_budget_bytes:
            raise ValueError("Cache reservations must leave memory for active inference")
        self.memory_budget_bytes = memory_budget_bytes
        self.allocator_cache_bytes = allocator_cache_bytes
        self.models = BoundedCache(model_cache_bytes, max_entries=16)
        self.features = BoundedCache(feature_cache_bytes)
        self.camera = BoundedCache(1024 * 1024)
        self.records = []
        self.lock = threading.RLock()
        self.previous_limits = None
        self.last_stage = None
        self.stage_started = None
        self.model_loads = 0
        self.closed = False
        self.owns_runtime_lock = False
        self.job_metadata = {}

    def __enter__(self):
        if self.closed or self.previous_limits is not None:
            raise RuntimeError("Runtime is closed or already active")
        _RUNTIME_LOCK.acquire()
        self.owns_runtime_lock = True
        previous_memory = None
        try:
            previous_memory = mx.set_memory_limit(self.memory_budget_bytes)
            previous_cache = mx.set_cache_limit(self.allocator_cache_bytes)
        except BaseException:
            if previous_memory is not None:
                mx.set_memory_limit(previous_memory)
            self.owns_runtime_lock = False
            _RUNTIME_LOCK.release()
            raise
        self.previous_limits = (previous_memory, previous_cache)
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()

    @contextmanager
    def job(self, report_path=None, metadata=None):
        with self.lock:
            if self.closed or self.previous_limits is None:
                raise RuntimeError("Enter the runtime before submitting jobs")
            self.records = []
            self.last_stage = None
            self.stage_started = None
            self.job_metadata = dict(metadata or {})
            mx.reset_peak_memory()
            status = "failed"
            try:
                yield self
                status = "complete"
            finally:
                self._finish_stage()
                report = self.report(status)
                if report_path is not None:
                    path = Path(report_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(report, indent=2) + "\n")
                if status != "complete":
                    self.trim()

    def _finish_stage(self):
        if self.last_stage is not None:
            mx.synchronize()
            self.records.append({
                "stage": self.last_stage,
                "seconds": time.perf_counter() - self.stage_started,
                "active_mlx_bytes": mx.get_active_memory(),
                "cached_mlx_bytes": mx.get_cache_memory(),
                "peak_mlx_bytes": mx.get_peak_memory(),
                "max_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            })
            self.last_stage = None

    def advance(self, name):
        self._finish_stage()
        if mx.get_active_memory() > self.memory_budget_bytes * 0.8:
            self.trim()
        self.last_stage = name
        self.stage_started = time.perf_counter()

    def load_model(self, key, factory, loader, retain=True):
        cached = self.models.get(key)
        if cached is not None:
            return cached
        if mx.get_active_memory() > self.memory_budget_bytes * 0.7:
            self.trim()
        model = factory()
        loader(model)
        if hasattr(model, "parameters"):
            mx.eval(model.parameters())
        self.model_loads += 1
        try:
            object.__setattr__(model, "_runtime_fingerprint", key)
        except (TypeError, AttributeError):
            pass
        if retain:
            self.models[key] = model
        return model

    def trim(self):
        self.models.clear()
        self.features.clear()
        self.camera.clear()
        gc.collect()
        mx.clear_cache()

    def report(self, status="complete"):
        peak_memory = mx.get_peak_memory()
        return {
            "schema": "pixal3d.runtime.v1", "status": status,
            "mlx_version": mx.__version__, "device": mx.device_info(),
            "job": dict(self.job_metadata),
            "memory_budget_bytes": self.memory_budget_bytes,
            "memory_budget_is_hard_limit": False,
            "memory_budget_exceeded": peak_memory > self.memory_budget_bytes,
            "memory_budget_excess_bytes": max(0, peak_memory - self.memory_budget_bytes),
            "peak_mlx_bytes": peak_memory, "stages": list(self.records),
            "model_loads": self.model_loads, "model_cache_hits": self.models.hits,
            "feature_cache_hits": self.features.hits,
            "resident_model_bytes": self.models.used_bytes,
            "resident_feature_bytes": self.features.used_bytes,
            "total_process_memory_is_not_capped": True,
        }

    def close(self):
        if self.closed:
            return
        try:
            self.trim()
            mx.synchronize()
        finally:
            if self.previous_limits is not None:
                previous_memory, previous_cache = self.previous_limits
                mx.set_memory_limit(previous_memory)
                mx.set_cache_limit(previous_cache)
            self.previous_limits = None
            self.closed = True
            if self.owns_runtime_lock:
                self.owns_runtime_lock = False
                _RUNTIME_LOCK.release()
