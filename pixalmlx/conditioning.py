"""Sampling-scoped, bounded caches for timestep-invariant conditioning."""

import mlx.core as mx


def array_bytes(value):
    if isinstance(value, mx.array):
        return value.nbytes
    if isinstance(value, dict):
        return sum(array_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(array_bytes(item) for item in value)
    return 0


class ZeroConditioning(dict):
    pass


class PreparedConditioning(dict):
    def __init__(self, context, max_bytes):
        super().__init__(context)
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self.values_cache = {}
        self.hits = 0
        self.zero_conditioning = isinstance(context, ZeroConditioning)

    def invariant(self, key, compute):
        if key in self.values_cache:
            self.hits += 1
            return self.values_cache[key]
        result = compute()
        size = array_bytes(result)
        if self.used_bytes + size <= self.max_bytes:
            self.values_cache[key] = result
            self.used_bytes += size
        return result

    def clear_invariants(self):
        self.values_cache.clear()
        self.used_bytes = 0


def prepare_sampling(model, noise, cond, neg_cond, model_kwargs, cache_bytes):
    if cache_bytes < 0:
        raise ValueError("Conditioning cache budget cannot be negative")
    contexts = []
    if cache_bytes and isinstance(cond, dict) and isinstance(neg_cond, dict):
        positive_budget = (cache_bytes - min(cache_bytes // 8, 4 * 1024 ** 2)
                           if isinstance(neg_cond, ZeroConditioning) else cache_bytes * 2 // 3)
        cond = PreparedConditioning(cond, positive_budget)
        neg_cond = PreparedConditioning(neg_cond, cache_bytes - positive_budget)
        contexts = [cond, neg_cond]
    if hasattr(model, "_coords_to_rope_phases") and model_kwargs.get("coords") is not None:
        model_kwargs = {**model_kwargs, "rope_phases": model._coords_to_rope_phases(model_kwargs["coords"])}
    elif hasattr(model, "proj_in_channels") and noise.ndim == 5:
        from .modules.rope import build_rope_phases

        model_kwargs = {**model_kwargs, "rope_phases": build_rope_phases(noise.shape[2], model.model_channels // model.num_heads)}
    return cond, neg_cond, model_kwargs, contexts
