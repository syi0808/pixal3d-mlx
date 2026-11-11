"""Selective compilation of pure, bounded Transformer operations."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


@lru_cache(maxsize=1)
def compiled_ffn():
    def forward(features, first_weight, first_bias, second_weight, second_bias):
        hidden = nn.gelu_approx(mx.addmm(first_bias, features, first_weight.T))
        return mx.addmm(second_bias, hidden, second_weight.T)

    return mx.compile(forward)


@lru_cache(maxsize=1)
def compiled_modulation():
    def forward(features, scale, shift):
        normalized = mx.fast.layer_norm(features.astype(mx.float32), weight=None, bias=None, eps=1e-6)
        return normalized.astype(features.dtype) * (1 + scale) + shift

    return mx.compile(forward)


@lru_cache(maxsize=1)
def compiled_gated_residual():
    return mx.compile(lambda features, hidden, gate: features + hidden * gate)


def configure_selective_compile(model, enabled):
    for block in getattr(model, "blocks", ()):
        if hasattr(block, "selective_compile"):
            block.selective_compile = enabled
        if hasattr(block, "mlp") and hasattr(block.mlp, "selective_compile"):
            block.mlp.selective_compile = enabled
