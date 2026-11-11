"""Numerically stable normalization layers for MLX."""

import mlx.core as mx
import mlx.nn as nn


class LayerNorm32(nn.Module):
    """LayerNorm that accumulates in float32 for numerical stability.

    TRELLIS.2 uses elementwise_affine=False (controlled by adaLN-Zero),
    so this is just normalize + optional affine.
    """

    def __init__(self, dims: int, eps: float = 1e-6, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        x = mx.fast.layer_norm(x, weight=None, bias=None, eps=self.eps)
        x = x.astype(orig_dtype)
        if self.affine:
            x = x * self.weight + self.bias
        return x
