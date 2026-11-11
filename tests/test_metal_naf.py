import mlx.core as mx
import numpy as np
import pytest

from pixalmlx.models.naf import NAFCrossAttention, fused_neighborhood_attention, neighborhood_attention


@pytest.mark.parametrize("shape,kernel,dilation", [
    ((1, 2, 7, 6, 4, 5), (3, 3), (1, 1)),
    ((1, 1, 7, 7, 33, 35), (3, 3), (1, 1)),
    ((2, 1, 13, 11, 7, 9), (3, 5), (3, 2)),
    ((1, 4, 18, 18, 64, 256), (9, 9), (2, 2)),
])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_fused_naf_matches_gather_reference(shape, kernel, dilation, dtype):
    batch, heads, height, width, query_channels, value_channels = shape
    mx.random.seed(12)
    queries = mx.random.normal((batch, heads, height, width, query_channels)).astype(dtype)
    keys = mx.random.normal(queries.shape).astype(dtype)
    values = mx.random.normal((batch, heads, height, width, value_channels)).astype(dtype)
    scale = query_channels ** -0.5
    expected = neighborhood_attention(queries, keys, values, kernel, scale, dilation=dilation)
    actual = fused_neighborhood_attention(queries, keys, values, kernel, scale, dilation=dilation)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=4e-3, atol=4e-3 if dtype == mx.float16 else 2e-5)


def test_fused_naf_reads_nearest_upsampled_keys_without_materializing_them():
    queries = mx.random.normal((1, 2, 24, 30, 8))
    keys = mx.random.normal((1, 2, 8, 10, 8))
    values = mx.random.normal((1, 2, 8, 10, 12))
    expanded_keys = mx.repeat(mx.repeat(keys, 3, axis=2), 3, axis=3)
    expanded_values = mx.repeat(mx.repeat(values, 3, axis=2), 3, axis=3)
    expected = neighborhood_attention(queries, expanded_keys, expanded_values, (5, 5), 0.25, dilation=(3, 3))
    actual = fused_neighborhood_attention(queries, keys, values, (5, 5), 0.25, dilation=(3, 3), upsample=(3, 3))
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=2e-5, rtol=1e-4)


def test_fused_naf_validates_dilation_groups():
    values = mx.zeros((1, 1, 8, 8, 2))
    with pytest.raises(ValueError, match="dilation group"):
        fused_neighborhood_attention(values, values, values, (5, 5), 1.0, dilation=(2, 2))


@pytest.mark.parametrize("backend", ["auto", "metal"])
def test_cross_attention_uses_fused_backend_without_repeated_keys(monkeypatch, backend):
    queries = mx.random.normal((1, 8, 27, 30))
    keys = mx.random.normal((1, 8, 9, 10))
    values = mx.random.normal((1, 12, 9, 10))
    reference = NAFCrossAttention(8, 2, kernel_size=(3, 3), backend="reference")
    expected = np.array(reference(queries, keys, values))

    def reject_repeat(*args, **kwargs):
        raise AssertionError("Fused attention must not materialize upsampled keys or values")

    monkeypatch.setattr(mx, "repeat", reject_repeat)
    fused = NAFCrossAttention(8, 2, kernel_size=(3, 3), backend=backend)
    np.testing.assert_allclose(np.array(fused(queries, keys, values)), expected, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("threadgroup_size", [64, 128, 256])
def test_naf_threadgroup_sizes_preserve_partial_group_results(threadgroup_size):
    queries = mx.random.normal((1, 1, 7, 7, 33))
    keys = mx.random.normal(queries.shape)
    values = mx.random.normal((1, 1, 7, 7, 35))
    actual = fused_neighborhood_attention(queries, keys, values, (3, 3), 33 ** -0.5,
                                          threadgroup_size=threadgroup_size)
    expected = neighborhood_attention(queries, keys, values, (3, 3), 33 ** -0.5)
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=2e-5, rtol=1e-4)
