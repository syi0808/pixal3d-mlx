import mlx.core as mx
import numpy as np
import pytest

from pixalmlx.models.naf import NAF
from pixalmlx.modules.proj_grid import _grid_sample_bilinear


@pytest.mark.parametrize("chunk_size", [1, 3, 20])
def test_naf_projects_only_bilinear_corners_with_border_padding(chunk_size):
    mx.random.seed(9)
    model = NAF(dim=16, heads_attn=2, heads_rope=2, kernel_size=3, img_layers=1)
    guide = mx.random.uniform(shape=(1, 3, 24, 24))
    features = mx.random.normal((1, 12, 6, 6))
    grid = mx.array([[[-2.0, -2.0], [-1.0, 1.0], [0.25, -0.5], [1.5, 2.0], [0.0, 0.0]]])
    model.upsampler.backend = "reference"
    expected = _grid_sample_bilinear(model(guide, features, (12, 12)), grid).transpose(0, 2, 1)
    model.upsampler.backend = "metal"
    actual = model.project(guide, features, (12, 12), grid, chunk_size=chunk_size)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-4, atol=2e-5)


def test_sparse_naf_does_not_invoke_dense_upsampler(monkeypatch):
    model = NAF(dim=16, heads_attn=2, heads_rope=2, kernel_size=3, img_layers=1)

    def reject_dense(*args, **kwargs):
        raise AssertionError("Sparse NAF must not build dense output features")

    monkeypatch.setattr(type(model.upsampler), "__call__", reject_dense)
    result = model.project(mx.ones((1, 3, 24, 24)), mx.ones((1, 12, 6, 6)),
                           (12, 12), mx.zeros((1, 5, 2)))
    assert result.shape == (1, 5, 12)
    assert bool(mx.all(mx.isfinite(result)).item())


def test_tiled_encoder_preserves_global_group_norm_and_border_convolutions():
    model = NAF(dim=16, heads_attn=2, heads_rope=2, kernel_size=3, img_layers=2)
    guide = mx.random.uniform(shape=(1, 3, 32, 32))
    features = mx.random.normal((1, 12, 8, 8))
    model.encoder_tile_rows = 0
    expected = model(guide, features, (16, 16))
    mx.eval(expected)
    model.encoder_tile_rows = 5
    actual = model(guide, features, (16, 16))
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-4, atol=2e-5)
