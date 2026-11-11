import itertools

import mlx.core as mx
import numpy as np
import pytest

from pixalmlx.modules.sparse_conv import build_neighbor_map
from pixalmlx.sparse_runtime import morton_permutation, prepare_gpu_subdivision


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_gpu_neighbors_match_cpu_without_coordinate_download(kernel_size, monkeypatch):
    coordinates = np.array(list(itertools.product((0, 65535), (0, 1, 65534, 65535),
                                                 (0, 1, 65534, 65535), (0, 1, 65534, 65535))), dtype=np.int32)
    np.random.default_rng(4).shuffle(coordinates)
    coordinates = mx.array(coordinates)
    expected = tuple(np.array(indices) for indices in build_neighbor_map(coordinates, kernel_size, backend="cpu"))

    def reject_asarray(*args, **kwargs):
        raise AssertionError("GPU topology must not download coordinates")

    monkeypatch.setattr(np, "asarray", reject_asarray)
    actual = build_neighbor_map(coordinates, kernel_size, backend="gpu")
    monkeypatch.undo()
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(np.array(result), reference)


@pytest.mark.parametrize("coordinates,message", [
    ([[0, 0, 0, 0], [0, 0, 0, 0]], "unique"),
    ([[0, -1, 0, 0]], "16-bit"),
    ([[65536, 0, 0, 0]], "16-bit"),
])
def test_gpu_neighbor_validation(coordinates, message):
    with pytest.raises(ValueError, match=message):
        build_neighbor_map(mx.array(coordinates), backend="gpu")


def test_gpu_subdivision_order_and_morton_order():
    coordinates = mx.array([[1, 3, 2, 1], [0, 0, 0, 0]])
    mask = mx.array([[True] * 8, [False, True, False, True, False, True, False, True]])
    indices, children = prepare_gpu_subdivision(coordinates, mask)
    expected_indices = np.flatnonzero(np.array(mask).reshape(-1))
    np.testing.assert_array_equal(np.array(indices), expected_indices)
    expected = np.array(coordinates)[expected_indices // 8].copy()
    child_indices = expected_indices % 8
    expected[:, 1:] = expected[:, 1:] * 2 + np.column_stack([child_indices & 1, (child_indices >> 1) & 1, (child_indices >> 2) & 1])
    np.testing.assert_array_equal(np.array(children), expected)
    permutation = morton_permutation(children)
    restored = children[permutation][mx.argsort(permutation)]
    np.testing.assert_array_equal(np.array(restored), expected)
    assert np.all(np.array(children[permutation])[:4, 0] == 0)


def test_morton_decoder_preserves_features_at_each_coordinate():
    from pixalmlx.models.shape_slat_decoder import SLatDecoder

    model = SLatDecoder(out_channels=7, latent_channels=8, model_channels=[16, 8], num_blocks=[0, 0])
    coordinates = mx.array([[0, 0, 0, 1], [0, 1, 0, 0]])
    features = mx.random.normal((2, 8))
    expected, expected_coordinates = model(features, coordinates)
    mx.eval(expected, expected_coordinates)
    model.spatial_order = "morton"
    actual, actual_coordinates = model(features, coordinates)

    def ordered(values, coords):
        coords = np.array(coords)
        order = np.lexsort(tuple(coords[:, axis] for axis in reversed(range(4))))
        return np.array(values)[order], coords[order]

    expected, expected_coordinates = ordered(expected, expected_coordinates)
    actual, actual_coordinates = ordered(actual, actual_coordinates)
    np.testing.assert_array_equal(actual_coordinates, expected_coordinates)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-5)
