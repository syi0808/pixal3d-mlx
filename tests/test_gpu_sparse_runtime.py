import itertools

import mlx.core as mx
import numpy as np
import pytest

from trellmlx.modules.sparse_conv import build_neighbor_map
from trellmlx.sparse_runtime import morton_permutation, prepare_gpu_subdivision


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
