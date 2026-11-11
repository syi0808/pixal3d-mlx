import itertools

import mlx.core as mx
import numpy as np
import pytest

from pixalmlx.models.shape_slat_decoder import SparseChannel2Spatial
from pixalmlx.modules.sparse_conv import NeighborMap, SparseConv3d, build_neighbor_map


def reference_neighbors(coordinates, kernel_size):
    lookup = {tuple(row): index for index, row in enumerate(coordinates.tolist())}
    sources, targets, kernels = [], [], []
    for kernel_index, offset in enumerate(itertools.product(range(kernel_size), repeat=3)):
        offset = np.array(offset) - kernel_size // 2
        for target, row in enumerate(coordinates):
            neighbor = (int(row[0]), *(row[1:] + offset).tolist())
            if neighbor in lookup:
                sources.append(lookup[neighbor])
                targets.append(target)
                kernels.append(kernel_index)
    return tuple(np.array(values, dtype=np.int32) for values in (sources, targets, kernels))


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_neighbor_map_matches_coordinate_reference(kernel_size):
    coordinates = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0],
                            [1, 0, 0, 0], [0, 65535, 0, 0]], dtype=np.int32)
    actual = build_neighbor_map(mx.array(coordinates), kernel_size)
    assert isinstance(actual, NeighborMap)
    for result, expected in zip(actual, reference_neighbors(coordinates, kernel_size)):
        np.testing.assert_array_equal(np.array(result), expected)
    assert len(actual.groups) == kernel_size ** 3


def test_empty_neighbor_map_and_duplicate_rejection():
    actual = build_neighbor_map(mx.zeros((0, 4), dtype=mx.int32))
    assert all(indices.size == 0 for indices in actual)
    with pytest.raises(ValueError, match="unique"):
        build_neighbor_map(mx.zeros((2, 4), dtype=mx.int32))


def test_neighbor_map_handles_shuffled_coordinates_and_field_boundaries():
    coordinates = np.array(list(itertools.product((0, 65535), (0, 1, 65534, 65535),
                                                 (0, 1, 65534, 65535),
                                                 (0, 1, 65534, 65535))), dtype=np.int32)
    np.random.default_rng(8).shuffle(coordinates)
    actual = build_neighbor_map(mx.array(coordinates))
    for result, expected in zip(actual, reference_neighbors(coordinates, 3)):
        np.testing.assert_array_equal(np.array(result), expected)


def test_cached_neighbor_groups_preserve_convolution():
    coordinates = mx.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    neighbors = build_neighbor_map(coordinates)
    convolution = SparseConv3d(4, 6)
    features = mx.random.normal((3, 4))
    actual = convolution(features, neighbors)
    expected = convolution(features, tuple(neighbors))
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-6)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("occupied", [False, True])
def test_subdivision_preserves_order_values_and_dtype(dtype, occupied):
    coordinates = np.array([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=np.int32)
    features = np.arange(2 * 32, dtype=np.float32).reshape(2, 32)
    mask = np.array([[True, False, True, False, False, True, False, True],
                     [False, True, False, True, True, False, True, False]]) & occupied
    actual_features, actual_coordinates = SparseChannel2Spatial.upsample(
        mx.array(features).astype(dtype), mx.array(coordinates), mx.array(mask))
    expected_coordinates = []
    expected_features = []
    for parent, child in zip(*np.nonzero(mask)):
        row = coordinates[parent].copy()
        row[1:] = row[1:] * 2 + [child % 2, (child // 2) % 2, child // 4]
        expected_coordinates.append(row)
        expected_features.append(features.reshape(2, 8, 4)[parent, child])
    np.testing.assert_array_equal(np.array(actual_coordinates), np.array(expected_coordinates, dtype=np.int32).reshape(-1, 4))
    np.testing.assert_array_equal(np.array(actual_features), np.array(expected_features).reshape(-1, 4))
    assert actual_features.dtype == dtype
