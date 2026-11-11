"""GPU topology construction with only compact shape metadata read by Python."""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _neighbor_kernel():
    return mx.fast.metal_kernel(
        name="pixal3d_sparse_neighbors",
        input_names=["coordinates", "sorted_keys", "order"],
        output_names=["neighbors"],
        source=r"""
        uint edge = thread_position_in_grid.x;
        if (edge >= Count * KernelSize * KernelSize * KernelSize) return;
        uint target = edge % Count;
        uint kernel_index = edge / Count;
        int first = coordinates[target * 4 + 1] + int(kernel_index / (KernelSize * KernelSize)) - KernelSize / 2;
        int second = coordinates[target * 4 + 2] + int((kernel_index / KernelSize) % KernelSize) - KernelSize / 2;
        int third = coordinates[target * 4 + 3] + int(kernel_index % KernelSize) - KernelSize / 2;
        int result = -1;
        if (first >= 0 && first <= 65535 && second >= 0 && second <= 65535 && third >= 0 && third <= 65535) {
            ulong key = (ulong(coordinates[target * 4]) << 48) | (ulong(first) << 32) | (ulong(second) << 16) | ulong(third);
            uint low = 0;
            uint high = Count;
            while (low < high) {
                uint middle = low + (high - low) / 2;
                if (sorted_keys[middle] < key) low = middle + 1;
                else high = middle;
            }
            if (low < Count && sorted_keys[low] == key) result = int(order[low]);
        }
        neighbors[edge] = result;
        """,
    )


@lru_cache(maxsize=1)
def _compact_neighbors_kernel():
    return mx.fast.metal_kernel(
        name="pixal3d_compact_neighbors",
        input_names=["neighbors", "prefix", "offsets"],
        output_names=["sources", "targets", "kernels"],
        source=r"""
        uint edge = thread_position_in_grid.x;
        if (edge >= Count * KernelCount || neighbors[edge] < 0) return;
        uint kernel_index = edge / Count;
        uint destination = offsets[kernel_index] + prefix[edge] - 1;
        sources[destination] = neighbors[edge];
        targets[destination] = edge % Count;
        kernels[destination] = kernel_index;
        """,
    )


def build_gpu_neighbor_map(coordinates, kernel_size):
    from .modules.sparse_conv import NeighborMap

    if not mx.issubdtype(coordinates.dtype, mx.integer):
        raise ValueError("Sparse coordinates must be integers")
    count = coordinates.shape[0]
    kernel_count = kernel_size ** 3
    if not count:
        empty = mx.zeros((0,), dtype=mx.int32)
        return NeighborMap(empty, empty, empty, [0] * kernel_count)
    packed_coordinates = coordinates.astype(mx.uint64)
    packed = ((packed_coordinates[:, 0] << 48) | (packed_coordinates[:, 1] << 32) |
              (packed_coordinates[:, 2] << 16) | packed_coordinates[:, 3])
    order = mx.argsort(packed)
    sorted_keys = packed[order]
    neighbors = _neighbor_kernel()(
        inputs=[coordinates, sorted_keys, order],
        template=[("Count", count), ("KernelSize", kernel_size)],
        grid=(count * kernel_count, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(kernel_count, count)], output_dtypes=[mx.int32],
    )[0]
    present = (neighbors >= 0).astype(mx.int32)
    counts = mx.sum(present, axis=1)
    invalid = mx.any((coordinates < 0) | (coordinates > 65535)).astype(mx.int32)
    duplicate = mx.any(sorted_keys[1:] == sorted_keys[:-1]).astype(mx.int32)
    metadata = mx.concatenate([counts, invalid[None], duplicate[None]]).tolist()
    if metadata[-2]:
        raise ValueError("Sparse coordinates must fit unsigned 16-bit fields")
    if metadata[-1]:
        raise ValueError("Sparse coordinates must be unique")
    counts = metadata[:-2]
    offsets = [0]
    for size in counts:
        offsets.append(offsets[-1] + size)
    prefix = mx.cumsum(present, axis=1)
    sources, targets, kernels = _compact_neighbors_kernel()(
        inputs=[neighbors, prefix, mx.array(offsets[:-1], dtype=mx.int32)],
        template=[("Count", count), ("KernelCount", kernel_count)],
        grid=(count * kernel_count, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(offsets[-1],)] * 3, output_dtypes=[mx.int32] * 3,
    )
    return NeighborMap(sources, targets, kernels, counts)


@lru_cache(maxsize=1)
def _subdivision_kernel():
    return mx.fast.metal_kernel(
        name="pixal3d_sparse_subdivision",
        input_names=["coordinates", "mask", "prefix"],
        output_names=["indices", "children"],
        source=r"""
        uint selection = thread_position_in_grid.x;
        if (selection >= Count * 8 || !mask[selection]) return;
        uint parent = selection / 8;
        uint child = selection % 8;
        uint destination = prefix[selection] - 1;
        indices[destination] = selection;
        children[destination * 4] = coordinates[parent * 4];
        children[destination * 4 + 1] = coordinates[parent * 4 + 1] * 2 + (child & 1);
        children[destination * 4 + 2] = coordinates[parent * 4 + 2] * 2 + ((child >> 1) & 1);
        children[destination * 4 + 3] = coordinates[parent * 4 + 3] * 2 + ((child >> 2) & 1);
        """,
    )


def prepare_gpu_subdivision(coordinates, mask):
    if coordinates.ndim != 2 or coordinates.shape[1] != 4 or mask.shape != (coordinates.shape[0], 8):
        raise ValueError("Subdivision requires coordinates [N, 4] and a mask [N, 8]")
    if not coordinates.shape[0]:
        return mx.zeros((0,), dtype=mx.int32), mx.zeros((0, 4), dtype=mx.int32)
    mask = mask.astype(mx.bool_).reshape(-1)
    prefix = mx.cumsum(mask.astype(mx.int32))
    count = int(prefix[-1].item())
    if not count:
        return mx.zeros((0,), dtype=mx.int32), mx.zeros((0, 4), dtype=mx.int32)
    return tuple(_subdivision_kernel()(
        inputs=[coordinates, mask, prefix], template=[("Count", coordinates.shape[0])],
        grid=(mask.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(count,), (count, 4)], output_dtypes=[mx.int32, mx.int32],
    ))


def morton_permutation(coordinates):
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("Morton coordinates must have shape [N, 4]")
    coordinates = coordinates.astype(mx.uint64)
    keys = coordinates[:, 0] << 48
    for bit in range(16):
        for axis in range(3):
            keys = keys | (((coordinates[:, axis + 1] >> bit) & 1) << (3 * bit + axis))
    return mx.argsort(keys)
