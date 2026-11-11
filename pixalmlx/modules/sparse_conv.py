"""Sparse 3D submanifold convolution for MLX.

Implements the gather-scatter pattern: for each active voxel, gather
features from its 3x3x3 neighborhood (only where other active voxels
exist), multiply by kernel weights, and scatter-add results back.

This is the same algorithm as trellis-mac's conv_none.py but in MLX.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class NeighborMap(tuple):
    def __new__(cls, source_indices, target_indices, kernel_indices, counts):
        instance = super().__new__(cls, (source_indices, target_indices, kernel_indices))
        offset = 0
        groups = []
        for count in counts:
            groups.append((source_indices[offset:offset + count], target_indices[offset:offset + count]))
            offset += count
        instance.groups = tuple(groups)
        return instance


def build_neighbor_map(coords: mx.array, kernel_size: int = 3, backend="auto") -> tuple:
    """Build source/target index pairs for sparse convolution.

    For each active voxel, find which of its 3³=27 neighbors are also active.
    Returns (src_indices, tgt_indices, kernel_indices) as MLX arrays.

    Uses sorted packed uint64 coordinates with vectorized neighbor searches.

    Args:
        coords: [N, 4] as (batch_idx, z, y, x), int
        kernel_size: convolution kernel size (3)

    Returns:
        src_idx: [E] source voxel indices
        tgt_idx: [E] target voxel indices
        k_idx: [E] kernel position indices (0..26 for 3x3x3)
    """
    if backend not in ("auto", "gpu", "cpu"):
        raise ValueError(f"Unknown sparse backend: {backend}")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("Sparse coordinates must have shape [N, 4]")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Sparse kernel size must be a positive odd integer")
    if backend == "gpu" or (backend == "auto" and mx.default_device() == mx.gpu):
        from ..sparse_runtime import build_gpu_neighbor_map

        return build_gpu_neighbor_map(coords, kernel_size)
    coordinates = np.asarray(coords)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("Sparse coordinates must have shape [N, 4]")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Sparse kernel size must be a positive odd integer")
    if np.any(coordinates < 0) or np.any(coordinates > 65535):
        raise ValueError("Sparse coordinates must fit unsigned 16-bit fields")
    coordinates = coordinates.astype(np.int64, copy=False)
    shifts = np.array([48, 32, 16, 0], dtype=np.uint64)

    def pack(values):
        return np.bitwise_or.reduce(values.astype(np.uint64) << shifts, axis=1)

    packed = pack(coordinates)
    order = np.argsort(packed, kind="stable")
    sorted_keys = packed[order]
    if np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError("Sparse coordinates must be unique")
    source_groups = []
    target_groups = []
    kernel_groups = []
    counts = []
    half = kernel_size // 2
    for kernel_index, offset in enumerate(np.ndindex((kernel_size,) * 3)):
        offsets = tuple(value - half for value in offset)
        in_bounds = np.ones(len(coordinates), dtype=bool)
        for axis, displacement in enumerate(offsets, start=1):
            if displacement < 0:
                in_bounds &= coordinates[:, axis] >= -displacement
            elif displacement > 0:
                in_bounds &= coordinates[:, axis] <= 65535 - displacement
        packed_offset = sum(displacement * (1 << shift)
                            for displacement, shift in zip(offsets, (32, 16, 0)))
        neighbor_keys = packed + np.uint64(packed_offset % (1 << 64))
        positions = np.searchsorted(sorted_keys, neighbor_keys)
        present = in_bounds & (positions < len(sorted_keys))
        if len(sorted_keys):
            present &= sorted_keys[np.minimum(positions, len(sorted_keys) - 1)] == neighbor_keys
        targets = np.flatnonzero(present).astype(np.int32)
        sources = order[positions[present]].astype(np.int32)
        source_groups.append(sources)
        target_groups.append(targets)
        kernel_groups.append(np.full(len(targets), kernel_index, dtype=np.int32))
        counts.append(len(targets))
    return NeighborMap(
        mx.array(np.concatenate(source_groups)),
        mx.array(np.concatenate(target_groups)),
        mx.array(np.concatenate(kernel_groups)),
        counts,
    )


class SparseConv3d(nn.Module):
    """Submanifold sparse 3D convolution.

    Weight layout matches TRELLIS.2 checkpoints: [out_C, kD, kH, kW, in_C]
    (flex_gemm convention, NOT PyTorch's [out_C, in_C, kD, kH, kW]).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 eval_interval: int = 0):
        super().__init__()
        if not isinstance(eval_interval, int) or eval_interval < 0:
            raise ValueError("Convolution evaluation interval must be a nonnegative integer")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eval_interval = eval_interval
        K = kernel_size

        scale = (2.0 / (in_channels * K * K * K)) ** 0.5
        # Weight: [out_C, kD, kH, kW, in_C] — matches checkpoint layout
        self.weight = mx.random.normal((out_channels, K, K, K, in_channels)) * scale
        self.bias = mx.zeros((out_channels,))

    def __call__(self, feats: mx.array, neighbor_map: tuple) -> mx.array:
        """
        Args:
            feats: [N, in_C] features at active voxels
            neighbor_map: (src_idx, tgt_idx, k_idx) from build_neighbor_map

        Returns:
            [N, out_C] convolved features
        """
        src_idx, tgt_idx, k_idx = neighbor_map
        N = feats.shape[0]
        Co = self.out_channels
        Ci = self.in_channels
        K_total = self.kernel_size ** 3

        # Reshape weight: [Co, K, K, K, Ci] → [K_total, Ci, Co]
        w = self.weight.reshape(Co, K_total, Ci)
        w = w.transpose(1, 2, 0)  # [K_total, Ci, Co]

        # Process per kernel position to keep memory bounded
        # and avoid scatter_add issues. 27 positions for 3x3x3.
        out = mx.zeros((N, Co))

        if isinstance(neighbor_map, NeighborMap):
            groups = neighbor_map.groups
        else:
            sources = np.asarray(src_idx)
            targets = np.asarray(tgt_idx)
            kernels = np.asarray(k_idx)
            groups = tuple((mx.array(sources[kernels == index]), mx.array(targets[kernels == index]))
                           for index in range(K_total))
        for kernel_index, (sources, targets) in enumerate(groups):
            if sources.size:
                edge_out = feats[sources] @ w[kernel_index]
                out = out.at[targets].add(edge_out)
                del edge_out
            if self.eval_interval and (kernel_index + 1) % self.eval_interval == 0:
                mx.eval(out)

        return out + self.bias
