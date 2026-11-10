"""Memory-bounded Metal kernels for the Pixal3D inference path."""

from functools import lru_cache

import mlx.core as mx


_NAF_SOURCE = r"""
    constexpr uint queries_per_group = 4;
    constexpr uint neighbors = KernelHeight * KernelWidth;
    uint lane = thread_index_in_simdgroup;
    uint query_slot = simdgroup_index_in_threadgroup;
    uint query_index = threadgroup_position_in_grid.x * queries_per_group + query_slot;
    uint query_count = BatchHeads * Height * Width;
    threadgroup float probabilities[queries_per_group * neighbors];
    bool valid = query_index < query_count;
    uint batch_head = query_index / (Height * Width);
    uint row = (query_index / Width) % Height;
    uint column = query_index % Width;
    uint row_group = row % DilationHeight;
    uint column_group = column % DilationWidth;
    uint row_group_length = (Height - row_group + DilationHeight - 1) / DilationHeight;
    uint column_group_length = (Width - column_group + DilationWidth - 1) / DilationWidth;
    int row_start = clamp(int(row / DilationHeight) - KernelHeight / 2,
                          0, int(row_group_length) - KernelHeight);
    int column_start = clamp(int(column / DilationWidth) - KernelWidth / 2,
                             0, int(column_group_length) - KernelWidth);
    if (valid) {
        for (uint neighbor = 0; neighbor < neighbors; ++neighbor) {
            uint source_row = (row_group + (row_start + neighbor / KernelWidth) * DilationHeight) / UpsampleHeight;
            uint source_column = (column_group + (column_start + neighbor % KernelWidth) * DilationWidth) / UpsampleWidth;
            uint source_index = (batch_head * SourceHeight + source_row) * SourceWidth + source_column;
            float product = 0.0f;
            for (uint channel = lane; channel < QueryChannels; channel += 32) {
                product += float(queries[query_index * QueryChannels + channel]) *
                           float(keys[source_index * QueryChannels + channel]);
            }
            float score = simd_sum(product) * scale_value[0];
            if (lane == 0) {
                probabilities[query_slot * neighbors + neighbor] = score;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid && lane == 0) {
        float maximum = -INFINITY;
        for (uint neighbor = 0; neighbor < neighbors; ++neighbor) {
            maximum = max(maximum, probabilities[query_slot * neighbors + neighbor]);
        }
        float denominator = 0.0f;
        for (uint neighbor = 0; neighbor < neighbors; ++neighbor) {
            float probability = exp(probabilities[query_slot * neighbors + neighbor] - maximum);
            probabilities[query_slot * neighbors + neighbor] = probability;
            denominator += probability;
        }
        for (uint neighbor = 0; neighbor < neighbors; ++neighbor) {
            probabilities[query_slot * neighbors + neighbor] /= denominator;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
        for (uint channel = lane; channel < ValueChannels; channel += 32) {
            float accumulator = 0.0f;
            for (uint neighbor = 0; neighbor < neighbors; ++neighbor) {
                uint source_row = (row_group + (row_start + neighbor / KernelWidth) * DilationHeight) / UpsampleHeight;
                uint source_column = (column_group + (column_start + neighbor % KernelWidth) * DilationWidth) / UpsampleWidth;
                uint source_index = (batch_head * SourceHeight + source_row) * SourceWidth + source_column;
                accumulator += probabilities[query_slot * neighbors + neighbor] *
                               float(values[source_index * ValueChannels + channel]);
            }
            output[query_index * ValueChannels + channel] = OutputType(accumulator);
        }
    }
"""


@lru_cache(maxsize=1)
def _naf_kernel():
    return mx.fast.metal_kernel(
        name="pixal3d_neighborhood_attention",
        input_names=["queries", "keys", "values", "scale_value"],
        output_names=["output"], source=_NAF_SOURCE,
    )


def neighborhood_attention_metal(queries, keys, values, kernel_size, scale,
                                 dilation=(1, 1), upsample=(1, 1)):
    batch, heads, height, width, query_channels = queries.shape
    kernel_height, kernel_width = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
    dilation_height, dilation_width = dilation
    source_height, source_width = keys.shape[2:4]
    if kernel_height <= 0 or kernel_width <= 0 or dilation_height <= 0 or dilation_width <= 0:
        raise ValueError("Neighborhood sizes and dilations must be positive")
    if height // dilation_height < kernel_height or width // dilation_width < kernel_width:
        raise ValueError("Neighborhood kernel exceeds a dilation group")
    if keys.shape[:2] != (batch, heads) or values.shape[:4] != keys.shape[:4]:
        raise ValueError("Neighborhood key/value batch and spatial shapes must match")
    if keys.shape[-1] != query_channels:
        raise ValueError("Neighborhood query/key channel counts must match")
    if (source_height * upsample[0], source_width * upsample[1]) != (height, width):
        raise ValueError("Neighborhood source shape does not match nearest upsampling")
    dtype = (mx.zeros((), dtype=queries.dtype) + mx.zeros((), dtype=keys.dtype) +
             mx.zeros((), dtype=values.dtype)).dtype
    template = [
        ("OutputType", dtype), ("BatchHeads", batch * heads),
        ("Height", height), ("Width", width),
        ("SourceHeight", source_height), ("SourceWidth", source_width),
        ("QueryChannels", query_channels), ("ValueChannels", values.shape[-1]),
        ("KernelHeight", kernel_height), ("KernelWidth", kernel_width),
        ("DilationHeight", dilation_height), ("DilationWidth", dilation_width),
        ("UpsampleHeight", upsample[0]), ("UpsampleWidth", upsample[1]),
    ]
    query_count = batch * heads * height * width
    return _naf_kernel()(
        inputs=[queries, keys, values, mx.array([scale], dtype=mx.float32)],
        template=template, grid=(((query_count + 3) // 4) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[queries.shape[:-1] + (values.shape[-1],)],
        output_dtypes=[dtype],
    )[0]
