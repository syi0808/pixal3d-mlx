"""Research-only decoder ablations; generation entrypoints do not import these."""

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .models.shape_slat_decoder import SLatDecoder, SparseChannel2Spatial, SparseResBlockC2S3d, _layernorm_noaffine
from .modules.sparse_conv import NeighborMap, SparseConv3d, build_neighbor_map


@dataclass(frozen=True)
class DecoderExperiment:
    phase_barriers: bool = False
    early_topology: bool = False
    fused_skip: bool = False
    terminal_chunk_size: int = 0
    convolution_eval_interval: int = 0
    convolution_scope: str = "all"


class BoundedSparseConv3d(SparseConv3d):
    def __init__(self, in_channels, out_channels, kernel_size=3, eval_interval=1):
        super().__init__(in_channels, out_channels, kernel_size)
        if eval_interval <= 0:
            raise ValueError("Convolution evaluation interval must be positive")
        self.eval_interval = eval_interval

    def __call__(self, feats, neighbor_map):
        if not isinstance(neighbor_map, NeighborMap):
            raise ValueError("Bounded convolution requires grouped neighbor indices")
        weights = self.weight.reshape(self.out_channels, self.kernel_size ** 3, self.in_channels).transpose(1, 2, 0)
        output = mx.zeros((feats.shape[0], self.out_channels))
        for kernel_index, (sources, targets) in enumerate(neighbor_map.groups):
            if sources.size:
                edge_output = feats[sources] @ weights[kernel_index]
                output = output.at[targets].add(edge_output)
                del edge_output
            if (kernel_index + 1) % self.eval_interval == 0:
                mx.eval(output)
        return output + self.bias


@lru_cache(maxsize=1)
def _skip_kernel():
    return mx.fast.metal_kernel(
        name="pixal3d_research_skip_add",
        input_names=["convolved", "parents", "selection"], output_names=["output"],
        source=r"""
        uint element = thread_position_in_grid.x;
        if (element >= Rows * OutputChannels) return;
        uint row = element / OutputChannels;
        uint channel = element % OutputChannels;
        uint selected = selection[row];
        uint parent = selected / 8;
        uint child = selected % 8;
        uint source_channel = child * ChildChannels + channel / Repeat;
        output[element] = OutputType(convolved[element]) + OutputType(parents[parent * ParentChannels + source_channel]);
        """,
    )


def fused_skip_add(convolved, parents, selection):
    parent_channels = parents.shape[1]
    output_channels = convolved.shape[1]
    child_channels = parent_channels // 8
    if parent_channels % 8 or not child_channels or selection.size != convolved.shape[0]:
        raise ValueError("Invalid channel-to-spatial skip layout")
    if output_channels > child_channels and output_channels % child_channels:
        raise ValueError("Skip repeats must match the output channels")
    if not convolved.shape[0]:
        return convolved
    dtype = (mx.zeros((), dtype=convolved.dtype) + mx.zeros((), dtype=parents.dtype)).dtype
    return _skip_kernel()(
        inputs=[convolved, parents, selection],
        template=[("Rows", convolved.shape[0]), ("OutputChannels", output_channels),
                  ("ParentChannels", parent_channels), ("ChildChannels", child_channels),
                  ("Repeat", max(1, output_channels // child_channels)), ("OutputType", dtype)],
        grid=(convolved.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[convolved.shape], output_dtypes=[dtype],
    )[0]


class ResearchUpsampleBlock(SparseResBlockC2S3d):
    def __init__(self, channels, out_channels, pred_subdiv, experiment, record=None, label="upsample"):
        super().__init__(channels, out_channels, pred_subdiv=pred_subdiv)
        object.__setattr__(self, "experiment", experiment)
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "label", label)

    def _barrier(self, phase, *values):
        if self.experiment.phase_barriers:
            mx.eval(*values)
            if self.record is not None:
                self.record(self.label + "/" + phase)

    def __call__(self, feats, coords, neighbor_map, subdiv=None, return_neighbor_map=False):
        options = self.experiment
        if not (options.phase_barriers or options.early_topology or options.fused_skip):
            return super().__call__(feats, coords, neighbor_map, subdiv=subdiv, return_neighbor_map=return_neighbor_map)
        if self.pred_subdiv:
            logits = self.to_subdiv(feats)
            mask = logits > 0
        else:
            logits = subdiv
            mask = mx.ones((feats.shape[0], 8), dtype=mx.bool_) if subdiv is None else subdiv > 0
        selection = None
        new_neighbors = None
        if options.early_topology:
            selection = SparseChannel2Spatial.prepare(coords, mask)
            new_neighbors = build_neighbor_map(selection[1])
            mx.eval(new_neighbors)
            if self.record is not None:
                self.record(self.label + "/early_topology")
        hidden = self.conv1(nn.silu(self.norm1(feats)), neighbor_map)
        self._barrier("conv1", hidden)
        if selection is None:
            selection = SparseChannel2Spatial.prepare(coords, mask)
        expanded, new_coords = SparseChannel2Spatial.upsample(hidden, coords, mask, selection=selection)
        self._barrier("selected_children", expanded)
        del hidden
        if not expanded.shape[0]:
            if return_neighbor_map:
                return expanded, new_coords, logits, build_neighbor_map(new_coords)
            return expanded, new_coords, logits
        skip = None
        if not options.fused_skip:
            skip, _ = SparseChannel2Spatial.upsample(feats, coords, mask, selection=selection)
            repeat = self.out_channels // (self.channels // 8)
            if repeat > 1:
                skip = mx.repeat(skip, repeat, axis=1)
            skip = skip[:, :self.out_channels]
            self._barrier("materialized_skip", skip)
        if new_neighbors is None:
            new_neighbors = build_neighbor_map(new_coords)
        self._barrier("topology", new_neighbors)
        expanded = nn.silu(self.norm2(expanded))
        self._barrier("norm2", expanded)
        expanded = self.conv2(expanded, new_neighbors)
        self._barrier("conv2", expanded)
        result = fused_skip_add(expanded, feats, selection[0]) if options.fused_skip else expanded + skip
        self._barrier("skip_added", result)
        if return_neighbor_map:
            return result, new_coords, logits, new_neighbors
        return result, new_coords, logits


class ResearchDecoder(SLatDecoder):
    def __init__(self, experiment=DecoderExperiment(), record=None, **kwargs):
        super().__init__(**kwargs)
        if experiment.terminal_chunk_size < 0:
            raise ValueError("Terminal chunk size cannot be negative")
        if experiment.convolution_eval_interval < 0:
            raise ValueError("Convolution evaluation interval cannot be negative")
        if experiment.convolution_scope not in ("all", "final-projection", "final-upsample"):
            raise ValueError("Unknown bounded convolution scope")
        object.__setattr__(self, "experiment", experiment)
        object.__setattr__(self, "record", record)
        for level_index, level in enumerate(self.blocks):
            for block_index, block in enumerate(level):
                if isinstance(block, SparseResBlockC2S3d):
                    level[block_index] = ResearchUpsampleBlock(block.channels, block.out_channels, block.pred_subdiv,
                                                              experiment, record, f"level_{level_index}")
                if experiment.convolution_eval_interval:
                    current_block = level[block_index]
                    for name, convolution in list(current_block.items()):
                        if experiment.convolution_scope != "all":
                            if level_index != len(self.blocks) - 2 or not isinstance(current_block, SparseResBlockC2S3d):
                                continue
                            if experiment.convolution_scope == "final-projection" and name != "conv1":
                                continue
                        if isinstance(convolution, SparseConv3d):
                            current_block[name] = BoundedSparseConv3d(convolution.in_channels, convolution.out_channels,
                                                                     convolution.kernel_size, experiment.convolution_eval_interval)

    def __call__(self, feats, coords, guide_subs=None, return_subs=False):
        chunk_size = self.experiment.terminal_chunk_size
        if not chunk_size:
            return super().__call__(feats, coords, guide_subs=guide_subs, return_subs=return_subs)
        feats, coords = self._order_input(feats, coords)
        feats = self.from_latent(feats)
        result = self._forward_levels(feats, coords, guide_subs=guide_subs, return_subs=return_subs)
        subdivisions = result[2] if return_subs else None
        feats, coords = result[:2]
        del result
        mx.eval(feats)
        if self.record is not None:
            self.record("terminal_input_materialized", rows=feats.shape[0], feature_bytes=feats.nbytes)
        outputs = []
        for start in range(0, feats.shape[0], chunk_size):
            output = self.output_layer(_layernorm_noaffine(feats[start:start + chunk_size], eps=1e-5))
            mx.eval(output)
            outputs.append(output)
        output = mx.concatenate(outputs, axis=0) if outputs else self.output_layer(feats)
        return (output, coords, subdivisions) if return_subs else (output, coords)
