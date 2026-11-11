"""Opt-in Transformer execution experiments; production entrypoints do not import this."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .models.pixal3d_flow import ProjModulatedBlock
from .models.sparse_structure_flow import FeedForward, _layernorm_noaffine
from .transformer_runtime import (
    compiled_ffn as _compiled_ffn,
    compiled_gated_residual as _compiled_gated_residual,
    compiled_modulation as _compiled_modulation,
)


@dataclass(frozen=True)
class TransformerExperiment:
    ffn_chunk_size: int = 2048
    eval_every: int = 2
    async_ffn: bool = False
    compiled_ffn: bool = False
    compiled_pointwise: bool = False
    compiled_block: bool = False
    cache_bytes: int = 128 * 1024 ** 2


VARIANTS = {
    "reference": TransformerExperiment(),
    "async-ffn": TransformerExperiment(async_ffn=True),
    "full-ffn": TransformerExperiment(ffn_chunk_size=0),
    "full-ffn-eval1": TransformerExperiment(ffn_chunk_size=0, eval_every=1),
    "compiled-ffn": TransformerExperiment(compiled_ffn=True),
    "compiled-pointwise": TransformerExperiment(compiled_pointwise=True),
    "compiled-both": TransformerExperiment(compiled_ffn=True, compiled_pointwise=True),
    "cache-512": TransformerExperiment(cache_bytes=512 * 1024 ** 2),
    "compiled-block": TransformerExperiment(ffn_chunk_size=0, eval_every=1, compiled_block=True),
    "compiled-block-pairs": TransformerExperiment(ffn_chunk_size=0, compiled_block=True),
}


class ResearchFeedForward(FeedForward):
    def __init__(self, original, options):
        nn.Module.__init__(self)
        self.mlp_0 = original.mlp_0
        self.mlp_2 = original.mlp_2
        self.chunk_size = options.ffn_chunk_size
        object.__setattr__(self, "options", options)

    def _chunk(self, features):
        if self.options.compiled_ffn:
            return _compiled_ffn()(features, self.mlp_0.weight, self.mlp_0.bias, self.mlp_2.weight, self.mlp_2.bias)
        return self.mlp_2(nn.gelu_approx(self.mlp_0(features)))

    def __call__(self, features):
        if not self.chunk_size or features.ndim != 2 or features.shape[0] <= self.chunk_size:
            return self._chunk(features)
        outputs = []
        evaluate = mx.async_eval if self.options.async_ffn else mx.eval
        for start in range(0, features.shape[0], self.chunk_size):
            output = self._chunk(features[start:start + self.chunk_size])
            evaluate(output)
            outputs.append(output)
        return mx.concatenate(outputs, axis=0)


class ResearchBlock(ProjModulatedBlock):
    def __init__(self, original):
        nn.Module.__init__(self)
        for name, value in original.items():
            self[name] = value

    def __call__(self, features, modulation, context, rope_phases=None):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mx.split(
            modulation + self.modulation, 6)
        hidden = _compiled_modulation()(features, scale_attn, shift_attn)
        hidden = self.self_attn(hidden, rope_phases=rope_phases)
        features = _compiled_gated_residual()(features, hidden, gate_attn)
        features = features + self.cross_attn(self.norm2(features), context)
        hidden = _compiled_modulation()(features, scale_mlp, shift_mlp)
        return _compiled_gated_residual()(features, self.mlp(hidden), gate_mlp)


class CompiledPureBlock(ProjModulatedBlock):
    def __init__(self, original):
        nn.Module.__init__(self)
        for name, value in original.items():
            self[name] = value
        object.__setattr__(self, "positive", mx.compile(self._positive))
        object.__setattr__(self, "negative", mx.compile(self._negative))

    def _self_attention(self, features, modulation, rope_phases):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mx.split(
            modulation + self.modulation, 6)
        hidden = _layernorm_noaffine(features) * (1 + scale_attn) + shift_attn
        features = features + self.self_attn(hidden, rope_phases=rope_phases) * gate_attn
        return features, shift_mlp, scale_mlp, gate_mlp

    def _positive(self, features, modulation, rope_phases, global_context, keys, values, projection):
        features, shift, scale, gate = self._self_attention(features, modulation, rope_phases)
        global_output = self.cross_attn.cross_attn_block(self.norm2(features), global_context, prepared_kv=(keys, values))
        features = features + (projection + global_output)
        hidden = _layernorm_noaffine(features) * (1 + scale) + shift
        return features + self.mlp(hidden) * gate

    def _negative(self, features, modulation, rope_phases, constant):
        features, shift, scale, gate = self._self_attention(features, modulation, rope_phases)
        features = features + mx.broadcast_to(constant, features.shape)
        hidden = _layernorm_noaffine(features) * (1 + scale) + shift
        return features + self.mlp(hidden) * gate

    def __call__(self, features, modulation, context, rope_phases=None):
        from .conditioning import PreparedConditioning

        if not isinstance(context, PreparedConditioning):
            raise ValueError("Pure compiled blocks require sampling-scoped prepared conditioning")
        inner = self.cross_attn.cross_attn_block
        global_context = context["global"]
        if context.zero_conditioning:
            def unconditional():
                projected = inner.to_kv(global_context[:, :1])
                values = projected.reshape(1, 1, 2, inner.num_heads, inner.head_dim)[:, :, 1]
                constant = inner.to_out(values.reshape(1, inner.channels))
                return constant + self.cross_attn.proj_linear.bias

            constant = context.invariant((id(self.cross_attn), "unconditional"), unconditional)
            return self.negative(features, modulation, rope_phases, constant)
        keys, values = context.invariant((id(self.cross_attn), "global_kv"), lambda: inner.prepare_context(global_context))
        projection = context.invariant((id(self.cross_attn), "projection"),
                                        lambda: self.cross_attn.proj_linear(context["proj"]))
        if projection.ndim == 3:
            projection = projection.reshape(-1, projection.shape[-1])
        return self.positive(features, modulation, rope_phases, global_context, keys, values, projection)


def configure_transformer(model, options):
    if options.ffn_chunk_size < 0 or options.eval_every < 1 or options.cache_bytes < 0:
        raise ValueError("Invalid Transformer experiment limits")
    if options.compiled_block and (options.ffn_chunk_size or not options.cache_bytes):
        raise ValueError("Pure compiled blocks require unchunked FFN and prepared conditioning")
    model.eval_every = options.eval_every
    for index, block in enumerate(model.blocks):
        block.mlp = ResearchFeedForward(block.mlp, options)
        if options.compiled_block:
            model.blocks[index] = CompiledPureBlock(block)
        elif options.compiled_pointwise:
            model.blocks[index] = ResearchBlock(block)
    return model
