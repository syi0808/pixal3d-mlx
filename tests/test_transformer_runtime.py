import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import pytest

from pixalmlx.conditioning import ZeroConditioning
from pixalmlx.models.pixal3d_flow import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel
from pixalmlx.models.sparse_structure_flow import FeedForward
from pixalmlx.samplers import flow_euler_sample


@pytest.mark.parametrize("dense", [True, False])
def test_selective_compilation_keeps_twelve_steps_forward_schedule_and_parameters(dense):
    arguments = dict(in_channels=8, out_channels=8, model_channels=32, num_heads=4, num_blocks=2,
                     mlp_hidden=64, context_channels=16, proj_in_channels=12)
    model = (Pixal3DSparseStructureFlowModel(**arguments, resolution=2) if dense else
             Pixal3DSLatFlowModel(**arguments))
    noise = mx.random.normal((1, 8, 2, 2, 2)) if dense else mx.random.normal((9, 8))
    tokens = 8 if dense else 9
    context = {"global": mx.random.normal((1, 5, 16)), "proj": mx.random.normal((1, tokens, 12))}
    negative = ZeroConditioning({name: mx.zeros_like(value) for name, value in context.items()})
    kwargs = {} if dense else {"coords": mx.arange(27).reshape(9, 3)}
    before = {name: np.array(value) for name, value in mlx.utils.tree_flatten(model.parameters())}

    class RecordingFlow:
        def __init__(self, inner):
            self.inner = inner
            self.timesteps = []

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __call__(self, features, timestep, conditioning, **options):
            self.timesteps.append(float(timestep.item()))
            return self.inner(features, timestep, conditioning, **options)

    recording = RecordingFlow(model)
    expected = flow_euler_sample(recording, noise, context, negative, steps=12,
                                 ffn_chunk_size=4, verbose=False, selective_compile=False, **kwargs)
    expected_schedule = list(recording.timesteps)
    recording.timesteps.clear()
    actual = flow_euler_sample(recording, noise, context, negative, steps=12,
                               ffn_chunk_size=4, verbose=False, selective_compile=True, **kwargs)
    assert len(recording.timesteps) == 22
    assert recording.timesteps == expected_schedule
    assert actual.dtype == mx.float32
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-4, atol=2e-5)
    assert all(block.selective_compile and block.mlp.selective_compile for block in model.blocks)
    assert all(block.mlp.chunk_size == 4 for block in model.blocks)
    after = dict(mlx.utils.tree_flatten(model.parameters()))
    assert before.keys() == after.keys()
    for name, value in before.items():
        np.testing.assert_array_equal(np.array(after[name]), value)
    flow_euler_sample(model, noise, context, negative, steps=1, verbose=False, selective_compile=False, **kwargs)
    assert all(not block.selective_compile and not block.mlp.selective_compile for block in model.blocks)


@pytest.mark.parametrize("bits", [4, 8])
def test_selective_ffn_compilation_falls_back_for_packed_quantized_weights(monkeypatch, bits):
    import pixalmlx.transformer_runtime as runtime

    module = FeedForward(64, 128)
    nn.quantize(module, group_size=64, bits=bits)
    features = mx.random.normal((9, 64))
    expected = module(features)
    mx.eval(expected)

    def unsupported():
        raise AssertionError("Packed weights must not reach the floating GEMM compiler")

    monkeypatch.setattr(runtime, "compiled_ffn", unsupported)
    module.selective_compile = True
    module.chunk_size = 4
    actual = module(features)
    assert module.mlp_0.weight.dtype == mx.uint32
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=2e-6)


def test_compiled_ffn_reads_explicit_current_weights_instead_of_capturing_a_model():
    module = FeedForward(32, 64)
    module.selective_compile = True
    features = mx.random.normal((5, 32))
    before = np.array(module(features))
    module.mlp_2.bias = module.mlp_2.bias + 1
    after = np.array(module(features))
    np.testing.assert_allclose(after, before + 1, rtol=1e-5, atol=2e-6)
