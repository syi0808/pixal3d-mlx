import mlx.core as mx
import mlx.utils
import numpy as np
import pytest

from pixalmlx.conditioning import PreparedConditioning
from pixalmlx.models.pixal3d_flow import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel
from pixalmlx.models.sparse_structure_flow import FeedForward
from pixalmlx.samplers import flow_euler_sample


def test_bounded_cache_admits_fixed_working_set_without_scan_thrashing():
    context = PreparedConditioning({}, 16)
    calls = []
    for _ in range(3):
        for index in range(4):
            context.invariant(index, lambda: calls.append(index) or mx.ones((4,)))
    assert calls.count(0) == 1
    assert len(calls) == 10
    assert context.used_bytes == 16
    context.clear_invariants()
    assert not context.values_cache


def test_chunked_ffn_matches_reference_and_keeps_parameters_unchanged():
    module = FeedForward(32, 64)
    features = mx.random.normal((19, 32))
    keys = [name for name, _ in mlx.utils.tree_flatten(module.parameters())]
    expected = module(features)
    module.chunk_size = 5
    actual = module(features)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=2e-6)
    assert keys == [name for name, _ in mlx.utils.tree_flatten(module.parameters())]


@pytest.mark.parametrize("dense", [False, True])
def test_sampling_cache_preserves_outputs_and_computes_rope_once(monkeypatch, dense):
    arguments = dict(model_channels=32, num_heads=4, num_blocks=2, mlp_hidden=64,
                     context_channels=16, proj_in_channels=12, in_channels=8, out_channels=8)
    if dense:
        model = Pixal3DSparseStructureFlowModel(**arguments, resolution=2)
        noise = mx.random.normal((1, 8, 2, 2, 2))
        coordinates = {}
        tokens = 8
    else:
        model = Pixal3DSLatFlowModel(**arguments)
        noise = mx.random.normal((9, 8))
        coordinates = {"coords": mx.arange(27).reshape(9, 3)}
        tokens = 9
    cond = {"global": mx.random.normal((1, 5, 16)), "proj": mx.random.normal((1, tokens, 12))}
    neg = {name: mx.zeros_like(value) for name, value in cond.items()}
    expected = flow_euler_sample(model, noise, cond, neg, steps=3, verbose=False,
                                 conditioning_cache_bytes=0, ffn_chunk_size=0, **coordinates)
    if not dense:
        calls = []
        original = model._coords_to_rope_phases

        def capture(coords):
            calls.append(1)
            return original(coords)

        monkeypatch.setattr(model, "_coords_to_rope_phases", capture)
    actual = flow_euler_sample(model, noise, cond, neg, steps=3, verbose=False,
                               conditioning_cache_bytes=1024 * 1024, ffn_chunk_size=4, **coordinates)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-4, atol=2e-5)
    if not dense:
        assert len(calls) == 1


def test_sampler_clears_invariant_cache_when_a_block_fails(monkeypatch):
    import pixalmlx.conditioning as conditioning

    model = Pixal3DSLatFlowModel(in_channels=8, out_channels=8, model_channels=32,
                                 num_heads=4, num_blocks=1, mlp_hidden=64,
                                 context_channels=16, proj_in_channels=12)
    cond = {"global": mx.ones((1, 5, 16)), "proj": mx.ones((1, 4, 12))}
    captured = []
    original = conditioning.prepare_sampling

    def prepare(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.extend(result[-1])
        return result

    def fail(*args, **kwargs):
        raise RuntimeError("block failed")

    monkeypatch.setattr(conditioning, "prepare_sampling", prepare)
    monkeypatch.setattr(model.blocks[0].mlp, "__call__", fail)
    monkeypatch.setattr(type(model.blocks[0].mlp), "__call__", fail)
    with pytest.raises(RuntimeError, match="block failed"):
        flow_euler_sample(model, mx.ones((4, 8)), cond, cond, steps=2, verbose=False)
    assert captured
    assert all(not context.values_cache for context in captured)


@pytest.mark.parametrize("dense", [True, False])
def test_experimental_mixed_precision_preserves_fp32_sampler_state(dense):
    arguments = dict(in_channels=8, out_channels=8, model_channels=32, num_heads=4,
                     num_blocks=1, mlp_hidden=64, context_channels=16, proj_in_channels=12)
    model = Pixal3DSparseStructureFlowModel(**arguments, resolution=2) if dense else Pixal3DSLatFlowModel(**arguments)
    model.apply(lambda value: value.astype(mx.float16))
    model.compute_dtype = mx.float16
    noise = mx.random.normal((1, 8, 2, 2, 2)) if dense else mx.random.normal((8, 8))
    cond = {"global": mx.ones((1, 5, 16)), "proj": mx.ones((1, 8, 12))}
    output = flow_euler_sample(model, noise, cond, cond, steps=2, verbose=False, compile_step=True)
    assert output.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(output)).item())


def test_symbolic_unconditional_context_eliminates_query_and_projection_gemms(monkeypatch):
    from pixalmlx.conditioning import ZeroConditioning

    model = Pixal3DSLatFlowModel(in_channels=8, out_channels=8, model_channels=32,
                                 num_heads=4, num_blocks=1, mlp_hidden=64,
                                 context_channels=16, proj_in_channels=12)
    attention = model.blocks[0].cross_attn
    features = mx.random.normal((9, 32))
    zeros = ZeroConditioning({"global": mx.zeros((1, 5, 16)), "proj": mx.zeros((1, 9, 12))})
    expected = attention(features, dict(zeros))
    mx.eval(expected)
    calls = []
    original = mx.fast.scaled_dot_product_attention

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", capture)
    context = PreparedConditioning(zeros, 1024)
    actual = attention(features, context)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=2e-6)
    assert not calls
    assert context.used_bytes <= 32 * 4


def test_zero_conditioning_sampling_matches_explicit_zero_features():
    from pixalmlx.conditioning import ZeroConditioning

    model = Pixal3DSLatFlowModel(in_channels=8, out_channels=8, model_channels=32,
                                 num_heads=4, num_blocks=2, mlp_hidden=64,
                                 context_channels=16, proj_in_channels=12)
    noise = mx.random.normal((9, 8))
    cond = {"global": mx.random.normal((1, 5, 16)), "proj": mx.random.normal((1, 9, 12))}
    neg = {name: mx.zeros_like(value) for name, value in cond.items()}
    expected = flow_euler_sample(model, noise, cond, neg, steps=4, verbose=False, conditioning_cache_bytes=0)
    actual = flow_euler_sample(model, noise, cond, ZeroConditioning(neg), steps=4, verbose=False)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-4, atol=2e-5)
