import mlx.core as mx
import mlx.utils
import numpy as np
import pytest

from pixalmlx.models.pixal3d_flow import Pixal3DSLatFlowModel, Pixal3DSparseStructureFlowModel
from pixalmlx.samplers import flow_euler_sample
from pixalmlx.transformer_experiments import VARIANTS, configure_transformer
from pixalmlx.conditioning import ZeroConditioning


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_transformer_experiments_preserve_weights_fp32_state_and_close_values(dense, variant):
    configuration = dict(in_channels=8, out_channels=8, model_channels=32, num_heads=4, num_blocks=2,
                         mlp_hidden=64, context_channels=16, proj_in_channels=12)
    factory = Pixal3DSparseStructureFlowModel if dense else Pixal3DSLatFlowModel
    reference = factory(**configuration, **({"resolution": 2} if dense else {}))
    experiment = factory(**configuration, **({"resolution": 2} if dense else {}))
    experiment.load_weights(mlx.utils.tree_flatten(reference.parameters()))
    before = {name: np.array(value) for name, value in mlx.utils.tree_flatten(experiment.parameters())}
    options = VARIANTS[variant]
    configure_transformer(experiment, options)
    features = mx.random.normal((1, 8, 2, 2, 2)) if dense else mx.random.normal((9, 8))
    tokens = 8 if dense else 9
    context = {"global": mx.random.normal((1, 5, 16)), "proj": mx.random.normal((1, tokens, 12))}
    negative = ZeroConditioning({name: mx.zeros_like(value) for name, value in context.items()})
    coordinates = {} if dense else {"coords": mx.arange(27).reshape(9, 3)}
    expected = flow_euler_sample(reference, features, context, negative, steps=3, ffn_chunk_size=4, verbose=False, **coordinates)
    actual = flow_euler_sample(experiment, features, context, negative, steps=3, ffn_chunk_size=options.ffn_chunk_size,
                               conditioning_cache_bytes=options.cache_bytes, verbose=False, **coordinates)
    assert actual.dtype == mx.float32
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-4, atol=2e-5)
    after = {name: np.array(value) for name, value in mlx.utils.tree_flatten(experiment.parameters())}
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])


def test_asynchronous_ffn_evaluation_preserves_sync_reference_with_real_chunks():
    from pixalmlx.models.sparse_structure_flow import FeedForward
    from pixalmlx.transformer_experiments import ResearchFeedForward, TransformerExperiment

    original = FeedForward(32, 64)
    original.chunk_size = 4
    experiment = ResearchFeedForward(original, TransformerExperiment(ffn_chunk_size=4, async_ffn=True))
    features = mx.random.normal((19, 32))
    expected = original(features)
    mx.eval(expected)
    np.testing.assert_array_equal(np.array(experiment(features)), np.array(expected))


def test_pure_compiled_blocks_do_not_capture_a_previous_image_context():
    configuration = dict(in_channels=8, out_channels=8, model_channels=32, num_heads=4,
                         num_blocks=2, mlp_hidden=64, context_channels=16, proj_in_channels=12)
    reference = Pixal3DSLatFlowModel(**configuration)
    compiled = Pixal3DSLatFlowModel(**configuration)
    compiled.load_weights(mlx.utils.tree_flatten(reference.parameters()))
    options = VARIANTS["compiled-block"]
    configure_transformer(compiled, options)
    noise = mx.random.normal((9, 8))
    coordinates = mx.arange(27).reshape(9, 3)
    outputs = []
    for value in (0.5, -1.0):
        context = {"global": mx.full((1, 5, 16), value), "proj": mx.full((1, 9, 12), value)}
        negative = ZeroConditioning({name: mx.zeros_like(array) for name, array in context.items()})
        expected = flow_euler_sample(reference, noise, context, negative, steps=3, verbose=False, coords=coordinates)
        actual = flow_euler_sample(compiled, noise, context, negative, steps=3, verbose=False,
                                   coords=coordinates, ffn_chunk_size=0)
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=3e-4, atol=2e-5)
        outputs.append(np.array(actual))
    assert not np.array_equal(outputs[0], outputs[1])
