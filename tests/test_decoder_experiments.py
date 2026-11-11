import mlx.core as mx
import mlx.utils
import numpy as np
import pytest

from scripts.investigate_decoder import preserve_rope_dtype
from pixalmlx.decoder_experiments import DecoderExperiment, ResearchDecoder, fused_skip_add
from pixalmlx.models.shape_slat_decoder import SLatDecoder


@pytest.mark.parametrize("parent_channels,output_channels", [(32, 16), (32, 4), (32, 2)])
def test_fused_skip_preserves_channel_repeat_order(parent_channels, output_channels):
    parents = mx.random.normal((3, parent_channels))
    selection = mx.array([0, 2, 5, 9, 13, 23], dtype=mx.int32)
    convolved = mx.random.normal((6, output_channels))
    skip = parents.reshape(-1, parent_channels // 8)[selection]
    repeat = output_channels // (parent_channels // 8)
    if repeat > 1:
        skip = mx.repeat(skip, repeat, axis=1)
    expected = convolved + skip[:, :output_channels]
    actual = fused_skip_add(convolved, parents, selection)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("options", [
    DecoderExperiment(phase_barriers=True),
    DecoderExperiment(early_topology=True),
    DecoderExperiment(fused_skip=True),
    DecoderExperiment(terminal_chunk_size=3),
    DecoderExperiment(convolution_eval_interval=1),
    DecoderExperiment(convolution_eval_interval=3),
    DecoderExperiment(convolution_eval_interval=9, phase_barriers=True),
    DecoderExperiment(convolution_eval_interval=3, convolution_scope="final-projection"),
    DecoderExperiment(convolution_eval_interval=1, convolution_scope="final-upsample"),
    DecoderExperiment(phase_barriers=True, early_topology=True, fused_skip=True, terminal_chunk_size=3),
])
def test_decoder_ablation_preserves_weights_coordinates_and_values(options):
    arguments = dict(out_channels=6, latent_channels=8, model_channels=[32, 16, 8], num_blocks=[1, 1, 0],
                     pred_subdiv=False, spatial_order="morton")
    reference = SLatDecoder(**arguments)
    experiment = ResearchDecoder(experiment=options, **arguments)
    experiment.load_weights(mlx.utils.tree_flatten(reference.parameters()))
    features = mx.random.normal((3, 8))
    coordinates = mx.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 1, 0]])
    expected, expected_coordinates = reference(features, coordinates)
    mx.eval(expected, expected_coordinates)
    actual, actual_coordinates = experiment(features, coordinates)
    np.testing.assert_array_equal(np.array(actual_coordinates), np.array(expected_coordinates))
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-4, atol=2e-5)


def test_experimental_rotary_cast_is_scoped_and_restored_on_failure():
    from pixalmlx.models import sparse_structure_flow

    original = sparse_structure_flow.apply_rope
    query = mx.ones((2, 1, 8), dtype=mx.float16)
    phases = mx.ones((2, 4, 2), dtype=mx.float32)
    assert original(query, phases).dtype == mx.float32
    with pytest.raises(RuntimeError):
        with preserve_rope_dtype(True):
            actual = sparse_structure_flow.apply_rope(query, phases)
            assert actual.dtype == mx.float16
            np.testing.assert_array_equal(np.array(actual), np.array(original(query, phases).astype(mx.float16)))
            raise RuntimeError("research interrupted")
    assert sparse_structure_flow.apply_rope is original


def test_experimental_rotary_cast_keeps_block_activations_half_precision():
    from pixalmlx.models.pixal3d_flow import ProjModulatedBlock

    block = ProjModulatedBlock(24, 1, 8, 32, 8)
    block.apply(lambda value: value.astype(mx.float16))
    features = mx.zeros((4, 24), dtype=mx.float16)
    phases = mx.zeros((4, 12, 2), dtype=mx.float32)
    modulation = mx.zeros((144,), dtype=mx.float16)
    context = {"global": mx.zeros((1, 5, 8), dtype=mx.float16),
               "proj": mx.zeros((4, 8), dtype=mx.float16)}
    assert block(features, modulation, context, rope_phases=phases).dtype == mx.float32
    with preserve_rope_dtype(True):
        assert block(features, modulation, context, rope_phases=phases).dtype == mx.float16


@pytest.mark.parametrize("options", [
    DecoderExperiment(terminal_chunk_size=-1),
    DecoderExperiment(convolution_eval_interval=-1),
    DecoderExperiment(convolution_scope="unknown"),
])
def test_invalid_research_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        ResearchDecoder(experiment=options, model_channels=[8], num_blocks=[0])
