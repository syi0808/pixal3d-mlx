import mlx.core as mx
import mlx.utils
import numpy as np
import pytest

from trellmlx.decoder_experiments import DecoderExperiment, ResearchDecoder, fused_skip_add
from trellmlx.models.shape_slat_decoder import SLatDecoder


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
