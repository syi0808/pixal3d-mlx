import mlx.core as mx
import mlx.utils
import numpy as np
import pytest

from pixalmlx.models.shape_slat_decoder import SLatDecoder, SparseResBlockC2S3d
from pixalmlx.modules.sparse_conv import SparseConv3d, build_neighbor_map


@pytest.mark.parametrize("interval", [1, 3, 9])
@pytest.mark.parametrize("weight_dtype", [mx.float16, mx.float32])
def test_bounded_sparse_convolution_keeps_kernel_accumulation_values(interval, weight_dtype):
    coordinates = mx.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 65535, 0, 0]])
    neighbors = build_neighbor_map(coordinates)
    reference = SparseConv3d(8, 16)
    reference.apply(lambda value: value.astype(weight_dtype))
    bounded = SparseConv3d(8, 16, eval_interval=interval)
    bounded.load_weights(mlx.utils.tree_flatten(reference.parameters()))
    features = mx.random.normal((4, 8))
    expected = reference(features, neighbors)
    mx.eval(expected)
    actual = bounded(features, neighbors)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("spatial_order", ["preserve", "morton"])
@pytest.mark.parametrize("empty", [False, True])
def test_shape_subdivision_and_guided_texture_match_with_final_upsample_barriers(spatial_order, empty):
    configuration = dict(latent_channels=8, model_channels=[32, 16, 8],
                         num_blocks=[0, 0, 0] if empty else [1, 1, 0], spatial_order=spatial_order)
    reference = SLatDecoder(out_channels=7, pred_subdiv=True, **configuration)
    for level in reference.blocks:
        for block in level:
            if isinstance(block, SparseResBlockC2S3d):
                block.to_subdiv.weight = mx.zeros_like(block.to_subdiv.weight)
                block.to_subdiv.bias = mx.array([-1.0] * 8 if empty else [-1.0, 1.0] * 4)
    bounded = SLatDecoder(out_channels=7, pred_subdiv=True, final_upsample_eval_interval=1, **configuration)
    bounded.load_weights(mlx.utils.tree_flatten(reference.parameters()))
    features = mx.random.normal((3, 8))
    coordinates = mx.array([[1, 16383, 16382, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    expected, expected_coordinates, expected_subdivisions = reference(features, coordinates, return_subs=True)
    mx.eval(expected, expected_coordinates, expected_subdivisions)
    actual, actual_coordinates, actual_subdivisions = bounded(features, coordinates, return_subs=True)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    np.testing.assert_array_equal(np.array(actual_coordinates), np.array(expected_coordinates))
    for actual_subdivision, expected_subdivision in zip(actual_subdivisions, expected_subdivisions, strict=True):
        np.testing.assert_array_equal(np.array(actual_subdivision), np.array(expected_subdivision))
    np.testing.assert_array_equal(np.array(bounded.upsample(features, coordinates, upsample_times=2)),
                                  np.array(expected_coordinates))
    texture = SLatDecoder(out_channels=6, pred_subdiv=False, **configuration)
    bounded_texture = SLatDecoder(out_channels=6, pred_subdiv=False, final_upsample_eval_interval=1, **configuration)
    bounded_texture.load_weights(mlx.utils.tree_flatten(texture.parameters()))
    expected_texture, expected_texture_coordinates = texture(features, coordinates, guide_subs=expected_subdivisions)
    mx.eval(expected_texture, expected_texture_coordinates)
    actual_texture, actual_texture_coordinates = bounded_texture(features, coordinates, guide_subs=actual_subdivisions)
    np.testing.assert_array_equal(np.array(actual_texture), np.array(expected_texture))
    np.testing.assert_array_equal(np.array(actual_texture_coordinates), np.array(expected_texture_coordinates))


def test_only_last_upsample_convolutions_have_evaluation_barriers():
    decoder = SLatDecoder(model_channels=[32, 16, 8], num_blocks=[1, 1, 0], final_upsample_eval_interval=3)
    for level_index, level in enumerate(decoder.blocks):
        for block in level:
            if isinstance(block, SparseResBlockC2S3d):
                expected_interval = 3 if level_index == 1 else 0
                assert block.conv1.eval_interval == block.conv2.eval_interval == expected_interval
            else:
                assert block.conv.eval_interval == 0


@pytest.mark.parametrize("interval", [-1, 0.5])
def test_invalid_decoder_intervals_are_rejected(interval):
    with pytest.raises(ValueError, match="nonnegative integer"):
        SLatDecoder(final_upsample_eval_interval=interval)
    with pytest.raises(ValueError, match="nonnegative integer"):
        SparseConv3d(8, 8, eval_interval=interval)


@pytest.mark.parametrize("interval", [None, 0, 3])
def test_resume_forwarding_preserves_decoder_policy(monkeypatch, interval):
    import generate_pixal3d
    import resume_pixal3d

    forwarded = []
    monkeypatch.setattr(resume_pixal3d, "main", lambda argv, runtime=None: forwarded.extend(argv))
    arguments = ["--image", "unused.png", "--resume", "unused", "--max-faces", "5000"]
    if interval is not None:
        arguments.extend(["--decoder-eval-interval", str(interval)])
    generate_pixal3d.main(arguments)
    assert forwarded[forwarded.index("--decoder-eval-interval") + 1] == str(1 if interval is None else interval)
    assert forwarded[forwarded.index("--max-faces") + 1] == "5000"


@pytest.mark.parametrize("entrypoint", ["generate_pixal3d", "resume_pixal3d"])
def test_cli_rejects_negative_decoder_interval_before_loading_models(entrypoint):
    from importlib import import_module

    arguments = ["--image", "unused.png", "--decoder-eval-interval", "-1"]
    if entrypoint == "resume_pixal3d":
        arguments.extend(["--ckpt", "unused"])
    with pytest.raises(SystemExit) as error:
        import_module(entrypoint).main(arguments)
    assert error.value.code == 2


def test_decoder_array_signatures_compare_raw_values_and_empty_shapes():
    from scripts.validate_decoder_memory import array_signature

    original = np.array([[0.0, -0.0], [1.0, 2.0]], dtype=np.float32)
    same = array_signature(original.copy())
    assert same == array_signature(original)
    assert same["all_finite"]
    changed = original.copy()
    changed[0, 1] = 0.0
    assert array_signature(changed)["sha256"] != same["sha256"]
    assert array_signature(np.empty((0, 7), dtype=np.float32))["shape"] == [0, 7]
    assert not array_signature(np.array([np.nan]))["all_finite"]
