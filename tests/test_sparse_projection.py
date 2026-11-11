import mlx.core as mx
import numpy as np
import pytest

from generate_pixal3d import extract_proj_features, index_proj_by_coords
from pixalmlx.modules.proj_grid import ProjGrid


@pytest.mark.parametrize("resolution", [4, 8])
def test_active_projection_matches_dense_selection(resolution):
    projector = ProjGrid(grid_resolution=resolution, image_resolution=32)
    features = mx.random.normal((1, 8, 8, 6))
    coordinates = np.array([[0, 0, 0, 0], [0, resolution - 1, 2, 1], [0, 1, 0, 3]])
    camera = (mx.array([0.7]), mx.array([2.0]), mx.array([1.0]))
    dense = projector(features, *camera)
    sparse = projector(features, *camera, coords=coordinates)
    linear_indices = (coordinates[:, 1] * resolution + coordinates[:, 2]) * resolution + coordinates[:, 3]
    np.testing.assert_allclose(np.array(sparse), np.array(dense[:, mx.array(linear_indices)]), atol=1e-6)


def test_projection_rejects_invalid_coordinates():
    projector = ProjGrid(grid_resolution=4)
    with pytest.raises(ValueError, match="outside"):
        projector(mx.zeros((1, 4, 4, 2)), mx.array([0.7]), mx.array([2.0]), mx.array([1.0]),
                  coords=np.array([[0, 4, 0, 0]]))


def test_projection_indexing_keeps_features_on_mlx():
    dense = mx.arange(4 ** 3 * 3).reshape(1, 4 ** 3, 3).astype(mx.float16)
    coordinates = np.array([[0, 2, 1, 3], [0, 0, 0, 0]])
    actual, negative = index_proj_by_coords(
        {"global": mx.ones((1, 5, 3)), "proj": dense},
        {"global": mx.zeros((1, 5, 3)), "proj": mx.zeros_like(dense)},
        coordinates, 4, target_dim=5)
    expected = np.array(dense).reshape(1, 4, 4, 4, 3)[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]]
    np.testing.assert_array_equal(np.array(actual["proj"])[0, :, :3], expected)
    assert actual["proj"].shape == (1, 2, 5)
    assert actual["proj"].dtype == mx.float16
    assert not bool(mx.any(negative["proj"]))


def test_backbone_cache_reuses_encoding_without_reusing_projection(tmp_path):
    from PIL import Image

    class TinyBackbone:
        patch_size = 8
        hidden_size = 6
        num_prefix_tokens = 1

        def __init__(self):
            self.calls = 0

        def __call__(self, image):
            self.calls += 1
            tokens = (image.shape[1] // self.patch_size) ** 2 + self.num_prefix_tokens
            return mx.arange(tokens * self.hidden_size).reshape(1, tokens, self.hidden_size).astype(mx.float32)

    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 16), color=(50, 100, 150)).save(image_path)
    backbone = TinyBackbone()
    cache = {}
    camera = {"camera_angle_x": 0.7, "distance": 2.0, "mesh_scale": 1.0}
    extract_proj_features(image_path, backbone, 2, 16, camera, no_rembg=True, feature_cache=cache)
    coordinates = np.array([[0, 0, 0, 0], [0, 2, 1, 0]])
    sparse, _ = extract_proj_features(image_path, backbone, 3, 16, camera, no_rembg=True,
                                     coords=coordinates, feature_cache=cache)
    assert backbone.calls == 1
    dense, _ = extract_proj_features(image_path, backbone, 3, 16, camera, no_rembg=True)
    indices = (coordinates[:, 1] * 3 + coordinates[:, 2]) * 3 + coordinates[:, 3]
    np.testing.assert_allclose(np.array(sparse["proj"]), np.array(dense["proj"][:, mx.array(indices)]), atol=1e-6)
    extract_proj_features(image_path, backbone, 3, 32, camera, no_rembg=True, feature_cache=cache)
    assert backbone.calls == 3
