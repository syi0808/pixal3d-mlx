"""Verify moge-mlx package is importable and re-exports work.

The full MoGe model test suite lives in the standalone moge-mlx package.
This file only checks that pixal3d-mlx can reach MoGe through moge_mlx.
"""

import pytest


def test_moge_mlx_import():
    """moge_mlx package should be importable with expected symbols."""
    from moge_mlx import MoGeModel, load_moge_weights, estimate_camera_params_mlx

    assert callable(MoGeModel)
    assert callable(load_moge_weights)
    assert callable(estimate_camera_params_mlx)


def test_moge_camera_reexports():
    """pixalmlx.moge_camera should re-export MLX functions from moge_mlx.camera."""
    from pixalmlx.moge_camera import (
        estimate_camera_params_mlx,
        _compute_distance_from_fov,
    )

    assert callable(estimate_camera_params_mlx)
    assert callable(_compute_distance_from_fov)


def test_moge_camera_native_adapter_preserves_upstream_fallback():
    from pixalmlx.moge_camera import _upstream_estimate_camera_params_mlx as from_pixalmlx
    from moge_mlx.camera import estimate_camera_params_mlx as from_moge

    assert from_pixalmlx is from_moge


def test_moge_camera_adapter_delegates_when_native_weights_are_missing(tmp_path, monkeypatch):
    import pixalmlx.moge_camera as camera

    monkeypatch.setattr(camera, "NATIVE_CAMERA_WEIGHTS", tmp_path / "absent.safetensors")
    captured = []
    monkeypatch.setattr(camera, "_upstream_estimate_camera_params_mlx",
                        lambda image, **kwargs: captured.append((image, kwargs)) or {"distance": 2.0})
    result = camera.estimate_camera_params_mlx("image.png", mesh_scale=1.5)
    assert result == {"distance": 2.0}
    assert captured[0][1]["mesh_scale"] == 1.5
