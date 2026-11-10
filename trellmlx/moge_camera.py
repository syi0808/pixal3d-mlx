"""MoGe-2 camera estimation for Pixal3D-MLX.

Estimates camera intrinsics (FOV) from a single image using MoGe-2,
matching the upstream Pixal3D camera conditioning pipeline.

MLX backend only — pixal3d-mlx is a pure-MLX pipeline. The MoGe model
and weights live in the standalone moge-mlx package.

Flags:
  --no-moge: skip camera estimation, use fixed default FOV (~49°).
  --fov <radians>: manual FOV override (takes priority).
"""

from pathlib import Path

from moge_mlx.camera import (  # noqa: F401
    estimate_camera_params_mlx as _upstream_estimate_camera_params_mlx,
    _compute_distance_from_fov,
)

NATIVE_CAMERA_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "moge_release.safetensors"


def estimate_camera_params_mlx(image_path, *, mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    """Estimate a camera with the pure MLX MoGe-2 port and native cached weights."""
    if not NATIVE_CAMERA_WEIGHTS.is_file():
        return _upstream_estimate_camera_params_mlx(
            image_path, mesh_scale=mesh_scale, extend_pixel=extend_pixel, image_resolution=image_resolution)
    import gc
    import math
    import time

    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from moge_mlx.model import MoGeModel
    from moge_mlx.weights import load_moge_weights

    started = time.perf_counter()
    model = MoGeModel()
    try:
        load_moge_weights(model, weights_path=str(NATIVE_CAMERA_WEIGHTS), verbose=False)
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        result = model.infer(mx.array(image.transpose(2, 0, 1)))
        intrinsics = np.array(result["intrinsics"])
        focal = float(intrinsics[0, 0])
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError("MoGe returned invalid camera intrinsics")
        field_of_view = 2 * math.atan(1 / (2 * focal))
        distance = _compute_distance_from_fov(field_of_view, mesh_scale=mesh_scale,
                                               extend_pixel=extend_pixel, image_resolution=image_resolution)
        print(f"  MoGe MLX camera: FOV={math.degrees(field_of_view):.1f}°, distance={distance:.4f} "
              f"({time.perf_counter() - started:.1f}s, native checkpoint)", flush=True)
        return {"camera_angle_x": field_of_view, "distance": distance, "mesh_scale": mesh_scale}
    finally:
        del model
        gc.collect()
        mx.clear_cache()
