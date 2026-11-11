"""Generate a 3D mesh from a single image using Pixal3D on MLX.

Pixal3D pipeline with projection-based conditioning:
  1. Image → DINOv3 + ProjGrid → global + projected features
  2. Sparse structure flow (proj) → occupancy grid → LR coordinates
  3. LR SLat flow (proj) → denormalize → decoder upsample → HR coordinates
  4. HR SLat flow (proj) → denormalize → full decode → mesh extraction
  5. Texture SLat flow (proj) → texture decode → PBR bake → GLB

Usage:
    PYTHONPATH=. python generate_pixal3d.py --image photo.png --output mesh.glb
"""

import argparse
import gc
import math
import os
import time

import mlx.core as mx
import numpy as np
from PIL import Image

from pixalmlx.cgal_holes import add_hole_fill_argument, validate_hole_fill_backend


# === Normalization constants from Pixal3D pipeline.json ===
# (These will be loaded from the actual pipeline.json when available)
# Using TRELLIS.2-4B values as placeholder — will need to verify against Pixal3D
SHAPE_SLAT_MEAN = np.array([
    0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252,
    1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904,
    0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463,
    1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
    -0.059621, 2.22078, 1.621212, 0.87723, 0.567247, -3.175944,
    -3.186688, 1.578665,
], dtype=np.float32)

SHAPE_SLAT_STD = np.array([
    5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237,
    5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469,
    5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013,
    5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194,
    5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207,
    5.347008, 5.405691,
], dtype=np.float32)

TEX_SLAT_MEAN = np.array([
    3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364,
    0.439898, -0.928075, 0.029398, -0.339596, -0.869527, 1.038479,
    -0.972385, 0.126042, -1.129303, 0.455149, -1.209521, 2.069067,
    0.544735, 2.569128, -0.323407, 2.293, -1.925608, -1.217717,
    1.213905, 0.971588, -0.023631, 0.10675, 2.021786, 0.250524,
    -0.662387, -0.768862,
], dtype=np.float32)

TEX_SLAT_STD = np.array([
    2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316,
    2.144656, 2.911822, 2.969419, 2.501689, 2.154811, 3.163343,
    2.621215, 2.381943, 3.186697, 3.021588, 2.295916, 3.234985,
    3.233086, 2.26014, 2.874801, 2.810596, 3.29272, 2.674999,
    2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849,
    2.786195, 2.77519,
], dtype=np.float32)


# === Camera parameter estimation ===

def compute_distance_from_fov(camera_angle_x, mesh_scale=1.0, image_resolution=512, extend_pixel=0):
    """Compute camera distance from FOV using Pixal3D's projection math.

    Uses the same geometry as Pixal3D's inference.py distance_from_fov().
    """
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    grid_point = np.array([-1.0, 0.0, 0.0]) @ rotation.T
    grid_point = grid_point / mesh_scale / 2.0

    focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
    f_pixels = focal_length * image_resolution / 32.0

    # Target pixel for the grid corner
    xt = 0 - extend_pixel
    yt = image_resolution - 1 + extend_pixel

    x_ndc = xt - image_resolution / 2.0
    xw, yw = grid_point[0], grid_point[1]

    distance = f_pixels * xw / x_ndc - yw
    return distance


def get_default_camera_params(fov_rad=0.8575560450553894, mesh_scale=1.0):
    """Get default camera parameters (Pixal3D's default front view)."""
    distance = compute_distance_from_fov(fov_rad, mesh_scale)
    return {
        'camera_angle_x': fov_rad,
        'distance': distance,
        'mesh_scale': mesh_scale,
    }


PIXAL3D_CROP_SCALE = 1.1  # upstream pixal3d_image_to_3d.py:177


def prepare_conditioning_image(image_path, output_path, no_rembg=False):
    """Preprocess once, as upstream inference.py does, and return the image path
    that both MoGe-2 and every projection stage must consume."""
    if no_rembg:
        return image_path
    from pixalmlx.preprocess import preprocess_image as rembg_preprocess
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(output_path))[0]
    prepared_path = os.path.join(out_dir, f"{stem}_preprocessed.png")
    rembg_preprocess(image_path, crop_scale=PIXAL3D_CROP_SCALE).save(prepared_path)
    print(f"  Conditioning image: {prepared_path}", flush=True)
    return prepared_path


def load_checkpointed_camera(ckpt_dir):
    """Return the camera saved with a generation checkpoint, or None."""
    from pixalmlx.checkpoint import load_checkpoint
    if not os.path.exists(os.path.join(ckpt_dir, "camera.json")):
        return None
    meta = load_checkpoint(ckpt_dir, "camera")
    return {key: float(meta[key]) for key in ("camera_angle_x", "distance", "mesh_scale")}


def to_pixal3d_glb_frame(vertices):
    """Map mesh-grid vertices into the frame upstream Pixal3D exports.

    Upstream inference.py first runs o_voxel ``to_glb`` (x, y, z) -> (x, z, -y),
    then applies its own rotation (inference.py:271-278). Composite: (-x, y, -z),
    a proper rotation (no winding flip).
    """
    v = np.asarray(vertices)
    return v * np.array([-1, 1, -1], dtype=v.dtype)


# === Helpers ===

def _denormalize_slat(slat, mean=SHAPE_SLAT_MEAN, std=SHAPE_SLAT_STD):
    return slat * mx.array(std) + mx.array(mean)


def _normalize_slat(slat, mean=SHAPE_SLAT_MEAN, std=SHAPE_SLAT_STD):
    return (slat - mx.array(mean)) / mx.array(std)


def _requantize_coords(hr_coords_np, lr_resolution, hr_resolution):
    """Requantize decoder output coords to target resolution.

    Matches upstream Pixal3D formula:
        ((coord + 0.5) / lr_resolution * (grid_res - 1)).round().int()
    where grid_res = hr_resolution // 16.
    """
    grid_res = hr_resolution // 16
    spatial = hr_coords_np[:, 1:4].astype(np.float64)
    spatial = np.round((spatial + 0.5) / lr_resolution * (grid_res - 1)).astype(np.int32)
    result = hr_coords_np.copy()
    result[:, 1:4] = spatial
    return np.unique(result, axis=0)


# === Feature extraction ===

def extract_proj_features(image_path, dinov3_model, grid_resolution, image_size,
                          camera_params, no_rembg=False,
                          use_naf_upsample=False, use_bilinear_upsample=False,
                          upsample_target_size=128, naf_model=None,
                          coords=None, feature_cache=None):
    """Extract DINOv3 + projection features for one stage.

    Returns:
        cond: dict {'global': [1, 5, 1024], 'proj': [1, R^3, D]}
              D = 1024 without upsample, 2048 with bilinear upsample
        neg_cond: dict with zeros
    """
    from pixalmlx.models.dinov3_proj import DinoV3ProjFeatureExtractor, preprocess_image

    extractor = DinoV3ProjFeatureExtractor(
        dinov3_model, image_size=image_size, grid_resolution=grid_resolution,
        use_naf_upsample=use_naf_upsample,
        use_bilinear_upsample=use_bilinear_upsample,
        upsample_target_size=upsample_target_size,
        naf_model=naf_model,
    )

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    if not no_rembg:
        try:
            from pixalmlx.preprocess import preprocess_image as rembg_preprocess
            img = rembg_preprocess(image_path, crop_scale=PIXAL3D_CROP_SCALE)
        except ImportError:
            pass

    img_tensor = preprocess_image(img, size=image_size)

    fov = mx.array([camera_params['camera_angle_x']])
    dist = mx.array([camera_params['distance']])
    scale = mx.array([camera_params['mesh_scale']])

    options = {}
    if coords is not None:
        options["coords"] = coords
    if feature_cache is not None:
        from pixalmlx.runtime import Stage, StageGraph, image_fingerprint

        model_key = getattr(dinov3_model, "_runtime_fingerprint", id(dinov3_model))
        cache_key = (image_fingerprint(image_path), image_size, no_rembg, model_key)

        def encode():
            encoded = feature_cache.get(cache_key)
            if encoded is not None:
                return encoded
            encoded = extractor.encode_image(img_tensor)
            feature_cache[cache_key] = encoded
            return encoded

        graph = StageGraph([
            Stage("image_encoding", (), ("backbone",), encode),
            Stage("active_projection", ("backbone",), ("global", "projection"),
                  lambda encoded: extractor(img_tensor, fov, dist, scale, backbone_features=encoded, **options)),
        ], outputs=("global", "projection"))
        z_global, z_proj = graph.execute()
    else:
        z_global, z_proj = extractor(img_tensor, fov, dist, scale, **options)
    mx.eval(z_global, z_proj)

    cond = {'global': z_global, 'proj': z_proj}
    from pixalmlx.conditioning import ZeroConditioning

    neg_cond = ZeroConditioning({
        'global': mx.zeros_like(z_global),
        'proj': mx.zeros_like(z_proj),
    })

    return cond, neg_cond


def cast_conditioning_to_dtype(cond, dtype):
    """Cast an MLX conditioning payload without changing its container shape."""
    if isinstance(cond, dict):
        from pixalmlx.conditioning import ZeroConditioning

        result = {key: cast_conditioning_to_dtype(value, dtype) for key, value in cond.items()}
        return ZeroConditioning(result) if isinstance(cond, ZeroConditioning) else result
    return cond.astype(dtype)


def index_proj_by_coords(cond, neg_cond, coords_4d, grid_resolution, target_dim=None,
                         preindexed=False):
    """Index full-grid projection features by sparse coordinates.

    For SLat stages, we need per-token projection features indexed by
    the sparse occupancy coordinates.

    Args:
        cond: dict with 'proj' of shape [1, R^3, D]
        neg_cond: dict with 'proj' of shape [1, R^3, D]
        coords_4d: [N, 4] int array (batch, x, y, z)
        grid_resolution: R (grid was R^3)
        target_dim: If set, zero-pad proj features to this dimension.
                    Used when model expects NAF-upsampled 2048-dim features
                    but we provide 1024-dim features without NAF.

    Returns:
        sparse_cond, sparse_neg_cond: dicts with 'proj' of shape [1, N, D]
    """
    R = grid_resolution
    if preindexed:
        return cond, neg_cond
    z_proj = cond['proj']  # [1, R^3, D]
    z_proj_grid = z_proj.reshape(1, R, R, R, -1)

    batch_idx = coords_4d[:, 0].astype(np.int32)
    x_idx = coords_4d[:, 1].astype(np.int32)
    y_idx = coords_4d[:, 2].astype(np.int32)
    z_idx = coords_4d[:, 3].astype(np.int32)

    # Index into the grid
    flat_indices = (((batch_idx * R + x_idx) * R + y_idx) * R + z_idx).astype(np.int32)
    z_proj_sparse = z_proj_grid.reshape(-1, z_proj.shape[-1])[mx.array(flat_indices)]

    # Zero-pad if model expects wider features (NAF upsample path)
    if target_dim is not None and z_proj_sparse.shape[-1] < target_dim:
        pad_width = target_dim - z_proj_sparse.shape[-1]
        z_proj_sparse = mx.concatenate([
            z_proj_sparse,
            mx.zeros((*z_proj_sparse.shape[:-1], pad_width), dtype=z_proj_sparse.dtype),
        ], axis=-1)

    z_proj_sparse = z_proj_sparse[None]

    sparse_cond = {'global': cond['global'], 'proj': z_proj_sparse}
    sparse_neg_cond = {
        'global': neg_cond['global'],
        'proj': mx.zeros_like(z_proj_sparse),
    }
    from pixalmlx.conditioning import ZeroConditioning

    if isinstance(neg_cond, ZeroConditioning):
        sparse_neg_cond = ZeroConditioning(sparse_neg_cond)

    return sparse_cond, sparse_neg_cond


# === Main ===

def main(argv=None, runtime=None):
    parser = argparse.ArgumentParser(description="Pixal3D on MLX: Image to 3D")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="/tmp/pixal3d-mlx-mesh.glb")
    parser.add_argument("--seed", type=int, default=42)
    for stage in ("ss", "lr", "hr", "texture"):
        parser.add_argument(f"--{stage}-steps", type=int, default=12,
                            help=f"{stage.upper()} sampling steps (1-100, default: 12)")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual FOV in radians (default: MoGe auto-estimate)")
    parser.add_argument("--no-moge", action="store_true",
                        help="Skip MoGe camera estimation, use fixed default FOV (~49 deg)")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=49152)
    parser.add_argument("--target-faces", type=int, default=200_000,
                        help="Soft triangle target before final mesh repairs; use --max-faces for a strict cap")
    parser.add_argument("--max-faces", type=int, default=0,
                        help="Strict final triangle limit after mesh repair (0=disabled)")
    parser.add_argument("--no-rembg", action="store_true")
    parser.add_argument("--save-checkpoints", metavar="DIR",
                        help="Save intermediate results after each stage")
    parser.add_argument("--resume", metavar="DIR",
                        help="Resume from checkpoints — redirects to resume_pixal3d.py")
    parser.add_argument("--no-cleanup", action="store_true")
    add_hole_fill_argument(parser)
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8],
                        help="Quantize flow models (0=off, 4=INT4, 8=INT8). Reduces memory ~4-6x.")
    parser.add_argument("--ss-only", action="store_true",
                        help="Run only sparse structure stage (for debugging)")
    parser.add_argument("--dense-conditioning", action="store_true",
                        help="Use the dense projection reference instead of active coordinates")
    parser.add_argument("--naf-backend", choices=["auto", "metal", "reference"], default="auto")
    parser.add_argument("--memory-budget-gb", type=float, default=10.0,
                        help="MLX graph-evaluation guideline in decimal GB, not a hard allocation cap")
    parser.add_argument("--model-cache-gb", type=float, default=2.0)
    parser.add_argument("--feature-cache-mb", type=float, default=256.0)
    parser.add_argument("--runtime-report")
    parser.add_argument("--ffn-chunk-size", type=int, default=2048)
    parser.add_argument("--conditioning-cache-mb", type=float, default=128.0)
    parser.add_argument("--naf-tile-rows", type=int, default=128)
    parser.add_argument("--naf-threads", type=int, choices=[64, 128, 256], default=128)
    parser.add_argument("--compile-step", action="store_true")
    parser.add_argument("--compile-transformer", action=argparse.BooleanOptionalAction, default=True,
                        help="Compile bounded FFN and modulation/residual operations (default: enabled)")
    parser.add_argument("--sparse-order", choices=["preserve", "morton"], default="morton")
    parser.add_argument("--decoder-eval-interval", type=int, default=1,
                        help="Evaluate every N sparse kernels in the final shape/texture upsample (0=legacy)")
    parser.add_argument("--slat-precision", choices=["float32", "mixed"], default="float32",
                        help="Experimental FP16 SLat compute; accumulations and sampling stay FP32")
    parser.add_argument("--ss-precision", choices=["float32", "mixed"], default="float32",
                        help="Mixed FP16 compute is experimental; FP32 remains the stable default")
    args = parser.parse_args(argv)
    if any(not 1 <= steps <= 100 for steps in (args.ss_steps, args.lr_steps, args.hr_steps, args.texture_steps)):
        parser.error("Sampling steps must be integers from 1 to 100")
    if min(args.feature_cache_mb, args.conditioning_cache_mb, args.naf_tile_rows, args.ffn_chunk_size,
           args.decoder_eval_interval, args.max_faces) < 0:
        parser.error("Cache and chunk sizes must be nonnegative")

    # Redirect --resume to the dedicated resume script
    if args.resume:
        import resume_pixal3d
        cmd = [
               "--image", args.image,
               "--ckpt", args.resume,
               "--output", args.output,
               "--seed", str(args.seed),
               "--texture-size", str(args.texture_size),
               "--texture-steps", str(args.texture_steps),
               "--target-faces", str(args.target_faces),
               "--max-faces", str(args.max_faces),
               "--hole-fill-backend", args.hole_fill_backend,
               "--memory-budget-gb", str(args.memory_budget_gb),
               "--model-cache-gb", str(args.model_cache_gb),
               "--feature-cache-mb", str(args.feature_cache_mb),
               "--conditioning-cache-mb", str(args.conditioning_cache_mb),
               "--ffn-chunk-size", str(args.ffn_chunk_size),
               "--naf-backend", args.naf_backend,
               "--naf-tile-rows", str(args.naf_tile_rows),
               "--naf-threads", str(args.naf_threads),
               "--sparse-order", args.sparse_order,
               "--decoder-eval-interval", str(args.decoder_eval_interval),
               "--slat-precision", args.slat_precision,
               "--quantize", str(args.quantize)]
        if args.runtime_report:
            cmd.extend(["--runtime-report", args.runtime_report])
        if args.compile_step:
            cmd.append("--compile-step")
        cmd.append("--compile-transformer" if args.compile_transformer else "--no-compile-transformer")
        if args.dense_conditioning:
            cmd.append("--dense-conditioning")
        if args.no_rembg:
            cmd.append("--no-rembg")
        if args.no_cleanup:
            cmd.append("--no-cleanup")
        if args.keep_largest:
            cmd.append("--keep-largest")
        if args.fov > 0:
            cmd.extend(["--fov", str(args.fov)])
        print(f"Resuming via resume_pixal3d.py --ckpt {args.resume}", flush=True)
        return resume_pixal3d.main(cmd, runtime=runtime)

    if not args.no_cleanup and not args.ss_only:
        try:
            validate_hole_fill_backend(args.hole_fill_backend)
        except RuntimeError as error:
            parser.error(str(error))

    from pixalmlx.runtime import InferenceRuntime

    if runtime is None:
        with InferenceRuntime(memory_budget_bytes=int(args.memory_budget_gb * 1e9),
                              model_cache_bytes=int(args.model_cache_gb * 1e9),
                              feature_cache_bytes=int(args.feature_cache_mb * 1024 ** 2)) as owned_runtime:
            with owned_runtime.job(args.runtime_report, metadata=vars(args)):
                return _generate(args, owned_runtime)
    with runtime.job(args.runtime_report, metadata=vars(args)):
        return _generate(args, runtime)


def _generate(args, runtime):
    from pixalmlx.runtime import checkpoint_fingerprint, image_fingerprint

    runtime.advance("preprocess_and_camera")
    mx.random.seed(args.seed)
    t_total = time.perf_counter()
    mx.metal.reset_peak_memory()

    # One preprocessed image feeds MoGe and every stage (upstream inference.py:192-215)
    cond_image = prepare_conditioning_image(args.image, args.output, no_rembg=args.no_rembg)

    # Camera parameters: manual FOV > MoGe auto > default
    if args.fov > 0:
        camera_params = get_default_camera_params(fov_rad=args.fov)
        print(f"Manual FOV: {math.degrees(args.fov):.1f} deg, distance: {camera_params['distance']:.4f}")
    elif not args.no_moge:
        try:
            print("=== MoGe-2 Camera Estimation (MLX) ===", flush=True)
            from pixalmlx.moge_camera import estimate_camera_params_mlx
            camera_key = image_fingerprint(cond_image)
            camera_params = runtime.camera.get(camera_key)
            if camera_params is None:
                camera_params = estimate_camera_params_mlx(cond_image)
                runtime.camera[camera_key] = dict(camera_params)
        except Exception as e:
            print(f"  MoGe unavailable ({e}), falling back to default FOV.", flush=True)
            camera_params = get_default_camera_params()
            print(f"Default FOV: {math.degrees(camera_params['camera_angle_x']):.1f} deg, "
                  f"distance: {camera_params['distance']:.4f}")
    else:
        camera_params = get_default_camera_params()
        print(f"Default FOV: {math.degrees(camera_params['camera_angle_x']):.1f} deg, "
              f"distance: {camera_params['distance']:.4f}")

    # Resolve checkpoint paths
    HF_PIXAL3D = os.path.expanduser(
        "~/.cache/huggingface/hub/models--TencentARC--Pixal3D/"
        "snapshots/0b31f9160aa400719af409098bff7936a932f726/ckpts/"
    )

    from pixalmlx.weight_loader import load_weights
    from pixalmlx.samplers import flow_euler_sample
    from pixalmlx.cleanup import cleanup_model, cleanup
    from pixalmlx.checkpoint import save_checkpoint, load_checkpoint, has_checkpoint

    ckpt_dir = args.save_checkpoints
    save_ckpts = ckpt_dir is not None
    if save_ckpts:
        save_checkpoint(ckpt_dir, "camera", **{k: float(camera_params[k]) for k in ("camera_angle_x", "distance", "mesh_scale")})

    # === DINOv3 backbone (shared across stages) ===
    print("=== Loading DINOv3 backbone ===", flush=True)
    from pixalmlx.models.dinov3 import DINOv3ViT, load_dinov3_weights

    DINOV3_WEIGHTS = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/"
        "snapshots/"
    )
    if os.path.isdir(DINOV3_WEIGHTS):
        snapshots = [d for d in os.listdir(DINOV3_WEIGHTS) if not d.startswith(".")]
        if snapshots:
            DINOV3_WEIGHTS = os.path.join(DINOV3_WEIGHTS, snapshots[0])

    dinov3 = DINOv3ViT()
    dinov3 = runtime.load_model(("dinov3", checkpoint_fingerprint(DINOV3_WEIGHTS)),
                                lambda: dinov3, lambda model: load_dinov3_weights(model, DINOV3_WEIGHTS))
    print("  DINOv3 loaded.", flush=True)
    feature_cache = runtime.features

    # === Stage 1: Sparse Structure (proj) ===
    print("\n=== Stage 1: Sparse Structure (proj) ===", flush=True)
    runtime.advance("sparse_structure")
    from pixalmlx.models.pixal3d_flow import Pixal3DSparseStructureFlowModel
    from pixalmlx.models.sparse_structure_decoder import SparseStructureDecoder

    # Extract features for SS stage (grid_resolution=16, image_size=512)
    print("  Extracting proj features (SS, 512, grid=16)...", flush=True)
    ss_cond, ss_neg_cond = extract_proj_features(
        cond_image, dinov3, grid_resolution=16, image_size=512,
        camera_params=camera_params, no_rembg=True,
        feature_cache=feature_cache,
    )

    ss_flow = Pixal3DSparseStructureFlowModel(
        in_channels=8, out_channels=8,
        model_channels=1536, num_heads=12,
        num_blocks=30, mlp_hidden=8192,
        context_channels=1024, proj_in_channels=1024,
        resolution=16,
    )
    ss_flow = runtime.load_model(("ss_flow", args.ss_precision), lambda: ss_flow,
                                 lambda model: load_weights(model, HF_PIXAL3D + "ss_flow_img_dit_1_3B_64_bf16.safetensors",
                                                            verbose=False, streaming=True,
                                                            target_dtype=mx.float32 if args.ss_precision == "float32" else mx.float16),
                                 retain=False)
    if args.quantize:
        from pixalmlx.quantize import quantize_model
        quantize_model(ss_flow, bits=args.quantize)
    # Upcast sparse structure flow to fp32 for CFG rescale stability.
    # bf16 precision noise gets amplified ~5x per Euler step through the
    # std ratio in CFG rescale, producing catastrophic divergence after
    # 4 steps. fp32 reduces this. Cost: ~2.4GB extra for ~20s.
    ss_dtype = mx.float32 if args.ss_precision == "float32" else mx.float16
    ss_flow.apply(lambda x: x.astype(ss_dtype) if mx.issubdtype(x.dtype, mx.floating) else x)
    if args.ss_precision == "mixed":
        ss_flow.compute_dtype = mx.float16

    # SS sampler params from Pixal3D pipeline.json
    SS_SAMPLER = dict(steps=args.ss_steps, guidance_strength=7.5, guidance_rescale=0.7,
                      guidance_interval=(0.6, 1.0), rescale_t=5.0)

    noise = mx.random.normal((1, 8, 16, 16, 16), key=mx.random.key(args.seed)).astype(mx.float32)
    t0 = time.perf_counter()
    # Cast conditioning to fp32 to match the fp32 sparse structure model
    z_s = flow_euler_sample(ss_flow, noise,
                            cast_conditioning_to_dtype(ss_cond, ss_dtype),
                            cast_conditioning_to_dtype(ss_neg_cond, ss_dtype),
                            conditioning_cache_bytes=int(args.conditioning_cache_mb * 1024 ** 2),
                            ffn_chunk_size=args.ffn_chunk_size, compile_step=args.compile_step,
                            selective_compile=args.compile_transformer,
                            verbose=False, **SS_SAMPLER)
    mx.eval(z_s)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s", flush=True)

    # Decode sparse structure
    # Use TRELLIS.2 decoder (same architecture, different checkpoint)
    HF_LARGE = os.path.expanduser(
        "~/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/"
        "snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/"
    )
    ss_dec_ckpt = HF_PIXAL3D + "ss_dec_conv3d_16l8_fp16.safetensors"
    if not os.path.exists(ss_dec_ckpt):
        ss_dec_ckpt = HF_LARGE + "ss_dec_conv3d_16l8_fp16.safetensors"

    ss_dec = SparseStructureDecoder()
    ss_dec = runtime.load_model(("ss_decoder", checkpoint_fingerprint(ss_dec_ckpt)), lambda: ss_dec,
                                lambda model: load_weights(model, ss_dec_ckpt, verbose=False, streaming=True))

    logits = ss_dec(z_s.astype(mx.float32))
    mx.eval(logits)
    decoded = np.array(logits[0, 0] > 0)

    lr_resolution = 32
    ratio = decoded.shape[0] // lr_resolution
    decoded_ds = decoded.reshape(
        lr_resolution, ratio, lr_resolution, ratio, lr_resolution, ratio
    ).any(axis=(1, 3, 5))
    lr_coords = np.argwhere(decoded_ds)
    print(f"  {len(lr_coords)} sparse voxels at {lr_resolution}³", flush=True)

    cleanup_model(ss_flow, ss_dec)
    del ss_flow, ss_dec, ss_cond, ss_neg_cond
    gc.collect()
    mx.metal.clear_cache()

    if save_ckpts:
        save_checkpoint(ckpt_dir, "ss", lr_coords=lr_coords)

    if args.ss_only:
        print(f"\n  SS-only mode. {len(lr_coords)} voxels found.")
        print(f"  Total: {time.perf_counter()-t_total:.1f}s")
        return

    runtime.advance("lr_conditioning_and_shape")
    from pixalmlx.models.naf import NAF as NAFModel
    from pixalmlx.models.naf_loader import load_naf_weights

    naf_path = os.path.join(os.path.dirname(__file__), "weights", "naf_release.safetensors")
    naf = NAFModel()
    naf = runtime.load_model(("naf", checkpoint_fingerprint(naf_path)), lambda: naf,
                             lambda model: load_naf_weights(model, naf_path, verbose=False))
    if hasattr(naf, "upsampler"):
        naf.upsampler.backend = args.naf_backend
        naf.upsampler.threadgroup_size = args.naf_threads
        naf.encoder_tile_rows = args.naf_tile_rows

    # === Stage 2a: LR Shape Latent (proj, 512) ===
    print("\n=== Stage 2a: LR Shape Latent (proj, 512) ===", flush=True)
    from pixalmlx.models.pixal3d_flow import Pixal3DSLatFlowModel
    from pixalmlx.models.shape_slat_decoder import SLatDecoder

    SHAPE_SAMPLER = dict(guidance_strength=7.5, guidance_rescale=0.5,
                         guidance_interval=(0.6, 1.0), rescale_t=3.0)

    # Extract proj features for shape LR stage (grid=32, image=512, NAF→proj_in=2048)
    # Note: without NAF upsampling, proj_in=1024. The Pixal3D shape models
    # use proj_in_channels=2048 (NAF), but we can try with 1024 initially
    # and zero-pad the second half. For now, use grid_resolution=32.
    print("  Extracting proj features (shape LR, 512, grid=32, NAF upsample to 512)...", flush=True)
    N_lr = len(lr_coords)
    lr_coords_4d = np.column_stack([np.zeros(N_lr, dtype=np.int32), lr_coords])
    shape_lr_cond, shape_lr_neg_cond = extract_proj_features(
        cond_image, dinov3, grid_resolution=32, image_size=512,
        camera_params=camera_params, no_rembg=True,
        use_naf_upsample=True, upsample_target_size=512, naf_model=naf,
        coords=None if args.dense_conditioning else lr_coords_4d,
        feature_cache=feature_cache,
    )

    # Index projection features by sparse coordinates
    N_lr = len(lr_coords)
    lr_coords_4d = np.column_stack([np.zeros(N_lr, dtype=np.int32), lr_coords])
    shape_lr_cond_sparse, shape_lr_neg_cond_sparse = index_proj_by_coords(
        shape_lr_cond, shape_lr_neg_cond, lr_coords_4d, grid_resolution=32,
        preindexed=not args.dense_conditioning,
    )

    # Load LR shape flow (proj_in_channels=2048 for NAF, but we provide 1024)
    # The model will accept whatever proj_in_channels we configure
    lr_slat_flow = Pixal3DSLatFlowModel(
        in_channels=32, out_channels=32,
        model_channels=1536, num_heads=12,
        num_blocks=30, mlp_hidden=8192,
        context_channels=1024, proj_in_channels=2048,
    )
    lr_slat_flow = runtime.load_model("lr_flow", lambda: lr_slat_flow,
                                      lambda model: load_weights(model, HF_PIXAL3D + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
                                                                 verbose=False, streaming=True), retain=False)
    if args.slat_precision == "mixed":
        lr_slat_flow.compute_dtype = mx.float16
    if args.quantize:
        quantize_model(lr_slat_flow, bits=args.quantize)

    lr_noise = mx.random.normal((N_lr, 32), key=mx.random.key(args.seed + 1))
    t0 = time.perf_counter()
    lr_slat = flow_euler_sample(
        lr_slat_flow, lr_noise, shape_lr_cond_sparse, shape_lr_neg_cond_sparse,
        verbose=False, steps=args.lr_steps, coords=mx.array(lr_coords),
        conditioning_cache_bytes=int(args.conditioning_cache_mb * 1024 ** 2),
        ffn_chunk_size=args.ffn_chunk_size, compile_step=args.compile_step,
        selective_compile=args.compile_transformer,
        **SHAPE_SAMPLER,
    )
    mx.eval(lr_slat)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s ({N_lr} tokens)", flush=True)

    lr_slat = _denormalize_slat(lr_slat)
    mx.eval(lr_slat)

    cleanup_model(lr_slat_flow)
    del lr_slat_flow, shape_lr_cond, shape_lr_neg_cond, shape_lr_cond_sparse, shape_lr_neg_cond_sparse
    gc.collect()
    mx.metal.clear_cache()

    # === Stage 2b: Upsample to HR coordinates ===
    print("\n=== Stage 2b: Upsample → HR coordinates ===", flush=True)

    shape_dec_ckpt = HF_PIXAL3D + "shape_dec_next_dc_f16c32_fp16.safetensors"
    decoder = SLatDecoder(out_channels=7, pred_subdiv=True, spatial_order=args.sparse_order,
                          final_upsample_eval_interval=args.decoder_eval_interval)
    runtime.advance("coordinate_subdivision")
    decoder = runtime.load_model(("shape_decoder", checkpoint_fingerprint(shape_dec_ckpt), args.sparse_order, args.decoder_eval_interval), lambda: decoder,
                                 lambda model: load_weights(model, shape_dec_ckpt, verbose=False, streaming=True))

    t0 = time.perf_counter()
    hr_coords_raw = decoder.upsample(lr_slat, mx.array(lr_coords_4d), upsample_times=4)
    mx.eval(hr_coords_raw)
    print(f"  Upsampled: {time.perf_counter()-t0:.1f}s ({hr_coords_raw.shape[0]:,} voxels)", flush=True)

    decoder_output_res = lr_resolution * 16  # 32 * 16 = 512
    hr_resolution = args.resolution
    hr_coords_np = np.array(hr_coords_raw)
    while True:
        quant_coords = _requantize_coords(hr_coords_np, decoder_output_res, hr_resolution)
        num_tokens = len(quant_coords)
        if num_tokens < args.max_tokens or hr_resolution == 1024:
            if hr_resolution != args.resolution:
                print(f"  Resolution reduced to {hr_resolution} ({num_tokens:,} tokens)", flush=True)
            break
        hr_resolution -= 128

    hr_coords_3d = quant_coords[:, 1:4]
    print(f"  HR coords: {num_tokens:,} tokens at res {hr_resolution}", flush=True)

    cleanup_model(decoder)
    del decoder
    gc.collect()
    mx.metal.clear_cache()

    # === Stage 2c: HR Shape Latent (proj, 1024) ===
    print("\n=== Stage 2c: HR Shape Latent (proj, 1024) ===", flush=True)
    runtime.advance("hr_conditioning_and_shape")

    # Projection and sparse coordinates must describe the same physical grid.
    # Upstream overrides the projection grid with actual_hr_resolution // 16.
    hr_grid_resolution = hr_resolution // 16
    hr_image_size = 1024
    print(f"  Extracting proj features (shape HR, {hr_image_size}, grid={hr_grid_resolution}, NAF upsample to 512)...", flush=True)
    shape_hr_cond, shape_hr_neg_cond = extract_proj_features(
        cond_image, dinov3, grid_resolution=hr_grid_resolution, image_size=hr_image_size,
        camera_params=camera_params, no_rembg=True,
        use_naf_upsample=True, upsample_target_size=512, naf_model=naf,
        coords=None if args.dense_conditioning else quant_coords,
        feature_cache=feature_cache,
    )

    # Index by HR sparse coordinates
    shape_hr_cond_sparse, shape_hr_neg_cond_sparse = index_proj_by_coords(
        shape_hr_cond, shape_hr_neg_cond, quant_coords, grid_resolution=hr_grid_resolution,
        preindexed=not args.dense_conditioning,
    )
    shared_hr_conditioning = (shape_hr_cond_sparse, shape_hr_neg_cond_sparse)
    del dinov3, naf
    gc.collect()
    mx.clear_cache()

    hr_slat_flow = Pixal3DSLatFlowModel(
        in_channels=32, out_channels=32,
        model_channels=1536, num_heads=12,
        num_blocks=30, mlp_hidden=8192,
        context_channels=1024, proj_in_channels=2048,
    )
    hr_slat_flow = runtime.load_model("hr_flow", lambda: hr_slat_flow,
                                      lambda model: load_weights(model, HF_PIXAL3D + "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
                                                                 verbose=False, streaming=True), retain=False)
    if args.slat_precision == "mixed":
        hr_slat_flow.compute_dtype = mx.float16
    if args.quantize:
        quantize_model(hr_slat_flow, bits=args.quantize)

    hr_noise = mx.random.normal((num_tokens, 32), key=mx.random.key(args.seed + 2))
    t0 = time.perf_counter()
    hr_slat = flow_euler_sample(
        hr_slat_flow, hr_noise, shape_hr_cond_sparse, shape_hr_neg_cond_sparse,
        verbose=False, steps=args.hr_steps, coords=mx.array(hr_coords_3d),
        conditioning_cache_bytes=int(args.conditioning_cache_mb * 1024 ** 2),
        ffn_chunk_size=args.ffn_chunk_size, compile_step=args.compile_step,
        selective_compile=args.compile_transformer,
        **SHAPE_SAMPLER,
    )
    mx.eval(hr_slat)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s ({num_tokens:,} tokens)", flush=True)

    # Diagnostic: shape latent stats
    hr_slat_np = np.array(hr_slat)
    print(f"  Shape latent (normalized): shape={hr_slat_np.shape}", flush=True)
    print(f"    per-channel std: min={hr_slat_np.std(axis=0).min():.4f} "
          f"max={hr_slat_np.std(axis=0).max():.4f} "
          f"mean={hr_slat_np.std(axis=0).mean():.4f}", flush=True)

    hr_slat = _denormalize_slat(hr_slat)
    mx.eval(hr_slat)

    cleanup_model(hr_slat_flow)
    del hr_slat_flow, shape_hr_cond, shape_hr_neg_cond, shape_hr_cond_sparse, shape_hr_neg_cond_sparse
    gc.collect()
    mx.metal.clear_cache()

    if save_ckpts:
        save_checkpoint(ckpt_dir, "shape_latent",
                        hr_slat=np.array(hr_slat),
                        quant_coords=quant_coords,
                        hr_resolution=hr_resolution)

    # === Stage 3: Shape Decode ===
    print("\n=== Stage 3: Decode Shape ===", flush=True)
    runtime.advance("shape_decode_and_mesh")

    shape_decoder = SLatDecoder(out_channels=7, pred_subdiv=True, spatial_order=args.sparse_order,
                                final_upsample_eval_interval=args.decoder_eval_interval)
    shape_decoder = runtime.load_model(("shape_decoder", checkpoint_fingerprint(shape_dec_ckpt), args.sparse_order, args.decoder_eval_interval), lambda: shape_decoder,
                                       lambda model: load_weights(model, shape_dec_ckpt, verbose=False, streaming=True))

    t0 = time.perf_counter()
    dec_out, dec_coords, shape_subs = shape_decoder(
        hr_slat, mx.array(quant_coords), return_subs=True,
    )
    mx.eval(dec_out)
    print(f"  Decoded: {time.perf_counter()-t0:.1f}s ({dec_out.shape[0]:,} voxels)", flush=True)

    cleanup_model(shape_decoder)
    del shape_decoder
    gc.collect()
    mx.metal.clear_cache()

    # === Mesh Extraction ===
    print("\n=== Mesh Extraction ===", flush=True)
    from pixalmlx.mesh_extract import decoder_output_to_mesh

    dec_coords_np = np.array(dec_coords)
    dec_feats_np = np.array(dec_out)
    del dec_out, dec_coords  # free after converting to numpy

    mesh_grid_size = hr_resolution
    t0 = time.perf_counter()
    vertices, faces = decoder_output_to_mesh(
        dec_feats_np, dec_coords_np, resolution=mesh_grid_size,
    )
    print(f"  Extracted: {time.perf_counter()-t0:.1f}s ({len(vertices):,}V {len(faces):,}F)", flush=True)

    # Use the same multi-pass cleanup+simplify as generate.py
    from generate import _cleanup_and_simplify_mesh
    vertices, faces = _cleanup_and_simplify_mesh(
        vertices, faces,
        target_faces=args.target_faces,
        no_cleanup=args.no_cleanup,
        keep_largest=args.keep_largest,
        hole_fill_backend=args.hole_fill_backend,
    )
    if args.max_faces:
        from pixalmlx.mesh_budget import enforce_face_limit
        vertices, faces = enforce_face_limit(vertices, faces, args.max_faces)

    if save_ckpts:
        # Save subdivision masks as list of arrays
        subs_list = [np.array(s) for s in shape_subs] if shape_subs else []
        save_checkpoint(ckpt_dir, "mesh",
                        vertices=vertices, faces=faces,
                        quant_coords=quant_coords,
                        hr_slat=np.array(hr_slat),
                        hr_resolution=hr_resolution)
        save_checkpoint(ckpt_dir, "shape_subs", subs=subs_list)

    # === Stage 4: Texture (proj, 1024) ===
    print("\n=== Stage 4: Texture SLat (proj, 1024) ===", flush=True)
    runtime.advance("texture_sampling")

    TEX_SAMPLER = dict(steps=args.texture_steps, guidance_strength=1.0, guidance_rescale=0.0,
                       guidance_interval=(0.6, 0.9), rescale_t=3.0)

    # Extract proj features for texture stage
    # Same image_size tradeoff as shape HR
    tex_image_size = 1024
    # Upstream uses NAF to 1024 for texture, but that's too large for
    # windowed attention on unified memory (GPU page fault). Use 512.
    tex_naf_size = 512
    print(f"  Extracting proj features (tex, {tex_image_size}, grid={hr_grid_resolution}, NAF upsample to {tex_naf_size})...", flush=True)
    print("  Reusing HR image conditioning for texture.", flush=True)
    tex_cond, tex_neg_cond = shared_hr_conditioning
    tex_cond_sparse, tex_neg_cond_sparse = index_proj_by_coords(
        tex_cond, tex_neg_cond, quant_coords, grid_resolution=hr_grid_resolution,
        preindexed=True,
    )

    tex_flow = Pixal3DSLatFlowModel(
        in_channels=64, out_channels=32,
        model_channels=1536, num_heads=12,
        num_blocks=30, mlp_hidden=8192,
        context_channels=1024, proj_in_channels=2048,
    )
    tex_flow = runtime.load_model("texture_flow", lambda: tex_flow,
                                  lambda model: load_weights(model, HF_PIXAL3D + "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
                                                             verbose=False, streaming=True), retain=False)
    if args.slat_precision == "mixed":
        tex_flow.compute_dtype = mx.float16
    if args.quantize:
        quantize_model(tex_flow, bits=args.quantize)

    # Re-normalize shape SLat for texture conditioning
    shape_cond = _normalize_slat(hr_slat)
    mx.eval(shape_cond)

    tex_noise = mx.random.normal((num_tokens, 32), key=mx.random.key(args.seed + 3))
    t0 = time.perf_counter()
    tex_slat = flow_euler_sample(
        tex_flow, tex_noise, tex_cond_sparse, tex_neg_cond_sparse,
        verbose=False, coords=mx.array(hr_coords_3d),
        concat_cond=shape_cond,
        conditioning_cache_bytes=int(args.conditioning_cache_mb * 1024 ** 2),
        ffn_chunk_size=args.ffn_chunk_size, compile_step=args.compile_step,
        selective_compile=args.compile_transformer,
        **TEX_SAMPLER,
    )
    mx.eval(tex_slat)
    print(f"  Sampled: {time.perf_counter()-t0:.1f}s ({num_tokens:,} tokens)", flush=True)

    # Diagnostic: texture latent stats before denormalization
    tex_slat_np = np.array(tex_slat)
    print(f"  Texture latent (normalized): shape={tex_slat_np.shape}", flush=True)
    print(f"    per-channel std: min={tex_slat_np.std(axis=0).min():.4f} "
          f"max={tex_slat_np.std(axis=0).max():.4f} "
          f"mean={tex_slat_np.std(axis=0).mean():.4f}", flush=True)

    tex_slat = _denormalize_slat(tex_slat, mean=TEX_SLAT_MEAN, std=TEX_SLAT_STD)
    mx.eval(tex_slat)

    # And after denormalization
    tex_dn_np = np.array(tex_slat)
    print(f"  Texture latent (denormalized):", flush=True)
    print(f"    per-channel std: min={tex_dn_np.std(axis=0).min():.4f} "
          f"max={tex_dn_np.std(axis=0).max():.4f}", flush=True)

    cleanup_model(tex_flow)
    del tex_flow, tex_cond, tex_neg_cond, tex_cond_sparse, tex_neg_cond_sparse
    del shared_hr_conditioning
    gc.collect()
    mx.metal.clear_cache()

    # === Stage 5: Texture Decode ===
    print("\n=== Stage 5: Texture Decode ===", flush=True)

    tex_dec_ckpt = HF_PIXAL3D + "tex_dec_next_dc_f16c32_fp16.safetensors"
    if not os.path.exists(tex_dec_ckpt):
        # Fall back to TRELLIS.2 tex decoder if Pixal3D's isn't downloaded yet
        HF_4B = os.path.expanduser(
            "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
            "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/"
        )
        tex_dec_ckpt = HF_4B + "tex_dec_next_dc_f16c32_fp16.safetensors"

    tex_decoder = SLatDecoder(out_channels=6, pred_subdiv=False, spatial_order=args.sparse_order,
                              final_upsample_eval_interval=args.decoder_eval_interval)
    runtime.advance("texture_decode_and_export")
    tex_decoder = runtime.load_model(("texture_decoder", checkpoint_fingerprint(tex_dec_ckpt), args.sparse_order, args.decoder_eval_interval), lambda: tex_decoder,
                                     lambda model: load_weights(model, tex_dec_ckpt, verbose=False, streaming=True))

    t0 = time.perf_counter()
    tex_out, tex_coords = tex_decoder(
        tex_slat, mx.array(quant_coords), guide_subs=shape_subs,
    )
    mx.eval(tex_out)

    # Diagnostic: raw decoder output before transform
    tex_raw_np = np.array(tex_out)
    print(f"  Raw decoder output stats (before * 0.5 + 0.5):", flush=True)
    for ch, name in enumerate(['R', 'G', 'B', 'metallic', 'roughness', 'alpha']):
        vals = tex_raw_np[:, ch]
        print(f"    {name}: [{vals.min():.3f}, {vals.max():.3f}] mean={vals.mean():.3f}", flush=True)

    tex_out = tex_out * 0.5 + 0.5
    mx.eval(tex_out)

    # Diagnostic: after transform
    tex_np_check = np.array(tex_out)
    print(f"  After * 0.5 + 0.5:", flush=True)
    for ch, name in enumerate(['R', 'G', 'B', 'metallic', 'roughness', 'alpha']):
        vals = tex_np_check[:, ch]
        print(f"    {name}: [{vals.min():.3f}, {vals.max():.3f}] mean={vals.mean():.3f}", flush=True)

    # Note: metallic/roughness may be near-constant due to degraded
    # conditioning (bilinear vs NAF). This is a known quality gap —
    # the fix is proper NAF upsampling, not output clamping.

    print(f"  Decoded: {time.perf_counter()-t0:.1f}s ({tex_out.shape[0]:,} voxels)", flush=True)

    cleanup_model(tex_decoder)
    del tex_decoder
    gc.collect()
    mx.metal.clear_cache()

    if save_ckpts:
        save_checkpoint(ckpt_dir, "texture",
                        tex_out=np.array(tex_out),
                        tex_coords=np.array(tex_coords))

    # === Stage 6: Texture Bake + GLB Export ===
    print("\n=== Stage 6: Texture Bake + GLB ===", flush=True)
    from pixalmlx.texture_bake import uv_unwrap, bake_texture
    import trimesh
    from trimesh.visual.material import PBRMaterial

    tex_np = np.array(tex_out)
    tex_coords_spatial = np.array(tex_coords)[:, 1:4]  # drop batch dim

    # UV unwrap
    t0 = time.perf_counter()
    uv_verts, uv_faces, uvs, vmapping = uv_unwrap(vertices, faces)
    if args.max_faces and len(uv_faces) > args.max_faces:
        raise RuntimeError("UV output exceeds the strict triangle budget")
    print(f"  UV unwrap: {len(uv_verts):,}V {len(uv_faces):,}F "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # Bake PBR textures
    base_color, metallic_roughness, alpha_mode = bake_texture(
        uv_verts, uv_faces, uvs, vmapping,
        tex_coords_spatial, tex_np, mesh_grid_size,
        texture_size=args.texture_size,
        backend="gpu",
    )
    print(f"  Baked textures ({args.texture_size}x{args.texture_size})", flush=True)

    # Export GLB in the upstream Pixal3D frame
    export_verts = to_pixal3d_glb_frame(uv_verts)

    export_uvs = uvs.copy()
    export_uvs[:, 1] = 1 - export_uvs[:, 1]

    mesh = trimesh.Trimesh(vertices=export_verts, faces=uv_faces, process=False)
    normals = mesh.vertex_normals

    material = PBRMaterial(
        baseColorTexture=Image.fromarray(base_color),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(metallic_roughness),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True,
    )

    textured_mesh = trimesh.Trimesh(
        vertices=export_verts,
        faces=uv_faces,
        vertex_normals=normals,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=export_uvs, material=material),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    from pixalmlx.glb_normals import export_with_corrected_normals

    runtime.advance("normal_correction")
    export_with_corrected_normals(textured_mesh, args.output, max_faces=args.max_faces)
    peak_mem = mx.metal.get_peak_memory() / 1e9
    print(f"\n  Saved: {args.output}")
    print(f"  Total: {time.perf_counter()-t_total:.1f}s")
    print(f"  Peak GPU memory: {peak_mem:.1f} GB")


if __name__ == "__main__":
    main()
