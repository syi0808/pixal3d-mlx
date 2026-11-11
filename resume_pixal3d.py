"""Resume Pixal3D pipeline from shape_latent checkpoint.

Skips stages 1-3 (geometry inference, ~35 min) and runs:
  Stage 3: Shape decode
  Stage 4: Texture SLat
  Stage 5: Texture decode
  Stage 6: Texture bake + GLB export

Usage:
    PYTHONPATH=. python resume_pixal3d.py --image photo.png --ckpt outputs/ckpt_dir --output mesh.glb
"""

import argparse
import gc
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from pixalmlx.cgal_holes import add_hole_fill_argument, validate_hole_fill_backend

from generate_pixal3d import (
    to_pixal3d_glb_frame,
    get_default_camera_params, extract_proj_features, index_proj_by_coords,
    _denormalize_slat, _normalize_slat, _requantize_coords,
    SHAPE_SLAT_MEAN, SHAPE_SLAT_STD, TEX_SLAT_MEAN, TEX_SLAT_STD,
)


def main(argv=None, runtime=None):
    parser = argparse.ArgumentParser(description="Resume Pixal3D from checkpoint")
    parser.add_argument("--image", required=True)
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory")
    parser.add_argument("--output", default="outputs/resumed.glb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--texture-steps", type=int, default=12,
                        help="Texture sampling steps (1-100, default: 12)")
    parser.add_argument("--target-faces", type=int, default=500_000,
                        help="Soft triangle target before final mesh repairs; use --max-faces for a strict cap")
    parser.add_argument("--max-faces", type=int, default=0,
                        help="Strict final triangle limit after mesh repair (0=disabled)")
    parser.add_argument("--no-rembg", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    add_hole_fill_argument(parser)
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--memory-budget-gb", type=float, default=10.0,
                        help="MLX graph-evaluation guideline in decimal GB, not a hard allocation cap")
    parser.add_argument("--model-cache-gb", type=float, default=2.0)
    parser.add_argument("--feature-cache-mb", type=float, default=256.0)
    parser.add_argument("--conditioning-cache-mb", type=float, default=128.0)
    parser.add_argument("--ffn-chunk-size", type=int, default=2048)
    parser.add_argument("--naf-backend", choices=["auto", "metal", "reference"], default="auto")
    parser.add_argument("--naf-tile-rows", type=int, default=128)
    parser.add_argument("--naf-threads", type=int, choices=[64, 128, 256], default=128)
    parser.add_argument("--sparse-order", choices=["preserve", "morton"], default="morton")
    parser.add_argument("--decoder-eval-interval", type=int, default=1,
                        help="Evaluate every N sparse kernels in the final shape/texture upsample (0=legacy)")
    parser.add_argument("--slat-precision", choices=["float32", "mixed"], default="float32")
    parser.add_argument("--quantize", type=int, choices=[0, 4, 8], default=0)
    parser.add_argument("--dense-conditioning", action="store_true")
    parser.add_argument("--compile-step", action="store_true")
    parser.add_argument("--compile-transformer", action=argparse.BooleanOptionalAction, default=True,
                        help="Compile bounded FFN and modulation/residual operations (default: enabled)")
    parser.add_argument("--runtime-report")
    args = parser.parse_args(argv)
    if not 1 <= args.texture_steps <= 100:
        parser.error("Texture sampling steps must be integers from 1 to 100")
    if min(args.feature_cache_mb, args.conditioning_cache_mb, args.ffn_chunk_size, args.naf_tile_rows,
           args.decoder_eval_interval, args.max_faces) < 0:
        parser.error("Cache and chunk sizes must be nonnegative")
    if not args.no_cleanup:
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
                return _resume(args, owned_runtime)
    with runtime.job(args.runtime_report, metadata=vars(args)):
        return _resume(args, runtime)


def _resume(args, runtime):
    from pixalmlx.runtime import checkpoint_fingerprint

    runtime.advance("shape_decode_and_mesh")

    mx.random.seed(args.seed)
    mx.metal.reset_peak_memory()
    t_total = time.perf_counter()

    from pixalmlx.weight_loader import load_weights
    from pixalmlx.samplers import flow_euler_sample
    from pixalmlx.cleanup import cleanup_model
    from pixalmlx.checkpoint import load_checkpoint

    # Camera: manual FOV > camera saved with the generation checkpoint > default
    from generate_pixal3d import load_checkpointed_camera
    saved_camera = load_checkpointed_camera(args.ckpt)
    if args.fov > 0:
        camera_params = get_default_camera_params(fov_rad=args.fov)
    elif saved_camera is not None:
        camera_params = saved_camera
        print(f"  Checkpointed camera: fov={camera_params['camera_angle_x']:.4f} rad, "
              f"distance={camera_params['distance']:.4f}", flush=True)
    else:
        camera_params = get_default_camera_params()
        print("  WARNING: checkpoint has no camera.json; resumed stages use the default FOV, "
              "not the camera that produced the shape.", flush=True)

    HF_PIXAL3D = os.path.expanduser(
        "~/.cache/huggingface/hub/models--TencentARC--Pixal3D/"
        "snapshots/0b31f9160aa400719af409098bff7936a932f726/ckpts/"
    )

    # === Load checkpoint ===
    print("=== Loading shape_latent checkpoint ===", flush=True)
    ckpt = load_checkpoint(args.ckpt, 'shape_latent')
    hr_slat = mx.array(ckpt['hr_slat'])
    quant_coords = ckpt['quant_coords']
    hr_coords_3d = quant_coords[:, 1:4]
    hr_resolution = int(ckpt['hr_resolution'])
    num_tokens = len(quant_coords)
    print(f"  {num_tokens:,} tokens, res {hr_resolution}", flush=True)

    # === Stage 3: Shape Decode ===
    print("\n=== Stage 3: Decode Shape ===", flush=True)
    from pixalmlx.models.shape_slat_decoder import SLatDecoder

    shape_dec_ckpt = HF_PIXAL3D + "shape_dec_next_dc_f16c32_fp16.safetensors"
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
    del dec_out, dec_coords
    gc.collect()

    mesh_grid_size = hr_resolution
    t0 = time.perf_counter()
    vertices, faces = decoder_output_to_mesh(
        dec_feats_np, dec_coords_np, resolution=mesh_grid_size,
    )
    print(f"  Extracted: {time.perf_counter()-t0:.1f}s ({len(vertices):,}V {len(faces):,}F)", flush=True)

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

    # === DINOv3 + NAF for texture conditioning ===
    print("\n=== Loading DINOv3 + NAF ===", flush=True)
    runtime.advance("texture_conditioning")
    from pixalmlx.models.dinov3 import DINOv3ViT, load_dinov3_weights
    from pixalmlx.models.naf import NAF as NAFModel
    from pixalmlx.models.naf_loader import load_naf_weights

    DINOV3_WEIGHTS = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/"
    )
    if os.path.isdir(DINOV3_WEIGHTS):
        snapshots = [d for d in os.listdir(DINOV3_WEIGHTS) if not d.startswith(".")]
        if snapshots:
            DINOV3_WEIGHTS = os.path.join(DINOV3_WEIGHTS, snapshots[0])

    dinov3 = DINOv3ViT()
    dinov3 = runtime.load_model(("dinov3", checkpoint_fingerprint(DINOV3_WEIGHTS)), lambda: dinov3,
                                lambda model: load_dinov3_weights(model, DINOV3_WEIGHTS))
    naf = NAFModel()
    naf_path = os.path.join(os.path.dirname(__file__), "weights", "naf_release.safetensors")
    naf = runtime.load_model(("naf", checkpoint_fingerprint(naf_path)), lambda: naf,
                             lambda model: load_naf_weights(model, naf_path, verbose=False))
    if hasattr(naf, "upsampler"):
        naf.upsampler.backend = args.naf_backend
        naf.upsampler.threadgroup_size = args.naf_threads
        naf.encoder_tile_rows = args.naf_tile_rows
    print("  Loaded.", flush=True)

    # === Stage 4: Texture SLat ===
    print("\n=== Stage 4: Texture SLat ===", flush=True)
    from pixalmlx.models.pixal3d_flow import Pixal3DSLatFlowModel

    TEX_SAMPLER = dict(steps=args.texture_steps, guidance_strength=1.0, guidance_rescale=0.0,
                       guidance_interval=(0.6, 0.9), rescale_t=3.0)

    tex_image_size = 1024
    tex_naf_size = 512
    tex_grid_resolution = hr_resolution // 16
    print(f"  Extracting proj features (tex, {tex_image_size}, grid={tex_grid_resolution}, NAF to {tex_naf_size})...", flush=True)
    tex_cond, tex_neg_cond = extract_proj_features(
        args.image, dinov3, grid_resolution=tex_grid_resolution, image_size=tex_image_size,
        camera_params=camera_params, no_rembg=args.no_rembg,
        use_naf_upsample=True, upsample_target_size=tex_naf_size, naf_model=naf,
        coords=None if args.dense_conditioning else quant_coords,
        feature_cache=runtime.features,
    )
    tex_cond_sparse, tex_neg_cond_sparse = index_proj_by_coords(
        tex_cond, tex_neg_cond, quant_coords, grid_resolution=tex_grid_resolution,
        preindexed=not args.dense_conditioning,
    )
    del dinov3, naf
    runtime.advance("texture_sampling")

    tex_flow = Pixal3DSLatFlowModel(
        in_channels=64, out_channels=32,
        model_channels=1536, num_heads=12,
        num_blocks=30, mlp_hidden=8192,
        context_channels=1024, proj_in_channels=2048,
    )
    tex_flow = runtime.load_model("texture_flow", lambda: tex_flow,
                                  lambda model: load_weights(model, HF_PIXAL3D + "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
                                                             verbose=False, streaming=True), retain=False)
    if args.quantize:
        from pixalmlx.quantize import quantize_model

        quantize_model(tex_flow, bits=args.quantize)
    if args.slat_precision == "mixed":
        tex_flow.compute_dtype = mx.float16

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

    tex_slat = _denormalize_slat(tex_slat, mean=TEX_SLAT_MEAN, std=TEX_SLAT_STD)
    mx.eval(tex_slat)

    cleanup_model(tex_flow)
    del tex_flow, tex_cond, tex_neg_cond, tex_cond_sparse, tex_neg_cond_sparse
    gc.collect()
    mx.metal.clear_cache()

    # === Stage 5: Texture Decode ===
    print("\n=== Stage 5: Texture Decode ===", flush=True)

    tex_dec_ckpt = HF_PIXAL3D + "tex_dec_next_dc_f16c32_fp16.safetensors"
    if not os.path.exists(tex_dec_ckpt):
        HF_4B = os.path.expanduser(
            "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/"
            "snapshots/af44b45f2e35a493886929c6d786e563ec68364d/ckpts/"
        )
        tex_dec_ckpt = HF_4B + "tex_dec_next_dc_f16c32_fp16.safetensors"

    runtime.advance("texture_decode_and_export")
    tex_decoder = SLatDecoder(out_channels=6, pred_subdiv=False, spatial_order=args.sparse_order,
                              final_upsample_eval_interval=args.decoder_eval_interval)
    tex_decoder = runtime.load_model(("texture_decoder", checkpoint_fingerprint(tex_dec_ckpt), args.sparse_order, args.decoder_eval_interval), lambda: tex_decoder,
                                     lambda model: load_weights(model, tex_dec_ckpt, verbose=False, streaming=True))

    t0 = time.perf_counter()
    tex_out, tex_coords = tex_decoder(
        tex_slat, mx.array(quant_coords), guide_subs=shape_subs,
    )
    mx.eval(tex_out)
    tex_out = tex_out * 0.5 + 0.5
    mx.eval(tex_out)
    print(f"  Decoded: {time.perf_counter()-t0:.1f}s ({tex_out.shape[0]:,} voxels)", flush=True)

    cleanup_model(tex_decoder)
    del tex_decoder
    gc.collect()
    mx.metal.clear_cache()

    # === Stage 6: Texture Bake + GLB ===
    print("\n=== Stage 6: Texture Bake + GLB ===", flush=True)
    from pixalmlx.texture_bake import uv_unwrap, bake_texture
    import trimesh
    from trimesh.visual.material import PBRMaterial

    tex_np = np.array(tex_out)
    tex_coords_spatial = np.array(tex_coords)[:, 1:4]

    t0 = time.perf_counter()
    uv_verts, uv_faces, uvs, vmapping = uv_unwrap(vertices, faces)
    if args.max_faces and len(uv_faces) > args.max_faces:
        raise RuntimeError("UV output exceeds the strict triangle budget")
    print(f"  UV unwrap: {len(uv_verts):,}V {len(uv_faces):,}F ({time.perf_counter()-t0:.1f}s)", flush=True)

    base_color, metallic_roughness, alpha_mode = bake_texture(
        uv_verts, uv_faces, uvs, vmapping,
        tex_coords_spatial, tex_np, mesh_grid_size,
        texture_size=args.texture_size,
        backend="gpu",
    )
    print(f"  Baked textures ({args.texture_size}x{args.texture_size})", flush=True)

    export_verts = to_pixal3d_glb_frame(uv_verts)
    export_uvs = uvs.copy()
    export_uvs[:, 1] = 1 - export_uvs[:, 1]

    mesh = trimesh.Trimesh(vertices=export_verts, faces=uv_faces, process=False)
    normals = mesh.vertex_normals

    material = PBRMaterial(
        baseColorTexture=Image.fromarray(base_color),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(metallic_roughness),
        metallicFactor=1.0, roughnessFactor=1.0,
        alphaMode=alpha_mode, doubleSided=True,
    )

    textured_mesh = trimesh.Trimesh(
        vertices=export_verts, faces=uv_faces, vertex_normals=normals,
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
