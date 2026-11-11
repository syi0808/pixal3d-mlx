"""Re-export an existing Pixal3D mesh/texture checkpoint without neural inference."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import mlx.core as mx
import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial

from generate import _cleanup_and_simplify_mesh
from generate_pixal3d import to_pixal3d_glb_frame
from pixalmlx.checkpoint import load_checkpoint
from pixalmlx.cgal_holes import add_hole_fill_argument, validate_hole_fill_backend
from pixalmlx.mesh_budget import enforce_face_limit
from pixalmlx.runtime import InferenceRuntime
from pixalmlx.texture_bake import bake_texture, uv_unwrap


def reexport(arguments):
    started = time.perf_counter()
    checkpoint = load_checkpoint(str(arguments.ckpt), "mesh")
    texture = load_checkpoint(str(arguments.ckpt), "texture")
    vertices, faces = checkpoint["vertices"], checkpoint["faces"]
    if arguments.raw_mesh is not None:
        with np.load(arguments.raw_mesh) as raw:
            vertices, faces = raw["vertices"], raw["faces"]
    resolution = int(checkpoint["hr_resolution"])
    input_faces = len(faces)
    timings = {"load_seconds": time.perf_counter() - started}
    stage_started = time.perf_counter()
    budget_diagnostics = {}
    if arguments.target_faces:
        vertices, faces = _cleanup_and_simplify_mesh(vertices, faces, target_faces=arguments.target_faces,
                                                     no_cleanup=False, hole_fill_backend=arguments.hole_fill_backend)
    if arguments.max_faces:
        vertices, faces = enforce_face_limit(vertices, faces, arguments.max_faces, diagnostics=budget_diagnostics)
    timings["simplify_seconds"] = time.perf_counter() - stage_started
    print(f"Triangle output: {input_faces:,} -> {len(faces):,} "
          f"(target {arguments.target_faces:,}, strict limit {arguments.max_faces:,}; 0=disabled)", flush=True)
    stage_started = time.perf_counter()
    uv_vertices, uv_faces, uvs, mapping = uv_unwrap(vertices, faces)
    timings["uv_seconds"] = time.perf_counter() - stage_started
    if arguments.max_faces and len(uv_faces) > arguments.max_faces:
        raise RuntimeError("UV output exceeds the triangle budget")
    stage_started = time.perf_counter()
    base_color, metallic_roughness, alpha_mode = bake_texture(
        uv_vertices, uv_faces, uvs, mapping, texture["tex_coords"][:, 1:4], texture["tex_out"],
        resolution, texture_size=arguments.texture_size, backend="gpu")
    mx.synchronize()
    timings["bake_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    export_vertices = to_pixal3d_glb_frame(uv_vertices)
    export_uvs = uvs.copy()
    export_uvs[:, 1] = 1 - export_uvs[:, 1]
    geometry = trimesh.Trimesh(vertices=export_vertices, faces=uv_faces, process=False)
    material = PBRMaterial(baseColorTexture=Image.fromarray(base_color),
                           baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
                           metallicRoughnessTexture=Image.fromarray(metallic_roughness),
                           metallicFactor=1.0, roughnessFactor=1.0, alphaMode=alpha_mode, doubleSided=True)
    mesh = trimesh.Trimesh(vertices=export_vertices, faces=uv_faces, vertex_normals=geometry.vertex_normals,
                           process=False, visual=trimesh.visual.TextureVisuals(uv=export_uvs, material=material))
    from pixalmlx.glb_normals import export_with_corrected_normals

    normal_report = export_with_corrected_normals(mesh, arguments.output, max_faces=arguments.max_faces)
    payload = arguments.output.read_bytes()
    scene = trimesh.load(io.BytesIO(payload), file_type="glb", force="scene", process=False)
    exported_faces = sum(len(part.faces) for part in scene.geometry.values())
    valid = exported_faces > 0 and (not arguments.max_faces or exported_faces <= arguments.max_faces)
    valid = valid and all(np.isfinite(part.vertices).all() and np.isfinite(part.visual.uv).all()
                          and part.visual.material.baseColorTexture.size == (arguments.texture_size,) * 2
                          and part.visual.material.metallicRoughnessTexture.size == (arguments.texture_size,) * 2
                          for part in scene.geometry.values())
    if not valid:
        raise RuntimeError("Exported GLB failed geometry, texture, or strict triangle validation")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(payload)
    timings["export_seconds"] = time.perf_counter() - stage_started
    report = {"status": "complete", "route": "mesh_texture_checkpoint_reexport", "neural_inference": False,
              "source_checkpoints": str(arguments.ckpt.resolve()), "output": str(arguments.output.resolve()),
              "max_faces": arguments.max_faces, "target_faces": arguments.target_faces,
              "hole_fill_backend": arguments.hole_fill_backend,
              "raw_mesh": None if arguments.raw_mesh is None else str(arguments.raw_mesh.resolve()),
              "raw_mesh_sha256": None if arguments.raw_mesh is None else hashlib.sha256(arguments.raw_mesh.read_bytes()).hexdigest(),
              "input_faces": input_faces, "exported_faces": exported_faces,
              "exported_vertices": sum(len(part.vertices) for part in scene.geometry.values()),
              "resolution": resolution, "texture_size": arguments.texture_size, "timings": timings,
              "budget_diagnostics": budget_diagnostics, "normal_correction": normal_report,
              "elapsed_seconds": time.perf_counter() - started, "peak_mlx_bytes": mx.get_peak_memory(),
              "output_bytes": len(payload), "output_sha256": hashlib.sha256(payload).hexdigest(),
              "source_sha256": {name: hashlib.sha256((arguments.ckpt / name).read_bytes()).hexdigest()
                                 for name in ("mesh.npz", "texture.npz")},
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())}
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, default=0, help="Strict final triangle limit (0=disabled)")
    parser.add_argument("--target-faces", type=int, default=0, help="Soft triangle target with standard mesh cleanup")
    add_hole_fill_argument(parser)
    parser.add_argument("--raw-mesh", type=Path,
                        help="Raw decoder geometry for soft-target cleanup, before component filtering/splitting")
    parser.add_argument("--texture-size", type=int, default=1024)
    arguments = parser.parse_args(argv)
    if min(arguments.max_faces, arguments.target_faces) < 0 or arguments.texture_size < 1:
        parser.error("Face settings must be nonnegative and texture size positive")
    if not arguments.max_faces and not arguments.target_faces:
        parser.error("Specify --target-faces or --max-faces")
    if arguments.target_faces and arguments.raw_mesh is None:
        parser.error("--target-faces requires --raw-mesh to avoid re-filtering already split surface parts")
    if arguments.output.exists() or arguments.report.exists():
        parser.error("Refusing to overwrite a model or report")
    if arguments.target_faces:
        try:
            validate_hole_fill_backend(arguments.hole_fill_backend)
        except RuntimeError as error:
            parser.error(str(error))
    with InferenceRuntime(model_cache_bytes=0, feature_cache_bytes=0):
        return reexport(arguments)


if __name__ == "__main__":
    main()
