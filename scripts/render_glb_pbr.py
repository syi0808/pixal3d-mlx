"""Render an actual GLB with Blender Cycles, embedded PBR maps, and imported normals.

Run using Blender's bundled Python: blender --background --factory-startup --python FILE -- ARGS.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector


def aim_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def area_light(name, location, energy, size, target):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    aim_at(obj, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", nargs="+", choices=["front", "three-quarter", "back"], default=["front", "three-quarter", "back"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-faces", type=int, default=0, help="Validate asset triangles only; staging floor is not exported")
    parser.add_argument("--save-blend", action="store_true")
    arguments = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if arguments.samples < 1 or arguments.resolution < 64 or arguments.threads < 1 or arguments.max_faces < 0:
        parser.error("Use positive samples/threads and resolution at least 64")
    if (arguments.output_dir / "render.json").exists():
        parser.error("Refusing to overwrite render evidence")
    started = time.perf_counter()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(arguments.input.resolve()), import_shading="NORMALS", merge_vertices=False)
    bpy.context.view_layer.update()
    assets = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not assets:
        raise RuntimeError("Expected nonempty GLB mesh geometry")
    triangle_count = 0
    for obj in assets:
        obj.data.calc_loop_triangles()
        triangle_count += len(obj.data.loop_triangles)
        for material in obj.data.materials:
            if material and material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        node.interpolation = "Linear"
    if arguments.max_faces and triangle_count > arguments.max_faces:
        raise RuntimeError("Imported asset exceeds the triangle limit")
    corners = [obj.matrix_world @ Vector(corner) for obj in assets for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    span = max(maximum - minimum)
    if not assets or triangle_count <= 0 or span <= 0:
        raise RuntimeError("Expected nonempty GLB mesh geometry")
    scene = bpy.context.scene
    bpy.ops.preferences.addon_enable(module="cycles")
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = arguments.samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = 42
    scene.cycles.max_bounces = 8
    scene.render.threads_mode = "FIXED"
    scene.render.threads = arguments.threads
    scene.render.resolution_x = scene.render.resolution_y = arguments.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.8, 0.82, 0.86, 1.0)
    background.inputs["Strength"].default_value = 0.3
    bpy.ops.mesh.primitive_plane_add(size=200 * span, location=(center.x, center.y, minimum.z - 0.002 * span))
    floor = bpy.context.object
    floor.name = "Render staging floor - not part of GLB"
    floor_material = bpy.data.materials.new("Neutral studio floor")
    floor_material.use_nodes = True
    principled = floor_material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.65, 0.65, 0.63, 1.0)
    principled.inputs["Roughness"].default_value = 0.9
    floor.data.materials.append(floor_material)
    area_light("Key", center + Vector((-2.0, 2.8, 3.5)) * span, 500, 3 * span, center)
    area_light("Fill", center + Vector((2.5, 1.8, 1.8)) * span, 300, 3 * span, center)
    area_light("Rim", center + Vector((-1.0, -2.5, 3.0)) * span, 400, 2.5 * span, center)
    camera_data = bpy.data.cameras.new("Asset camera")
    camera = bpy.data.objects.new("Asset camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span * 1.23
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100
    directions = {"front": Vector((0.0, 3.0, 0.08)), "three-quarter": Vector((1.6, 2.8, 0.25)),
                  "back": Vector((0.0, -3.0, 0.08))}
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    renders = []
    for view in arguments.views:
        camera.location = center + directions[view] * span
        aim_at(camera, center)
        output = arguments.output_dir / (view + ".png")
        scene.render.filepath = str(output.resolve())
        stage_started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        renders.append({"view": view, "image": str(output.resolve()), "seconds": time.perf_counter() - stage_started})
    if arguments.save_blend:
        camera.location = center + directions["front"] * span
        aim_at(camera, center)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str((arguments.output_dir / "studio.blend").resolve()))
    materials = []
    for obj in assets:
        for material in obj.data.materials:
            if material and material.use_nodes:
                nodes = material.node_tree.nodes
                shader = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
                materials.append({"name": material.name, "principled_bsdf": shader is not None,
                                  "textures": [{"name": node.image.name, "size": list(node.image.size),
                                                "color_space": node.image.colorspace_settings.name,
                                                "interpolation": node.interpolation}
                                               for node in nodes if node.type == "TEX_IMAGE" and node.image],
                                  "base_color_linked": bool(shader and shader.inputs["Base Color"].is_linked),
                                  "roughness_linked": bool(shader and shader.inputs["Roughness"].is_linked),
                                  "metallic_linked": bool(shader and shader.inputs["Metallic"].is_linked)})
    report = {"status": "complete", "blender_version": bpy.app.version_string, "engine": scene.render.engine,
              "device": scene.cycles.device, "samples": arguments.samples, "resolution": arguments.resolution,
              "input": str(arguments.input.resolve()), "input_sha256": hashlib.sha256(arguments.input.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "asset_triangles": triangle_count, "modifiers_added": False, "import_shading": "NORMALS",
              "uv_texture_sampling": "linear per-shading-point", "color_management": scene.view_settings.view_transform,
              "materials": materials, "renders": renders, "elapsed_seconds": time.perf_counter() - started,
              "staging_floor_in_exported_glb": False}
    (arguments.output_dir / "render.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PBR_RENDER_REPORT=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
