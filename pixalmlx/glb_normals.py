"""Correct static GLB normals while preserving triangle geometry, UVs, and PBR images."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from pixalmlx.mesh_normals import smooth_corner_normals


DTYPES = {5120: np.dtype("i1"), 5121: np.dtype("u1"), 5122: np.dtype("<i2"),
          5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path):
    data = Path(path).read_bytes()
    if struct.unpack_from("<4sII", data) != (b"glTF", 2, len(data)):
        raise ValueError("Expected a valid GLB v2 file")
    chunks = {}
    position = 12
    while position < len(data):
        length, kind = struct.unpack_from("<II", data, position)
        position += 8
        chunks[kind] = data[position:position + length]
        position += length
    return json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]


def accessor_array(document, binary, index):
    accessor = document["accessors"][index]
    if "sparse" in accessor:
        raise ValueError("Sparse accessors are not supported by this static-mesh correction tool")
    view = document["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise ValueError("Only embedded buffer zero is supported")
    dtype, width = DTYPES[accessor["componentType"]], WIDTHS[accessor["type"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * width)
    return np.ndarray((accessor["count"], width), dtype=dtype, buffer=binary, offset=offset,
                      strides=(stride, dtype.itemsize)).copy()


def append_accessor(document, binary, array, template, target):
    array = np.ascontiguousarray(array)
    binary.extend(b"\0" * (-len(binary) % 4))
    offset = len(binary)
    payload = array.tobytes()
    binary.extend(payload)
    view_index = len(document["bufferViews"])
    document["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload), "target": target})
    accessor = deepcopy(template)
    accessor.update({"bufferView": view_index, "byteOffset": 0, "count": len(array)})
    accessor.pop("sparse", None)
    accessor.pop("min", None)
    accessor.pop("max", None)
    if target == 34962 and accessor["type"] == "VEC3" and template.get("min") is not None:
        accessor["min"], accessor["max"] = array.min(axis=0).tolist(), array.max(axis=0).tolist()
    document["accessors"].append(accessor)
    return len(document["accessors"]) - 1


def image_hashes(document, binary):
    result = []
    for image in document.get("images", []):
        view = document["bufferViews"][image["bufferView"]]
        offset = view.get("byteOffset", 0)
        result.append(hashlib.sha256(binary[offset:offset + view["byteLength"]]).hexdigest())
    return result


def correct_glb(input_path, output_path, max_faces=0, crease_degrees=75.0):
    if max_faces < 0:
        raise ValueError("Triangle limit must be nonnegative")
    if Path(output_path).exists():
        raise ValueError("Refusing to overwrite a corrected model")
    original, original_binary = read_glb(input_path)
    if original.get("skins") or original.get("animations"):
        raise ValueError("Only static GLB meshes are supported")
    document, binary = deepcopy(original), bytearray(original_binary)
    reports = []
    total_faces = 0
    for mesh_index, mesh in enumerate(document["meshes"]):
        for primitive_index, primitive in enumerate(mesh["primitives"]):
            if primitive.get("mode", 4) != 4 or primitive.get("targets"):
                raise ValueError("Only unmorphed triangle primitives are supported")
            old_primitive = original["meshes"][mesh_index]["primitives"][primitive_index]
            if "TANGENT" in old_primitive["attributes"]:
                raise ValueError("Meshes with explicit tangents require tangent rebuilding, which this tool does not perform")
            attributes = {name: accessor_array(original, original_binary, index)
                          for name, index in old_primitive["attributes"].items()}
            faces = accessor_array(original, original_binary, old_primitive["indices"]).reshape(-1, 3)
            total_faces += len(faces)
            if max_faces and total_faces > max_faces:
                raise ValueError("Source model exceeds the requested triangle limit")
            normals, metrics = smooth_corner_normals(attributes["POSITION"], faces, crease_degrees)
            corners = faces.reshape(-1)
            keys = np.empty(len(corners), dtype=[("vertex", "<u4"), ("normal", "<f4", (3,))])
            keys["vertex"], keys["normal"] = corners, normals.reshape(-1, 3)
            _, representatives, inverse = np.unique(keys, return_index=True, return_inverse=True)
            for name, array in attributes.items():
                if name == "NORMAL":
                    continue
                template = original["accessors"][old_primitive["attributes"][name]]
                primitive["attributes"][name] = append_accessor(document, binary, array[corners[representatives]], template, 34962)
            normal_array = normals.reshape(-1, 3)[representatives]
            primitive["attributes"]["NORMAL"] = append_accessor(
                document, binary, normal_array, {"type": "VEC3", "componentType": 5126}, 34962)
            primitive["indices"] = append_accessor(
                document, binary, inverse.astype("<u4").reshape(-1, 1), {"type": "SCALAR", "componentType": 5125}, 34963)
            new_faces = inverse.reshape(-1, 3)
            for name, array in attributes.items():
                if name != "NORMAL":
                    restored = accessor_array(document, binary, primitive["attributes"][name])[new_faces]
                    if restored.tobytes() != array[faces].tobytes():
                        raise RuntimeError(f"Normal correction changed triangle attribute {name}")
            lengths = np.linalg.norm(normal_array, axis=1)
            if not np.isfinite(normal_array).all() or not np.allclose(lengths, 1.0, atol=1e-6):
                raise RuntimeError("Corrected normals must all be finite unit vectors")
            before = attributes.get("NORMAL", np.zeros_like(attributes["POSITION"]))
            reports.append({"mesh": mesh_index, "primitive": primitive_index, "triangles": len(faces),
                            "vertices_before": len(attributes["POSITION"]), "vertices_after": len(normal_array),
                            "zero_normals_before": int((np.linalg.norm(before, axis=1) < 1e-6).sum()),
                            "zero_normals_after": int((lengths < 1e-6).sum()),
                            "normal_length_min": float(lengths.min()), "normal_length_max": float(lengths.max()),
                            "triangle_geometry_and_attributes_identical": True, **metrics})
    if image_hashes(document, binary) != image_hashes(original, original_binary):
        raise RuntimeError("Embedded PBR image data changed")
    if document.get("materials") != original.get("materials"):
        raise RuntimeError("PBR material parameters changed")
    document["buffers"][0]["byteLength"] = len(binary)
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    payload = (struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary)) +
               struct.pack("<II", len(encoded), 0x4E4F534A) + encoded +
               struct.pack("<II", len(binary), 0x004E4942) + binary)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(payload)
    return {"status": "complete", "input": str(Path(input_path).resolve()), "output": str(Path(output_path).resolve()),
            "triangles": total_faces, "max_faces": max_faces, "geometry_uv_and_other_attributes_preserved": True,
            "embedded_pbr_image_bytes_preserved": True, "material_parameters_preserved": True,
            "source_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(payload).hexdigest(), "output_bytes": len(payload), "primitives": reports}


def export_with_corrected_normals(mesh, output_path, max_faces=0):
    import tempfile

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pixal3d-export-", dir=output.parent) as directory:
        source, corrected = Path(directory) / "source.glb", Path(directory) / "corrected.glb"
        source.write_bytes(mesh.export(file_type="glb"))
        report = correct_glb(source, corrected, max_faces=max_faces)
        report.update(input="in-memory mesh export", output=str(output))
        report_path = Path(directory) / "normals.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        corrected.replace(output)
        report_path.replace(output.with_suffix(".normals.json"))
    return report
