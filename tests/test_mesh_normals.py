import numpy as np
from PIL import Image
import pytest
import trimesh

from scripts.correct_glb_normals import accessor_array, correct_glb, image_hashes, read_glb
from pixalmlx.mesh_normals import smooth_corner_normals


def test_uv_split_sphere_corners_share_smooth_normals():
    sphere = trimesh.creation.icosphere(subdivisions=1)
    vertices = np.array(sphere.vertices)[np.array(sphere.faces).reshape(-1)]
    faces = np.arange(len(vertices)).reshape(-1, 3)
    normals, report = smooth_corner_normals(vertices, faces)
    assert report["smooth_edges"] > 0
    np.testing.assert_allclose(np.linalg.norm(normals, axis=2), 1, atol=1e-6)
    _, inverse = np.unique(vertices, axis=0, return_inverse=True)
    flat = normals.reshape(-1, 3)
    for group in np.unique(inverse):
        group_normals = flat[inverse == group]
        np.testing.assert_allclose(group_normals, np.broadcast_to(group_normals[0], group_normals.shape), atol=1e-7)


def test_cube_preserves_sharp_edges():
    cube = trimesh.creation.box()
    normals, report = smooth_corner_normals(cube.vertices, cube.faces, crease_degrees=30)
    expected = np.repeat(np.array(cube.face_normals)[:, None, :], 3, axis=1)
    np.testing.assert_allclose(normals, expected, atol=1e-7)
    assert report["sharp_edges"] == 12


def test_back_to_back_leaf_faces_do_not_cancel_normals():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]] * 2, dtype=np.float32)
    faces = np.array([[0, 1, 2], [5, 4, 3]])
    normals, _ = smooth_corner_normals(vertices, faces)
    np.testing.assert_allclose(normals[0], [[0, 0, 1]] * 3)
    np.testing.assert_allclose(normals[1], [[0, 0, -1]] * 3)


def test_degenerate_triangles_receive_finite_fallback_normals_without_geometry_changes():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 0, 1]])
    original_vertices, original_faces = vertices.copy(), faces.copy()
    normals, report = smooth_corner_normals(vertices, faces)
    assert report["degenerate_triangles"] == 1
    np.testing.assert_allclose(np.linalg.norm(normals, axis=2), 1, atol=1e-7)
    np.testing.assert_array_equal(vertices, original_vertices)
    np.testing.assert_array_equal(faces, original_faces)


def test_entirely_degenerate_geometry_is_rejected():
    with pytest.raises(ValueError, match="entirely degenerate"):
        smooth_corner_normals(np.zeros((3, 3)), np.array([[0, 1, 2]]))


def test_nonmanifold_sheets_are_not_blended_together():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    normals, report = smooth_corner_normals(vertices, faces)
    expected = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_normals
    np.testing.assert_allclose(normals, np.repeat(expected[:, None, :], 3, axis=1), atol=1e-7)
    assert report["nonmanifold_edges_not_joined"] == 1


@pytest.mark.parametrize("angle", [0, 180, -1])
def test_invalid_crease_angles_are_rejected(angle):
    with pytest.raises(ValueError, match="Crease angle"):
        smooth_corner_normals(np.eye(3), np.array([[0, 1, 2]]), crease_degrees=angle)


def test_glb_correction_preserves_triangles_uv_materials_and_image_bytes(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=0)
    vertices = np.array(sphere.vertices)[np.array(sphere.faces).reshape(-1)]
    faces = np.arange(len(vertices)).reshape(-1, 3)
    uvs = np.tile([[0, 0], [1, 0], [0.5, 1]], (len(faces), 1))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (8, 8), "magenta"),
                                                   metallicRoughnessTexture=Image.new("RGB", (8, 8), "green"))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=np.zeros_like(vertices), process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    source, target = tmp_path / "source.glb", tmp_path / "corrected.glb"
    source.write_bytes(mesh.export(file_type="glb"))
    report = correct_glb(source, target, max_faces=20)
    assert report["triangles"] == 20
    assert report["primitives"][0]["zero_normals_after"] == 0
    old_document, old_binary = read_glb(source)
    document, binary = read_glb(target)
    assert document["materials"] == old_document["materials"]
    assert image_hashes(document, binary) == image_hashes(old_document, old_binary)
    old_primitive = old_document["meshes"][0]["primitives"][0]
    primitive = document["meshes"][0]["primitives"][0]
    old_faces = accessor_array(old_document, old_binary, old_primitive["indices"]).reshape(-1, 3)
    new_faces = accessor_array(document, binary, primitive["indices"]).reshape(-1, 3)
    for attribute in ("POSITION", "TEXCOORD_0"):
        previous = accessor_array(old_document, old_binary, old_primitive["attributes"][attribute])[old_faces]
        actual = accessor_array(document, binary, primitive["attributes"][attribute])[new_faces]
        assert previous.tobytes() == actual.tobytes()
    with pytest.raises(ValueError, match="overwrite"):
        correct_glb(source, target)
    with pytest.raises(ValueError, match="triangle limit"):
        correct_glb(source, tmp_path / "over-budget.glb", max_faces=19)


def test_final_export_normal_correction_is_uncapped_and_atomic(tmp_path):
    import hashlib
    import json

    from pixalmlx.glb_normals import export_with_corrected_normals

    mesh = trimesh.creation.icosphere(subdivisions=5)
    output = tmp_path / "mesh.glb"
    report = export_with_corrected_normals(mesh, output)
    assert report["triangles"] == len(mesh.faces) > 5000
    assert report["max_faces"] == 0
    original = output.read_bytes()
    with pytest.raises(ValueError, match="triangle limit"):
        export_with_corrected_normals(mesh, output, max_faces=5000)
    assert output.read_bytes() == original
    assert json.loads(output.with_suffix(".normals.json").read_text())["output_sha256"] == hashlib.sha256(original).hexdigest()
