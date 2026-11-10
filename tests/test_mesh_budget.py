import json

import numpy as np
import pytest
import trimesh

from trellmlx.mesh_budget import enforce_face_limit


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
def test_invalid_strict_triangle_budgets(limit):
    with pytest.raises(ValueError, match="positive integer"):
        enforce_face_limit(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32), limit)


def test_mesh_within_budget_is_unchanged():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)
    actual_vertices, actual_faces = enforce_face_limit(vertices, faces, len(faces))
    assert actual_vertices is vertices and actual_faces is faces


def test_real_simplifier_satisfies_triangle_cap():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    vertices, faces = enforce_face_limit(np.array(mesh.vertices), np.array(mesh.faces), 100)
    assert 0 < len(faces) <= 100
    assert np.isfinite(vertices).all()
    assert faces.max() < len(vertices)


def test_welding_fallback_can_reduce_independently_split_triangles():
    import fast_simplification

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    vertices = np.array(mesh.vertices)[np.array(mesh.faces).reshape(-1)]
    faces = np.arange(len(vertices)).reshape(-1, 3)
    original = vertices.copy()
    diagnostics = {}

    def stalled_until_welded(points, triangles, **kwargs):
        if len(points) == len(original):
            return points, triangles
        return fast_simplification.simplify(points, triangles, **kwargs)

    output_vertices, output_faces = enforce_face_limit(vertices, faces, 100,
                                                       simplify=stalled_until_welded, diagnostics=diagnostics)
    assert 0 < len(output_faces) <= 100
    assert np.isfinite(output_vertices).all()
    assert diagnostics["fallback_position_decimals"] == 7
    np.testing.assert_array_equal(vertices, original)


def test_failure_to_meet_cap_raises_instead_of_truncating_faces():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    with pytest.raises(RuntimeError, match="could not meet"):
        enforce_face_limit(np.array(mesh.vertices), np.array(mesh.faces), 10,
                           simplify=lambda vertices, faces, **kwargs: (vertices, faces))


def test_nonfinite_simplification_is_rejected():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.5)

    def invalid(vertices, faces, **kwargs):
        return np.full_like(vertices, np.nan), faces[:5]

    with pytest.raises(RuntimeError, match="nonfinite"):
        enforce_face_limit(np.array(mesh.vertices), np.array(mesh.faces), 10, simplify=invalid)


def test_checkpoint_reexport_writes_valid_glb_without_inference(tmp_path, monkeypatch):
    import reexport_pixal3d

    mesh = trimesh.creation.icosphere(subdivisions=0, radius=0.5)
    np.savez(tmp_path / "mesh.npz", vertices=np.array(mesh.vertices), faces=np.array(mesh.faces))
    (tmp_path / "mesh.json").write_text(json.dumps({"hr_resolution": 512}))
    np.savez(tmp_path / "texture.npz", tex_out=np.ones((1, 6), dtype=np.float32),
             tex_coords=np.zeros((1, 4), dtype=np.int32))
    monkeypatch.setattr(reexport_pixal3d, "uv_unwrap", lambda vertices, faces: (
        vertices, faces, np.zeros((len(vertices), 2)), np.arange(len(vertices))))

    def bake(*args, **kwargs):
        assert args[-1] == 512
        return np.full((8, 8, 4), 255, dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8), "OPAQUE"

    monkeypatch.setattr(reexport_pixal3d, "bake_texture", bake)
    arguments = ["--ckpt", str(tmp_path), "--output", str(tmp_path / "model.glb"),
                 "--report", str(tmp_path / "report.json"), "--max-faces", "20", "--texture-size", "8"]
    report = reexport_pixal3d.main(arguments)
    assert report["status"] == "complete" and not report["neural_inference"]
    assert report["exported_faces"] == 20
    assert (tmp_path / "model.glb").exists()
    with pytest.raises(SystemExit) as error:
        reexport_pixal3d.main(arguments)
    assert error.value.code == 2
