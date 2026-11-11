from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from pixalmlx.cgal_holes import cgal_executable, fill_small_holes_cgal
from scripts.benchmark_hole_filling import analytic_patch_error, annulus_fixture, edge_groups


def native_available():
    try:
        cgal_executable()
        return True
    except RuntimeError:
        return False


def test_missing_native_binary_fails_explicitly_without_fan_fallback(tmp_path):
    vertices, faces = annulus_fixture("plane")
    with pytest.raises(RuntimeError, match="not built"):
        fill_small_holes_cgal(vertices, faces, executable=tmp_path / "missing", verbose=False)


def test_disabled_filling_does_not_require_native_binary(tmp_path):
    vertices, faces = annulus_fixture("plane")
    result = fill_small_holes_cgal(vertices, faces, max_hole_perimeter=0, executable=tmp_path / "missing")
    assert result[0] is vertices
    assert result[1] is faces


@pytest.mark.parametrize("options", [{"density_control_factor": 0}, {"density_control_factor": np.nan},
                                    {"fairing_continuity": 3}, {"max_boundary_vertices": 2},
                                    {"max_collar_faces": 0}, {"timeout": 0}, {"fallback": "unknown"}])
def test_invalid_native_options_are_rejected_before_execution(options):
    vertices, faces = annulus_fixture("plane")
    with pytest.raises(ValueError):
        fill_small_holes_cgal(vertices, faces, **options)


def test_invalid_geometry_is_rejected_before_execution():
    vertices, faces = annulus_fixture("plane")
    invalid = vertices.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fill_small_holes_cgal(invalid, faces)
    with pytest.raises(ValueError, match="bounds"):
        fill_small_holes_cgal(vertices, faces + len(vertices))


def test_bridge_rejects_corrupt_native_payload(tmp_path, monkeypatch):
    import pixalmlx.cgal_holes as module

    binary = tmp_path / "native"
    binary.write_text("unused")
    binary.chmod(0o755)

    def corrupt(command, **options):
        Path(command[2]).write_bytes(struct.pack("<8sQQ", b"PXHOUT01", 100, 0))
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(module.subprocess, "run", corrupt)
    vertices, faces = annulus_fixture("plane")
    with pytest.raises(RuntimeError, match="payload size"):
        fill_small_holes_cgal(vertices, faces, executable=binary)


def test_bridge_rejects_degenerate_added_triangles(tmp_path, monkeypatch):
    import pixalmlx.cgal_holes as module

    vertices, faces = annulus_fixture("plane")
    binary = tmp_path / "native"
    binary.write_text("unused")
    binary.chmod(0o755)

    def collapsed(command, **options):
        center = np.array([vertices[0]], dtype="<f8")
        triangle = np.array([[0, 1, len(vertices)]], dtype="<u8")
        Path(command[2]).write_bytes(struct.pack("<8sQQ", b"PXHOUT01", 1, 1) + center.tobytes() + triangle.tobytes())
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(module.subprocess, "run", collapsed)
    with pytest.raises(RuntimeError, match="degenerate"):
        fill_small_holes_cgal(vertices, faces, executable=binary, verbose=False)


def test_cleanup_default_remains_fan_and_cgal_is_explicit(monkeypatch):
    import pixalmlx.cgal_holes as cgal
    import pixalmlx.mesh_cleanup as cleanup

    vertices, faces = annulus_fixture("plane", sides=8, rings=3, inner=0.001, outer=0.01)
    calls = []

    def native(vertices, faces, **options):
        calls.append(options)
        return vertices, faces

    monkeypatch.setattr(cgal, "fill_small_holes_cgal", native)
    cleanup.cleanup_mesh(vertices, faces, do_fix_normals=False, verbose=False)
    assert not calls
    cleanup.cleanup_mesh(vertices, faces, do_fix_normals=False, hole_fill_backend="cgal", verbose=False)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="hole_fill_backend"):
        cleanup.cleanup_mesh(vertices, faces, hole_fill_backend="unknown")


@pytest.mark.skipif(not native_available(), reason="Optional CGAL executable is not built")
@pytest.mark.parametrize("surface", ["plane", "sphere", "saddle"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_patch_keeps_original_surface_and_closes_only_selected_hole(surface, dtype):
    vertices, faces = annulus_fixture(surface)
    vertices = vertices.astype(dtype)
    diagnostics = {}
    output_vertices, output_faces = fill_small_holes_cgal(
        vertices, faces, max_hole_perimeter=4, diagnostics=diagnostics, verbose=False)
    assert output_vertices[:len(vertices)].tobytes() == vertices.tobytes()
    assert output_faces[:len(faces)].tobytes() == faces.tobytes()
    assert diagnostics["filled_loops"] == 1
    assert diagnostics["added_vertices"] > 0
    assert diagnostics["rejected"]["perimeter"] == 1
    keys, owners, counts = edge_groups(output_faces)
    assert np.count_nonzero(counts == 1) == 48
    assert not np.any(counts > 2)
    assert np.isfinite(output_vertices).all()
    errors = analytic_patch_error(output_vertices, output_faces, len(faces), surface)
    assert errors["samples"] > 0
    if surface == "plane":
        assert errors["height_max_absolute_error"] < 1e-10
    else:
        from pixalmlx.mesh_cleanup import fill_small_holes

        fan_vertices, fan_faces = fill_small_holes(vertices, faces, max_hole_perimeter=4, verbose=False)
        fan_error = analytic_patch_error(fan_vertices, fan_faces, len(faces), surface)
        assert errors["samples"] == fan_error["samples"]
        assert errors["height_rmse"] < fan_error["height_rmse"]


@pytest.mark.skipif(not native_available(), reason="Optional CGAL executable is not built")
def test_native_rejects_oversized_boundaries_without_silent_fallback():
    vertices, faces = annulus_fixture("sphere")
    diagnostics = {}
    output_vertices, output_faces = fill_small_holes_cgal(
        vertices, faces, max_hole_perimeter=3, max_boundary_vertices=8, diagnostics=diagnostics, verbose=False)
    np.testing.assert_array_equal(output_vertices, vertices)
    np.testing.assert_array_equal(output_faces, faces)
    assert diagnostics["filled_loops"] == 0
    assert diagnostics["rejected"]["boundary_budget"] == 1


@pytest.mark.skipif(not native_available(), reason="Optional CGAL executable is not built")
def test_explicit_fan_fallback_closes_native_rejected_hole_and_reports_it():
    vertices, faces = annulus_fixture("sphere")
    diagnostics = {}
    output_vertices, output_faces = fill_small_holes_cgal(
        vertices, faces, max_hole_perimeter=3, max_boundary_vertices=8,
        diagnostics=diagnostics, fallback="fan", verbose=False)
    assert diagnostics["cgal_filled_loops"] == 0
    assert diagnostics["fallback_filled_loops"] == 1
    assert diagnostics["filled_loops"] == 1
    assert output_vertices[:len(vertices)].tobytes() == vertices.tobytes()
    assert output_faces[:len(faces)].tobytes() == faces.tobytes()
    keys, owners, counts = edge_groups(output_faces)
    assert np.count_nonzero(counts == 1) == 48


@pytest.mark.skipif(not native_available(), reason="Optional CGAL executable is not built")
def test_native_shadow_orientation_does_not_reorder_original_faces():
    vertices, faces = annulus_fixture("sphere")
    faces[::3] = faces[::3][:, [0, 2, 1]]
    diagnostics = {}
    output_vertices, output_faces = fill_small_holes_cgal(
        vertices, faces, max_hole_perimeter=4, diagnostics=diagnostics, verbose=False)
    assert diagnostics["filled_loops"] == 1
    assert output_faces[:len(faces)].tobytes() == faces.tobytes()
    assert output_vertices[:len(vertices)].tobytes() == vertices.tobytes()


@pytest.mark.skipif(not native_available(), reason="Optional CGAL executable is not built")
def test_native_preserves_explicit_integer_byte_order():
    vertices, faces = annulus_fixture("sphere")
    faces = faces.astype(">i8")
    output_vertices, output_faces = fill_small_holes_cgal(vertices, faces, max_hole_perimeter=4, verbose=False)
    assert output_faces.dtype == faces.dtype
    assert output_faces[:len(faces)].tobytes() == faces.tobytes()


@pytest.mark.skipif(not native_available(), reason="Optional CGAL executable is not built")
def test_native_ambiguous_boundary_is_not_silently_filled():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 3, 4]])
    diagnostics = {}
    output_vertices, output_faces = fill_small_holes_cgal(
        vertices, faces, max_hole_perimeter=10, diagnostics=diagnostics, verbose=False)
    assert diagnostics["filled_loops"] == 0
    assert diagnostics["rejected"]["ambiguous_boundary"] == 1
    np.testing.assert_array_equal(output_vertices, vertices)
    np.testing.assert_array_equal(output_faces, faces)


@pytest.mark.skipif(not native_available(), reason="Optional CGAL executable is not built")
def test_native_degenerate_collar_rejection_preserves_input():
    vertices, faces = annulus_fixture("plane")
    vertices[0] = vertices[1]
    diagnostics = {}
    output_vertices, output_faces = fill_small_holes_cgal(
        vertices, faces, max_hole_perimeter=4, diagnostics=diagnostics, verbose=False)
    assert diagnostics["filled_loops"] == 0
    assert diagnostics["rejected"]["degenerate_collar"] == 1
    np.testing.assert_array_equal(output_vertices, vertices)
    np.testing.assert_array_equal(output_faces, faces)


@pytest.mark.skipif(not native_available(), reason="Optional CGAL executable is not built")
def test_native_rejects_truncated_input_without_writing_output(tmp_path):
    source, destination = tmp_path / "input.bin", tmp_path / "output.bin"
    source.write_bytes(b"PXH")
    process = subprocess.run([str(cgal_executable()), str(source), str(destination)], capture_output=True, text=True)
    assert process.returncode != 0
    assert "Truncated input header" in process.stderr
    assert not destination.exists()


def test_fixed_grid_analytic_metric_compares_the_same_locations():
    from pixalmlx.mesh_cleanup import fill_small_holes

    vertices, faces = annulus_fixture("sphere")
    output_vertices, output_faces = fill_small_holes(vertices, faces, max_hole_perimeter=4, verbose=False)
    error = analytic_patch_error(output_vertices, output_faces, len(faces), "sphere")
    assert error["samples"] == error["requested_samples"]
    assert error["samples"] > 7000
    assert error["ambiguous_projected_samples"] == 0
    assert 0.04 < error["height_rmse"] < 0.08
