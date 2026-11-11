from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import generate_pixal3d
import pixalmlx.cgal_holes as cgal
import resume_pixal3d
from pixalmlx.server.app import JobOptions
from pixalmlx.server.worker import generation_arguments


@pytest.mark.parametrize("backend,fallback", [("cgal", "skip"), ("cgal-fan", "fan")])
def test_cleanup_dispatches_native_fallback_policy(monkeypatch, backend, fallback):
    from pixalmlx.mesh_cleanup import cleanup_mesh
    from scripts.benchmark_hole_filling import annulus_fixture

    calls = []

    def native(vertices, faces, **options):
        calls.append(options)
        return vertices, faces

    monkeypatch.setattr(cgal, "fill_small_holes_cgal", native)
    vertices, faces = annulus_fixture("plane", sides=8, rings=3, inner=0.01, outer=0.05)
    cleanup_mesh(vertices, faces, do_fix_normals=False, hole_fill_backend=backend, verbose=False)
    assert len(calls) == 1
    assert calls[0]["fallback"] == fallback


@pytest.mark.parametrize("module,arguments,entry", [
    (generate_pixal3d, ["--image", "unused.png"], "_generate"),
    (resume_pixal3d, ["--image", "unused.png", "--ckpt", "unused"], "_resume"),
])
@pytest.mark.parametrize("backend", [None, "fan", "cgal"])
def test_generation_and_resume_default_hybrid_with_explicit_overrides(monkeypatch, module, arguments, entry, backend):
    native_checks = []
    monkeypatch.setattr(cgal, "cgal_executable", lambda: native_checks.append(True) or Path("native"))
    monkeypatch.setattr(module, entry, lambda args, runtime: args.hole_fill_backend)
    runtime = SimpleNamespace(job=lambda *args, **kwargs: nullcontext())
    if backend is not None:
        arguments = [*arguments, "--hole-fill-backend", backend]
    assert module.main(arguments, runtime=runtime) == (backend or "cgal-fan")
    assert bool(native_checks) == (backend != "fan")


@pytest.mark.parametrize("module,arguments,entry", [
    (generate_pixal3d, ["--image", "unused.png"], "_generate"),
    (resume_pixal3d, ["--image", "unused.png", "--ckpt", "unused"], "_resume"),
])
def test_missing_native_helper_fails_before_model_loading(monkeypatch, capsys, module, arguments, entry):
    def missing():
        raise RuntimeError("CGAL helper missing")

    def reject_loading(*args, **kwargs):
        raise AssertionError("Model loading must not start")

    monkeypatch.setattr(cgal, "cgal_executable", missing)
    monkeypatch.setattr(module, entry, reject_loading)
    with pytest.raises(SystemExit) as stopped:
        module.main(arguments)
    assert stopped.value.code == 2
    assert "--hole-fill-backend fan" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [["--no-cleanup"], ["--ss-only"]])
def test_unused_hole_filler_does_not_require_native_helper(monkeypatch, arguments):
    def reject_check():
        raise AssertionError("This route does not fill holes")

    monkeypatch.setattr(cgal, "cgal_executable", reject_check)
    monkeypatch.setattr(generate_pixal3d, "_generate", lambda args, runtime: args.hole_fill_backend)
    runtime = SimpleNamespace(job=lambda *args, **kwargs: nullcontext())
    assert generate_pixal3d.main(["--image", "unused.png", *arguments], runtime=runtime) == "cgal-fan"


@pytest.mark.parametrize("backend", [None, "fan", "cgal-fan", "cgal"])
def test_resume_redirect_forwards_hole_backend(monkeypatch, backend):
    captured = []
    monkeypatch.setattr(resume_pixal3d, "main", lambda arguments, runtime=None: captured.extend(arguments))
    arguments = ["--image", "unused.png", "--resume", "checkpoint"]
    if backend:
        arguments.extend(["--hole-fill-backend", backend])
    generate_pixal3d.main(arguments)
    assert captured[captured.index("--hole-fill-backend") + 1] == (backend or "cgal-fan")


def test_all_local_cleanup_passes_receive_selected_backend(monkeypatch):
    import generate
    import pixalmlx.mesh_cleanup as cleanup

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.zeros((20, 3), dtype=np.int64)
    calls = []

    def clean(vertices, faces, **options):
        calls.append(options)
        return vertices, faces

    monkeypatch.setattr(cgal, "cgal_executable", lambda: Path("native"))
    monkeypatch.setattr(cleanup, "cleanup_mesh", clean)
    generate._cleanup_and_simplify_mesh(vertices, faces, target_faces=2, no_cleanup=False,
                                      hole_fill_backend="cgal-fan", simplify=lambda vertices, faces, **kwargs: (vertices, faces[:len(faces) // 2]),
                                      log=lambda *args, **kwargs: None)
    assert len(calls) == 3
    assert all(options["hole_fill_backend"] == "cgal-fan" for options in calls)
    assert "min_component_area" not in calls[0]
    assert all(options["min_component_area"] == 0 for options in calls[1:])


def test_server_jobs_record_hybrid_default_and_legacy_jobs_adopt_it(tmp_path):
    options = JobOptions().model_dump()
    assert options["hole_fill_backend"] == "cgal-fan"
    for backend in ("cgal-fan", "fan", "cgal", None):
        if backend is None:
            options.pop("hole_fill_backend")
        else:
            options["hole_fill_backend"] = backend
        arguments = generation_arguments({"options": options}, tmp_path)
        assert arguments[arguments.index("--hole-fill-backend") + 1] == (backend or "cgal-fan")


@pytest.mark.parametrize("backend", ["unknown", None, 1, True])
def test_server_rejects_invalid_hole_backend(backend):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        JobOptions(hole_fill_backend=backend)


def test_reexport_default_is_forwarded_to_export(tmp_path, monkeypatch):
    import reexport_pixal3d

    observed = []
    monkeypatch.setattr(cgal, "cgal_executable", lambda: Path("native"))
    monkeypatch.setattr(reexport_pixal3d, "reexport", lambda args: observed.append(args.hole_fill_backend))
    reexport_pixal3d.main(["--ckpt", str(tmp_path), "--raw-mesh", str(tmp_path / "raw.npz"),
                          "--target-faces", "20000", "--output", str(tmp_path / "model.glb"),
                          "--report", str(tmp_path / "report.json")])
    assert observed == ["cgal-fan"]


@pytest.mark.parametrize("backend", [None, "fan"])
def test_legacy_generation_uses_the_same_cli_default(monkeypatch, backend):
    import sys

    import generate

    selected = []

    class StopBeforeModels(Exception):
        pass

    def stop(*args):
        raise StopBeforeModels

    monkeypatch.setattr(generate, "validate_hole_fill_backend", selected.append)
    monkeypatch.setattr(generate.mx.random, "seed", stop)
    arguments = ["generate.py", "--image", "unused.png"]
    if backend:
        arguments.extend(["--hole-fill-backend", backend])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(StopBeforeModels):
        generate.main()
    assert selected == [backend or "cgal-fan"]


def test_reexport_without_hole_filling_does_not_require_native_helper(tmp_path, monkeypatch):
    import reexport_pixal3d

    def reject_check():
        raise AssertionError("Strict-cap-only re-export does not fill holes")

    monkeypatch.setattr(cgal, "cgal_executable", reject_check)
    monkeypatch.setattr(reexport_pixal3d, "reexport", lambda args: args.hole_fill_backend)
    result = reexport_pixal3d.main(["--ckpt", str(tmp_path), "--max-faces", "5000",
                                   "--output", str(tmp_path / "model.glb"), "--report", str(tmp_path / "report.json")])
    assert result == "cgal-fan"
