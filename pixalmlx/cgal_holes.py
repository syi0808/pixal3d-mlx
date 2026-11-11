"""Optional native CGAL hole patches with immutable original surface arrays."""

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time

import numpy as np


DEFAULT_HOLE_FILL_BACKEND = "cgal-fan"
HOLE_FILL_BACKENDS = ("cgal-fan", "fan", "cgal")


def add_hole_fill_argument(parser):
    parser.add_argument("--hole-fill-backend", choices=HOLE_FILL_BACKENDS, default=DEFAULT_HOLE_FILL_BACKEND,
                        help="Hole filling: cgal-fan (default), reference fan, or CGAL without fan fallback")


def validate_hole_fill_backend(backend):
    if backend not in HOLE_FILL_BACKENDS:
        raise ValueError(f"Unknown hole filling backend: {backend}")
    if backend != "fan":
        try:
            cgal_executable()
        except RuntimeError as error:
            raise RuntimeError(f"{error} Use --hole-fill-backend fan for the reference implementation.") from error


def cgal_executable(executable=None):
    path = Path(executable or os.environ.get("PIXALMLX_CGAL_HOLE_FILLER") or
                Path(__file__).resolve().parents[1] / "build/cgal-holes/pixal-cgal-holes")
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError("CGAL hole filler is not built. See native/cgal_holes and "
                           "docs/cgal-hole-filling.md, or set PIXALMLX_CGAL_HOLE_FILLER.")
    return path.resolve()


def fill_small_holes_cgal(vertices, faces, *, max_hole_perimeter=3e-2, density_control_factor=2.0,
                          fairing_continuity=1, max_boundary_vertices=256, max_collar_faces=4096,
                          timeout=300, executable=None, diagnostics=None, fallback="skip", verbose=True):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Expected vertices [V, 3] and triangle faces [F, 3]")
    if vertices.dtype not in (np.dtype("float32"), np.dtype("float64")) or faces.dtype.kind not in "iu":
        raise ValueError("Expected float32/float64 vertices and integer faces")
    if not np.isfinite(vertices).all() or not np.isfinite(max_hole_perimeter):
        raise ValueError("Vertices and perimeter must be finite")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Triangle indices are out of bounds")
    if not np.isfinite(density_control_factor) or not 1 <= density_control_factor <= 4:
        raise ValueError("Density control factor must be between 1 and 4")
    if fairing_continuity not in (0, 1, 2) or not isinstance(fairing_continuity, int):
        raise ValueError("Fairing continuity must be 0, 1, or 2")
    if not isinstance(max_boundary_vertices, int) or not 3 <= max_boundary_vertices <= 1024:
        raise ValueError("Boundary vertex budget must be between 3 and 1024")
    if not isinstance(max_collar_faces, int) or max_collar_faces < 3:
        raise ValueError("Collar face budget must be at least 3")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite")
    if fallback not in ("skip", "fan"):
        raise ValueError("Fallback must be 'skip' or 'fan'")
    if not len(faces) or max_hole_perimeter <= 0:
        if diagnostics is not None:
            diagnostics.update(filled_loops=0, added_vertices=0, added_faces=0, skipped_disabled=True)
        return vertices, faces
    binary = cgal_executable(executable)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pixal-cgal-holes-") as directory:
        source, destination = Path(directory) / "input.bin", Path(directory) / "output.bin"
        with source.open("wb") as stream:
            stream.write(struct.pack("<8sQQddQQQ", b"PXHIN001", len(vertices), len(faces),
                                     max_hole_perimeter, density_control_factor, fairing_continuity,
                                     max_boundary_vertices, max_collar_faces))
            np.asarray(vertices, dtype="<f8").tofile(stream)
            np.asarray(faces, dtype="<u8").tofile(stream)
        process = subprocess.run([str(binary), str(source), str(destination)], capture_output=True,
                                 text=True, timeout=timeout, check=False)
        if process.returncode:
            raise RuntimeError(f"CGAL hole filler failed: {process.stderr.strip()[:2000]}")
        report = json.loads(process.stdout)
        with destination.open("rb") as stream:
            header = stream.read(24)
            if len(header) != 24:
                raise RuntimeError("Truncated CGAL output header")
            magic, vertex_count, face_count = struct.unpack("<8sQQ", header)
            if magic != b"PXHOUT01" or destination.stat().st_size != 24 + 24 * (vertex_count + face_count):
                raise RuntimeError("Invalid CGAL output protocol or payload size")
            added_vertices = np.fromfile(stream, dtype="<f8", count=vertex_count * 3).reshape(-1, 3)
            added_faces = np.fromfile(stream, dtype="<u8", count=face_count * 3).reshape(-1, 3)
    if not np.isfinite(added_vertices).all():
        raise RuntimeError("CGAL produced nonfinite vertices")
    if added_faces.size and added_faces.max() >= len(vertices) + len(added_vertices):
        raise RuntimeError("CGAL produced out-of-bounds triangle indices")
    if added_faces.size and added_faces.max() > np.iinfo(faces.dtype).max:
        raise RuntimeError("CGAL output does not fit the input index dtype")
    cast_vertices = added_vertices.astype(vertices.dtype)
    if not np.isfinite(cast_vertices).all():
        raise RuntimeError("CGAL vertices overflow the input dtype")
    output_vertices = np.concatenate((vertices, cast_vertices))
    output_faces = np.concatenate((faces, added_faces.astype(faces.dtype)), dtype=faces.dtype)
    if len(added_faces):
        triangles = output_vertices[added_faces.astype(np.int64)].astype(np.float64)
        crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        if not np.isfinite(crosses).all() or np.any(np.all(crosses == 0, axis=1)):
            raise RuntimeError("CGAL triangles degenerate or overflow at the input vertex precision")
    report["cgal_filled_loops"] = report["filled_loops"]
    report["fallback_filled_loops"] = 0
    if fallback == "fan":
        from .mesh_cleanup import fill_small_holes

        fallback_started = time.perf_counter()
        before_vertices, before_faces = len(output_vertices), len(output_faces)
        output_vertices, output_faces = fill_small_holes(
            output_vertices, output_faces, max_hole_perimeter=max_hole_perimeter, verbose=False)
        report.update(fallback_seconds=time.perf_counter() - fallback_started,
                      fallback_filled_loops=len(output_vertices) - before_vertices,
                      fallback_added_faces=len(output_faces) - before_faces)
        report["filled_loops"] += report["fallback_filled_loops"]
    report["added_vertices"] = len(output_vertices) - len(vertices)
    report["added_faces"] = len(output_faces) - len(faces)
    if diagnostics is not None:
        diagnostics.update(report, bridge_seconds=time.perf_counter() - started,
                           original_vertices_preserved=output_vertices[:len(vertices)].tobytes() == vertices.tobytes(),
                           original_faces_preserved=output_faces[:len(faces)].tobytes() == faces.tobytes())
    if verbose:
        print(f"  CGAL filled {report['cgal_filled_loops']} holes; fan fallback {report['fallback_filled_loops']}; "
              f"added {report['added_vertices']:,} vertices / {report['added_faces']:,} faces; "
              f"CGAL rejected {report['rejected']}", flush=True)
    return output_vertices, output_faces
