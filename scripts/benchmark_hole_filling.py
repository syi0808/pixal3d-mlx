"""Compare unchanged fan filling and fixed-boundary CGAL patches on frozen meshes."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import statistics
import subprocess
import time

import numpy as np

from pixalmlx.cgal_holes import cgal_executable, fill_small_holes_cgal
from pixalmlx.mesh_cleanup import fill_small_holes


def annulus_fixture(surface, sides=48, rings=8, inner=0.45, outer=0.9):
    angles = np.arange(sides) * (2 * np.pi / sides)
    radii = np.linspace(inner, outer, rings)
    horizontal = np.array([[radius * np.cos(angle), radius * np.sin(angle)]
                           for radius in radii for angle in angles])
    vertices = np.column_stack((horizontal, surface_height(surface, horizontal)))
    faces = []
    for ring in range(rings - 1):
        for side in range(sides):
            following = (side + 1) % sides
            inside, outside = ring * sides, (ring + 1) * sides
            faces.extend(((inside + side, outside + side, outside + following),
                          (inside + side, outside + following, inside + following)))
    return vertices, np.asarray(faces, dtype=np.int64)


def surface_height(surface, horizontal):
    horizontal = np.asarray(horizontal)
    if surface == "sphere":
        return np.sqrt(np.maximum(0, 1 - np.sum(horizontal ** 2, axis=-1)))
    if surface == "saddle":
        return 0.6 * (horizontal[..., 0] ** 2 - horizontal[..., 1] ** 2)
    if surface == "plane":
        return np.zeros(horizontal.shape[:-1])
    raise ValueError(f"Unknown analytic surface: {surface}")


def analytic_patch_error(vertices, faces, original_face_count, surface):
    from scipy.spatial import ConvexHull

    triangles = vertices[faces[original_face_count:]].astype(np.float64)
    if not len(triangles):
        return {"samples": 0}
    original = vertices[np.unique(faces[:original_face_count])]
    radii = np.linalg.norm(original[:, :2], axis=1)
    boundary = original[np.isclose(radii, radii.min()), :2]
    hull = ConvexHull(boundary)
    horizontal = np.linspace(boundary.min(axis=0), boundary.max(axis=0), 101)
    grid = np.stack(np.meshgrid(horizontal[:, 0], horizontal[:, 1]), axis=-1).reshape(-1, 2)
    queries = grid[np.all(grid @ hull.equations[:, :2].T + hull.equations[:, 2] < -1e-12, axis=1)]
    first = triangles[:, 1, :2] - triangles[:, 0, :2]
    second = triangles[:, 2, :2] - triangles[:, 0, :2]
    determinants = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    usable = np.abs(determinants) > 1e-15
    triangles, first, second, determinants = triangles[usable], first[usable], second[usable], determinants[usable]
    errors = []
    ambiguous = 0
    for start in range(0, len(queries), 256):
        current = queries[start:start + 256]
        offsets = current[:, None] - triangles[None, :, 0, :2]
        weights_first = (offsets[..., 0] * second[:, 1] - offsets[..., 1] * second[:, 0]) / determinants
        weights_second = (first[:, 0] * offsets[..., 1] - first[:, 1] * offsets[..., 0]) / determinants
        inside = (weights_first >= -1e-10) & (weights_second >= -1e-10) & (weights_first + weights_second <= 1 + 1e-10)
        strict = (weights_first > 1e-8) & (weights_second > 1e-8) & (weights_first + weights_second < 1 - 1e-8)
        ambiguous += int(np.sum(strict.sum(axis=1) > 1))
        covered = inside.any(axis=1)
        owners = np.argmax(inside, axis=1)
        rows = np.arange(len(current))
        predicted = (triangles[owners, 0, 2] + weights_first[rows, owners] * (triangles[owners, 1, 2] - triangles[owners, 0, 2])
                     + weights_second[rows, owners] * (triangles[owners, 2, 2] - triangles[owners, 0, 2]))
        errors.extend(predicted[covered] - surface_height(surface, current[covered]))
    errors = np.asarray(errors)
    return {"samples": int(errors.size), "requested_samples": len(queries), "ambiguous_projected_samples": ambiguous,
            "height_rmse": float(np.sqrt(np.mean(errors ** 2))),
            "height_max_absolute_error": float(np.max(np.abs(errors))),
            "note": "Identical fixed XY grid inside the original convex hole boundary; actual patch triangle interpolation"}


def edge_groups(faces):
    pairs = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2).astype(np.uint64)
    keys = (np.minimum(pairs[:, 0], pairs[:, 1]) << 32) | np.maximum(pairs[:, 0], pairs[:, 1])
    order = np.argsort(keys)
    ordered = keys[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1] if len(ordered) else np.empty(0, dtype=np.int64)
    counts = np.diff(np.r_[starts, len(ordered)])
    return ordered[starts], order[starts] // 3, counts


def triangle_normals(vertices, faces):
    triangles = vertices[faces].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    return normals / np.maximum(lengths[:, None], np.finfo(np.float64).tiny), lengths


def seam_angles(original_vertices, original_faces, vertices, faces, boundary_keys, boundary_owners):
    keys, owners, counts = edge_groups(faces[len(original_faces):])
    positions = np.searchsorted(boundary_keys, keys)
    positions = np.minimum(positions, max(0, len(boundary_keys) - 1))
    if not len(boundary_keys):
        return np.empty(0, dtype=np.uint64), np.empty(0)
    valid = (counts == 1) & (boundary_keys[positions] == keys)
    old_normals, old_lengths = triangle_normals(original_vertices, original_faces[boundary_owners[positions[valid]]])
    new_normals, new_lengths = triangle_normals(vertices, faces[len(original_faces) + owners[valid]])
    nondegenerate = (old_lengths > 0) & (new_lengths > 0)
    angles = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", old_normals, new_normals)), 0, 1)))
    return keys[valid][nondegenerate], angles[nondegenerate]


def angle_summary(angles):
    return {"edges": len(angles), "mean_degrees": float(np.mean(angles)) if len(angles) else None,
            "p95_degrees": float(np.percentile(angles, 95)) if len(angles) else None}


def prepare_real(raw_mesh):
    from pixalmlx.mesh_cleanup import remove_duplicate_faces, remove_small_components, repair_non_manifold_edges

    with np.load(raw_mesh) as raw:
        vertices, faces = raw["vertices"], raw["faces"]
    vertices, faces = remove_duplicate_faces(vertices, faces, verbose=False)
    vertices, faces = remove_small_components(vertices, faces, verbose=False)
    return repair_non_manifold_edges(vertices, faces, verbose=False)


def measure_backend(vertices, faces, backend, perimeter, repeats, options):
    function = fill_small_holes if backend == "fan" else fill_small_holes_cgal
    samples = []
    signatures = []
    result = None
    for iteration in range(repeats + 1):
        diagnostics = {}
        arguments = {} if backend == "fan" else {**options, "diagnostics": diagnostics,
                                                  "fallback": "fan" if backend == "cgal-fan" else "skip"}
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.perf_counter()
        cpu_started = time.process_time()
        result = function(vertices, faces, max_hole_perimeter=perimeter, verbose=False, **arguments)
        wall_seconds = time.perf_counter() - started
        python_cpu_seconds = time.process_time() - cpu_started
        child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu_seconds = child_after.ru_utime + child_after.ru_stime - child_usage.ru_utime - child_usage.ru_stime
        timing = {"wall_seconds": wall_seconds,
                  "python_process_cpu_seconds": python_cpu_seconds,
                  "parent_and_completed_children_cpu_seconds": python_cpu_seconds + child_cpu_seconds,
                  "diagnostics": diagnostics}
        if iteration:
            samples.append(timing)
        signatures.append([hashlib.sha256(np.ascontiguousarray(array)).hexdigest() for array in result])
    output_vertices, output_faces = result
    preserved_vertices = output_vertices[:len(vertices)].tobytes() == vertices.tobytes()
    preserved_faces = output_faces[:len(faces)].tobytes() == faces.tobytes()
    if not preserved_vertices or not preserved_faces:
        raise RuntimeError("Hole filling changed the original surface")
    report = {"samples": samples, "median_seconds": statistics.median(sample["wall_seconds"] for sample in samples),
              "added_vertices": len(output_vertices) - len(vertices), "added_faces": len(output_faces) - len(faces),
              "original_vertex_bytes_preserved": preserved_vertices, "original_face_bytes_preserved": preserved_faces,
              "repeated_output_hashes_equal": all(signature == signatures[0] for signature in signatures),
              "output_sha256": signatures[-1]}
    return report, result


def compare_case(name, vertices, faces, perimeter, arguments):
    keys, owners, counts = edge_groups(faces)
    boundary_keys, boundary_owners = keys[counts == 1], owners[counts == 1]
    options = {"density_control_factor": arguments.density, "fairing_continuity": arguments.continuity,
               "executable": arguments.executable, "timeout": arguments.timeout}
    case = {"name": name, "input_vertices": len(vertices), "input_faces": len(faces),
            "max_hole_perimeter": perimeter, "input_boundary_edges": len(boundary_keys),
            "input_nonmanifold_edges": int(np.sum(counts > 2)), "backends": {}}
    results = {}
    seams = {}
    for backend in ("fan", "cgal", "cgal-fan"):
        report, result = measure_backend(vertices, faces, backend, perimeter, arguments.repeats, options)
        output_vertices, output_faces = result
        output_keys, _, output_counts = edge_groups(output_faces)
        report.update(output_boundary_edges=int(np.sum(output_counts == 1)),
                      output_nonmanifold_edges=int(np.sum(output_counts > 2)))
        _, patch_lengths = triangle_normals(output_vertices, output_faces[len(faces):])
        report["degenerate_added_triangles"] = int(np.sum(patch_lengths == 0))
        seams[backend] = seam_angles(vertices, faces, output_vertices, output_faces, boundary_keys, boundary_owners)
        report["all_filled_seam_angles"] = angle_summary(seams[backend][1])
        if name in ("sphere", "saddle", "plane"):
            report["analytic_error"] = analytic_patch_error(output_vertices, output_faces, len(faces), name)
        np.savez_compressed(arguments.output_dir / f"{name}-{backend}.npz", vertices=output_vertices, faces=output_faces)
        case["backends"][backend] = report
        results[backend] = result
        print(json.dumps({"case": name, "backend": backend, "seconds": report["median_seconds"],
                          "added_faces": report["added_faces"], "diagnostics": report["samples"][-1]["diagnostics"]}), flush=True)
    common, fan_indices, cgal_indices = np.intersect1d(seams["fan"][0], seams["cgal"][0], return_indices=True)
    case["same_filled_boundary_edges"] = {"edges": len(common),
        "fan": angle_summary(seams["fan"][1][fan_indices]), "cgal": angle_summary(seams["cgal"][1][cgal_indices]),
        "note": "Orientation-invariant normal angle across identical original boundary edges. Continuity proxy, not ground truth."}
    common, fan_indices, hybrid_indices = np.intersect1d(seams["fan"][0], seams["cgal-fan"][0], return_indices=True)
    case["same_filled_boundary_edges_hybrid"] = {"edges": len(common),
        "fan": angle_summary(seams["fan"][1][fan_indices]),
        "cgal-fan": angle_summary(seams["cgal-fan"][1][hybrid_indices])}
    np.savez_compressed(arguments.output_dir / f"{name}-input.npz", vertices=vertices, faces=faces)
    return case, results


def cross_section(vertices, faces):
    segments = []
    for triangle in vertices[faces]:
        intersections = []
        for corner in range(3):
            first, second = triangle[corner], triangle[(corner + 1) % 3]
            if abs(first[1]) < 1e-12:
                intersections.append(first[[0, 2]])
            if first[1] * second[1] < 0:
                fraction = -first[1] / (second[1] - first[1])
                intersections.append((first + fraction * (second - first))[[0, 2]])
        if len(intersections) >= 2:
            unique = np.unique(np.round(intersections, 12), axis=0)
            if len(unique) >= 2:
                segments.append(unique[[0, -1]])
    return segments


def plot_synthetic(examples, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    colors = {"fan": "#c97831", "cgal": "#2874a6"}
    for row, surface in enumerate(("sphere", "saddle")):
        vertices, faces, results = examples[surface]
        for column, backend in enumerate(("fan", "cgal")):
            axis = figure.add_subplot(2, 3, row * 3 + column + 1, projection="3d")
            output_vertices, output_faces = results[backend]
            axis.add_collection3d(Poly3DCollection(vertices[faces], facecolor="#dddddd", edgecolor="#aaaaaa", linewidth=0.25, alpha=0.5))
            axis.add_collection3d(Poly3DCollection(output_vertices[output_faces[len(faces):]], facecolor=colors[backend],
                                                  edgecolor="#444444", linewidth=0.4, alpha=0.95))
            axis.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(0.35, 1.1) if surface == "sphere" else (-0.6, 0.6),
                     title=f"{surface.title()} — {backend.upper()}", xlabel="x", ylabel="y", zlabel="z")
            axis.view_init(elev=25, azim=-65)
            axis.set_box_aspect((1, 1, 0.7))
        axis = figure.add_subplot(2, 3, row * 3 + 3)
        positions = np.linspace(-0.9, 0.9, 400)
        axis.plot(positions, surface_height(surface, np.column_stack((positions, np.zeros_like(positions)))),
                  color="#333333", linestyle="--", label="Known surface", linewidth=2)
        axis.add_collection(LineCollection(cross_section(vertices, faces), colors="#999999", linewidths=4))
        for backend in ("fan", "cgal"):
            output_vertices, output_faces = results[backend]
            axis.add_collection(LineCollection(cross_section(output_vertices, output_faces[len(faces):]),
                                               colors=colors[backend], linewidths=2.5, label=backend.upper()))
        axis.autoscale()
        axis.set(title=f"{surface.title()} cross section (y = 0)", xlabel="x", ylabel="z")
        axis.legend()
        axis.grid(alpha=0.2)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main():
    from scripts.investigate_python_overhead import other_inference_processes, source_snapshot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--continuity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--raw-mesh", type=Path, default=Path("outputs/runtime/postprocess-investigation/raw-mesh.npz"))
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--plot", action="store_true")
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.output_dir.exists():
        parser.error("Use positive repeats and a fresh output directory")
    if other_inference_processes():
        parser.error("Another inference/benchmark process is active")
    binary = cgal_executable(arguments.executable)
    arguments.output_dir.mkdir(parents=True)
    sources = source_snapshot()
    native_sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in Path("native/cgal_holes").glob("*") if path.is_file()}
    report = {"schema": "pixalmlx.hole_fill_comparison.v1", "repeats": arguments.repeats,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "working_tree_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
              "python_source_sha256": sources, "native_source_sha256": native_sources,
              "executable_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "options": {"density": arguments.density, "continuity": arguments.continuity}, "cases": [],
              "scope": "Hole filling only. One warm-up per backend followed by serial repeats. Native bridge I/O and validation included. "
                       "Not full generation, not a like-for-like algorithm speedup; CGAL has conservative rejection policies. "
                       "Original triangle/vertex prefixes preserved; global inter-patch self-intersections are not certified."}
    examples = {}
    for surface in ("sphere", "saddle", "plane"):
        vertices, faces = annulus_fixture(surface)
        case, results = compare_case(surface, vertices, faces, 4.0, arguments)
        report["cases"].append(case)
        examples[surface] = (vertices, faces, results)
        (arguments.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not arguments.skip_real:
        vertices, faces = prepare_real(arguments.raw_mesh)
        report["raw_mesh_sha256"] = hashlib.sha256(arguments.raw_mesh.read_bytes()).hexdigest()
        case, _ = compare_case("turnip-prefill", vertices, faces, 0.03, arguments)
        report["cases"].append(case)
    if arguments.plot:
        plot_synthetic(examples, arguments.output_dir / "synthetic-comparison.png")
    after = source_snapshot()
    report["changed_python_paths_during_run"] = sorted(path for path in sources.keys() | after.keys()
                                                       if sources.get(path) != after.get(path))
    report["other_inference_processes_after"] = other_inference_processes()
    (arguments.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(arguments.output_dir / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
