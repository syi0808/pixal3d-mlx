"""Strict triangle budgets applied after face-adding mesh repairs."""

import numpy as np


def enforce_face_limit(vertices, faces, max_faces, simplify=None, diagnostics=None):
    """Cap normalized Pixal3D geometry, failing rather than dropping arbitrary faces."""
    if not isinstance(max_faces, int) or isinstance(max_faces, bool) or max_faces < 1:
        raise ValueError("Maximum face count must be a positive integer")
    if len(faces) <= max_faces:
        return vertices, faces
    source_vertices, source_faces = vertices, faces
    if simplify is None:
        import fast_simplification
        simplify = fast_simplification.simplify
    for attempt in range(3):
        previous_count = len(faces)
        vertices, faces = simplify(vertices, faces, target_count=max_faces)
        if not len(faces) or not np.isfinite(vertices).all():
            raise RuntimeError("Face limiting produced an empty or nonfinite mesh")
        if len(faces) <= max_faces:
            from .mesh_cleanup import orient_faces_by_adjacency
            return orient_faces_by_adjacency(vertices, faces, verbose=False)
        if len(faces) >= previous_count:
            break
    import trimesh
    print("Strict triangle budget: welding normalized positions at 7 decimals before retry", flush=True)
    mesh = trimesh.Trimesh(vertices=np.round(source_vertices, 7), faces=source_faces, process=False)
    mesh.merge_vertices(digits_vertex=7)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if diagnostics is not None:
        diagnostics["fallback_position_decimals"] = 7
        diagnostics["fallback_input_faces"] = len(mesh.faces)
    vertices, faces = simplify(np.array(mesh.vertices), np.array(mesh.faces), target_count=max_faces)
    if not len(faces) or not np.isfinite(vertices).all():
        raise RuntimeError("Face limiting produced an empty or nonfinite mesh")
    if len(faces) <= max_faces:
        from .mesh_cleanup import orient_faces_by_adjacency
        return orient_faces_by_adjacency(vertices, faces, verbose=False)
    raise RuntimeError(f"Simplifier could not meet the strict triangle limit: {len(faces)} > {max_faces}")
