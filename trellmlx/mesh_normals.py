"""Angle-weighted corner normals with smoothing across UV-split copies."""

import math

import numpy as np
from scipy.spatial import cKDTree


def smooth_corner_normals(vertices, faces, crease_degrees=75.0):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Expected vertices [V, 3] and triangles [F, 3]")
    if not len(faces) or not np.isfinite(vertices).all():
        raise ValueError("Mesh must contain finite, nonempty geometry")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Triangle vertex indices are out of bounds")
    if not 0 < crease_degrees < 180:
        raise ValueError("Crease angle must be between zero and 180 degrees")
    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(crosses, axis=1)
    valid = lengths > np.finfo(np.float64).eps * max(float(np.ptp(vertices, axis=0).max()) ** 2, 1e-30)
    if not valid.any():
        raise ValueError("Cannot derive normals from entirely degenerate geometry")
    face_normals = np.zeros_like(crosses)
    face_normals[valid] = crosses[valid] / lengths[valid, None]
    if not valid.all():
        centers = triangles.mean(axis=1)
        nearest = cKDTree(centers[valid]).query(centers[~valid], k=1)[1]
        face_normals[~valid] = face_normals[valid][nearest]
    angles = np.empty((len(faces), 3), dtype=np.float64)
    for corner in range(3):
        first = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        second = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        angles[:, corner] = np.arctan2(np.linalg.norm(np.cross(first, second), axis=1), np.einsum("ij,ij->i", first, second))
    _, geometric_ids = np.unique(vertices, axis=0, return_inverse=True)
    geometric_faces = geometric_ids[faces]
    parent = np.arange(len(faces) * 3, dtype=np.int64)

    def find(corner):
        while parent[corner] != corner:
            parent[corner] = parent[parent[corner]]
            corner = int(parent[corner])
        return corner

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    edges = {}
    for face_index, face in enumerate(geometric_faces):
        if not valid[face_index]:
            continue
        for local in range(3):
            following = (local + 1) % 3
            first, second = int(face[local]), int(face[following])
            key = (min(first, second), max(first, second))
            edge_corners = (face_index * 3 + local, face_index * 3 + following)
            edges.setdefault(key, []).append((face_index, first, second, edge_corners))
    threshold = math.cos(math.radians(crease_degrees))
    smooth_edges = sharp_edges = nonmanifold_edges = 0
    for entries in edges.values():
        if len(entries) > 2:
            nonmanifold_edges += 1
            continue
        if len(entries) != 2:
            continue
        first_face, first_start, first_end, first_corners = entries[0]
        second_face, second_start, second_end, second_corners = entries[1]
        if first_face == second_face or first_start != second_end or first_end != second_start:
            continue
        if np.dot(face_normals[first_face], face_normals[second_face]) < threshold:
            sharp_edges += 1
            continue
        union(first_corners[0], second_corners[1])
        union(first_corners[1], second_corners[0])
        smooth_edges += 1
    roots = np.array([find(corner) for corner in range(len(parent))])
    _, groups = np.unique(roots, return_inverse=True)
    accumulated = np.zeros((groups.max() + 1, 3), dtype=np.float64)
    np.add.at(accumulated, groups, np.repeat(face_normals, 3, axis=0) * angles.reshape(-1, 1))
    normal_lengths = np.linalg.norm(accumulated, axis=1)
    usable = normal_lengths > 1e-12
    accumulated[usable] /= normal_lengths[usable, None]
    normals = accumulated[groups]
    parent_normals = np.repeat(face_normals, 3, axis=0)
    fallback = ~usable[groups] | (np.einsum("ij,ij->i", normals, parent_normals) <= 0)
    normals[fallback] = parent_normals[fallback]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals.reshape(-1, 3, 3).astype(np.float32), {
        "crease_degrees": crease_degrees, "smooth_edges": smooth_edges, "sharp_edges": sharp_edges,
        "nonmanifold_edges_not_joined": nonmanifold_edges, "degenerate_triangles": int((~valid).sum()),
        "fallback_corners": int(fallback.sum()), "smoothing_groups": int(groups.max() + 1),
    }
