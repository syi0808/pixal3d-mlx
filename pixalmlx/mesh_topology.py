"""Vectorized CPU topology with canonical face-corner representatives."""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def _group_edges(faces):
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size and (faces.min() < 0 or faces.max() >= 2 ** 31):
        raise ValueError("Edge indices must fit nonnegative int31")
    pairs = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    lower = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.uint64)
    upper = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.uint64)
    keys = (lower << 32) | upper
    order = np.argsort(keys)
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(keys)])
    return order, starts, counts


def _components(count, rows, columns):
    if not len(rows):
        return count, np.arange(count, dtype=np.int64)
    adjacency = sparse.csr_matrix((np.ones(len(rows) * 2, dtype=np.int8),
                                   (np.r_[rows, columns], np.r_[columns, rows])), shape=(count, count))
    return connected_components(adjacency, directed=False)


def grouped_face_components(faces, face_count):
    if face_count != len(faces):
        raise ValueError("Face count must match the supplied triangle array")
    if not face_count:
        return 0, np.empty(0, dtype=np.int64)
    order, starts, counts = _group_edges(faces)
    owners = order // 3
    anchors = np.repeat(owners[starts], counts)
    nonfirst = np.ones(len(order), dtype=bool)
    nonfirst[starts] = False
    return _components(face_count, anchors[nonfirst], owners[nonfirst])


def sparse_corner_repair(vertices, faces, verbose=True):
    if not len(faces):
        return vertices, faces
    order, starts, counts = _group_edges(faces)
    if not np.any(counts > 2):
        return vertices, faces
    paired_starts = starts[counts == 2]

    def corners(occurrences):
        owners = occurrences // 3
        first = occurrences % 3
        second = (first + 1) % 3
        forward = faces[owners, first] <= faces[owners, second]
        return owners * 3 + np.where(forward, first, second), owners * 3 + np.where(forward, second, first)

    first_left, second_left = corners(order[paired_starts])
    first_right, second_right = corners(order[paired_starts + 1])
    corner_count = len(faces) * 3
    component_count, labels = _components(corner_count, np.r_[first_left, second_left], np.r_[first_right, second_right])
    representatives = np.full(component_count, corner_count, dtype=np.int64)
    np.minimum.at(representatives, labels, np.arange(corner_count))
    component_order = np.argsort(representatives)
    ranks = np.empty(component_count, dtype=np.int64)
    ranks[component_order] = np.arange(component_count)
    new_vertices = vertices[faces.reshape(-1)[representatives[component_order]]]
    new_faces = ranks[labels].reshape(-1, 3).astype(faces.dtype, copy=False)
    if verbose:
        print(f"  Split {len(new_vertices) - len(vertices)} vertices across {int(np.sum(counts > 2))} non-manifold edges", flush=True)
    return new_vertices.astype(vertices.dtype, copy=False), new_faces
