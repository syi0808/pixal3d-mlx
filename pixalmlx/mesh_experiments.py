"""Scoped CPU topology backend selectors for reference performance comparisons."""

from contextlib import contextmanager

from .mesh_topology import grouped_face_components, sparse_corner_repair


@contextmanager
def reference_cleanup():
    from . import mesh_cleanup

    original_components = mesh_cleanup._face_connected_components
    original_repair = mesh_cleanup.repair_non_manifold_edges
    mesh_cleanup._face_connected_components = mesh_cleanup._face_connected_components_reference
    mesh_cleanup.repair_non_manifold_edges = mesh_cleanup._repair_non_manifold_edges_reference
    try:
        yield
    finally:
        mesh_cleanup._face_connected_components = original_components
        mesh_cleanup.repair_non_manifold_edges = original_repair


@contextmanager
def experimental_cleanup(enabled=True):
    from . import mesh_cleanup

    original_components = mesh_cleanup._face_connected_components
    original_repair = mesh_cleanup.repair_non_manifold_edges
    if enabled:
        mesh_cleanup._face_connected_components = grouped_face_components
        mesh_cleanup.repair_non_manifold_edges = sparse_corner_repair
    try:
        yield
    finally:
        mesh_cleanup._face_connected_components = original_components
        mesh_cleanup.repair_non_manifold_edges = original_repair
