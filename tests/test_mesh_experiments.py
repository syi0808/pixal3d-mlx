import numpy as np
import pytest

from pixalmlx import mesh_cleanup
from pixalmlx.mesh_experiments import experimental_cleanup, grouped_face_components, reference_cleanup, sparse_corner_repair


@pytest.mark.parametrize("seed", range(12))
def test_experimental_topology_preserves_component_labels_and_corner_repair(seed):
    random = np.random.default_rng(seed)
    vertices = random.normal(size=(24, 3)).astype(np.float32)
    faces = random.integers(0, len(vertices), size=(30, 3), dtype=np.int32)
    expected_count, expected_labels = mesh_cleanup._face_connected_components_reference(faces, len(faces))
    count, labels = grouped_face_components(faces, len(faces))
    assert count == expected_count
    np.testing.assert_array_equal(labels, expected_labels)
    expected_vertices, expected_faces = mesh_cleanup._repair_non_manifold_edges_reference(vertices, faces, verbose=False)
    actual_vertices, actual_faces = sparse_corner_repair(vertices, faces, verbose=False)
    np.testing.assert_array_equal(actual_vertices, expected_vertices)
    np.testing.assert_array_equal(actual_faces, expected_faces)


def test_topology_experiment_handles_empty_and_disconnected_triangles():
    vertices = np.zeros((6, 3), dtype=np.float64)
    empty = np.empty((0, 3), dtype=np.int64)
    count, labels = grouped_face_components(empty, 0)
    assert count == labels.size == 0
    actual_vertices, actual_faces = sparse_corner_repair(vertices, empty)
    assert actual_vertices is vertices and actual_faces is empty
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    count, labels = grouped_face_components(faces, 2)
    assert count == 2
    np.testing.assert_array_equal(labels, [0, 1])


def test_topology_experiment_restores_functions_on_failure():
    original_components = mesh_cleanup._face_connected_components
    original_repair = mesh_cleanup.repair_non_manifold_edges
    with pytest.raises(RuntimeError):
        with experimental_cleanup():
            assert mesh_cleanup._face_connected_components is grouped_face_components
            assert mesh_cleanup.repair_non_manifold_edges is sparse_corner_repair
            raise RuntimeError("interrupted")
    assert mesh_cleanup._face_connected_components is original_components
    assert mesh_cleanup.repair_non_manifold_edges is original_repair


def test_production_uses_vectorized_topology_and_reference_scope_restores_it():
    assert mesh_cleanup._face_connected_components is grouped_face_components
    assert mesh_cleanup.repair_non_manifold_edges is sparse_corner_repair
    with pytest.raises(RuntimeError):
        with reference_cleanup():
            assert mesh_cleanup._face_connected_components is mesh_cleanup._face_connected_components_reference
            assert mesh_cleanup.repair_non_manifold_edges is mesh_cleanup._repair_non_manifold_edges_reference
            raise RuntimeError("interrupted")
    assert mesh_cleanup._face_connected_components is grouped_face_components
    assert mesh_cleanup.repair_non_manifold_edges is sparse_corner_repair
