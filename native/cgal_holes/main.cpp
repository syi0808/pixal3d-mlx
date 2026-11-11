#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/version.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;
using Vertex = Mesh::Vertex_index;
using Face = Mesh::Face_index;
using Triangle = std::array<std::uint32_t, 3>;
using OutputTriangle = std::array<std::uint64_t, 3>;
using Clock = std::chrono::steady_clock;

struct Input {
    std::vector<Point> points;
    std::vector<Triangle> faces;
    double perimeter;
    double density;
    unsigned continuity;
    std::size_t max_edges;
    std::size_t max_collar_faces;
};

struct Statistics {
    std::uint64_t boundary_edges = 0;
    std::uint64_t closed_loops = 0;
    std::uint64_t eligible_loops = 0;
    std::uint64_t filled_loops = 0;
    std::uint64_t refined_loops = 0;
    std::map<std::string, std::uint64_t> rejected;
    double topology_seconds = 0;
    double refinement_seconds = 0;
    double fairing_seconds = 0;
    double validation_seconds = 0;
};

struct Patch {
    std::vector<Point> points;
    std::vector<OutputTriangle> faces;
};

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::uint64_t edge_key(std::uint32_t first, std::uint32_t second) {
    return (std::uint64_t(std::min(first, second)) << 32) | std::max(first, second);
}

template<class Value>
Value read_value(std::istream& source) {
    Value value;
    if (!source.read(reinterpret_cast<char*>(&value), sizeof(value)))
        throw std::runtime_error("Truncated input");
    return value;
}

template<class Value>
void write_value(std::ostream& destination, const Value& value) {
    destination.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

Input read_input(const std::string& path) {
    if (std::endian::native != std::endian::little)
        throw std::runtime_error("The bridge requires a little-endian host");
    std::ifstream source(path, std::ios::binary);
    std::array<char, 8> magic{};
    if (!source.read(magic.data(), magic.size()))
        throw std::runtime_error("Truncated input header");
    if (std::string(magic.data(), magic.size()) != "PXHIN001")
        throw std::runtime_error("Invalid input protocol");
    const auto vertex_count = read_value<std::uint64_t>(source);
    const auto face_count = read_value<std::uint64_t>(source);
    Input input;
    input.perimeter = read_value<double>(source);
    input.density = read_value<double>(source);
    const auto continuity = read_value<std::uint64_t>(source);
    if (continuity > 2)
        throw std::runtime_error("Invalid fairing continuity");
    input.continuity = unsigned(continuity);
    input.max_edges = read_value<std::uint64_t>(source);
    input.max_collar_faces = read_value<std::uint64_t>(source);
    if (vertex_count >= std::numeric_limits<std::uint32_t>::max() ||
        face_count >= std::numeric_limits<std::uint32_t>::max() ||
        !std::isfinite(input.perimeter) || input.perimeter < 0 ||
        !std::isfinite(input.density) || input.density < 1 || input.density > 4 ||
        input.continuity > 2 || input.max_edges < 3 || input.max_edges > 1024 ||
        input.max_collar_faces < 3)
        throw std::runtime_error("Invalid input dimensions or options");
    const auto payload_start = source.tellg();
    source.seekg(0, std::ios::end);
    const auto expected = (vertex_count + face_count) * 3 * sizeof(double);
    if (std::uint64_t(source.tellg() - payload_start) != expected)
        throw std::runtime_error("Input payload size mismatch");
    source.seekg(payload_start);
    input.points.reserve(vertex_count);
    for (std::uint64_t index = 0; index < vertex_count; ++index) {
        std::array<double, 3> point;
        for (auto& coordinate : point) {
            coordinate = read_value<double>(source);
            if (!std::isfinite(coordinate))
                throw std::runtime_error("Nonfinite input vertex");
        }
        input.points.emplace_back(point[0], point[1], point[2]);
    }
    input.faces.resize(face_count);
    for (auto& face : input.faces) {
        for (auto& vertex : face) {
            const auto identifier = read_value<std::uint64_t>(source);
            if (identifier >= vertex_count)
                throw std::runtime_error("Input triangle index out of bounds");
            vertex = std::uint32_t(identifier);
        }
    }
    return input;
}

bool orient_collar(std::vector<Triangle>& faces) {
    std::unordered_map<std::uint64_t, std::vector<std::pair<std::size_t, bool>>> edges;
    for (std::size_t face_index = 0; face_index < faces.size(); ++face_index) {
        const auto& face = faces[face_index];
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const auto first = face[corner], second = face[(corner + 1) % 3];
            if (first == second)
                return false;
            edges[edge_key(first, second)].push_back({face_index, first < second});
        }
    }
    std::vector<std::vector<std::pair<std::size_t, bool>>> adjacency(faces.size());
    for (const auto& [key, uses] : edges) {
        if (uses.size() > 2)
            return false;
        if (uses.size() == 2) {
            const bool different = uses[0].second == uses[1].second;
            adjacency[uses[0].first].push_back({uses[1].first, different});
            adjacency[uses[1].first].push_back({uses[0].first, different});
        }
    }
    std::vector<int> flips(faces.size(), -1);
    for (std::size_t root = 0; root < faces.size(); ++root) {
        if (flips[root] != -1)
            continue;
        flips[root] = 0;
        std::vector<std::size_t> pending{root};
        while (!pending.empty()) {
            const auto current = pending.back();
            pending.pop_back();
            for (const auto& [neighbor, different] : adjacency[current]) {
                const int desired = flips[current] ^ int(different);
                if (flips[neighbor] != -1 && flips[neighbor] != desired)
                    return false;
                if (flips[neighbor] == -1) {
                    flips[neighbor] = desired;
                    pending.push_back(neighbor);
                }
            }
        }
    }
    for (std::size_t index = 0; index < faces.size(); ++index)
        if (flips[index])
            std::swap(faces[index][1], faces[index][2]);
    return PMP::is_polygon_soup_a_polygon_mesh(faces);
}

Patch fill_loop(const Input& input, const std::vector<std::uint32_t>& loop,
                const std::vector<std::vector<std::uint32_t>>& incident_faces,
                std::uint64_t output_offset, Statistics& statistics) {
    std::set<std::uint32_t> selected_faces;
    std::set<std::uint32_t> selected_vertices(loop.begin(), loop.end());
    std::vector<std::uint32_t> frontier(loop);
    for (unsigned ring = 0; ring < input.continuity + 1; ++ring) {
        std::vector<std::uint32_t> next;
        for (auto vertex : frontier) {
            for (auto face : incident_faces[vertex]) {
                selected_faces.insert(face);
                if (selected_faces.size() > input.max_collar_faces)
                    throw std::runtime_error("collar_budget");
                for (auto neighbor : input.faces[face])
                    if (selected_vertices.insert(neighbor).second)
                        next.push_back(neighbor);
            }
        }
        frontier.swap(next);
    }
    std::vector<std::uint32_t> global_ids(selected_vertices.begin(), selected_vertices.end());
    std::unordered_map<std::uint32_t, std::uint32_t> local_ids;
    std::vector<Point> points;
    for (auto global : global_ids) {
        local_ids.emplace(global, std::uint32_t(points.size()));
        points.push_back(input.points[global]);
    }
    std::vector<Triangle> faces;
    for (auto identifier : selected_faces) {
        const auto& original = input.faces[identifier];
        if (CGAL::collinear(input.points[original[0]], input.points[original[1]], input.points[original[2]]))
            throw std::runtime_error("degenerate_collar");
        faces.push_back({local_ids.at(original[0]), local_ids.at(original[1]), local_ids.at(original[2])});
    }
    if (!orient_collar(faces))
        throw std::runtime_error("nonmanifold_collar");
    Mesh mesh;
    std::vector<std::pair<std::size_t, Vertex>> vertex_pairs;
    PMP::polygon_soup_to_polygon_mesh(points, faces, mesh,
        CGAL::parameters::point_to_vertex_output_iterator(std::back_inserter(vertex_pairs)));
    std::vector<Vertex> original_vertices(points.size());
    for (const auto& [index, vertex] : vertex_pairs)
        original_vertices.at(index) = vertex;
    auto border = mesh.halfedge(original_vertices[local_ids.at(loop[0])], original_vertices[local_ids.at(loop[1])]);
    if (border == Mesh::null_halfedge())
        throw std::runtime_error("missing_boundary");
    if (!mesh.is_border(border))
        border = mesh.opposite(border);
    if (!mesh.is_border(border))
        throw std::runtime_error("missing_boundary");
    std::set<std::size_t> expected, actual;
    for (auto global : loop)
        expected.insert(original_vertices[local_ids.at(global)].idx());
    std::size_t border_count = 0;
    for (auto vertex : CGAL::vertices_around_face(border, mesh)) {
        actual.insert(vertex.idx());
        ++border_count;
    }
    if (actual != expected || border_count != loop.size())
        throw std::runtime_error("boundary_mismatch");
    std::vector<Face> patch_faces;
    std::vector<Vertex> patch_vertices;
    auto started = Clock::now();
    PMP::triangulate_and_refine_hole(mesh, border,
        CGAL::parameters::face_output_iterator(std::back_inserter(patch_faces))
                         .vertex_output_iterator(std::back_inserter(patch_vertices))
                         .density_control_factor(input.density)
                         .do_not_use_cubic_algorithm(true));
    statistics.refinement_seconds += seconds(started);
    if (patch_faces.empty())
        throw std::runtime_error("triangulation_failed");
    if (patch_vertices.empty())
        throw std::runtime_error("no_interior_vertices");
    if (patch_vertices.size() > 4096)
        throw std::runtime_error("patch_budget");
    started = Clock::now();
    const bool faired = PMP::fair(mesh, patch_vertices, CGAL::parameters::fairing_continuity(input.continuity));
    statistics.fairing_seconds += seconds(started);
    if (!faired)
        throw std::runtime_error("fairing_failed");
    started = Clock::now();
    for (std::size_t index = 0; index < points.size(); ++index)
        if (mesh.point(original_vertices[index]) != points[index])
            throw std::runtime_error("original_vertex_moved");
    std::vector<std::pair<Face, Face>> intersections;
    PMP::self_intersections(mesh, std::back_inserter(intersections));
    std::unordered_set<std::uint32_t> new_faces;
    for (auto face : patch_faces)
        new_faces.insert(face.idx());
    for (const auto& pair : intersections)
        if (new_faces.contains(pair.first.idx()) || new_faces.contains(pair.second.idx()))
            throw std::runtime_error("patch_intersection");
    Patch patch;
    std::vector<std::uint64_t> mapping(mesh.number_of_vertices(), std::numeric_limits<std::uint64_t>::max());
    for (std::size_t index = 0; index < original_vertices.size(); ++index)
        mapping[original_vertices[index].idx()] = global_ids[index];
    for (auto vertex : mesh.vertices()) {
        if (mapping[vertex.idx()] != std::numeric_limits<std::uint64_t>::max())
            continue;
        const auto& point = mesh.point(vertex);
        if (!std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z()))
            throw std::runtime_error("nonfinite_patch");
        mapping[vertex.idx()] = output_offset + patch.points.size();
        patch.points.push_back(point);
    }
    std::unordered_set<std::uint32_t> boundary(loop.begin(), loop.end());
    std::map<std::pair<std::uint64_t, std::uint64_t>, unsigned> patch_edges;
    for (auto face : patch_faces) {
        OutputTriangle triangle;
        std::size_t corner = 0;
        for (auto vertex : CGAL::vertices_around_face(mesh.halfedge(face), mesh)) {
            if (corner >= 3)
                throw std::runtime_error("nontriangle_patch");
            const auto identifier = mapping[vertex.idx()];
            if (identifier < input.points.size() && !boundary.contains(std::uint32_t(identifier)))
                throw std::runtime_error("patch_outside_boundary");
            triangle[corner++] = identifier;
        }
        if (corner != 3)
            throw std::runtime_error("nontriangle_patch");
        for (std::size_t index = 0; index < 3; ++index)
            ++patch_edges[std::minmax(triangle[index], triangle[(index + 1) % 3])];
        patch.faces.push_back(triangle);
    }
    std::set<std::pair<std::uint64_t, std::uint64_t>> expected_edges, actual_edges;
    for (std::size_t index = 0; index < loop.size(); ++index)
        expected_edges.insert(std::minmax(std::uint64_t(loop[index]), std::uint64_t(loop[(index + 1) % loop.size()])));
    for (const auto& [edge, count] : patch_edges) {
        if (count > 2)
            throw std::runtime_error("nonmanifold_patch");
        if (count == 1)
            actual_edges.insert(edge);
    }
    if (actual_edges != expected_edges)
        throw std::runtime_error("patch_boundary_changed");
    statistics.validation_seconds += seconds(started);
    return patch;
}

Patch process(const Input& input, Statistics& statistics) {
    auto started = Clock::now();
    std::vector<std::uint64_t> edges;
    edges.reserve(input.faces.size() * 3);
    std::vector<std::vector<std::uint32_t>> incident_faces(input.points.size());
    for (std::size_t index = 0; index < input.faces.size(); ++index) {
        const auto& face = input.faces[index];
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const auto first = face[corner], second = face[(corner + 1) % 3];
            edges.push_back(edge_key(first, second));
            incident_faces[first].push_back(std::uint32_t(index));
        }
    }
    std::sort(edges.begin(), edges.end());
    std::map<std::uint32_t, std::vector<std::uint32_t>> adjacency;
    for (std::size_t first = 0; first < edges.size();) {
        std::size_t end = first + 1;
        while (end < edges.size() && edges[end] == edges[first])
            ++end;
        if (end == first + 1) {
            const auto lower = std::uint32_t(edges[first] >> 32);
            const auto upper = std::uint32_t(edges[first]);
            adjacency[lower].push_back(upper);
            adjacency[upper].push_back(lower);
            ++statistics.boundary_edges;
        }
        first = end;
    }
    statistics.topology_seconds = seconds(started);
    std::unordered_set<std::uint32_t> visited;
    Patch output;
    for (const auto& [root, neighbors] : adjacency) {
        if (visited.contains(root))
            continue;
        std::vector<std::uint32_t> pending{root}, component;
        visited.insert(root);
        bool simple = true;
        while (!pending.empty()) {
            const auto current = pending.back();
            pending.pop_back();
            component.push_back(current);
            if (adjacency.at(current).size() != 2)
                simple = false;
            for (auto neighbor : adjacency.at(current))
                if (visited.insert(neighbor).second)
                    pending.push_back(neighbor);
        }
        if (!simple || component.size() < 3) {
            ++statistics.rejected["ambiguous_boundary"];
            continue;
        }
        ++statistics.closed_loops;
        std::vector<std::uint32_t> loop{root};
        auto previous = root;
        auto current = adjacency.at(root)[0];
        while (current != root) {
            loop.push_back(current);
            const auto& next = adjacency.at(current);
            const auto following = next[0] == previous ? next[1] : next[0];
            previous = current;
            current = following;
        }
        double perimeter = 0;
        for (std::size_t index = 0; index < loop.size(); ++index)
            perimeter += std::sqrt(CGAL::squared_distance(input.points[loop[index]], input.points[loop[(index + 1) % loop.size()]]));
        if (perimeter > input.perimeter) {
            ++statistics.rejected["perimeter"];
            continue;
        }
        ++statistics.eligible_loops;
        if (loop.size() > input.max_edges) {
            ++statistics.rejected["boundary_budget"];
            continue;
        }
        try {
            auto patch = fill_loop(input, loop, incident_faces, input.points.size() + output.points.size(), statistics);
            output.points.insert(output.points.end(), patch.points.begin(), patch.points.end());
            output.faces.insert(output.faces.end(), patch.faces.begin(), patch.faces.end());
            ++statistics.filled_loops;
            ++statistics.refined_loops;
        } catch (const CGAL::Failure_exception&) {
            ++statistics.rejected["cgal_precondition"];
        } catch (const std::runtime_error& error) {
            ++statistics.rejected[error.what()];
        }
    }
    return output;
}

void write_output(const std::string& path, const Patch& patch) {
    std::ofstream destination(path, std::ios::binary);
    destination.write("PXHOUT01", 8);
    write_value(destination, std::uint64_t(patch.points.size()));
    write_value(destination, std::uint64_t(patch.faces.size()));
    for (const auto& point : patch.points) {
        write_value(destination, double(point.x()));
        write_value(destination, double(point.y()));
        write_value(destination, double(point.z()));
    }
    for (const auto& face : patch.faces)
        for (auto vertex : face)
            write_value(destination, vertex);
    if (!destination)
        throw std::runtime_error("Could not write output");
}

int main(int argument_count, char** arguments) {
    try {
        if (argument_count != 3)
            throw std::runtime_error("Usage: pixal-cgal-holes INPUT OUTPUT");
        const auto input = read_input(arguments[1]);
        Statistics statistics;
        const auto started = Clock::now();
        const auto patch = process(input, statistics);
        const auto native_seconds = seconds(started);
        write_output(arguments[2], patch);
        std::cout << std::setprecision(17) << "{\"cgal_version\":" << std::quoted(CGAL_VERSION_STR)
                  << ",\"native_seconds\":" << native_seconds
                  << ",\"topology_seconds\":" << statistics.topology_seconds
                  << ",\"refinement_seconds\":" << statistics.refinement_seconds
                  << ",\"fairing_seconds\":" << statistics.fairing_seconds
                  << ",\"validation_seconds\":" << statistics.validation_seconds
                  << ",\"boundary_edges\":" << statistics.boundary_edges
                  << ",\"closed_loops\":" << statistics.closed_loops
                  << ",\"eligible_loops\":" << statistics.eligible_loops
                  << ",\"filled_loops\":" << statistics.filled_loops
                  << ",\"refined_loops\":" << statistics.refined_loops
                  << ",\"added_vertices\":" << patch.points.size()
                  << ",\"added_faces\":" << patch.faces.size()
                  << ",\"rejected\":{";
        bool first = true;
        for (const auto& [reason, count] : statistics.rejected) {
            if (!first)
                std::cout << ',';
            first = false;
            std::cout << std::quoted(reason) << ':' << count;
        }
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
