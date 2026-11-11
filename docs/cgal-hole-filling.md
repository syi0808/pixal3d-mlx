# CGAL hole filling comparison

CGAL refinement/fairing followed by fan filling of rejected holes is now the
default for generation, checkpoint resume, soft-target re-export, and API jobs
(`cgal-fan`). Hole filling retains its input's existing vertex and triangle
arrays byte-for-byte. The original fan implementation remains available with
`--hole-fill-backend fan`.

The controlled curved-surface examples improve substantially, but the real
turnip comparison does **not** establish a universal improvement: CGAL alone
leaves rejected holes open, and CGAL-plus-fan fills the same number of holes
as the reference but takes longer without a consistent seam-angle improvement.
The default is an explicit processing-policy choice, not a new performance claim.

## Implementation and preservation

`pixalmlx.cgal_holes.fill_small_holes_cgal` invokes one optional native helper
per input mesh, not one process per hole. The helper uses CGAL's
`triangulate_and_refine_hole` followed by `fair` on **only newly created
interior vertices**. Defaults are density factor 2.0 and continuity 1.

For each unambiguous closed boundary below the perimeter limit:

1. Collect a local collar from the original mesh; continuity 1 uses two face
   rings. Copy and consistently orient this collar for CGAL without changing
   the caller's triangles.
2. Check that the collar is a valid polygon mesh and that its selected border
   cycle exactly matches the intended hole.
3. Triangulate/refine the hole, then solve the bi-Laplacian fairing problem
   for the newly inserted vertices. Boundary and collar vertices stay fixed.
4. Reject failed triangulation/fairing, insufficient refinement, degenerate
   collars, changed boundary connectivity, and detected patch/collar
   intersections. Verify that original collar points did not move.
5. Return only appended vertices/triangles. The Python bridge concatenates
   them after the original arrays and checks index bounds, finite values,
   index-dtype overflow, and triangle collapse at the caller's precision.

The implementation uses the algorithms described in the
[CGAL hole filling API](https://doc.cgal.org/latest/PMP_Mesh_repair/group__PMP__hole__filling__grp.html)
and [fairing API](https://doc.cgal.org/latest/PMP_Remeshing/group__PMP__local__remeshing__grp.html).
It is a local interpolation, not recovery of unknown ground-truth geometry.

Native limits default to 256 boundary vertices, 4096 collar faces, and 4096
new vertices per patch. Cubic triangulation fallback is disabled. Ambiguous
boundary components are skipped rather than guessed. A 300 s subprocess
timeout bounds a single native call. Failed native execution is an error,
not a silent fallback.

The standalone bridge retains `fallback="skip"` for compatibility. Production
entrypoints select `cgal-fan`, which explicitly passes `fallback="fan"` and runs the
unchanged reference filler over the remaining boundaries after CGAL has
finished. Diagnostics distinguish CGAL-filled and fan-filled holes. This
fallback preserves reference behavior, including its possible degenerate
caps; it is not a geometric validity guarantee.

Intersection checks concern each new patch and its local original collar.
They do not certify the absence of intersections with distant original
triangles or between separately generated patches. A rejection can also
involve coincident/split geometry, not necessarily an obviously folded cap.
Original winding can be inconsistent; only the shadow collar is reoriented.
The normal orientation cleanup remains a separate step.

The byte-preservation contract applies to the **hole filling input**. Earlier
duplicate removal/component filtering/non-manifold repair and later
simplification or normal repair retain their existing behavior. This backend
does not promise that a whole generation pipeline never modifies old geometry.

## Build and use

The Python package does not build or download CGAL. Default generation now
requires the helper below; alternatively select `--hole-fill-backend fan`.
Installed packages can locate a separately built helper using
`PIXALMLX_CGAL_HOLE_FILLER`. Native availability is checked before expensive
generation/resume work. `--no-cleanup`, `--ss-only`, and strict-cap-only
re-exports do not require an unused helper. A missing helper or native process
failure is an error, rather than an implicit whole-job fallback.

```bash
HOMEBREW_NO_AUTO_UPDATE=1 brew install cgal eigen
cmake -S native/cgal_holes -B build/cgal-holes \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(brew --prefix)"
cmake --build build/cgal-holes -j 1
```

CGAL 5.6 or newer, Eigen, CMake 3.22 or newer, and a C++20 compiler are
required. The tested build uses CGAL 6.2.1, Eigen 5.0.1, CMake, and Apple
Clang 17 on Apple Silicon. Eigen is discovered without a major-version pin
so both Eigen 3 and the tested Eigen 5 package can be found.

```python
from pixalmlx.cgal_holes import fill_small_holes_cgal

diagnostics = {}
vertices, faces = fill_small_holes_cgal(
    vertices, faces,
    max_hole_perimeter=0.03,
    density_control_factor=2.0,
    fairing_continuity=1,
    fallback="fan",
    diagnostics=diagnostics,
)
```

All generation/resume entrypoints and soft-target re-export accept
`--hole-fill-backend cgal-fan|fan|cgal`, defaulting to `cgal-fan`. The server's
`hole_fill_backend` JSON option has the same choices and default, is recorded
in job options, and is forwarded explicitly to the worker. Older queued jobs
without the field also use `cgal-fan`. Runtime/re-export reports record the choice.

For direct library use, `cleanup_mesh(..., hole_fill_backend="cgal-fan")`
selects the combined mode. `cgal` leaves rejected holes open, and `fan` uses
only the unchanged reference implementation. The low-level `cleanup_mesh`
and `_cleanup_and_simplify_mesh` APIs retain their `fan` defaults to preserve
reference experiments and custom callers; production entrypoints pass the
selected backend explicitly through every cleanup pass.

The production-default integration was exercised without neural resampling
using the archived turnip raw geometry and mesh/texture checkpoints:
`outputs/runtime/cgal-hybrid-integration/export.json`. The command omitted
`--hole-fill-backend`, recorded `cgal-fan`, ran combined filling through the
cleanup/simplification path, and exported a validated **20,044-triangle** GLB
with 1024 PBR maps from a soft 20,000-face target. Existing assets were not
overwritten. This checks checkpoint re-export, not a new image-to-3D run or
multi-asset quality equivalence.

Default-policy validation reports **566 passed and the same three existing
README failures**, logged in `cgal-hybrid-integration/full-tests.log`. A second
targeted run with the helper intentionally unavailable reports **104 passed /
13 native cases skipped**, checking that explicit fan mode and non-filling
routes remain usable without CGAL.

CGAL's relevant headers identify their license as
`GPL-3.0-or-later OR LicenseRef-Commercial`; CGAL is not an MIT dependency.
No CGAL headers or prebuilt helper are bundled here. See the
[CGAL licensing terms](https://www.cgal.org/license.html) before distributing
software built with these components.

## Measurement conditions

Measured 2026-11-11, Asia/Seoul, Apple M3 Pro, 14 GPU cores, 18 GiB physical
memory, Python 3.12.10. Accepted evidence is under
`outputs/runtime/cgal-hole-filling/comparison-final/`, with external monitoring
in `comparison-final-monitor/`. Earlier `synthetic-v1` and `comparison-v2`
through `comparison-v4` directories are retained as preliminary experiments,
not the timing table below.

Each backend gets one untimed warm-up and five timed serial repetitions.
Wall times include the Python/native bridge, temporary-file I/O, native
startup, native patch checks, and Python output validation. Metrics, hashing,
NPZ output, and plotting occur outside timed calls. Parent CPU and completed
child CPU are reported separately from wall time; parent CPU alone would
omit the native process's work. Native phase counters are partial stage
instrumentation; use the outer wall times for comparisons.

Source/binary hashes and working-tree state are recorded. No inference
process was detected at the start or end, Python source hashes remained
unchanged, sampled system pressure stayed normal, and no new system swap
growth was observed. Desktop scheduling is not controlled. The repetitions
are not a randomized confidence interval. The process monitor does not
establish a hard memory cap for the entire subprocess tree.

### Known curved surfaces

The sphere, saddle, and plane fixtures each have 384 vertices and 672
triangles in an annulus, with 48 edges around the intended hole. Its inner
radius is 0.45, outer radius 0.9, and the synthetic perimeter cutoff is 4.0,
not the production 0.03 cutoff. The larger outer boundary remains open.

Height RMSE is evaluated at the **same 7809 XY query positions** inside the
original hole polygon for both outputs, interpolating the actual output
triangles. Both cover all positions with no ambiguous projected overlap.
The sphere has radius 1; the saddle is `z = 0.6 * (x*x - y*y)`. These are
synthetic interpolation tests, not object-generation quality scores.

| Surface | Fan height RMSE | CGAL height RMSE | Fan added triangles | CGAL added triangles |
| --- | ---: | ---: | ---: | ---: |
| Sphere | 0.062807 | 0.006542 | 48 | 482 |
| Saddle | 0.015533 | 0.007546 | 48 | 534 |
| Plane | 0 | 0 | 48 | 562 |

The RMSE reduction is approximately 89.6% on the sphere and 51.4% on the
saddle, with substantially more triangles. Sphere mean seam angle decreases
from 28.90 to 3.70 degrees; saddle mean decreases from 9.63 to 2.70 degrees.
The plane has no quality gain. Fan takes approximately 0.77–0.79 ms per
fixture; CGAL takes 7.59 / 8.25 / 9.16 ms for sphere/saddle/plane, including
native process overhead. For these small individual holes it is slower.

`synthetic-comparison.png` shows the actual meshes and their `y=0` cross
sections: grey is the fixed original surface, orange is fan, blue is CGAL,
and the dashed line is the known analytic surface. Normal smoothing alone
cannot produce the blue geometric cross section.

### Actual turnip geometry

The raw seed-42, resolution-512 decoder mesh is frozen in
`outputs/runtime/postprocess-investigation/raw-mesh.npz`. Existing duplicate
removal, area filtering, and vectorized non-manifold repair are run once
before benchmarking. Both backends receive the same **724,113 vertices /
1,344,002 triangles**, with 96,704 boundary edges. Detection finds 14,078
closed loops; 13,949 pass the 0.03 perimeter limit and 129 are too large.

| Backend | Median seconds | Range seconds | Holes filled | Added vertices | Added triangles | Remaining boundary edges |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Original fan | 2.196 | 2.060–2.241 | 13,949 | 13,949 | 85,562 | 11,142 |
| CGAL only | 1.299 | 1.259–1.384 | 9,463 | 20,477 | 79,808 | 38,924 |
| CGAL then explicit fan fallback | 3.260 | 3.200–3.326 | 13,949 | 24,963 | 107,590 | 11,142 |

CGAL alone processes only 67.8% of eligible holes, so its smaller wall time
is **not** an equal-coverage acceleration claim. Its 4,486 eligible rejections
are: 2,607 detected patch/collar intersections, 1,750 patches without new
interior vertices, 125 failed triangulations, and 4 non-manifold collars.
The combined backend fills these remaining loops with the reference fan
algorithm. It takes approximately 48.4% / 1.064 s longer than fan alone for
this first filling pass and adds approximately 25.7% more triangles.

Compare continuity only on original boundary edges filled by both methods:

| Comparison scope | Edges | Fan mean angle | Alternative mean angle | Fan p95 | Alternative p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fan versus CGAL-accepted patches | 57,780 | 1.787° | 1.891° | 4.636° | 5.127° |
| Fan versus combined backend | 85,534 | 3.293° | 3.363° | 12.360° | 11.969° |

Angles use the absolute normal dot product because fan's winding is corrected
later. They are continuity proxies, not ground-truth surface error or global
visual quality. The combined mean is slightly worse while its p95 is slightly
better; there is no consistent measured continuity improvement on this mesh.
Fan and the combined mode each add 28 exactly degenerate triangles. CGAL-only
adds none, but leaves rejected holes open. All outputs have zero edges with
more than two incident triangles, which does not prove manifold vertices,
watertightness, or absence of geometric intersections.

All backends preserve every original vertex/triangle byte and produce
identical output hashes across all six calls, including warm-up. No original
GLB, pretrained weights, or saved checkpoint is overwritten. This is a
single hole-filling stage, not cleanup/simplification, texture export, or
end-to-end generation timing. Later simplification could change the apparent
benefit and has not been evaluated for the alternative backend.

## Reproduce and validate

```bash
.venv/bin/python -m pytest tests/test_cgal_holes.py -q

# Optional, for the static comparison figure:
uv pip install --python .venv/bin/python matplotlib

PYTHONPATH=. .venv/bin/python scripts/benchmark_hole_filling.py \
  --repeats 5 --plot --output-dir outputs/runtime/cgal-hole-filling/new-comparison
```

Use a fresh output directory; evidence is not overwritten. The benchmark
rejects detected concurrent inference. It stores original and filled mesh
arrays, timings, native rejection diagnostics, preservation checks, analytic
errors, same-edge seam metrics, and source/binary hashes. `--skip-real` runs
only analytic fixtures; `--density` and `--continuity` expose explicit research
parameters. Plotting is optional and not a runtime dependency.

The native tests cover float32/float64 preservation, native opt-in, fixed
boundaries, analytic curvature improvement, planar output, rejected/ambiguous
holes, degenerate collars, malformed protocol/payloads, post-cast triangle
collapse, explicit fallback, original winding preservation, and integer byte
order. Without a built helper, native integration cases are skipped while
bridge/dispatch validation still runs.

Original comparison validation: **27 CGAL-focused tests passed**. The full suite
reports **540 passed and the same three pre-existing README onboarding
failures** (`test_readme_uses_current_huggingface_cli`,
`test_readme_tests_section_avoids_stale_exact_count`, and
`test_readme_quantization_section_states_measured_tradeoff`). The full output
is retained as `comparison-final/full-tests.log`.

CGAL-plus-fan is the configured production default. Keep the fan override
available for reference comparison, environments without a native helper, or
assets where fairing is undesirable. This is not a demonstrated universal
quality improvement or a whole-pipeline speed optimization.
