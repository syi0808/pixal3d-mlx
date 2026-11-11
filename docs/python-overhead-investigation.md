# Python overhead investigation

On the current M3 Pro checkout, the clearest remaining Python-heavy work is
CPU mesh processing, not Transformer dispatch. Frozen turnip replays spend
4.565 s in winding/normal repair, 2.985 s in hole filling, and 1.389 s in final
corner-normal calculation. In contrast, constructing one warm, unchunked
Transformer block takes approximately 0.1 ms before approximately 133 ms of
MLX evaluation and synchronization. These are component observations, not a
measured speedup from C++ or a new whole-generation benchmark.

## Measurement scope

- Measured on 2026-11-11, Asia/Seoul, Apple M3 Pro, 14 GPU cores, 18 GiB RAM,
  Python 3.12.10, MLX 0.31.2, NumPy 2.4.6, SciPy 1.17.1, trimesh 4.12.2,
  and NetworkX 3.6.1.
- The checkout already includes uncommitted vectorized topology and selective
  Transformer compilation changes. This investigation does not implement or
  change those production policies. Each report records revision, working-tree
  status, Python source hashes, versions, and input signatures.
- Cleanup and extraction reuse turnip seed 42, resolution 512: 663,831 raw
  vertices and 1,344,004 triangles. Cleanup uses the existing soft 200,000-face
  target and finishes at 105,040 vertices and 201,195 triangles.
- Corner normals reuse the actual generated GLB, with 148,062 UV-split
  vertices and 201,195 triangles. Texture baking reuses archived mesh and
  voxel attributes at 1024 texture resolution.
- CPU cases have one untimed warm-up, three unprofiled repetitions, then a
  separate cProfile run. Wall, main-thread CPU, and process CPU clocks are
  recorded separately. The Transformer cases have 15 timed repetitions after
  warm-up. Timers include evaluation/synchronization where stated.
- The accepted runs have no other detected inference processes at their start
  or end, no changed Python source hashes during execution, normal sampled
  memory pressure, and no observed new system swap. Desktop applications are
  not controlled; this is not a statistical confidence interval.

The accepted reports are under `outputs/runtime/python-overhead-investigation/`:
`cleanup-idle`, `normals-idle`, `extract-idle`, `graph-ss-idle`, and
`texture-idle`. Earlier `cleanup`, `cleanup-stable`, and `graph-ss` screens are
excluded because source changes or concurrent inference were observed.

## CPU hotspots

| Active operation | Unprofiled median seconds | Attribution and priority |
| --- | ---: | --- |
| `mesh_cleanup.fix_normals` | 4.565 | First priority: trimesh winding traversal, NetworkX graph objects, and tiny per-edge NumPy calls |
| `mesh_cleanup.fill_small_holes`, three passes combined | 2.985 | First priority: triangle-edge dictionaries, tuple sorting, loop traversal, and small per-hole array operations |
| `mesh_normals.smooth_corner_normals` | 1.389 | Second priority: edge dictionaries and Python corner union/find in final GLB normal correction |
| `mesh_extract.decoder_output_to_mesh` | 0.443 | Lower absolute cost: spatial dictionary construction and batched queries still implemented as Python iteration |

The first two rows are **inside**, not additional to, the 9.456 s median
cleanup-and-simplification replay. Individual cleanup times are aggregated
within each repetition before taking the median. Nested component-connectivity
timings must not be added again.

All three cleanup and extraction results, and their profiled replays, match
the archived vertex/triangle arrays exactly. Repeated corner normals and baked
texture arrays match their respective warm-up arrays exactly. This verifies
replay stability; it does not validate an alternative algorithm or backend.

### Winding and normal repair

`mesh_cleanup.fix_normals` delegates to `trimesh.repair.fix_normals`. The
installed `fix_winding` builds a NetworkX face graph, traverses it with BFS,
and calls `faces_to_edges`, `group_rows`, sorting, tiling, and hashing for
individual face pairs.

The cleanup profile contains **199,426 `group_rows` calls** and approximately
199,425 face-pair conversions. `fix_winding` has 8.000 s cumulative time in
the profiled run, versus 4.565 s for the encompassing normal-fix stage in
unprofiled repetitions. Profiling these many small calls is expensive; the
8.000 s is diagnostic, not the performance baseline.

An effective replacement would group edges once and traverse compact face
adjacency rather than repeatedly allocating tiny arrays. Preserve winding
consistency, component traversal/orientation, and trimesh's signed-volume
outward-orientation policy. Substituting winding-only
`orient_faces_by_adjacency` or omitting repairs is not equivalent.

### Hole filling

`fill_small_holes` visits all triangle edges using Python dictionaries and
sorted tuples, even when relatively few boundary loops need filling. It runs
on 1,344,002, 598,411, and 199,463 triangles in this replay. Its cumulative
profile time is 4.689 s, of which 2.761 s is attributed to the function frame
itself. The entire cleanup profile records 6,838,426 dictionary `get` calls
and 6,715,810 `sorted` calls; these are whole-cleanup counts, not exclusively
hole-filling counts.

Candidate work is vectorized edge counting followed by compact boundary-loop
traversal, or a narrow C++ extension for those loops. Preserve perimeter
thresholds, branching/non-manifold boundary behavior, centroid generation,
triangle ordering, and the existing cap winding/orientation contract. Skipping
small components or leaving holes open is a quality/policy change.

### Corner normals

Final normal correction still has its own union/find implementation, separate
from the already-vectorized non-manifold cleanup. Its profile records
**1,779,033 `find` calls and 587,724 `union` calls**, over 603,585 corners.
The 1.389 s median is corner-normal computation only: parsing the GLB,
repacking attributes, validating textures, and writing files are excluded.

A compact graph or native union/find is a candidate. It must retain
geometric welding across UV seams, opposite-directed manifold-edge checks,
the 75-degree crease threshold, corner-angle weighting, degenerate-face
fallbacks, and canonical grouping/accumulation behavior.

### Mesh extraction

The frozen decoder replay reproduces the archived raw mesh exactly. Its
4.642 s decoder preparation, including loading and conversion to NumPy, is
excluded from the 0.443 s CPU extraction median. Extraction makes
**2,688,016 dictionary lookups** in the profile, in addition to inserting
663,831 voxel keys.

Sorted packed-key lookup, as already used by the fast texture sampler, is a
candidate. Preserve coordinate packing, missing-neighbor filtering,
duplicate-key semantics, quad split decisions, and output ordering. Even
eliminating this entire operation would save less than half a second for
this input.

## MLX dispatch and waiting

The graph probe loads the first real pretrained SS block and frozen
4096-token, 1536-channel conditional inputs. Weights, RoPE, modulation, and
reusable conditioning are prepared before timing. The remaining 29 blocks,
the unconditional pass, loading, and the full pipeline caches are absent.

| Warm block configuration | Call median ms | Evaluate and synchronize median ms | Total median ms | Explicit evaluations inside call |
| --- | ---: | ---: | ---: | ---: |
| Reference, FFN chunks of 2048 | 136.153 | 1.543 | 137.704 | 2 |
| Current selective compile, FFN chunks of 2048 | 132.480 | 1.177 | 133.657 | 2 |
| Reference, unchunked lazy FFN | 0.085 | 136.557 | 136.662 | 0 |
| Current selective compile, unchunked lazy FFN | 0.115 | 133.344 | 133.443 | 0 |

Each column's median is computed independently; their displayed sum need not
equal the total median. The chunked call includes execution forced by two
internal `mx.eval` calls. Its approximately 132 ms must **not** be labeled
Python execution. For example, the first current chunked repetition records
about 0.500 ms main-thread CPU inside that call despite approximately 133 ms
wall time.

The unchunked case separates lazy graph construction from evaluation. The
call includes Python **and native MLX graph construction**, yet is below
0.1% of block latency here. It is not a measurement of pure interpreter
overhead, nor a promise about every token count, cold compilation, or a
memory-constrained full model. Removing FFN evaluation boundaries also
changes activation lifetime; this probe does not justify changing production
chunking or extrapolating one-block timings to a whole sampler.

The existing [hardware audit](transformer-sampling-investigation.md#hardware-bottleneck-audit-2026-11-11)
independently finds that resident dense computation dominates. C++ host
translation does not remove its GEMMs or quadratic full attention.

## Texture and native work

The frozen GPU-backend bake takes 2.125 s median, excluding UV unwrap. Its
profile records 402,390 `max` calls in the adaptive face-chunk planner in
`rasterize_uv_mlx`, identifying another Python-loop candidate. The rasterizer
also performs GPU evaluation and NumPy scatter/conversion: its 0.636 s
profiled function self time is not a pure-interpreter measurement.

`sample_voxel_attrs_fast` already uses vectorized sorted lookup. The slower
dictionary-based `sample_voxel_attrs` and per-triangle CPU `rasterize_uv`
exist as reference paths but are **not used** by production's `backend="gpu"`.
Likewise, `_face_connected_components_reference` and
`_repair_non_manifold_edges_reference` are no longer the current cleanup
backend. Their old call counts should not be presented as remaining costs.

UV unwrap takes 18.424 s externally, with 18.162 s main-thread CPU and
76.915 s process CPU. `xatlas.Atlas.generate` is a compiled extension binding,
not a Python triangle loop. The local cProfile run accounts for only 0.00291 s
and omits the encompassing UV frame/native call; it cannot apportion that
18.424 s. The diagnostic records the accounted-wall fraction in subsequent
texture reports rather than treating this profiler blind spot as evidence
of zero cost. Cython and extension operations can also have Python-looking
filenames, and ndarray operators can be charged to a Python frame's self
time. External clocks and source inspection remain necessary.

Model loading, cold compilation, background removal, camera estimation,
checkpoint compression, and server queueing are not independently attributed
by these replays. In particular, `mx.eval`, `.item()`, `.tolist()`, and NumPy
conversion can wait for native/GPU work; their wall time alone is not
evidence of an interpreter bottleneck.

## Recommended order

1. Replace per-face-pair winding work with compact adjacency and equivalent
   orientation handling; benchmark against the frozen mesh and multiple
   component/non-manifold fixtures.
2. Replace hole-filling edge dictionaries with grouped edges and bounded
   boundary traversal while preserving the existing surface policy.
3. Consider reusing vectorized topology primitives or a narrow native
   union/find for corner normals, with seam and crease regression checks.
4. Treat extraction dictionaries and raster chunk planning as smaller
   follow-ups, not reasons for a full rewrite.

These stage times include necessary native arithmetic and cannot all be
claimed as removable Python overhead. A full C++ port has not been measured.
Optimizing these CPU stages does not remove the dominant GPU inference work.

## Reproduction

Use a fresh output directory; the diagnostic refuses to overwrite reports
or profile files and rejects detected concurrent inference unless explicitly
allowed. Run cases serially, not alongside generation or another benchmark.

```bash
.venv/bin/python -m pytest tests/test_investigate_python_overhead.py -q

PYTHONPATH=. .venv/bin/python scripts/investigate_python_overhead.py cleanup \
  --report outputs/runtime/python-replay/cleanup.json --repeats 3
PYTHONPATH=. .venv/bin/python scripts/investigate_python_overhead.py normals \
  --report outputs/runtime/python-replay/normals.json --repeats 3
PYTHONPATH=. .venv/bin/python scripts/investigate_python_overhead.py extract \
  --report outputs/runtime/python-replay/extract.json --repeats 3
PYTHONPATH=. .venv/bin/python scripts/investigate_python_overhead.py graph \
  --report outputs/runtime/python-replay/ss.json --repeats 15
PYTHONPATH=. .venv/bin/python scripts/investigate_python_overhead.py texture \
  --report outputs/runtime/python-replay/texture.json --repeats 3
```

The original runs wrap these commands with `scripts/measure_generation.py`
for process-footprint and system-pressure/swap monitoring. JSON reports,
full `.prof` files, generation logs, and resource samples remain in the
ignored runtime evidence directory; no GLB or checkpoint is overwritten.
