# Bounded Pixal3D runtime for Apple Silicon

This implementation keeps pretrained weights, full global self-attention,
geometry token selection, and the 12-step sampling schedules unchanged by
default. It does not implement distillation, local attention substitution,
token pruning, or automatic precision reductions.

The completed native-MoGe 1024 experiment, including process footprint, swap,
and pressure measurements, is documented in [1024-validation.md](1024-validation.md).
Follow-up decoder liveness ablations and transformer precision/cache findings
are documented in [decoder-investigation.md](decoder-investigation.md).
The validated production shape/texture policy is documented in
[shape-decoder-integration.md](shape-decoder-integration.md).
The attached turnip's 512 GLB and full-pipeline resource measurements are in
[turnip512-validation.md](turnip512-validation.md).
Strict triangle caps, checkpoint-only LOD export, and additional CPU topology
experiments are documented in [performance-followup.md](performance-followup.md).
Frozen-input SS/HR compilation, synchronization, and FFN/cache experiments are
documented in [transformer-sampling-investigation.md](transformer-sampling-investigation.md).
Its [hardware bottleneck audit](transformer-sampling-investigation.md#hardware-bottleneck-audit-2026-11-11)
adds empirical compute/bandwidth ceilings, GPU execution coverage, dtype
promotion, and separately scoped compression/swap observations.

## Execution and memory ownership

`pixalmlx.runtime.StageGraph` compiles explicit stage dependencies, prunes dead
nodes, orders producers before consumers, and releases owned resources after
their last use. Nested array outputs are evaluated before inputs are released.
Image encoding and active projection use this graph in the live generation
path. It is an explicit DAG executor, **not** an automatic tracer/compiler for
arbitrary Python or the entire neural network.

The main and checkpoint-resume entrypoints share `InferenceRuntime`:

- A default **10,000,000,000-byte MLX graph-evaluation guideline**, restored when
  the runtime closes. This is not a hard allocation cap. Configured guidelines
  above Metal's recommended working set are rejected.
- Bounded LRU model and feature caches, with oversized entries bypassed rather
  than admitted. Checkpoint fingerprints include file size and modification
  time; decoder cache identities also include spatial ordering and evaluation
  interval.
- Content-addressed image encoding and camera caches. Separate seed outputs
  can reuse features even when preprocessed images have different filenames.
- One active job per runtime, and serialized runtime ownership of process-wide
  MLX limit settings. Exceptions clear reusable caches and restore limits.
- Stage boundaries report elapsed time, live/pooled/peak MLX allocation, and
  process high-water RSS. Reports distinguish these different measurements.
- Streamed checkpoint loading materializes and assigns one tensor at a time.
  Legacy FP32-to-FP16 rounding is preserved, including when a tensor is then
  upcast for the stable FP32 sparse-structure stage.

The setting guides MLX graph evaluation; it does **not** guarantee either an
MLX allocation ceiling or a process memory ceiling. Allocations may exceed it
when RAM or swap remains available. The native 1024 run measured 11.44 GB peak
MLX allocation despite a 10 GB setting. RSS, NumPy arrays, ONNX sessions,
file-backed pages, and system memory require separate accounting. There is no
CPU-offload claim that expands physical unified memory. See
[the MLX API contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html).
Reports now explicitly mark the guideline as non-hard and record overshoot.

Buffer reuse is handled by MLX's bounded allocator pool plus shorter live
ranges and bounded operation sizes. There is no custom alias-aware raw-buffer
arena, nor a claim to release references still owned by external Python code.

## Sparse execution

Sparse topology now stays in MLX/Metal:

1. Pack and sort unsigned 64-bit coordinate keys on GPU.
2. Binary-search neighbors in a custom Metal kernel.
3. Prefix-scan and compact valid edges on GPU.
4. Reuse per-kernel groups across convolutions at a spatial level.
5. Prefix-scan subdivision masks and produce child coordinates and feature
   gather indices directly on GPU.

Python reads only kernel edge counts and validation flags once per topology,
and one scalar child count per subdivision, because MLX array output shapes
must be known to Python. Full coordinate or feature payloads are not downloaded
for these operations. Requantization, mesh extraction, UV processing, and GLB
export still have CPU stages; this is not an entirely GPU-resident pipeline.

Decoder storage can use a Morton permutation without changing which voxels
are evaluated. Shape and texture use the same ordering for guide subdivision
alignment. `--sparse-order preserve` retains legacy coordinate order.

Pixal3D generation/resume default to `--decoder-eval-interval 1`: only the two
convolutions in the last shape/texture upsample evaluate between kernel
contributions, shortening large gather/GEMM/scatter live ranges. The predictive
shape path and guided texture path retain the same accumulation order and
weights. `--decoder-eval-interval 0` restores the legacy evaluation schedule.
Plain decoder construction remains legacy by default for API compatibility.

## NAF without dense high-resolution feature output

Sparse conditioning projects voxels to image coordinates first. For each voxel
it computes NAF only at the four bilinear image neighbors, then interpolates
them with the original border-clamping convention. A bounded query chunk avoids
both a dense `512 x 512 x 1024` output and large neighborhood gathers.

The guide encoder uses row-tiled convolutions with reflected halos. GroupNorm
still sees the **entire spatial image**, not per-tile statistics. Each encoder
branch is pooled before concatenation, avoiding the full-resolution combined
feature tensor. DINO also evaluates between pairs of blocks to bound activation
lifetimes. These changes preserve the model's operations rather than resizing
the guide or substituting bilinear upsampling.

The NAF kernel supports 64, 128, and 256 threads per threadgroup. This M3 Pro
comparison favored the retained 128-thread default; there is no claim that one
setting is optimal for every Apple GPU or shape.

## DiT static work and bounded activations

- Sparse or dense RoPE is prepared once per sampling invocation.
- Timestep modulation is prepared once per step, shared by positive and
  negative passes, and released with the sampler scope.
- Global K/V and per-block projected image features use sampling-scoped bounded
  caches. Admission keeps a fixed subset instead of LRU-thrashing through all
  transformer blocks when the full projection cache does not fit.
- Known zero negative conditioning is explicitly tagged. All five global
  tokens then have identical K/V, making cross-attention independent of the
  query. Its output becomes one constant per block, including the projection
  bias. This eliminates negative-branch query/output/projection GEMMs and
  attention without dropping a guidance pass. Arbitrary user contexts are not
  inspected and assumed to be zero.
- FFN row chunking limits `tokens x hidden_channels` intermediates. Evaluation
  between pairs of blocks shortens the remaining live graph.
- Pixal3D generation/resume enable selective compilation of pure FFN chunks,
  AdaLN modulation, and gated residuals by default. `--no-compile-transformer`
  restores uncompiled execution without changing the 12-step schedules,
  attention, chunk sizes, or precision. Library sampler calls remain opt-in
  through `selective_compile=True`; raw model construction stays uncompiled.
  Packed quantized FFNs retain their quantized operators instead of routing
  integer storage through a floating-point GEMM compiler.
- Optional compilation fuses the pure Euler update. MLX GEMM and Flash
  Attention remain the execution backends for dense transformer computation.

CPU mesh cleanup uses sorted edge groups and SciPy sparse connectivity for
face components and non-manifold corner repair. Component filtering, canonical
vertex/face ordering, simplification, hole filling and normal policy are
unchanged. The retained reference backend is for scoped benchmark comparisons,
not a process-wide production switch. See the
[production performance comparison](performance-followup.md#production-integration-2026-11-11).

Stable SS compute and CFG remain FP32. Experimental `--ss-precision mixed`
and `--slat-precision mixed` keep FP32 sampler state and normalization
accumulation while using FP16 model activations. Small numerical/finite tests
do **not** establish image-to-mesh quality equivalence for these modes, so
neither is enabled by default. Existing INT4/INT8 quantization is also opt-in.

## Persistent asset batches

```bash
.venv/bin/python -m pixalmlx.batch_inference \
  --engine pixal3d-persistent --image assets/shoe_input.png \
  --seeds 42,43,44 --resolution 512 --max-concurrent 1 \
  --output-dir outputs/asset-batch --report outputs/asset-batch/report.json
```

This route calls Pixal3D in the same process, never changes global `sys.argv`
or working directory, retains compiled kernels, and reuses admitted encoder,
camera, feature, and decoder resources. It reports effective concurrency of
one even if a larger concurrency was requested. Each job validates its output,
writes a runtime report, and updates a durable batch report. Multi-view inputs
are rejected rather than silently ignored.

The bounded cache does **not** keep every multi-GB flow model resident. Those
models may reload between jobs. Stage-major multi-asset scheduling and
disk-backed prepacked weight caches are not implemented or benchmarked.
Cold and warm runs use explicit per-stage noise keys (`seed` through
`seed + 3`) so cache hits cannot change randomness. The same seed is not
guaranteed to reproduce an asset from the older initialization-dependent RNG.

## M3 Pro measurements

Measured on this M3 Pro, 18 GB, MLX 0.31.2. Medians use an untimed warm-up and
five timed repeats, with no concurrent GPU benchmark process. They are
**component measurements, not overall generation speedups**.

| Case | Reference | Optimized | Peak MLX allocation |
| --- | ---: | ---: | ---: |
| 12K-voxel neighbor map | CPU 16.82 ms | GPU 1.79 ms | GPU 4.47 MB |
| Pretrained NAF, 512 guide, 6K projected points | Dense 567.00 ms | Sparse/tiled 491.46 ms | 4.17 GB to 1.73 GB |
| Tiny DiT, 2048 tokens, six CFG steps | Uncached 376.50 ms | Static cache/folding 366.74 ms | 1.14 GB to 449 MB |
| Same DiT, FFN chunks of 512 | Uncached 376.50 ms | Bounded 398.67 ms | 1.14 GB to 234 MB |

The last case deliberately trades about 6% more time for about 80% lower
peak MLX allocation. Dense NAF with tiling alone also got slower (701.67 ms);
the sparse output path is important rather than assuming tiling is a speedup.
GPU neighbor construction allocates more MLX scratch than the CPU reference;
host NumPy allocations are not included in that comparison.

NAF uses official pretrained weights with synthetic DINO features and image
queries. Maximum absolute difference was `1.23e-5`. Tiny DiT uses random
weights; its maximum absolute difference was `1.44e-6`.

A separate 1024-guide / 512-output sparse NAF run completed in 1.80 s with
3.39 GB peak MLX allocation. This is **not** a full 1024-resolution generation.
The attention threadgroup comparison measured 11.79 / 11.31 / 11.37 ms at
64 / 128 / 256 threads, respectively.

Reproduce with:

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_architecture.py --output outputs/runtime/architecture.json --repeats 5
PYTHONPATH=. .venv/bin/python scripts/benchmark_architecture.py --output outputs/runtime/architecture-hr-guide.json --guide-size 1024 --only naf-sparse
PYTHONPATH=. .venv/bin/python scripts/benchmark_architecture.py --output outputs/runtime/architecture-threadgroups.json --only kernel --repeats 5
```

Raw reports include revision, working-tree state, per-repeat times, and device
metadata. Benchmark inputs remain synthetic even when checkpoint weights are
real. Full-pipeline quality and multi-asset throughput require separate
validation. A completed single-input 1024 memory experiment is now available
in [1024-validation.md](1024-validation.md).

## Completed 512 generation

An actual `assets/shoe_input.png` generation completed with pretrained weights,
the normal 12-step schedules, FP32 SS, no weight quantization, and a 10 GB MLX
evaluation guideline:

| Measurement | Observed |
| --- | ---: |
| Pipeline-reported elapsed time | 368.7 s |
| Peak MLX allocation | 8,173,536,474 bytes |
| Process high-water RSS (`time -l`) | 6,826,115,072 bytes |
| Peak process memory footprint (`time -l`) | 9,501,707,072 bytes |
| Shape / texture latent tokens | 2,219 |
| Exported faces | 209,727 |
| Base color and metallic/roughness maps | 1024 x 1024 |

These memory figures describe different accounting domains and must not be
added together. The run demonstrates the 10 GB MLX target for **one input**,
not every 512 input. It is not a before/after whole-pipeline speed comparison.
The run used the supported default FOV, 49.1 degrees, because upstream MoGe
checkpoint loading initially required a missing Torch installation. Background
removal and its first model download are included in the cold-run time.

Artifacts are `outputs/runtime/e2e512/shoe.glb`, `runtime.json`,
`generation.log`, `validation.json`, and `shoe-witness.png`. Geometry coordinates
are finite and nonempty, both PBR maps load successfully, and the three-view
witness render is nonblank. This is not a rigorous quality-equivalence study
against an older-runtime render or a watertightness guarantee.

## Native MoGe checkpoint preparation

Pixal3D, DINOv3, MoGe, and NAF downloads are now present locally. The official
MoGe checkpoint was checksum-verified and converted once to raw safetensors,
so the runtime adapter can use the original `moge_mlx` model and weight mapping
without importing Torch. Torch is used only by the isolated conversion tool.

```bash
moge_source=$(.venv/bin/python -c 'import json; print(json.load(open("weights/download-manifest.json"))["models"]["moge"]["path"])')
uv run --no-project --python .venv/bin/python --with torch --with numpy --with safetensors \
  python scripts/convert_moge_weights.py --input "$moge_source" \
  --expected-sha256 3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5
```

The converter uses `weights_only=True`, verifies the pinned source hash, and
writes `weights/moge_release.safetensors` atomically with a provenance manifest.
The native camera path was independently exercised on the preprocessed shoe:
2.46 s, 4.17 GB peak MLX allocation, FOV 37.0 degrees, with no Torch module
imported. Its report is `outputs/runtime/native-camera.runtime.json`.
The six-minute GLB above is from the earlier default-FOV run, not a rerun with
this newly available camera estimate. If the native checkpoint is absent, the
adapter preserves the upstream loader/fallback behavior.

## Running and comparing

```bash
.venv/bin/python generate_pixal3d.py --image assets/shoe_input.png \
  --resolution 512 --memory-budget-gb 10 \
  --output outputs/shoe.glb --save-checkpoints outputs/shoe-checkpoints \
  --runtime-report outputs/shoe.runtime.json
```

Relevant isolated-reference switches are `--dense-conditioning`,
`--naf-backend reference`, `--naf-tile-rows 0`, `--conditioning-cache-mb 0`,
`--ffn-chunk-size 0`, and `--sparse-order preserve`. A full dense reference
may exceed this machine's memory budget; use small controlled comparisons.
`--naf-threads` exposes the tested threadgroup sizes. Checkpoint resume accepts
the same applicable controls and runs in-process.

The default generation resolution is now 512. Neither resolution nor sampling
steps are silently reduced to rescue an allocation failure. A memory guideline
is not a watchdog; use `scripts/measure_generation.py` for external process and
system-pressure guards. The existing explicit `--max-tokens` behavior is unchanged.
