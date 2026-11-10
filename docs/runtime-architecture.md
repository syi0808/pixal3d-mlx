# Bounded Pixal3D runtime for Apple Silicon

This implementation keeps pretrained weights, full global self-attention,
geometry token selection, and the 12-step sampling schedules unchanged by
default. It does not implement distillation, local attention substitution,
token pruning, or automatic precision reductions.

## Execution and memory ownership

`trellmlx.runtime.StageGraph` compiles explicit stage dependencies, prunes dead
nodes, orders producers before consumers, and releases owned resources after
their last use. Nested array outputs are evaluated before inputs are released.
Image encoding and active projection use this graph in the live generation
path. It is an explicit DAG executor, **not** an automatic tracer/compiler for
arbitrary Python or the entire neural network.

The main and checkpoint-resume entrypoints share `InferenceRuntime`:

- A default **10,000,000,000-byte MLX allocation limit**, restored when the
  runtime closes. Limits above Metal's recommended working set are rejected.
- Bounded LRU model and feature caches, with oversized entries bypassed rather
  than admitted. Checkpoint fingerprints include file size and modification
  time; decoder cache identities also include spatial ordering.
- Content-addressed image encoding and camera caches. Separate seed outputs
  can reuse features even when preprocessed images have different filenames.
- One active job per runtime, and serialized runtime ownership of process-wide
  MLX limit settings. Exceptions clear reusable caches and restore limits.
- Stage boundaries report elapsed time, live/pooled/peak MLX allocation, and
  process high-water RSS. Reports distinguish these different measurements.
- Streamed checkpoint loading materializes and assigns one tensor at a time.
  Legacy FP32-to-FP16 rounding is preserved, including when a tensor is then
  upcast for the stable FP32 sparse-structure stage.

The limit applies to MLX allocations, **not** all process RSS, NumPy arrays,
ONNX sessions, file-backed pages, or system memory. Exceeding the allocation
budget produces an error; setting a 10 GB budget does not prove every 512
input can complete within it. There is no CPU-offload claim that expands
physical unified memory. See [MLX memory management](https://ml-explore.github.io/mlx/build/html/python/memory_management.html).

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
- Optional compilation fuses the pure Euler update. MLX GEMM and Flash
  Attention remain the execution backends for dense transformer computation.

Stable SS compute and CFG remain FP32. Experimental `--ss-precision mixed`
and `--slat-precision mixed` keep FP32 sampler state and normalization
accumulation while using FP16 model activations. Small numerical/finite tests
do **not** establish image-to-mesh quality equivalence for these modes, so
neither is enabled by default. Existing INT4/INT8 quantization is also opt-in.

## Persistent asset batches

```bash
.venv/bin/python -m trellmlx.batch_inference \
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
real. Full-pipeline quality, 1024 memory feasibility, and multi-asset throughput
require separate validation.

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
steps are silently reduced to rescue an allocation failure. The existing
explicit `--max-tokens` behavior is unchanged.
