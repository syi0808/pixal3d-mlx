# Apple Silicon runtime optimization

This is the initial continuation checkpoint. The current runtime, memory
controls, sparse NAF projection, persistent batching, and newer measurements
are documented in [runtime-architecture.md](runtime-architecture.md).

## Local continuation

Work resumed on 2026-11-10 from the Codex chat **Pixal3D 실행 가능 여부 확인**.
The checkout, Git history, uncommitted work, and benchmark artifacts were moved
from `/Users/classting/Workspace/pixal3d-mlx` to
`/Users/classting/Workspace/temp/pixal3d`. The virtual environment was recreated
at the new location with `uv sync --frozen --extra dev`.

Measured environment: Apple M3 Pro, 14 GPU cores, 18 GB unified memory,
macOS 26.2, Python 3.12.10, MLX 0.31.2. The Metal recommended working set is
approximately 13.3 GiB; memory limits have not been raised.

## Implemented changes

- NAF attention uses a custom Metal kernel with SIMD dot products and
  threadgroup-local attention probabilities. It preserves shifted NATTEN border
  windows and reads nearest-upsampled keys and values directly from their
  low-resolution source instead of materializing repeated maps or gathered
  neighborhoods.
- Sparse subdivision prepares coordinate selection once and gathers feature
  slices on MLX instead of copying feature tensors through NumPy.
- Sparse convolution caches per-kernel index groups. Decoder blocks reuse
  neighbor maps until coordinates change. Map construction uses sorted packed
  coordinates and vectorized searches, including explicit field-boundary checks.
- Conditioning projects only active voxel coordinates, caches DINO encodings
  across compatible stages, and reuses identical HR conditioning for texture.
  DINO and NAF models are released before HR flow sampling. Camera transforms
  no longer require a CPU round trip.

`--naf-backend reference` selects the gather-based NAF implementation;
`--dense-conditioning` selects dense projection before sparse indexing.
These switches do not disable the other optimizations.

## Measurements

All numbers below are **synthetic component measurements**, not end-to-end
generation speedups. Five timed repeats follow an untimed warm-up. The original
component baseline used three repeats in the preceding session, so small timing
differences should not be interpreted as controlled performance guarantees.

| Component | Original baseline | Current median | Ratio |
| --- | ---: | ---: | ---: |
| Neighbor map, 12K coordinates | 27.36 ms | 17.24 ms | 1.59x |
| Subdivision, 12K parents | 59.48 ms | 2.01 ms | 29.66x |
| Sparse convolution, 2K voxels | 2.33 ms | 0.89 ms | 2.62x |
| Dense projection, grid 32 | 4.02 ms | 2.93 ms | 1.37x |

A separate same-process comparison at 128 x 128, four heads, 32 query channels
and 64 value channels per head measured:

| NAF implementation | Median | Peak MLX allocation |
| --- | ---: | ---: |
| Gather reference | 105.91 ms | 567.3 MiB |
| Fused Metal | 6.24 ms | 51.7 MiB |

This is approximately **17.0x faster with 90.9% lower peak MLX allocation** for
that case. Peak values include persistent benchmark inputs and are not process
RSS, system memory pressure, swap usage, or full pipeline peaks. These attention
dimensions are smaller than the production NAF model.

Raw reports are retained locally in the ignored `outputs/runtime/` directory:
`baseline-components.json`, `resumed-components.json`, and
`resumed-metal-naf-128.json`.

The official NAF checkpoint was also checksum-verified, converted, and loaded
with all 37 parameters present. With a 512 x 512 guide image and synthetic
`1024 x 32 x 32` DINO features, the complete pretrained NAF module measured
576.77 ms with the reference backend versus 249.98 ms with Metal at output
size 128 x 128: approximately **2.31x**, not 17x. Maximum absolute output
difference was `1.46e-5`. Peak MLX allocations were 3.65 GiB and 2.93 GiB.
Metal output at 512 x 512 completed in 456.76 ms with a 3.86 GiB peak.
These measurements include the image encoder and use three timed repeats;
they still do not use real DINO features or validate the full pipeline's
1024 x 1024 HR guide image. Details are in `outputs/runtime/pretrained-naf.json`.

```bash
uv sync --frozen --extra dev
.venv/bin/python -m pytest tests/test_sparse_runtime.py tests/test_sparse_projection.py tests/test_metal_naf.py tests/test_pixal3d_proj.py tests/test_generate_pixal3d_contract.py tests/test_naf_natten_borders.py -q
PYTHONPATH=. .venv/bin/python scripts/benchmark_runtime.py --output outputs/runtime/components.json --repeats 5
PYTHONPATH=. .venv/bin/python scripts/benchmark_runtime.py --output outputs/runtime/naf-128.json --repeats 5 --naf-size 128 --only naf_128,naf_fused_128
```

The focused suite passes, including partial Metal threadgroups, non-contiguous
cross-attention inputs, cached feature extraction, projection equivalence,
shuffled sparse coordinates, and unsigned 16-bit coordinate boundaries.
The resumed full suite reports 232 passed, 6 skipped, and three pre-existing
README onboarding failures:
`test_readme_uses_current_huggingface_cli`,
`test_readme_tests_section_avoids_stale_exact_count`, and
`test_readme_quantization_section_states_measured_tradeoff`.
They have not been changed as part of runtime optimization.

## Model downloads and execution gate

The public model download was restarted with `--skip-dino --verify`. It runs
independently of the terminal session; progress and PID are recorded in
`outputs/runtime/model-download.log` and `outputs/runtime/model-download.pid`.
HF checkpoints stay in the shared Hugging Face cache rather than being copied
into this checkout. `weights/download-manifest.json` records completed model
groups and checksum verification.

```bash
tail -f outputs/runtime/model-download.log
ps -p "$(cat outputs/runtime/model-download.pid)" -o pid,etime,command
```

At the continuation checkpoint, Hugging Face login works but the official
DINOv3 repository still returns HTTP 403: access is awaiting author review.
No substitute checkpoint has been used. Full image-to-GLB generation remains
unvalidated until access is granted and all checkpoints are present.

After the current public download exits, the same script can resume missing
files and include DINOv3:

```bash
.venv/bin/python scripts/download_models.py --verify
```

NAF's official `.pth` checksum is verified separately. The runtime requires its
one-time safetensors conversion; PyTorch is only needed for this conversion,
not installed into the inference environment:

```bash
uv run --no-project --python .venv/bin/python --with torch --with safetensors python -c 'import torch; from safetensors.torch import save_file; save_file(torch.load("weights/naf_release.pth", map_location="cpu", weights_only=True), "weights/naf_release.safetensors")'
```

## Next validation and architectural work

1. Finish public downloads, obtain DINOv3 access, and validate a 512-resolution
   checkpointed run before attempting 1024 resolution on this machine.
2. Measure stage timings, process RSS, Metal allocations, memory pressure, and
   swap alongside output quality. Compare quantization modes separately from
   changes to sampling steps or resolution.
3. Consider fusing NAF evaluation with sparse bilinear projection. The current
   kernel removes neighborhood intermediates but still emits a dense
   `512 x 512 x 1024` feature map, approximately 1 GiB in FP32. Evaluating only
   the four image neighbors required per active projected voxel could remove
   this remaining dense output, subject to numerical equivalence tests.
4. Evaluate timestep-invariant per-block conditioning caches only against
   their additional unified-memory cost; do not assume more caching is faster
   on an 18 GB machine.
