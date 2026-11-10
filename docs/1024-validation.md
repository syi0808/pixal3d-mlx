# Native MoGe / 1024 validation on the 18 GB M3 Pro

## Result

**Generation completed, but this is a memory-constrained operating point, not
a comfortable all-in-RAM configuration.** The process exported a valid GLB at
the requested geometry resolution without quantization, mixed-precision opt-in,
sampling-step reduction, resolution fallback, or a raised MLX memory setting.
System swap increased and a brief critical-pressure sample was observed.

This is one shoe input and one active generation on this particular desktop.
It does not establish that all 1024 inputs, or concurrent jobs, fit in 18 GB.

## Conditions

- Date: 2026-11-10, 20:38:38 to approximately 20:53:30, Asia/Seoul.
- Apple M3 Pro, 14 GPU cores, physical memory 19,327,352,832 bytes (18 GiB),
  macOS 26.2, MLX 0.31.2.
- Revision: `85cfd22d396da2af687bd1735bb8e54d890c415b`.
- Input `assets/shoe_input.png`, seed 42, geometry resolution 1024, exported
  textures 1024 x 1024, default target of 200,000 faces.
- Native safetensors MoGe path: FOV 36.9804 degrees, distance 1.4951931.
  The log confirms `native checkpoint` and contains no MoGe fallback.
- Original pretrained weights, 12-step schedules, FP32 SS and sampler state,
  no INT4/INT8 quantization, no experimental mixed-precision mode.
- Default sparse NAF projection, Morton decoder storage, FFN chunks of 2048,
  and a configured 10,000,000,000-byte MLX evaluation guideline.
- NAF output feature resolution remains the port's bounded 512 setting.
  “1024” here is the geometry grid, not a claim that every intermediate feature
  map or texture is rendered at a higher resolution.
- The system already had approximately 5.02 GB of swap in use before the run.
  Other desktop applications were not closed or otherwise controlled.

## Measurements

All GB values below are decimal, 1 GB = 1,000,000,000 bytes. Different accounting
domains overlap and **must not be added together**.

| Measurement | Observed |
| --- | ---: |
| Child-process elapsed time | 892.18 s, 14 min 52 s |
| Pipeline-reported elapsed time | 891.6 s |
| Peak MLX allocation | 11,438,883,722 bytes, 11.44 GB |
| Peak macOS process footprint | 13,999,821,768 bytes, 14.00 GB |
| Child high-water RSS | 7,790,428,160 bytes, 7.79 GB |
| Highest sampled RSS | 7,731,871,744 bytes |
| System swap before | 5,024,713,277 bytes |
| Highest system swap during run | 6,256,129,474 bytes |
| System swap after | 6,239,352,258 bytes |
| Maximum swap-used growth | 1,231,416,197 bytes, 1.23 GB |
| Memory-pressure warning samples | Approximately 19.3 s combined |
| Critical-pressure samples | One sample, approximately 1 s |
| Safety watchdog termination | None |
| Exit code | 0 |

The memory monitor samples `proc_pid_rusage` version 4 once per second. The
footprint figure uses its lifetime high-water field, not only a maximum of
instantaneous samples. RSS high-water comes from `getrusage(RUSAGE_CHILDREN)`
in a fresh monitoring process. `vm_stat` and `sysctl` provide system-wide
compression, swap, and pressure observations. Pressure durations are sampling
estimates; shorter events between samples may be missed.

System `Swapouts` increased by 77,720 page units and `Swapins` by 4,113 during
the run. These and the swap-used totals are **system-wide**, so the observations
are not exclusive per-process swap attribution. The important distinction from
the earlier partial-progress updates is that the major increase happened near
the final texture decode, after those updates.

The watchdog was configured to stop this process if its observed current
footprint exceeded 14 GB, critical pressure persisted for 10 s, or warning
pressure plus at least 2 GB of new swap persisted for 30 s. The lifetime peak
was very close to the footprint threshold. Completing without watchdog action
is not a claim of generous memory headroom.

## The 10 GB setting is not a hard cap

The run exceeded its configured MLX value by **1,438,883,722 bytes**. The local
MLX 0.31.2 API docstring and the [official API documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html)
describe `set_memory_limit` as an evaluation guideline: allocations can exceed
it while RAM or swap remains available. The earlier documentation's hard-cap
interpretation was incorrect and has been corrected.

Runtime reports now include `memory_budget_is_hard_limit: false`, an exceeded
flag, and excess bytes. The historical raw report for this experiment is
preserved unchanged. The external pressure watchdog is separate from MLX's
guideline and does not force every individual allocation to remain below it.

## Stage timing and bottlenecks

| Runtime stage, including its loading/cleanup work | Seconds |
| --- | ---: |
| Preprocess, native camera, backbone loading | 7.29 |
| Sparse structure | 105.55 |
| LR conditioning and shape | 66.39 |
| Coordinate subdivision | 5.05 |
| HR conditioning and shape | 367.47 |
| Shape decode and mesh processing | 70.78 |
| Texture sampling | 220.71 |
| Texture decode and export | 48.33 |

Within those stages, HR flow sampling took 354.8 s and texture flow sampling
211.0 s. The two enclosing stages account for about 66% of pipeline time.
They are the primary latency targets.

The shape decoder raised the running MLX peak to 10.97 GB; the texture decoder
raised it to 11.44 GB. The highest process footprint and critical-pressure
sample occurred near the last texture-decoder expansion, approximately 857 s
after launch. Decoder working sets, rather than the HR transformer stage,
are the immediate memory target for a stricter 1024 memory budget.

## Output validation

- The saved latent metadata reports `hr_resolution: 1024`; the log reports
  projection grid 64 and 10,784 HR tokens. No reduced-resolution message occurs.
- Full decode produces 2,814,231 active voxels.
- Exported GLB contains 152,293 vertices and 200,301 faces after cleanup,
  simplification, and UV splitting.
- Vertex coordinates are finite, geometry is nonempty, and both base-color
  and metallic/roughness maps load at 1024 x 1024.
- A nonblank three-view witness render is retained. These checks do not prove
  watertightness or numerical/visual equivalence to an older runtime.

Artifacts under `outputs/runtime/e2e1024-native/`:

- `shoe.glb`, `shoe-witness.png`, `validation.json`, `witness.json`.
- `runtime.json`: MLX peaks, effective arguments, and stage timings.
- `resources.json`: process/system summary and exact command.
- `resources.jsonl`: 880 raw process/system samples.
- `generation.log` and intermediate `checkpoints/`.
- `memory-timeline.png`: footprint/RSS timeline with pressure annotations.

The previous 512 run took 368.7 s with 8.17 GB peak MLX allocation, but used
the default 49.1-degree FOV, not native MoGe. Its first background-removal
download was also included. It is therefore **not** a controlled
resolution-only speed comparison.

## Reproduction

Use a fresh output directory; the monitor refuses to overwrite prior evidence.

```bash
.venv/bin/python scripts/measure_generation.py \
  --output-dir outputs/runtime/e2e1024-native --interval 1 -- \
  .venv/bin/python -u generate_pixal3d.py \
  --image assets/shoe_input.png --seed 42 --resolution 1024 \
  --texture-size 1024 --memory-budget-gb 10 \
  --output outputs/runtime/e2e1024-native/shoe.glb \
  --save-checkpoints outputs/runtime/e2e1024-native/checkpoints \
  --runtime-report outputs/runtime/e2e1024-native/runtime.json
```

Conclusion: **1024 can finish on this 18 GB Mac for this input with the optimized
single-job runtime, but it uses swap and briefly reaches critical pressure.**
Treat it as a constrained mode, not proof of swap-free operation or reliable
headroom for larger assets or concurrent generation.

Follow-up frozen-latent memory ablations are in
[decoder-investigation.md](decoder-investigation.md). Those isolated replays do
not replace this full-pipeline measurement or establish a new end-to-end peak.
