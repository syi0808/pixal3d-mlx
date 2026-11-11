# Frozen-input Transformer sampling investigation

The experiments and hardware audit below describe their historical execution
paths. Pixal3D generation/resume now enable only the narrower FFN/pointwise
selective compilation path by default; whole-block and QK/RoPE compilation
remain experimental. See the [production integration measurements](performance-followup.md#production-integration-2026-11-11).

## Scope

Apple M3 Pro / 18 GiB, MLX 0.31.2. The primary experiments keep pretrained
weights, FP32 activations/sampler state, 12 Euler steps, guidance, global
attention, and token selection unchanged. SS weights use the production FP32
loading path; HR weights retain their production FP16 storage with FP32
activations. There is no new precision conversion, quantization, or training.

**Production generation defaults are unchanged.** The new code is isolated in
`pixalmlx/transformer_experiments.py` and `scripts/investigate_sampling.py`.
The previously generated high-poly, 5K, and 20K GLBs are not modified.

The frozen source is the turnip / seed 42 run:

- SS: 4,096 dense tokens, 1,536 channels, 30 blocks, MLP width 8,192;
  projected input width 1,024. Twelve steps execute 22 model forwards with
  the existing guidance interval.
- HR: 2,185 sparse tokens, the same Transformer width/depth, projected input
  width 2,048. Twelve steps execute 21 model forwards.
- The prepared image and saved native-MoGe camera are reused. Image features,
  explicit noise, and sparse/concat conditioning are saved to NPZ before any
  comparison. SS reference decoding exactly reproduces the archived 2,333
  LR coordinates; reference HR denormalization reproduces the archived shape
  latent exactly.

Model loading, image-feature extraction, and decoder validation are excluded
from sampling time. Outputs are evaluated and GPU execution synchronized
before stopping the timer. MLX peaks reset after weights/inputs are loaded.
These isolated processes do not retain the pipeline's DINO/decoder model
caches, so their memory peaks must not be compared directly with the original
8.34 GB whole-pipeline peak. Process lifetime footprint includes loading and
validation. Memory accounting domains overlap and must not be added.

Each case uses a fresh process and no concurrent GPU benchmark. Whole-sampler
measurements include the first forward's compilation/static-cache work, but
are not a claim about a cold machine's Metal shader cache. Reports snapshot
revision, dirty-tree status, and source hashes at execution start.
The initial forward-screen reports predate the start-snapshot enhancement;
their source remained fixed during the screening sequence. GB figures are
decimal; cache settings expressed in MiB use binary units.

## Conditional-forward screening

One warm-up/cold forward and two timed forwards per setting, at the real
first SS timestep and fixed inputs. The table shows the median of those two
timed forwards. This screens candidates; it is not a full sampler benchmark
or a confidence interval.

| SS execution path | Warm forward seconds | Peak MLX GB | Difference from reference forward |
| --- | ---: | ---: | --- |
| Reference, FFN chunks 2,048, evaluate every two blocks | 4.450 | 6.334 | — |
| Async FFN chunks, same block backpressure | 4.351 | 7.278 | Exactly equal |
| Unchunked FFN, evaluate every two blocks | 4.357 | 7.068 | Exactly equal |
| Unchunked FFN, evaluate every block | 4.392 | 6.430 | Exactly equal |
| Compile each pure FFN chunk | 4.324 | 6.333 | Exactly equal |
| Compile AdaLN/gated-residual pointwise functions | 4.442 | 6.334 | Exactly equal |
| Compile FFN and pointwise functions | 4.282 | 6.267 | Exactly equal |
| Conditioning cache 512 MiB, otherwise reference | 4.403 | 6.737 | Exactly equal |
| Compile pure whole blocks, evaluate every block | 4.055 | 6.246 | Max 8.46e-6, RMSE 7.13e-7 |
| Compile pure whole blocks, evaluate every two blocks | 4.019 | 6.960 | Same numerical difference |

Simply using `async_eval` or larger FFN graphs is not a free improvement:
the approximately 2% forward gain costs 0.7–0.94 GB of extra allocation.
The 512 MiB static cache adds approximately 0.40 GB for about 1% in this
conditional-only screen; the negative CFG branch already has its zero-context
cross-attention folded, so that is not a corresponding whole-sampler gain.

The reference performs two FFN chunk evaluations per block. There are 1,320
such chunk boundaries over the 22 SS forwards, in addition to model-level
evaluation boundaries. Async chunk evaluation preserves the existing
every-two-block synchronous backpressure rather than enqueueing an unbounded
whole sampler. Even so, its measured memory/time tradeoff is unattractive here.

MLX documents asynchronous evaluation in its
[API reference](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html)
and graph fusion in the [compilation guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html).
Actual measurements use the installed 0.31.2 runtime, not presumed behavior or
speed of the newer documentation release.

## Full 12-step sampling

First observations with the same frozen inputs and schedules:

| Stage/path | Sampling seconds | Peak MLX GB | Output check |
| --- | ---: | ---: | --- |
| SS reference | 93.568 | 6.336 | Archived LR coordinates equal |
| SS compiled FFN + pointwise | 89.537 | 6.269 | Final latent exactly equal |
| SS pure whole-block compile | 84.817 | 6.248 | Latent differs; LR coordinates equal |
| HR reference | 49.007 | 3.610 | Archived shape latent exactly equal |
| HR compiled FFN + pointwise | 45.873 | 3.605 | Final latent exactly equal |
| HR pure whole-block compile | 42.693 | 3.605 | Nonzero latent differences |

The narrower compiled path saves 4.31% / 4.03 s on SS and 6.40% / 3.13 s on
HR for these observations, with exact final latent arrays. This is the
lower-risk candidate for wider asset/seed/shape testing. Exactness on these
cases does not prove bitwise equality for every shape or dtype.

The whole-block path saves 9.35% on SS and 12.88% on HR in the first
observations, but is **not an exact-output optimization**:

- SS final normalized latent: maximum absolute error 6.58e-5, RMSE 1.72e-6.
  The decoded 32-cubed LR coordinates exactly match the original 2,333 active
  voxels, with zero changed voxels and IoU 1.0 for this input.
- HR normalized latent: maximum absolute error 5.99e-4, RMSE 7.81e-6.
  After denormalization, maximum error is 0.0032263 and RMSE 4.13e-5. Geometry
  subdivision/edge decisions can be threshold-sensitive; no new mesh or visual
  quality equivalence has been established for this HR variant.

Pure compiled blocks separate static conditioning/cache mutation from the
compiled tensor graph. Explicit global K/V, per-token projection, and the
known-zero unconditional constant are prepared outside it. Positive and
negative branches have separate pure compiled functions. FFN chunking is
disabled inside these blocks, with a synchronous boundary after each block
to bound the working set. GEMM and Flash Attention remain MLX operations;
this is not a persistent Metal kernel or an entire pipeline compiler.

The sampling loop, all CFG forwards, and pretrained parameters are retained.
Compilation can fuse/reassociate floating-point operations, explaining why
FP32 alone does not imply bitwise equivalence. Tests also exercise changing
image contexts in the same compiled model to detect stale conditioning capture.

SS was repeated in fresh processes with identical inputs:

| Path | Sampling seconds, two observations | Median seconds |
| --- | --- | ---: |
| Reference | 93.568, 93.529 | 93.548 |
| Pure whole-block compile | 84.817, 84.864 | 84.841 |

The repeated median reduction is 9.31% / 8.71 s. Both reference latents match
exactly; both compiled runs produce the same reported error metrics and the
same archived LR coordinates. These are sequential repetitions on one image
and seed, not an image-quality benchmark or statistical confidence interval.
All full-sampler cases in the table, their SS repeats, and the eight-step
case below complete with normal pressure and no observed new system swap.

## Separate eight-step tradeoff

Changing only the SS schedule to eight steps, with the reference FP32
execution path and frozen noise/conditioning, takes **63.620 s**: about 32%
less than the 12-step reference median. The schedule runs 15 model forwards
instead of 22. Peak MLX allocation remains approximately 6.336 GB: fewer
steps do not materially reduce the per-forward working set or weight memory.

This is an algorithm/quality tradeoff, not execution-equivalent optimization:

- Final latent maximum absolute difference 1.8285, RMSE 0.08354.
- LR active voxels change from 2,333 to 2,323.
- 44 voxel positions change occupancy; IoU with the original is 0.98128.
- Occupancy IoU is not a 3D visual-quality score. Changed structure/coordinates
  can change subsequent conditioning and generated geometry. No new GLB or
  visual quality comparison was produced for this schedule.

The experiment does not require retraining, but is **not enabled by default**.
The research CLI's `--steps` defaults to 12; the eight-step case explicitly
uses `--steps 8` and is reported separately from compilation variants.

## Where the remaining work is

The first real conditional SS block, at 4,096 tokens, has the following
warm-input medians from five timed repeats after a warm-up:

| Component | Milliseconds |
| --- | ---: |
| AdaLN normalization/modulation | 1.413 |
| Self-attention, including projections/QK normalization/RoPE | 66.085 |
| Attention gated residual | 1.187 |
| Cross-attention with cached image projection | 13.361 |
| FFN, including both GEMMs and GELU | 58.269 |
| Whole block | 145.315 |

These component calls have separate evaluation boundaries and materialized
inputs. They are not a GPU timeline profile; do not add them or multiply them
into an asserted whole-pipeline time. They show that the main costs are the
Transformer attention/FFN computations, not merely the scalar Euler update or
host synchronization.

Next priorities:

1. Validate the narrower exact-observed compiled path across LR, texture,
   larger 1024 token counts, more images, and seeds before production adoption.
2. Check shape-decoder subdivision and mesh/texture quality for pure compiled
   HR blocks. Tiny latent error and unchanged SS occupancy are not enough to
   prove downstream geometry equivalence.
3. Investigate fused QK normalization/RoPE/layout preparation while retaining
   full attention. Measure it independently; replacing MLX GEMM/Flash
   Attention wholesale is not justified by these results.
4. Keep asynchronous work bounded and stage/token-aware. Evaluate batch
   throughput and memory headroom separately from single-asset latency.
5. Treat selective mixed precision, quantization, and fewer steps as separate
   accuracy/quality experiments. SS has a known FP32 stability requirement.
   Reusing weights or avoiding retraining does not mean unchanged outputs.

There is no new end-to-end generation-time claim. In particular, a 13% HR
sampling gain is not a 13% reduction of the 333 s full turnip generation.

## Hardware bottleneck audit: 2026-11-11

### Conclusions and measurement limits

The dominant resident Transformer work is **compute/kernel-throughput limited,
not an autoregressive weight-streaming workload**. Capacity is a separate,
real constraint: loading the full SS model on the current busy desktop causes
compression and, in one run, substantial new system swap. This does not imply
that every subsequent forward is SSD-swap-bound. QK preparation and FFN
intermediates remain useful memory-traffic/kernel-fusion targets.

The audit uses MLX 0.31.2 on the actual Apple M3 Pro: 11 CPU cores, 14 GPU
cores, 19,327,352,832 bytes of physical unified memory (18 GiB), and a Metal
recommended working set of 14,302,248,960 bytes. AC power is connected and
`pmset -g therm` reports no recorded thermal/performance warning; GPU clocks
and sustained thermal throttling have **not** been measured. Two external
displays and other desktop apps remain active. Apple's specification lists
[150 GB/s memory bandwidth](https://support.apple.com/en-au/117736), shared
by the system rather than dedicated GPU VRAM.

These are empirical roofline/operator measurements, not a direct DRAM-busy,
ALU-occupancy, or hardware-stall counter attribution. Instruments' default
Metal System Trace exposes GPU execution intervals, but its exported counter
metadata contains only `RT Unit Active`, not the required DRAM/ALU counters.
Unprivileged `powermetrics` refuses to run. No privilege, memory-limit, power,
precision, production model, or sampling-default changes are made.

### Empirical roofline

Synthetic warm resident tensors are evaluated and synchronized before stopping
each timer. Two fresh-process sweeps use seven and nine timed repeats after
one warm-up. Short sweeps show some clock/scheduling variability; these are
not sustained peak specifications or confidence intervals.

| Measurement | Observed |
| --- | ---: |
| 256 MiB explicit Metal copy, read + write | 123.2 / 126.4 GB/s |
| 256 MiB elementwise add, two reads + write | 126.5 / 127.9 GB/s |
| FP32 FFN-up GEMM, M=4096, N=8192, K=1536 | 4.08 / 3.84 TFLOP/s |
| FP16 same GEMM | 4.37 / 4.38 TFLOP/s |
| Already-evaluated array + synchronization | 0.016 / 0.017 ms |
| One-element add + evaluation + synchronization | 0.131 / 0.145 ms |

Streamed-byte rates count logical reads/writes, not measured external-memory
transactions. The 256 MiB input alone is much larger than small operator
working sets; the smaller 64 MiB copy is more variable and is not used as the
bandwidth ceiling. The larger add reaches about 85% of nominal bandwidth.

An SS FFN-up matrix multiply performs **103.079 GFLOP**. Its ideal minimum
tensor traffic is 209.715 MB, counting one read of each operand and one output
write. Arithmetic intensity is **491.52 FLOP/byte**, versus the empirical
ridge of approximately **30–33 FLOP/byte** from these compute/stream tests.
At 127.9 GB/s its ideal memory-only time is 1.64 ms; the real first-block
GEMM takes 26.45 ms at 3.90 effective TFLOP/s. Tiling rereads, casts and caches
mean the minimum is not actual DRAM traffic, but the intensity, GEMM plateau,
and M-scaling strongly favor a compute-throughput diagnosis over a global
weight-bandwidth bottleneck.

For contrast, the same FP32 weight matrix with M=1 takes about 0.55 ms, versus
25–27 ms for 4096 rows. Thousands of tokens reuse the weights within a forward;
the latency is not 4096 separate weight-streaming GEMVs. Consequently, LLM
single-token-decode heuristics such as expecting proportional speedups from
INT4 weight compression do not transfer to this DiT workload.

### Actual first-block operators and dtype promotion

Only the first real pretrained block and the input/timestep projections are
loaded for operator isolation. SS uses the frozen 4096-token input; HR uses
2185 tokens. Other 29 blocks and pipeline model caches are absent. Precomputed
intermediate tensors are materialized before each individual operator test.
Separate operator timings must not be summed or multiplied into a claimed
full-sampler latency. SS repeat 2 uses nine timed observations:

| SS operator | Median milliseconds |
| --- | ---: |
| QKV projection | 15.23 |
| Q/K normalization, both tensors | 3.91 |
| Q/K RoPE, both tensors | 6.58 |
| Fused full self-attention core | 36.56 |
| Attention output projection | 5.20 |
| Cached conditional cross-attention | 15.63 |
| FFN up projection | 26.45 |
| FFN GELU | 5.40 |
| FFN down projection | 30.43 |
| FFN with 2048-row chunks | 62.19 |
| Whole block with 2048-row chunks | 159.01 |

The preceding SS observation gives 66.88 ms for full self-attention and
70.71 ms for chunked FFN, versus 73.73 / 62.19 ms in repeat 2. This variability
is another reason not to treat a component table as a production GPU timeline.

HR stores FFN weights in **FP16**, but its input activation, QKV output, all
SDPA inputs, and FFN hidden output are **FP32**. Mixed input/weight promotion
therefore preserves FP32 computation. Smaller checkpoint/storage dtype is not
proof of FP16 compute. Even genuinely FP16 synthetic GEMM is only about 7–14%
faster for this shape in these sweeps, not an assumed 2x hardware speedup.

Synthetic contiguous-input SDPA isolates scaling without model loading:

| Tokens | FP32 milliseconds | FP16 milliseconds |
| --- | ---: | ---: |
| 1024 | 2.40 | 1.89 |
| 2185 | 12.32 | 7.80 |
| 4096 | 35.38 | 26.06 |
| 10784 | 261.85 | 178.93 |

The 4096-to-10784 token change is 2.63x in length, 6.93x in QK/PV FLOPs, and
7.40x in measured FP32 time. Fused attention avoids allocating the full score
matrix, but does not remove quadratic computation. Effective QK/PV throughput
is 2.73–2.91 TFLOP/s for these larger FP32 cases; the count excludes softmax.
No direct register-pressure/occupancy attribution is established. Neither
replacing global attention nor enabling reduced precision follows from this
measurement without separate quality validation.

### GPU timeline: gaps are secondary, not zero

A separate Instruments Metal System Trace records the warmed, chunked first
SS block in a bounded eight-second replay. In the final five seconds of target
GPU execution, from trace time 6.245989083 to 11.245989083 seconds:

- Union of target Python GPU execution intervals: **4.391442 s / 87.83%**.
- Union of all processes' GPU execution intervals: **4.456210 s / 89.12%**.
- The remaining intervals are GPU-uncovered time, not automatically Python
  overhead; CPU dispatch, synchronization, scheduling and profiling can all
  contribute. Concurrent desktop GPU work is present.

Interval unions avoid double-counting overlapping/nested submissions. These
figures measure time with submitted GPU execution, **not** 87.83% ALU
utilization or memory-controller utilization. The trace is instrumented and
its block timings are not the uninstrumented timing baseline. Even ideal
gap elimination has limited scope here; synchronization is not a credible
explanation for the majority of current Transformer latency.

### Capacity, compression, swap and file I/O are different

The initial desktop snapshot already uses approximately 1.35 decimal GB of
swap. Small operator tests show no new swap-used growth and normal pressure;
that does **not** prove a full resident model has enough RAM.

The first new full SS screen retains **5.358 GB of parameters**, peaks at
**6.334 GB MLX allocation / 6.683 GB process footprint**, and takes 30.06 s
including loading and three conditional forwards. System swap-used rises by
**1.920 GB**. Swapouts increase by 117,164 page units; pressure reaches warning.
The sampled swapout burst is concentrated at approximately 7.6–15.3 s while
the resident model/workspace grows, then stops. The two warm forwards take
4.682 / 4.697 s. This is real capacity contention on this desktop, but the
system-wide counters do not identify whose pages are evicted.

A second, explicitly phase-scoped full-model replay follows on the already
altered desktop/cache state. It is **not** a controlled cold/warm speedup:

| Scope | Seconds | System Swapins / Swapouts | Process disk-read bytes |
| --- | ---: | ---: | ---: |
| Model construction/loading/evaluation | 9.754 | 0 / 0 | Not separately sampled |
| First forward | 5.064 | 32 / 0 | 622,592 |
| Warm forward 1 | 5.036 | 4 / 0 | 0 |
| Warm forward 2 | 4.950 | 4 / 0 | 0 |

Neither warm forward incurs process pageins in `proc_pid_rusage`. Four system
Swapins are only 64 KiB at the reported 16 KiB page size, not multi-GB SSD
thrashing. Conversely, warm forward 1 coincides with 130,330 system
decompressions, approximately 2.14 GB of logical page units, despite no process
disk reads. Compression/decompression is RAM/CPU work and must not be mislabeled
as SSD swap I/O. These counters are system-wide and cannot quantify the exact
latency attributable to this process.

Checkpoint/file `Pageins` also must not be treated as `Swapins`: the scoped
load has 327,247 file/page-in units but zero Swapins. Larger RAM and shorter
model/cache lifetimes can improve admission, compression pressure and reloads;
they do not eliminate the resident GEMMs and quadratic attention. The older
[full 1024 run](1024-validation.md) separately shows that decoder expansion can
be the peak-capacity stage even when Transformers dominate elapsed time.

### Additional optimization headroom

1. **QK normalization + RoPE fusion is a measured candidate.** A pure
   `mx.compile` probe reduces their combined first-block cost from 11.80 to
   **7.08 ms / 40.0%**, with maximum absolute difference **1.43e-6**. The
   improvement is local, not a full-sampler saving. It is not bitwise equal,
   is not installed into attention, and has no downstream geometry/texture
   validation. A fused layout-aware Metal kernel is a reasonable next
   experiment, not an already demonstrated gain.
2. **FFN activation/epilogue traffic remains worth targeting.** The unchunked
   FP32 hidden tensor is 134.2 MB; standalone GELU costs 5.4 ms. Fusion or
   carefully tuned bounded chunks can reduce intermediate traffic. GEMM
   itself is already near the empirical large-matrix compute plateau, so
   rewriting matrix multiplication is a lower-priority, higher-risk task.
3. **Keep full-attention kernel tuning separate from approximation.** Larger
   token counts disproportionately increase SDPA computation. True FP16 SDPA
   is faster synthetically, but dtype, FP32 SS stability, and downstream
   quality require explicit validation; FP16 weight storage alone is not it.
4. **Budget model admission against the busy desktop, not just MLX bytes.**
   Shorten backbone/decoder/cache lifetimes and measure loading separately
   from resident execution. Avoid keeping all multi-GB flows resident on
   18 GiB, raising the memory guideline as a supposed capacity fix, or
   enabling parallel generations. A cache/quantization memory win is not
   automatically an inference-throughput win.
5. **Do not prioritize removing every host boundary.** The GPU timeline and
   earlier async/unchunked ablations indicate smaller potential than the
   main dense computations, with an observed memory-liveness penalty.

Raw reports, scoped memory samples, logs, exported GPU XML and the Instruments
trace are retained under `outputs/runtime/hardware-investigation/`. Reports
preserve per-repeat observations, device metadata, revision/dirty-tree status,
source hashes, input hashes, and system counter deltas. The diagnostic script
evolves during the audit; use each report's source hash rather than assuming
all observations used the final script text. Generated GLBs and production
inference defaults remain unchanged. There is no new image-to-3D latency claim.

Hardware diagnostics reproduction, with fresh report paths:

```bash
PYTHONPATH=. .venv/bin/python scripts/measure_generation.py \
  --output-dir outputs/runtime/hardware-new --interval 0.5 -- \
  .venv/bin/python -u scripts/investigate_hardware.py roofline \
  --repeats 9 --report outputs/runtime/hardware-new/roofline.json

PYTHONPATH=. .venv/bin/python scripts/investigate_hardware.py block \
  --stage ss --repeats 9 --report outputs/runtime/hardware-new/ss-block.json

PYTHONPATH=. .venv/bin/python scripts/measure_generation.py \
  --output-dir outputs/runtime/hardware-forward-new --interval 0.5 -- \
  .venv/bin/python -u scripts/investigate_hardware.py forward --stage ss \
  --repeats 2 --report outputs/runtime/hardware-forward-new/ss.json

xcrun xctrace record --template 'Metal System Trace' --time-limit 20s \
  --no-prompt --output outputs/runtime/hardware-new/ss.trace \
  --env PYTHONPATH="$PWD" --launch -- "$PWD/.venv/bin/python" \
  "$PWD/scripts/investigate_hardware.py" block --stage ss --repeats 1 \
  --trace-seconds 8 --report "$PWD/outputs/runtime/hardware-new/traced.json"

xcrun xctrace export --input outputs/runtime/hardware-new/ss.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output outputs/runtime/hardware-new/gpu.xml

PYTHONPATH=. .venv/bin/python scripts/investigate_hardware.py trace \
  --gpu-intervals outputs/runtime/hardware-new/gpu.xml \
  --report outputs/runtime/hardware-new/timeline.json
```

Run these sequentially, not concurrently with other GPU benchmarks. Optional
`block --capture PATH.gputrace` additionally requires `MTL_CAPTURE_ENABLED=1`,
as described by the [MLX Metal debugger guide](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html).
The present evidence uses Instruments, not a separate `.gputrace` capture.

Hardware diagnostic tests: **8 passed**, covering GEMM work accounting,
swap-used versus swap-I/O deltas, evaluation/repeat counting, interval clipping
and overlap unions, exported XML references, and phase-scoped full forwards
on a tiny model. The full audit suite reports **429 passed**, with the same
three pre-existing README onboarding failures listed below. Raw test output is
`outputs/runtime/hardware-investigation/full-tests.log`.

## Reproduction

All frozen inputs, normalized outputs, per-case reports, process/system
measurements, and logs are retained in `outputs/runtime/sampling-investigation/`.
Use fresh report/output paths; evidence is not overwritten.

```bash
PYTHONPATH=. .venv/bin/python scripts/investigate_sampling.py prepare --stage ss

PYTHONPATH=. .venv/bin/python scripts/measure_generation.py \
  --output-dir outputs/runtime/ss-compile-new --interval 1 -- \
  .venv/bin/python -u scripts/investigate_sampling.py sample \
  --stage ss --variant compiled-both \
  --reference outputs/runtime/sampling-investigation/ss-reference.npz \
  --report outputs/runtime/ss-compile-new/sampling.json

PYTHONPATH=. .venv/bin/python scripts/investigate_sampling.py profile \
  --stage ss --repeats 5 --report outputs/runtime/ss-block-profile-new.json
```

Preparation refuses to replace existing frozen inputs; pass `--inputs` for a
new path. Other compatible runs can specify `--checkpoints` and their saved
preprocessed `--image`. `--stage hr` / `texture` freeze corresponding sparse
conditioning; those models are not substituted for the dense SS model.

## Validation

- Transformer experiment tests: 22 passed, covering dense/sparse models,
  unchanged parameter arrays/names, FP32 sampler state, real async chunks,
  positive/negative conditioning, multiple timesteps, and changing image
  contexts in the same pure compiled model.
- Focused Transformer/sampling suite: 31 passed.
- Full suite: 368 passed and the same three pre-existing README onboarding
  failures: `test_readme_uses_current_huggingface_cli`,
  `test_readme_tests_section_avoids_stale_exact_count`, and
  `test_readme_quantization_section_states_measured_tradeoff`.
- Frozen full-sampler references reproduce the archived SS coordinates and
  HR latent; numerical errors and occupancy changes are reported for every
  non-exact variant rather than treated as equivalent output quality.
