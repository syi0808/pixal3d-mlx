# Frozen-input Transformer sampling investigation

## Scope

Apple M3 Pro / 18 GiB, MLX 0.31.2. The primary experiments keep pretrained
weights, FP32 activations/sampler state, 12 Euler steps, guidance, global
attention, and token selection unchanged. SS weights use the production FP32
loading path; HR weights retain their production FP16 storage with FP32
activations. There is no new precision conversion, quantization, or training.

**Production generation defaults are unchanged.** The new code is isolated in
`trellmlx/transformer_experiments.py` and `scripts/investigate_sampling.py`.
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
