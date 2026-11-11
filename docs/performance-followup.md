# Performance follow-up and strict polygon budgets

The initial observations below predate production adoption. Selective
FFN/pointwise compilation and vectorized CPU topology are now integrated;
see [the production comparison](#production-integration-2026-11-11).

## Completed 5,000-triangle turnip

The background checkpoint-only export produced
`outputs/turnip5k/turnip-5k.glb`: **4,999 triangles**, 6,126 UV-split vertices,
2,184,444 bytes, embedded 1024 x 1024 base-color and metallic/roughness maps.
The GLB was reloaded and its total triangle count, finite positions/UVs, and
texture dimensions were checked before the output was written.

The existing 201,195-triangle model is preserved. No image encoder, camera
estimator, flow sampler, or decoder was rerun for the 5K asset: the saved
mesh and voxel texture attributes were reused, with new simplification, UVs,
and baking. This preserves the same generated object/seed, not every fine
geometric feature of the high-poly mesh.

| Checkpoint re-export stage | Seconds |
| --- | ---: |
| Checkpoint load | 0.049 |
| Strict simplification | 0.369 |
| UV unwrap | 0.059 |
| Texture bake | 2.008 |
| Export and validation | 0.155 |
| Internal elapsed | 2.639 |
| Whole monitored process | 4.050 |

Peak MLX allocation is 3,929,179,072 bytes and process lifetime footprint
4,739,697,784 bytes. Pressure remains normal, with no observed new system
swap. These are different overlapping memory-accounting domains. The 2.639 s
is **not** an image-to-3D generation time or a speedup of neural inference.

### Why a strict cap needs more than `--target-faces`

The existing `--target-faces` remains a soft simplification target: subsequent
hole filling/repair can add faces. The original 200,000 target exported 201,195
triangles. New `--max-faces` is an opt-in final cap, applied after face-adding
repairs and checked again after UV unwrapping. It defaults to 0 / disabled.
It is forwarded through checkpoint resume and included in runtime arguments.
Counts refer to **triangles**, not quads or vertices; UV seams can increase
vertex count without increasing triangles.

The first conservative re-export attempt correctly failed rather than
publishing an over-budget model: repeated direct edge-collapse simplification
stalled at 7,159 faces on this highly split mesh. The successful fallback
rounds normalized positions to seven decimal places, welds coincident points,
and removes duplicate/degenerate faces before retrying simplification. The
rounding alone changes each coordinate by at most approximately 5e-8 model
units; simplification itself has a much larger, intentional geometry effect.
This can change connectivity and does not guarantee watertightness or exact
visual preservation. No arbitrary triangle truncation or blanket removal of
small connected parts is used. Failure to meet the cap still raises an error.

The installed fast-simplification 0.1.13 documentation also describes a target
as something that may not be reached for some aggressiveness/settings; a final
count assertion is necessary for a strict limit. See the
[project API documentation](https://pyvista.github.io/fast-simplification/_autosummary/fast_simplification.simplify_mesh.html).
The local installed implementation, not the newer website version, was used.

Re-export without neural inference, using a fresh output path:

```bash
PYTHONPATH=. .venv/bin/python reexport_pixal3d.py \
  --ckpt outputs/turnip512/checkpoints --max-faces 5000 --texture-size 1024 \
  --output outputs/turnip5k-new/turnip-5k.glb \
  --report outputs/turnip5k-new/export.json
```

For a new image generation, use both `--target-faces 5000 --max-faces 5000`.
The former reduces intermediate postprocessing work; the latter enforces the
upper bound after repair. Neither reduces the model's token count, denoising
steps, or the dense mesh initially decoded by the neural network.

Raw evidence: `outputs/turnip5k/export.json`, `run2/{resources.json,
resources.jsonl,generation.log}`. The failed first attempt is retained in the
parent directory's measurement files. The successful export report correctly
records a dirty working tree while the cap implementation was being developed;
the subsequently committed implementation is `1b1d043`.

## Measured UV cost versus polygon budget

All cases start from the same already-cleaned 201,195-face checkpoint, not
from a new neural generation. The strict-cap path changes geometry and UVs.

| Strict cap | Actual triangles | Simplify seconds | UV seconds | Soft target after repair |
| --- | ---: | ---: | ---: | ---: |
| Original | 201,195 | — | 18.337 | — |
| 5,000 | 4,999 | 0.599 | 0.056 | 5,006 |
| 10,000 | 10,000 | 0.102 | 0.078 | 10,017 |
| 20,000 | 20,000 | 0.120 | 0.275 | 20,042 |
| 50,000 | 50,000 | 0.231 | 1.060 | 50,092 |

These are one observation per setting, excluding baking. The soft-target
column replays the existing cleanup/simplification helper on the cleaned
checkpoint; it is not a quality-equivalent comparison with reprocessing the
original 1.34-million-triangle raw mesh. Its purpose is to demonstrate that a
target is not an upper bound.

A subsequent requested `--target-faces 20000` export **without `--max-faces`**
uses the original raw geometry and finishes at 20,034 triangles. See
[turnip20k-validation.md](turnip20k-validation.md) for the model, exact command,
and measurements. The re-export CLI supports this soft-target-only route;
the strict-cap feature remains available but is not used for that asset.

The low-poly export primarily improves UV/baking cost, output size, and
downstream rendering. It does not turn the 333 s full generation into a 3 s
generation. Stored mesh/texture checkpoints do make additional LOD exports
dramatically cheaper than repeating geometry and texture sampling.

## Exact-output CPU cleanup optimization candidate

The actual raw turnip mesh was reconstructed from its frozen shape latent:
663,831 vertices and 1,344,004 triangles. No flow sampling was rerun. Replaying
the production cleanup reproduces the saved 105,040 vertices / 201,195 faces.

An opt-in experiment replaces two Python-heavy operations:

- Face connectivity: sort packed edge keys and connect each shared-edge group
  as a sparse star instead of building a Python dictionary and all face pairs.
  The connected-component partition is unchanged.
- Non-manifold repair: solve the same manifold-edge corner unions using a
  SciPy sparse graph. Canonical minimum corner representatives preserve the
  original vertex and face ordering, not only equivalent geometry.

The first unprofiled observations are 18.165 s for production cleanup and
9.702 s for the experimental path. Both vertex and face arrays match the
archived mesh exactly. This is an approximately 8.46 s component saving, not
a 47% speedup of full image-to-3D generation. The experiment is **not enabled
by production generation defaults**.

A separate cProfile replay takes 29.455 s and is diagnostic only: profiling
overhead makes it unsuitable as the timing baseline. It records approximately
19.05 million Python `find` and 6.31 million `union` calls, confirming the
topology loops as a real CPU bottleneck. Profiled cumulative times include
non-manifold repair 9.93 s, normal fixing 8.13 s, face connectivity 5.39 s, and
hole filling 4.53 s. Nested cumulative timings must not be added together.

Raw reports and profile: `outputs/runtime/postprocess-investigation/`.
`scripts/investigate_postprocess.py` provides `prepare`, `cleanup`, and
`budgets` modes; `--experimental` scopes and restores the experimental cleanup
functions. The raw geometry is retained for repeated CPU-only measurements.

Three fresh-process observations per path, including the first observations:

| Measurement | Production | Experimental sparse topology |
| --- | ---: | ---: |
| Median cleanup seconds | 18.390 | 9.702 |
| Cleanup seconds, range | 18.165–18.846 | 9.575–9.779 |
| Median process lifetime footprint GB | 1.551 | 1.367 |
| Match archived vertex/face arrays | All three | All three |

The median component saving is **47.2% / 8.688 s**. These sequential trials
are not a randomized confidence interval, and no whole-generation run with the
experimental backend has been measured. The existing cleanup test modules
also pass under the temporary experimental substitutions: 29 passed.

## Remaining priorities

1. **Transformer computation remains the largest latency target.** The turnip
   flow samplers total 226.6 s, approximately 68% of pipeline time: SS 93.9 s,
   LR 51.4 s, HR 49.2 s, texture 32.1 s. Low-poly export and decoder memory work
   do not remove these computations.
2. **Fuse bounded Transformer pointwise work.** AdaLN modulation and gated
   residual updates are candidates for selective `mx.compile` / Metal fusion.
   MLX documents graph simplification and operation fusion in its
   [compilation guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html).
   This is not yet benchmarked here; compiled arithmetic also needs numerical
   checks rather than assuming bitwise equivalence.
3. **Shorten backbone-weight lifetime.** The DINO checkpoint has 1,212,518,400
   tensor bytes, while the retained 512/1024 image-feature cache is only
   21,012,480 bytes. Pre-encoding both image sizes could allow releasing the
   approximately 1.21 GB backbone before SS and later projecting only cached
   features. Coordinate-dependent NAF projection still runs later. This is a
   design candidate, not a measured speedup or guaranteed peak reduction.
4. **Stage-aware model-cache admission.** The completed job retains about
   1.90 GB of decoder/other model parameters. Budgeting the incoming model and
   workspace before loading can be safer than checking only current active
   bytes; persistent batches should not keep an unused decoder through the
   next SS stage unnecessarily. Measure reload cost versus memory headroom.
5. **Normal fixing and hole filling are the next CPU targets.** Preserve
   component filtering, surface parts, winding, and final orientation policy.
   Skipping repairs or keeping only the largest part is not an equivalent
   optimization for leaf/stem geometry.
6. **Genuinely mixed precision / fewer sampling steps are quality tradeoffs.**
   The earlier fixed-shape texture experiment saved 23.3% sampling time after
   correcting RoPE dtype promotion, with nonzero errors. SS has a known
   numerical-stability reason for FP32. Do not enable these globally without
   voxel, silhouette, texture, and multi-seed validation. No retraining is
   required to experiment, but output quality is not guaranteed unchanged.

No new end-to-end speed or 1024 memory guarantee is inferred from these
component experiments. The original high-poly GLB and its measured evidence
remain unchanged.

## Validation and artifact integrity

The full suite reports **344 passed and the same three pre-existing README
onboarding failures**. The strict-cap/re-export tests cover no-op budgets,
real simplification, controlled welding fallback, nonfinite results, refusing
unmet caps instead of truncating triangles, GLB validation, no overwrites, and
CLI resume forwarding. Twelve randomized topology cases compare component
labels and repaired vertex/face arrays with the original implementation.

The 5K GLB's SHA-256 is
`059bfab5ae9f24dd6160cc72ffab86ddf8a330eaa01b3d1c075cc0b970ab4554`.
Front/back previews and a three-view mesh witness are retained in
`outputs/turnip5k/`. They render the actual low-poly GLB, not the source image;
the simple preview shades sampled triangle colors rather than doing a full
PBR render. The original high-poly GLB remains byte-identical.

## Production integration: 2026-11-11

Pixal3D generation and checkpoint resume now default to selectively compiling
pure FFN chunks, AdaLN modulation and gated residual updates. Shared compiled
functions take weights and biases explicitly rather than retaining a model.
Chunking and every-two-block evaluation remain bounded as before. Full blocks,
QK/RoPE and the Euler update are not newly compiled. The existing optional
`--compile-step` continues to control only the Euler update.

`--no-compile-transformer` disables this new default. The library sampler's
`selective_compile` argument defaults to false for compatibility; the CLI
passes its setting explicitly to all four samplers and through resume.
Reused models receive the current setting on every sampling call. Quantized
FFNs retain their packed quantized operators and do not enter the compiled
floating-point FFN path.

CPU cleanup now uses the previously isolated vectorized edge grouping and
sparse corner-connectivity implementation. Component filtering, corner
representatives, triangle/vertex ordering, simplification and normal/hole
policies are unchanged. Original Python topology remains available only as
a scoped benchmark reference. No global monkey-patching is used by production
generation.

### Same-input end-to-end observations

Both processes use `outputs/turnip512/input.png`, seed 42, resolution 512,
1024 x 1024 textures, the 200,000-face soft target and native MoGe. All four
samplers retain **12 steps**, the original guidance schedules, pretrained
weights and FP32 activation policy. No quantization, token pruning, alternate
attention, model-cache policy change, or step reduction is introduced.
The archived high-poly and LOD assets are not overwritten.

| Measurement | Before | Selective compile + vectorized cleanup |
| --- | ---: | ---: |
| Pipeline-reported seconds | 368.3 | 345.8 |
| Monitored whole-process seconds | 369.749 | 346.590 |
| SS sampling seconds | 109.9 | 106.4 |
| LR sampling seconds | 57.4 | 53.8 |
| HR sampling seconds | 51.1 | 53.3 |
| Texture sampling seconds | 34.0 | 30.5 |
| Shape decode + mesh stage seconds | 24.186 | 15.889 |
| UV unwrap seconds | 19.4 | 18.3 |
| Peak MLX allocation GB | 8.339 | 8.339 |
| Peak process footprint GB | 9.828 | 9.724 |
| Maximum new system swap-used GB | 2.904 | 1.280 |

The pipeline observation improves by **22.5 s / 6.1%**; monitored wall time by
23.16 s / 6.3%. Shape decode + mesh improves by 8.30 s. Rounded cleanup-pass
logs show pass 1 falling from 10.5 to 3.6 s; pass 2 remains about 6 s.

These are one sequential observation per configuration, not a controlled
confidence interval. The desktop/cache state differs, swap grows in both,
and HR sampling actually regresses in this pair. Preprocessing and UV time
also change despite not being optimized. Therefore not every second of the
end-to-end difference can be attributed to the two code changes, and a
universal 6.1% speedup is not claimed. MLX peak allocation is effectively
unchanged; this work is not a memory-capacity fix.

### Isolated performance controls

Fixed-input tests use the actual production classes, not the research wrappers.
Weights, input archives, 12-step schedules, guidance, precision, FFN chunking
and source hashes match within each pair. Model/input loading and downstream
decoding are outside sampling time; initial selective compilation is included.

| Component | Reference seconds | Optimized seconds | Reduction |
| --- | ---: | ---: | ---: |
| Full SS sampling, 4096 tokens, 12 steps | 94.838 | 90.421 | 4.66% |
| Full HR sampling, 2185 tokens, 12 steps | 48.950 | 46.536 | 4.93% |
| Complete raw-mesh cleanup/simplification, median of 3 | 18.799 | 9.730 | 48.24% |

The frozen raw mesh contains 663,831 vertices / 1,344,004 triangles. CPU
reference observations are 19.042 / 18.799 / 18.781 s; vectorized observations
are 9.725 / 9.748 / 9.730 s. Each uses a fresh process, with reference and
vectorized trials interleaved. Median cleanup saving is **9.068 s**. This is
not a 48% reduction of neural inference or full generation.

One initial HR-selective replay (55.491 s) overlaps a separately launched
Python-overhead benchmark and is **excluded**, not averaged into these results.
This task's benchmark queue is suspended while that independent batch finishes;
the selective HR measurement is then repeated at
`hr-production-selective-isolated/`. SS and the retained HR reference do not
overlap that independent batch. CPU comparisons and the HR retry run without
concurrent benchmark/test workloads. Desktop apps remain uncontrolled, and
sampling still has only one retained observation per path.

Isolated SS MLX peaks are 6.336 GB reference / 6.269 GB selective. These
component peaks exclude DINO and pipeline model caches and do not contradict
the unchanged 8.339 GB whole-generation peak.

### Reproduction and scope

Raw full-run reports/logs/outputs are under
`outputs/runtime/selective-cleanup-integration/{before,after}/`. Independent
fixed-input sampling and repeated raw-mesh cleanup reports use the same parent
directory. GPU benchmarks run sequentially, without concurrent cleanup or
test workloads.

```bash
PYTHONPATH=. .venv/bin/python generate_pixal3d.py \
  --image outputs/turnip512/input.png --seed 42 --resolution 512 \
  --texture-size 1024 --target-faces 200000 \
  --output outputs/runtime/new-optimized/turnip.glb \
  --runtime-report outputs/runtime/new-optimized/runtime.json

PYTHONPATH=. .venv/bin/python scripts/investigate_sampling.py time \
  --stage ss --variant production-reference --steps 12 \
  --report outputs/runtime/new-benchmark/ss-reference.json

PYTHONPATH=. .venv/bin/python scripts/investigate_sampling.py time \
  --stage ss --variant production-selective --steps 12 \
  --report outputs/runtime/new-benchmark/ss-selective.json

PYTHONPATH=. .venv/bin/python scripts/investigate_postprocess.py cleanup \
  --reference --report outputs/runtime/new-benchmark/cleanup-reference.json

PYTHONPATH=. .venv/bin/python scripts/investigate_postprocess.py cleanup \
  --report outputs/runtime/new-benchmark/cleanup-vectorized.json
```

Use fresh paths and an external `scripts/measure_generation.py` monitor for
pressure guards. `time` samples the real full model and 12-step schedule but
skips downstream decoding/quality comparisons; it is not a per-forward screen.
CPU comparisons start from the frozen raw mesh, not the already-cleaned mesh.
No additional visual, multi-asset, or multi-seed quality study is performed;
this integration's requested comparison is performance-focused.

Focused integration tests: **93 passed**. The full working-tree suite reports
**449 passed**, with the same three pre-existing README onboarding failures:
`test_readme_uses_current_huggingface_cli`,
`test_readme_tests_section_avoids_stale_exact_count`, and
`test_readme_quantization_section_states_measured_tradeoff`.
Tests cover unchanged 12-step/22-forward SS-style schedules, chunk sizes,
parameter arrays, repeated-model toggling, packed INT4/INT8 fallback, CLI/resume
forwarding, and canonical CPU topology against the original implementation.

The machine-readable comparison and exclusion record is
`outputs/runtime/selective-cleanup-integration/summary.json`; raw test logs are
`focused-tests.log`, `full-tests.log`, and `final-tests.log` in the same
directory. Concurrently
authored Python-overhead investigation files are outside this optimization
patch and are not committed with it.
