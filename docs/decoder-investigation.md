# Texture decoder memory and latency investigation

## Scope and method

Measured on 2026-11-10, Apple M3 Pro / 18 GiB, MLX 0.31.2. At the time of these
measurements these were **opt-in research paths**. The later validated
final-upsample policy is now integrated for production shape and texture
decoding; see [shape-decoder-integration.md](shape-decoder-integration.md).
The other ablations and precision variants remain research-only.
No retraining, token pruning, quantization, or sampling-step reduction is used.

The input is the native-MoGe 1024 shoe / seed 42 run documented in
[1024-validation.md](1024-validation.md). Its texture latent was regenerated
from the saved shape latent, camera, and preprocessed image. Replaying the
reference decoder reproduces the archived coordinates and all six transformed
texture attributes exactly: maximum absolute error 0.

Every ablation runs in a fresh process, with no concurrent GPU benchmark.
Decoder timing starts after weight loading, input materialization, allocator
cache clearing, and MLX peak reset; it ends after output evaluation and GPU
synchronization. Peak MLX bytes include live model weights and inputs.
The external monitor's lifetime process footprint also includes loading and
NumPy validation, so it is not an isolated GPU-allocation measurement.

The 10 GB MLX setting remains an evaluation guideline, not a hard cap.
Evaluation placement can change allocator scheduling and observed peaks.
Process footprint, MLX allocation, RSS, and system swap are overlapping but
different accounting domains; do not add them together.

## Where the memory goes

The largest expansion has 704,213 parent voxels / 128 channels and 2,814,231
child voxels / 64 channels. The final level has 41,996,311 sparse neighbor edges.
All sizes below are decimal GB and refer to individual array payloads, not
additional allocations guaranteed to overlap.

| Array | Shape | Bytes |
| --- | --- | ---: |
| Dense parent projection before child selection | 704,213 x 512, FP32 | 1,442,228,224 |
| One child feature / repeated skip / residual | 2,814,231 x 64, FP32 | 720,443,136 |
| Unrepeated skip | 2,814,231 x 16, FP32 | 180,110,784 |
| Final source, target, kernel index arrays | 3 x 41,996,311, INT32 | 503,955,732 |
| Each dense neighbor / presence / prefix scratch | 27 x 2,814,231, INT32 | 303,936,948 |
| Texture decoder parameters | Pretrained FP16 weights | 948,432,012 |
| Cached shape decoder parameters | Not used by texture decode | 948,462,926 |

The production sparse convolution already loops over kernel positions, but
does not evaluate the accumulated output inside that loop. Building 27
gather/GEMM/scatter contributions into one lazy graph can still produce a large
live working set. **A Python loop is not an execution-memory boundary.**

With phase barriers, the first convolution of the final upsample raises the
running peak from 5.75 GB to 10.28 GB. Moving neighbor construction earlier,
fusing skip expansion/add, or chunking terminal normalization does not remove
that convolution peak. These interventions should not be assumed additive.
Graph-built trace events are not operation execution timings; phase barriers
also alter the execution schedule being studied.

MLX documents the tradeoff between oversized graphs and overly frequent
evaluation in its [lazy evaluation guide](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).
The observations here are from the installed 0.31.2 runtime, not assumed
performance of a newer release.

## Decoder ablations

First fresh-process observations, without the unused resident shape decoder:

| Variant | Decoder seconds | Peak MLX GB | Maximum attribute error |
| --- | ---: | ---: | ---: |
| Reference | 13.37 | 10.994 | 0 |
| Upsample phase barriers | 13.67 | 10.276 | 0 |
| Early topology materialization | 13.48 | 10.706 | 0 |
| Fused skip gather/repeat/add | 13.26 | 10.784 | 0 |
| Terminal norm/output chunks of 65,536 | 13.24 | 10.784 | 0 |
| All four above | 13.50 | 10.783 | 0 |
| Evaluate every sparse kernel, all convolutions | 17.39 | 6.414 | 0 |
| Evaluate every 3 kernels, all convolutions | 15.84 | 8.180 | 0 |
| Evaluate every 9 kernels, all convolutions | 14.44 | 10.576 | 0 |
| Every 3 kernels plus phase barriers, all convolutions | 15.99 | 7.458 | 0 |
| Every 3 kernels, final parent projection only | 13.52 | 8.873 | 0 |
| Every kernel, final parent projection only | 13.76 | 8.873 | 0 |
| Every kernel, both final-upsample convolutions | 14.02 | 6.414 | 0 |
| Every 3 kernels, both final-upsample convolutions | 13.73 | 8.180 | 0 |

The promising choice is to bound **both convolutions in just the last
upsample**, not every convolution in the network. It reaches the same 6.414 GB
peak as globally bounded convolution while avoiding most synchronization
overhead. Every tested exact-math variant preserves coordinate order and
transformed texture attribute values for this input. This is not validation
across other assets or the predictive shape decoder.

Three fresh-process observations per selected path, including the first runs:

| Measurement | Reference | Bounded final upsample, interval 1 |
| --- | ---: | ---: |
| Median decoder seconds | 13.365 | 14.017 |
| Peak MLX GB, median | 10.784 | 6.414 |
| Peak MLX GB, range | 10.784–10.994 | 6.414–6.414 |
| Median process lifetime footprint GB | 10.958 | 7.439 |
| Process footprint GB, range | 10.559–11.394 | 7.401–7.482 |
| New system swap / highest pressure level | 0 / normal | 0 / normal |

This is approximately **40.5% lower median MLX peak**, with 0.652 s / 4.9%
more decoder time. It is not a 40.5% reduction in total Mac memory or the
end-to-end pipeline peak. The observations are sequential, not randomized
trials or a statistical confidence interval.

Retaining an otherwise unused shape decoder changes the reference measurement
to 13.60 s / 11.396 GB MLX / 11.453 GB process footprint. The cache owns 0.948 GB
of weights, but the observed MLX peak changes by only 0.402 GB: memory-guideline
scheduling means parameter bytes cannot simply be subtracted from a peak.
Cache eviction remains useful, especially for single-job latency, but its
reload cost and persistent-batch benefits must be considered separately.
The bounded final-upsample replay with those same resident weights takes
14.00 s / 7.363 GB MLX / 8.415 GB footprint, with exact transformed attributes,
normal pressure, and no observed increase in system swap.

## Latency outside the decoder

In the full 892 s run, HR flow sampling takes 354.8 s and texture sampling
211.0 s. The enclosing stages consume about 66% of pipeline time. Texture
decoding itself takes about 14 s, or 1.6%; UV work takes another 25.7 s.
Decoder memory work is valuable for feasibility, not a route to a large
end-to-end speedup on its own.

Controlled texture-sampling replays keep shape, image features, noise seed,
12 Euler steps, and FP32 sampler state fixed:

| Configuration | Replay seconds | Peak MLX GB | Attribute comparison |
| --- | ---: | ---: | --- |
| FP32 activations, 128 MiB conditioning cache | 210.86 | 4.706 | Exact reference |
| Existing mixed-precision option, 128 MiB cache | 210.28 | 4.793 | Max 0.002051, RMSE 0.0001522 |
| FP32 activations, 2,048 MiB cache | 205.10 | 6.561 | Latent and decoded attributes exactly equal |
| Mixed activations with rotary output cast back, 128 MiB cache | 161.63 | 4.608 | Max 0.003059, RMSE 0.0001575 |

The replay timer includes denormalization and writing the small compressed
latent archive; it excludes backbone/conditioning extraction and model loading.
Each sampling setting has one observation, not repeated speedup evidence.
The larger cache observes a 5.76 s saving at a cost of approximately 1.85 GB
additional MLX peak. That is not an attractive unconditional default for 18 GB.

As an arithmetic estimate, not a GPU-time profile, the 10,784-token texture
block's main dense MACs split approximately into self-attention QK/AV 43.8%,
MLP 33.3%, self-attention projections 12.5%, cross-attention query/output 6.2%,
and image projection 4.2%. This uses channels 1,536, MLP width 8,192, and
projected input width 2,048; it excludes normalization, small global K/V work,
and data movement. Caching image projections alone cannot remove the main
attention/MLP computation, consistent with the modest replay improvement.

### Mixed precision is promoted back to FP32

A symbolic dtype audit on the installed runtime finds:

1. Model activation and weights can start in FP16.
2. Rotary phases are FP32.
3. `apply_rope` returns FP32 Q/K after multiplication by those phases.
4. Attention and its output projection then return FP32; the residual promotes
   the block state, and subsequent blocks stay FP32.
5. The same toy attention without RoPE returns FP16.

This explains why simply selecting the existing mixed mode does not provide
a material speed benefit in this replay. It still changes values because
conditioning and early operations are rounded. No default precision change is
justified by that result.

The research harness can round FP32 rotary results back to their incoming
activation dtype with `--flow-mixed --preserve-rope-dtype`. This is scoped to
the standalone experiment process, restores the original function on exit,
and is **not** imported or enabled by generation entrypoints. It changes
numerical behavior and requires finite-output and visual quality validation.

The actual 12-step texture replay with this rotary cast takes **161.63 s**,
23.3% less than the 210.86 s reference observation. Latent and decoded outputs
are finite; coordinates remain equal. The six-channel attribute maximum error
is 0.0030593, mean absolute error 0.00009061, and RMSE 0.00015753. RGB mean
absolute errors are respectively 0.0002011, 0.0001677, and 0.0001373, measured
after `* 0.5 + 0.5`, before clipping or texture baking. This is not exact-output
optimization. The process footprint is 5.227 GB, with normal pressure and no
observed new system swap.

This is one fixed-shape texture case. It does not establish visual equivalence,
quality on other assets, safe FP16 SS/HR geometry, or a 23.3% end-to-end speedup.
No new GLB/render comparison was produced for the precision variant.

## Next priorities

1. Validate targeted final-upsample barriers across assets, seeds, 512/1024,
   and the shape decoder before integrating a default memory policy. Test
   empty subdivisions and boundary coordinates as well as typical objects.
2. Release or temporarily bypass cold model-cache entries before heavy decode,
   without discarding reusable image features unnecessarily. Measure cold and
   persistent-batch throughput separately.
3. If still needed, stream the parent convolution's output channels/selected
   children instead of materializing all 512 parent channels. A further option
   is target-row-chunked convolution with the full source halo. Preserve kernel
   accumulation order; Morton order does not make every neighbor chunk-local.
4. Reduce topology scratch, or lazily construct the unused kernel-index payload
   for grouped convolution. That one payload is about 168 MB here, smaller
   than the convolution-graph opportunity already measured.
5. Calibrate genuinely mixed transformer compute, including RoPE, against
   pretrained-output and visual quality. Keep FP32 normalization/CFG/state;
   do not treat finite tensors or small random tests as quality equivalence.
6. Tune conditioning-cache admission and FFN chunks by stage/token count.
   Fully caching projections costs about 1.99 GB at 10,784 FP32 tokens; the
   modest observed saving should be weighed against memory pressure.
7. Profile CPU mesh simplification and UV unwrapping separately. Repeated
   same-geometry texture variants may reuse UV layout, but arbitrary shape
   changes invalidate such a cache. This has not been implemented or timed.

Optimizing texture decode alone does **not** prove the whole pipeline now
stays under 10 GB. The original shape decode still peaks near 10.97 GB and the
HR transformer stage near 8.34 GB. No new end-to-end memory/swap claim is made.

## Reproduction and artifacts

Frozen latent SHA-256:
`423e7884fb8d30ae1cf3cce491eb8c8c0b3150a16db676a023bd645af0510432`.
All raw evidence is under the ignored `outputs/runtime/decoder-investigation/`
directory: per-run `decoder.json`, `resources.json`, `resources.jsonl`, and
`generation.log`, plus frozen latent archives and `dtype-audit.json`.
Decoder reports record revision and device; later ablations also record
experiment options, latent hash, and whether the source working tree was dirty.

Use a fresh directory per measurement:

```bash
PYTHONPATH=. .venv/bin/python scripts/investigate_decoder.py prepare

PYTHONPATH=. .venv/bin/python scripts/measure_generation.py \
  --output-dir outputs/runtime/decoder-replay-new --interval 1 -- \
  .venv/bin/python -u scripts/investigate_decoder.py decode \
  --variant bounded-upsample --convolution-eval-interval 1 \
  --report outputs/runtime/decoder-replay-new/decoder.json

PYTHONPATH=. .venv/bin/python scripts/investigate_decoder.py audit \
  --report outputs/runtime/decoder-dtype-audit-new.json

PYTHONPATH=. .venv/bin/python scripts/investigate_decoder.py prepare \
  --flow-mixed --preserve-rope-dtype \
  --latent outputs/runtime/texture-latent-mixed-rope-new.npz

.venv/bin/python -m pytest tests/test_decoder_experiments.py \
  tests/test_gpu_sparse_runtime.py tests/test_sparse_runtime.py \
  tests/test_sampling_runtime.py -q
```

Preparation refuses to overwrite an existing latent. Supply `--latent` for a
new path, and `--checkpoints` / `--image` when replaying a different archived
run. The checkpoint filenames, model architecture, and comparison format are
specific to the current Pixal3D experiment, not a general inference CLI.

## Validation

- Decoder/precision research tests: 18 passed, covering fused skip layouts,
  shared-weight decoder variants, invalid configurations, scoped restoration,
  and FP16 block dtype propagation.
- Focused decoder/sparse/sampling suite: 45 passed.
- Full suite: 300 passed, three pre-existing README onboarding failures:
  `test_readme_uses_current_huggingface_cli`,
  `test_readme_tests_section_avoids_stale_exact_count`, and
  `test_readme_quantization_section_states_measured_tradeoff`.
- Frozen 1024 replays validate all tested exact-math variants against archived
  coordinates and transformed texture attributes, with maximum error 0.
- Sampling precision experiments are finite for this case but intentionally
  remain opt-in; numerical differences are reported rather than suppressed.
