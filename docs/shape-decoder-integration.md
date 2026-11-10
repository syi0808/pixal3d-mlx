# Bounded shape/texture decoder integration

## Production policy

Pixal3D generation and checkpoint resume now default to
`--decoder-eval-interval 1`. This evaluates the accumulated sparse-convolution
output after each kernel position in **both convolutions of the last
channel-to-spatial upsample**. Other decoder convolutions remain unchanged.
The policy applies to predictive shape decoding, the coordinate-upsample pass,
and guided texture decoding.

The pretrained weights, kernel accumulation order, voxel selection, FP32
activations, normalization, sampling schedules, and texture transformation
remain unchanged. This is an execution/liveness change, not mixed precision.
The experimental RoPE precision change from the earlier investigation is not
enabled in production.

- `--decoder-eval-interval 0` selects the legacy evaluation schedule.
- Larger positive values group kernel contributions between evaluations.
- Plain `SLatDecoder` construction retains its legacy default of 0 for API
  compatibility; Pixal3D entrypoints explicitly pass their configured policy.
- Persistent model-cache identities include the interval, preventing a model
  cached under another evaluation policy from being silently reused.
- The same option is forwarded by `generate_pixal3d.py --resume`.

This is not a hard memory cap. MLX's configured 10 GB evaluation guideline and
the external process/pressure watchdog retain their existing meanings.

An attached-image 512 generation using this policy completed and exported a
validated GLB; its full-pipeline measurements are in
[turnip512-validation.md](turnip512-validation.md).

## Frozen 1024 shape validation

Measured on 2026-11-10, M3 Pro / 18 GiB, MLX 0.31.2, code revision `a51b5e6`.
Input: the native-MoGe 1024 shoe checkpoint, seed 42, 10,784 latent tokens,
2,814,231 final voxels, 948,462,926 bytes of shape-decoder parameters.

Each setting uses a fresh process. Weight loading is excluded from decoder
timing; inputs/weights are evaluated and MLX allocator caches/peak are reset
before the timed forward pass. Output evaluation and GPU synchronization are
included. Process lifetime footprint also includes loading and validation.
All GB figures are decimal and different accounting domains must not be added.

| Measurement | Legacy interval 0 | Final-upsample interval 1 |
| --- | ---: | ---: |
| Decoder time | 13.639 s | 14.055 s |
| Peak MLX allocation | 10,994,042,482 bytes | 6,414,282,194 bytes |
| Peak MLX GB | 10.994 | 6.414 |
| Process lifetime footprint | 10,981,983,864 bytes | 7,461,373,752 bytes |
| New observed system swap | 0 | 0 |
| Highest memory-pressure level | Normal | Normal |

The observed MLX reduction is **41.7%**, for approximately 0.416 s / 3.0%
additional decoder time. These are one observation per setting, not repeated
end-to-end speed measurements.

SHA-256 signatures of raw output bytes, coordinates, and every predicted
subdivision array are identical between settings. The shape outputs are also
finite; coordinates and subdivision values match the original archived run.
This includes the subdivision predictions that will guide texture decode,
not merely a visually similar mesh or matching output dimensions.

The production texture decoder was independently replayed with the same new
policy: 13.973 s, 6,414,251,280 bytes peak MLX, 7,511,082,616 bytes process
footprint. Its coordinates and all transformed texture attribute values match
the original archive exactly, with normal pressure and no new observed swap.

Artifacts: `outputs/runtime/shape-memory-validation/{reference,bounded,texture-production}/`
contain `decoder.json`, `resources.json`, `resources.jsonl`, and `generation.log`.

## Reproduction

Use fresh directories; validation and process-monitor reports are not
overwritten.

```bash
PYTHONPATH=. .venv/bin/python scripts/measure_generation.py \
  --output-dir outputs/runtime/shape-reference-new --interval 1 -- \
  .venv/bin/python -u scripts/validate_decoder_memory.py \
  --kind shape --eval-interval 0 \
  --report outputs/runtime/shape-reference-new/decoder.json

PYTHONPATH=. .venv/bin/python scripts/measure_generation.py \
  --output-dir outputs/runtime/shape-bounded-new --interval 1 -- \
  .venv/bin/python -u scripts/validate_decoder_memory.py \
  --kind shape --eval-interval 1 \
  --reference-report outputs/runtime/shape-reference-new/decoder.json \
  --report outputs/runtime/shape-bounded-new/decoder.json
```

The defaults use the retained shoe checkpoints and texture latent from the
earlier investigation. `--checkpoints`, `--texture-latent`, and `--kind texture`
select another compatible replay. The script returns a failure status on an
archived-output mismatch or a reference-signature mismatch.
