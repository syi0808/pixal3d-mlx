# Attached turnip image: 512 generation on 18 GiB

## Result

Generated and validated `outputs/turnip512/turnip.glb` from the attached image.
This is a **512 geometry-grid** result with **1024 x 1024 PBR textures**.
It completes on this 18 GiB M3 Pro with no observed increase in system swap,
although memory pressure briefly reaches warning level.

This is one input/seed and one active generation, not a guarantee for every
asset, concurrent jobs, or swap-free operation on a clean desktop.

## Conditions

- 2026-11-10, approximately 21:55:49–22:01:22 Asia/Seoul.
- Apple M3 Pro, 14 GPU cores, 18 GiB physical memory, MLX 0.31.2.
- Generation code: `a51b5e676ebb8b906a23dfdd295e2ddcbf3d8e8b`.
- The original 1312 x 1199 RGB attachment is copied to
  `outputs/turnip512/input.png`; its SHA-256 is
  `c7c8c760274e3f921d925f17e0d80f1379fb0bbf5425a89c23613eedd14b65c5`.
- Background removed using the existing preprocessing path; the saved
  conditioning image is 892 x 892 with the body and leaves retained.
- Native safetensors MoGe, no fallback: FOV approximately 38.7 degrees,
  distance 1.4226099.
- Seed 42, original pretrained weights, 12-step schedules, no quantization,
  and no mixed-precision or experimental RoPE casting.
- `--decoder-eval-interval 1` applies the validated last-upsample policy to
  predictive shape and guided texture decoding. See
  [shape-decoder-integration.md](shape-decoder-integration.md).
- MLX evaluation guideline 10 decimal GB; the external monitor uses the
  existing 14 GB process-footprint / sustained-pressure watchdog.
- Other desktop applications were not closed or otherwise controlled.

## Measurements

All GB values are decimal. MLX allocations, process footprint, and RSS are
different overlapping accounting domains and must not be added together.

| Measurement | Observed |
| --- | ---: |
| Monitored process elapsed | 333.094 s, 5 min 33 s |
| Pipeline-reported elapsed | 332.5 s |
| Peak MLX allocation | 8,338,815,618 bytes, 8.34 GB |
| Process lifetime footprint | 9,816,529,592 bytes, 9.82 GB |
| Child high-water RSS | 7,184,711,680 bytes |
| System swap before | 5,397,419,130 bytes |
| Highest system swap | 5,397,419,130 bytes |
| System swap after | 5,363,864,698 bytes |
| New observed swap-used growth | 0 bytes |
| Sampled normal pressure | Approximately 330.0 s |
| Sampled warning pressure | Approximately 3.05 s |
| Sampled critical pressure | 0 s |
| Watchdog termination | None |
| Process return code | 0 |

The desktop already had approximately 5.40 GB of swap in use. “No new swap”
does not mean the system had no swapped pages or that per-process swap was
measured. Pressure and swap are system-wide, sampled approximately once per
second; short events may be missed. The process footprint uses macOS's
lifetime high-water field rather than only sampled instantaneous maxima.

| Stage including loading/cleanup | Seconds |
| --- | ---: |
| Preprocess, camera, backbone | 7.21 |
| Sparse structure | 104.53 |
| LR conditioning and shape | 60.86 |
| Coordinate subdivision | 5.75 |
| HR conditioning and shape | 61.87 |
| Shape decode and mesh processing | 24.01 |
| Texture sampling | 41.53 |
| Texture decode and export | 26.74 |

Within those stages, SS sampling takes 93.9 s, LR sampling 51.4 s, HR sampling
49.2 s, shape decode 3.7 s, texture sampling 32.1 s, and texture decode 3.4 s.
UV unwrapping takes 18.4 s. The global MLX peak is already reached during SS;
the stage report's later repeated 8.34 GB values are cumulative peaks, not
independent measurements of each decoder's live working set.

This is not a controlled speed comparison with the earlier shoe: the input,
token count, camera, and decoder policy differ. The memory-policy comparison
uses the frozen same-input decoder replay documented separately.

## Artifact validation

- Saved metadata confirms geometry resolution 512 and 2,185 HR latent tokens;
  there is no resolution fallback. Final decode has 663,831 active voxels.
- GLB v2 header and byte length validate; the file reloads as a nonempty scene.
- Export: 148,062 UV-split vertices, 201,195 triangles, 9,583,168 bytes.
  Cleanup after the approximate 200,000-face target slightly increases faces.
- Vertex and UV arrays are finite; base-color and metallic/roughness textures
  both load at 1024 x 1024. Material alpha mode is opaque.
- The software witness is nonblank and samples colors from the embedded
  texture. Body and leaf geometry are present in the actual generated mesh.
- The UV-split exported mesh is not reported watertight. These checks do not
  certify a closed fabrication mesh, game-ready LOD, or exact input-image
  appearance.

GLB SHA-256:
`3c9c781624b11515dc7b5f07d13878afa40f11d49aefec210004c15af94875fc`.

Artifacts under `outputs/turnip512/`:

- `turnip.glb`: generated model with embedded textures.
- `turnip-preview.png`, `turnip-back.png`: front/back software previews of
  the actual GLB; triangle-centroid texture sampling and simple shading, not
  reference PBR renders or edited input images. `render_preview.py` reproduces
  them.
- `turnip-witness.png`, `witness.json`: deterministic three-plane mesh witness.
- `validation.json`: geometry/texture checks, hashes, and pressure summary.
- `runtime.json`, `resources.json`, `resources.jsonl`, `generation.log`:
  generation settings, stages, process/system observations, and exact command.
- `input.png`, `turnip_preprocessed.png`, `checkpoints/`: retained inputs and
  camera/shape/mesh/texture checkpoints.
- `full-tests.log`: validation suite output.

## Reproduction

The input copy above is retained; use a new output directory to avoid
overwriting measurement evidence.

```bash
PYTHONPATH=. .venv/bin/python scripts/measure_generation.py \
  --output-dir outputs/turnip512-rerun --interval 1 -- \
  .venv/bin/python -u generate_pixal3d.py \
  --image outputs/turnip512/input.png --seed 42 --resolution 512 \
  --texture-size 1024 --decoder-eval-interval 1 --memory-budget-gb 10 \
  --output outputs/turnip512-rerun/turnip.glb \
  --save-checkpoints outputs/turnip512-rerun/checkpoints \
  --runtime-report outputs/turnip512-rerun/runtime.json
```

## Tests

The full suite reports **319 passed and three pre-existing README onboarding
failures**. The new decoder-policy module has 19 passing tests, including
predictive/empty subdivisions, guided texture alignment, coordinate boundaries,
raw-byte signatures, policy scoping, and resume forwarding. The known failures
remain `test_readme_uses_current_huggingface_cli`,
`test_readme_tests_section_avoids_stale_exact_count`, and
`test_readme_quantization_section_states_measured_tradeoff`; no unrelated README
changes are included.
