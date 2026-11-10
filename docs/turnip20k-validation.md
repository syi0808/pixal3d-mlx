# Turnip 20K soft-target export

For the subsequent normal-corrected model and actual Cycles PBR views, see
[normal correction and PBR rendering](turnip20k-normal-correction.md). The
original export described here remains unchanged.

Generated `outputs/turnip20k/turnip-20k.glb` with `--target-faces 20000` and
**without `--max-faces`**. The final GLB contains **20,034 triangles**, 13,832
UV-split vertices, and 1024 x 1024 base-color and metallic/roughness maps.
It is 2,637,180 bytes. The small face-count overshoot is expected for a soft
target: simplification reaches 19,999 faces, then final repair produces 20,034.

## Method

- Same 512 geometry, image, seed 42, and voxel texture attributes as the
  original turnip. Neural inference is not rerun.
- Start from the retained raw decoder mesh: 663,831 vertices / 1,344,004 faces.
  Using raw geometry preserves the original component-filter-before-splitting
  policy instead of treating previously split surface fragments as debris.
- Standard production mesh cleanup, soft simplification, UV unwrap, texture
  rebaking, and GLB export. No experimental cleanup substitution is enabled.
- Reported `max_faces` is 0 / disabled, and the exact process command has no
  `--max-faces` argument. No strict-cap welding/rounding fallback is invoked.
- Original high-poly and 5K models remain byte-identical.
- Code revision: `a71fe43c48d5f2d6a54decd5c35407fcd4fad91d`.

## Measurements

| Operation | Seconds |
| --- | ---: |
| Checkpoint/raw-mesh load | 0.091 |
| Cleanup and soft simplification | 10.632 |
| UV unwrap | 0.487 |
| Texture bake | 1.944 |
| Export and validation | 0.156 |
| Internal elapsed | 13.315 |
| Whole monitored process | 14.187 |

MLX peak is 4,552,016,416 bytes; process lifetime footprint is 5,802,512,248
bytes. These are different overlapping accounting domains. Pressure remains
normal and no new system swap-used growth is observed. Timings describe
checkpoint reprocessing, **not** a new image-to-3D inference run.

The GLB reloads with finite geometry and UV arrays; both embedded PBR images
load at 1024 x 1024. Front/back previews and a three-plane witness render the
actual exported model. The previews use simple software shading, not a full
PBR renderer. This does not certify watertightness or identical high-poly
silhouettes.

Artifacts: `outputs/turnip20k/` contains `turnip-20k.glb`, `export.json`,
`validation.json`, `resources.json`, `resources.jsonl`, `generation.log`,
`turnip-preview.png`, `turnip-back.png`, `witness.png`, and `witness.json`.

GLB SHA-256:
`3f9899c60c5faaddd0f59744b61a5996517f3d96f63d4b5b0eb683a9477eaa8f`.

## Reproduction

Use new output paths; existing models/reports are not overwritten.

```bash
PYTHONPATH=. .venv/bin/python reexport_pixal3d.py \
  --ckpt outputs/turnip512/checkpoints \
  --raw-mesh outputs/runtime/postprocess-investigation/raw-mesh.npz \
  --target-faces 20000 --texture-size 1024 \
  --output outputs/turnip20k-new/turnip-20k.glb \
  --report outputs/turnip20k-new/export.json
```

The re-export CLI now accepts a soft target without a strict cap. Soft-target
cleanup requires the raw mesh explicitly to avoid re-filtering already split
surface parts. If raw geometry is unavailable, regenerate it from the saved
shape latent with `scripts/investigate_postprocess.py prepare`; flow sampling
is still unnecessary.

Validation: 13 mesh-budget/re-export tests pass, including soft-target-only
output above the target without invoking a strict cap. The full suite reports
346 passed and the same three pre-existing README onboarding failures.
