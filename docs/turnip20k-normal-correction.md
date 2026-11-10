# Turnip 20K normal correction and PBR rendering

The existing soft-target 20K model has been corrected and rendered with its
embedded PBR textures. This is asset postprocessing, not another neural
inference run. The latest request selects the 20K model instead of the earlier
5K model; the original **20,034 triangles** are retained, without subdivision
or another simplification pass.

## Result

| Property | Original | Corrected |
| --- | ---: | ---: |
| Triangles | 20,034 | 20,034 |
| Attribute-split vertices | 13,832 | 13,959 |
| Zero-length vertex normals | 52 | 0 |
| Embedded base-color image | 1024 x 1024 | Unchanged bytes |
| Embedded metallic/roughness image | 1024 x 1024 | Unchanged bytes |
| GLB size, bytes | 2,637,180 | 3,325,012 |

Triangle-expanded positions, UVs, and all other non-normal attributes are
byte-identical. Material parameters and embedded image bytes are also
identical. Additional vertex entries preserve sharp normal boundaries without
changing geometry. The patcher appends corrected attributes to the original
binary buffer, retaining unused original geometry bytes; file size growth is
not additional triangles or higher-resolution textures.

Normal correction groups coincident positions across UV seams, joins
consistently oriented manifold face fans below a 75-degree crease threshold,
and uses corner-angle-weighted normals. Non-manifold edges are not joined.
The source contains 28 degenerate triangles, which are retained to preserve
geometry and receive valid fallback normals. All output vertex normals are
finite and unit length. This does not certify watertightness or repair the
source topology.

## Rendering and limitations

The previous lightweight previews sampled one texture color per triangle and
used face-normal lighting. They therefore exaggerated faceting independently
of the actual exported material. The new renders use **Blender 5.1.2 Cycles**,
imported custom normals, linear UV texture sampling at shading points, and the
existing base-color, metallic, and roughness inputs on Principled BSDF.
Base color is sRGB; metallic/roughness is Non-Color. No normal map is invented,
and no material repainting or parameter adjustment is performed.

Each final view uses 1024 x 1024 pixels, 128 samples, denoising, AgX color
management, fixed seed 42, studio area lights, and CPU rendering. No mesh
modifiers are added. The staging floor is not part of the exported GLB.
Front, three-quarter, and back renders take 22.623, 23.503, and 21.428 seconds;
the reported rendering-stage total is **67.667 seconds**. These are render
times, not image-to-3D generation times.

An original-model front render uses the same renderer, lights, camera, and
sampling settings for a fair normal-only comparison. Much of the improvement
over the old approximate previews comes from proper rendering, not only the
normal correction. Low-poly silhouette edges, source texture blemishes, and
leaf geometry defects can still be visible: changing normals cannot recover
missing geometry or texture detail.

## Artifacts

- Corrected asset: `outputs/turnip20k-smooth/turnip-20k-smooth.glb`
- Correction audit: `outputs/turnip20k-smooth/normals.json`
- Final views: `outputs/turnip20k-smooth/pbr/{front,three-quarter,back}.png`
- Renderer audit: `outputs/turnip20k-smooth/pbr/render.json`
- Packed, reproducible scene: `outputs/turnip20k-smooth/pbr/studio.blend`
- Same-lighting original: `outputs/turnip20k-smooth/before-pbr/front.png`

Corrected GLB SHA-256:
`5653bd99da2ae0815187519c603cd4360e8554a642b4de80d0ac6998dc57414a`.
The original 20K, 5K, and high-poly GLB hashes remain unchanged.

## Reproduction

Use fresh output paths; existing assets and reports are not overwritten.
No polygon-budget option is necessary: normal correction preserves the input
triangle count. Its optional `--max-faces` is only a validation guard, not a
decimation operation or a change to the earlier soft-target export policy.

```bash
PYTHONPATH=. .venv/bin/python scripts/correct_glb_normals.py \
  --input outputs/turnip20k/turnip-20k.glb \
  --output outputs/turnip20k-smooth-new/turnip-20k-smooth.glb \
  --report outputs/turnip20k-smooth-new/normals.json \
  --crease-degrees 75

/Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --python scripts/render_glb_pbr.py -- \
  --input outputs/turnip20k-smooth-new/turnip-20k-smooth.glb \
  --output-dir outputs/turnip20k-smooth-new/pbr \
  --resolution 1024 --samples 128 --views front three-quarter back --save-blend
```

Ten focused normal-correction tests pass. The full suite reports **378 passed**
and the same three pre-existing README onboarding failures. Independent GLB
reload confirms 20,034 triangles, finite unit normals, UVs, and both embedded
1024 x 1024 PBR images. Production inference/export defaults are unchanged.
