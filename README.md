# pixal3d-mlx

MLX-native [Pixal3D](https://github.com/TencentARC/Pixal3D) (SIGGRAPH 2026) inference for Apple Silicon.

Image → textured 3D mesh with PBR materials. No NVIDIA GPU required. Neural inference runs through MLX on Metal — including camera estimation via a native MoGe-2 port ([details](docs/moge-camera-estimation.md)).

[Pixal3D](https://github.com/TencentARC/Pixal3D) uses pixel-aligned back-projection conditioning to establish direct pixel-to-3D correspondence, producing dramatically better geometry and texture fidelity than attention-based conditioning alone. This port runs the full pipeline natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx), including a pure-MLX port of [NAF](https://github.com/valeoai/NAF) (Neural Attention Fields) for feature upsampling.

### Input → Output

<table>
<tr>
<td><img src="assets/outputs/pixal3d_input.jpg" width="180" alt="Input image"></td>
<td><img src="assets/outputs/pixal3d_render_1.png" width="180" alt="Output 1"></td>
<td><img src="assets/outputs/pixal3d_render_2.png" width="180" alt="Output 2"></td>
<td><img src="assets/outputs/pixal3d_render_3.png" width="180" alt="Output 3"></td>
<td><img src="assets/outputs/pixal3d_render_4.png" width="180" alt="Output 4"></td>
</tr>
<tr>
<td align="center"><em>Input</em></td>
<td align="center"><em>Generated</em></td>
<td align="center"><em>Generated</em></td>
<td align="center"><em>Generated</em></td>
<td align="center"><em>Generated</em></td>
</tr>
</table>

*Single image → textured 3D mesh with PBR materials. ~10 min at 512 resolution on M4 Max. Fits in 16 GB unified memory. Higher resolutions need 64 GB.*

## Quick start

```bash
git clone https://github.com/syi0808/pixal3d-mlx.git
cd pixal3d-mlx
uv venv .venv --python python3.12
source .venv/bin/activate
uv pip install -e .

# Hugging Face auth (needed for gated DINOv3 weights):
huggingface-cli login
# Request access: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m

# Download model weights (~24 GB for Pixal3D, ~1 GB for DINOv3):
huggingface-cli download TencentARC/Pixal3D
huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m

# Download and convert NAF weights (~3 MB, needs torch for one-time conversion):
mkdir -p weights
curl -L https://github.com/valeoai/NAF/releases/download/model/naf_release.pth -o weights/naf_release.pth
pip install torch safetensors  # only needed for this conversion
python -c "import torch; from safetensors.torch import save_file; save_file(torch.load('weights/naf_release.pth', map_location='cpu', weights_only=True), 'weights/naf_release.safetensors')"

# Build default CGAL hole repair (or select --hole-fill-backend fan below):
HOMEBREW_NO_AUTO_UPDATE=1 brew install cgal eigen cmake
cmake -S native/cgal_holes -B build/cgal-holes -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(brew --prefix)"
cmake --build build/cgal-holes -j 1

# Generate:
PYTHONPATH=. python generate_pixal3d.py --image your_image.png --output output.glb

open output.glb
```

## What works now

### Private API and dashboard

Run generation from another computer through a private, token-free
API with a durable single-worker queue, a dashboard, and an interactive PBR
GLB viewer. Final exports automatically correct UV-seam normals without
changing triangles or embedded textures. See the
[local server and Tailscale setup](docs/private-server.md).

```bash
uv pip install --python .venv/bin/python -e '.[server]'
npm ci --prefix web && npm run build --prefix web
.venv/bin/python -m pixalmlx.server
```

The server binds only to `127.0.0.1:8000`; no login or API token is required.
Remote access uses Tailscale Serve and tailnet access policy, not public
Funnel exposure. All permitted clients share the same workspace. Server
startup and Tailscale configuration are explicit.

### Image-to-3D CLI

Full Pixal3D pipeline: image → textured GLB with PBR materials (base color, metallic, roughness).

This checkout defaults to 512 resolution and a 10 GB MLX evaluation guideline,
not a hard allocation cap. It includes sparse NAF projection, GPU sparse topology, static DiT
conditioning, and an in-process persistent batch engine. See the
[runtime implementation and measured tradeoffs](docs/runtime-architecture.md).
Pixal3D generation/resume enable bounded FFN and modulation/residual compilation
by default (`--no-compile-transformer` disables it). CPU mesh topology uses
vectorized edge grouping and sparse corner connectivity without reducing steps.
Hole filling defaults to CGAL refinement/fairing with fan fallback for rejected
holes. Build the native helper as above, or use `--hole-fill-backend fan` for
the original implementation. Missing native support is reported before model
loading. See [setup, preservation guarantees, and measured tradeoffs](docs/cgal-hole-filling.md).

```bash
# Full pipeline (resolution 512 by default):
PYTHONPATH=. python generate_pixal3d.py --image photo.png --output mesh.glb

# Opt into 1024 (memory and latency must be validated separately):
PYTHONPATH=. python generate_pixal3d.py --image photo.png --output mesh.glb --resolution 1024

# Faster run (resolution 512, ~10 min):
PYTHONPATH=. python generate_pixal3d.py --image photo.png --output mesh.glb --resolution 512

# High-res textures and more geometry:
PYTHONPATH=. python generate_pixal3d.py --image photo.png --output mesh.glb --texture-size 4096 --target-faces 500000

# Resume from checkpoint (skip geometry, redo texture):
PYTHONPATH=. python resume_pixal3d.py --image photo.png --ckpt outputs/ckpt_dir --output mesh.glb
```

### Pipeline

1. **Image conditioning** — DINOv3 ViT-L/16 + NAF feature upsampling + ProjGrid 3D projection
2. **Sparse structure** — 1.3B DiT flow model (12 Euler steps) → 64³ occupancy
3. **LR shape latent** — 1.3B DiT on sparse tokens (512 model)
4. **Upsample** — decoder subdivision → HR coordinates
5. **HR shape latent** — 1.3B DiT on sparse tokens (1024 model)
6. **Shape decode + mesh extraction** — sparse UNet decoder → dual-grid mesh
7. **Texture latent + decode** — 1.3B DiT → per-voxel PBR attributes
8. **UV unwrap + texture bake** — xatlas + trilinear voxel sampling + seam inpaint → GLB

### Performance (M4 Max, 128 GB)

| Resolution | Tokens | Time | Peak GPU | Texture |
|-----------|--------|------|----------|---------|
| 512 (FP16) | ~6K | ~10 min | ~12 GB | 2048 |
| 512 (INT4) | ~6K | ~12 min | ~12 GB | 2048 |
| 1024 (FP16) | ~15-26K | ~21 min | ~33 GB | 4096 |

Resolution 512 fits comfortably in 16 GB unified memory.
Resolution 1024 requires 64 GB unified memory.
INT4 quantization (`--quantize 4`) reduces flow model weights 3.5x with comparable quality.

### Key features

- **Pixel-aligned projection conditioning** — explicit pixel-to-voxel spatial correspondence, not loose attention
- **NAF upsampling** — pure MLX port of [valeoai/NAF](https://github.com/valeoai/NAF) with windowed neighborhood attention (no natten CUDA dependency)
- **No SDPA cliff** — MLX Flash Attention handles 26K+ tokens where PyTorch MPS silently breaks at 18K
- **Checkpoint/resume** — save intermediate results, resume from shape latent to skip geometry stages
- **Aggressive memory cleanup** — `mx.metal.clear_cache()` between stages, model offloading
- **MoGe-2 camera estimation** — pure MLX port of [MoGe-2](https://github.com/microsoft/MoGe) (326M params) for automatic per-image FOV estimation. No PyTorch dependency. Matches PyTorch output within 0.08° FOV / 0.9999973 correlation ([details](docs/moge-camera-estimation.md))
- **INT4 quantization** — `--quantize 4` reduces flow model weights 3.5x (2.7 GB → 755 MB per model)
- **47 tests** — projection modules, flow models, weight loading, NAF (neighborhood attention, RoPE, encoder, cross-attention, weight loader)

## Architecture

The distribution is `pixal3d-mlx`; Python imports and module commands use
`pixalmlx`. Shared runtime, sampling, mesh, and math modules use model-neutral
names. Model-specific implementations and official weight IDs retain their
upstream names.

After updating an existing checkout, reinstall the editable package with
`uv pip install --python .venv/bin/python -e '.[server,dev]'` and rebuild the
viewer with `npm run build --prefix web`. Update external imports and server
launch commands to `pixalmlx`. Batch and quantization reports now use the
`pixalmlx.batch_report.v1` and `pixalmlx.quant_bench.v1` schema identifiers;
the quantization checkpoint override is `PIXALMLX_CHECKPOINT_ROOT`. New batch
outputs default to the `pixal` prefix. Existing checkpoints and historical
validation artifacts are not renamed.

```
pixalmlx/
├── models/
│   ├── pixal3d_flow.py          # Pixal3D SS + SLat flow models with ProjectAttention
│   ├── dinov3_proj.py           # DINOv3 + NAF + ProjGrid feature extraction
│   ├── naf.py                   # NAF feature upsampler (windowed neighborhood attention)
│   ├── naf_loader.py            # NAF weight loader
│   ├── sparse_structure_flow.py # Base sparse-structure DiT flow model
│   ├── slat_flow.py             # Base sparse-latent DiT flow model
│   ├── shape_slat_decoder.py    # Sparse UNet decoder
│   ├── sparse_structure_decoder.py
│   └── dinov3.py                # DINOv3 ViT-L/16
├── modules/
│   ├── proj_grid.py             # 3D grid projection + bilinear sampling
│   ├── proj_attention.py        # ProjectAttention (cross-attn + proj_linear)
│   ├── attention.py             # MLX Flash Attention + MultiHeadRMSNorm
│   ├── rope.py                  # 3D Rotary Position Embedding
│   ├── norm.py                  # LayerNorm32
│   └── sparse_conv.py           # Submanifold sparse 3D convolution
├── mesh_extract.py              # Dual-grid mesh extraction
├── mesh_cleanup.py              # Dedup, non-manifold repair, hole fill, normal fix
├── texture_bake.py              # UV unwrap + MLX GPU rasterization + voxel sampling
├── samplers.py                  # Flow Euler sampler with CFG + guidance interval
├── weight_loader.py             # Safetensors loader (key remap, bf16→fp16)
└── checkpoint.py                # Stage checkpoint save/load
```

## How it differs from upstream Pixal3D

| | Upstream (CUDA) | This port (MLX) |
|---|---|---|
| Runtime | PyTorch + CUDA + natten + nvdiffrast + o_voxel | MLX + numpy + CGAL/fan mesh repair (no PyTorch) |
| Camera estimation | MoGe-2 via PyTorch CUDA | MoGe-2 ported to MLX (326M params) |
| Attention | Flash Attention 3 (CUDA) | MLX Flash Attention (Metal) |
| NAF | natten CUDA kernels | Windowed gather with dilation (pure MLX) |
| Sparse conv | flex_gemm CUDA | Gather-scatter sparse conv (MLX) |
| Mesh extraction | o_voxel CUDA + nvdiffrast | numpy dual-grid + xatlas + MLX GPU rasterizer |
| Install | Multiple compiled CUDA/C++ packages | `pip install -e .` + CGAL helper, or fan-only repair |

## MoGe-2 Camera & Geometry Estimation

Pure MLX port of [MoGe-2](https://github.com/microsoft/MoGe) (326M params, DINOv2-L/14 backbone + ConvStack decoder) for monocular geometry estimation on Apple Silicon. Supports both the base `moge-2-vitl` checkpoint (depth/points) and the `moge-2-vitl-normal` variant (adds direct normal prediction).

### Precision & Performance

Benchmarked on M-series Apple Silicon, BunnyCake fixture, `moge-2-vitl-normal` checkpoint. All parity measurements are against PyTorch fp32 as reference.

| | Inference | Max Angular Error | Mean Angular Error |
|---|---|---|---|
| **MLX fp32** | **0.60s** | **0.028°** | **0.0002°** |
| PyTorch MPS fp32 | 0.84s | — | — |
| PyTorch MPS fp16 (default) | 0.90s | 0.420° | 0.014° |

MLX loads weights in 0.27s vs 5.4s for PyTorch.

See [docs/moge-camera-estimation.md](docs/moge-camera-estimation.md) for implementation details.

## Credits

- [Original MLX port](https://github.com/lyonsno/pixal3d-mlx) by Noah Lyons — the upstream codebase for this fork
- [Pixal3D](https://github.com/TencentARC/Pixal3D) by TencentARC — the model and paper (SIGGRAPH 2026)
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research — the backbone architecture
- [NAF](https://github.com/valeoai/NAF) by Valeo.ai — Neural Attention Fields feature upsampler
- [Pixal3D-mac](https://github.com/pawel-mazurkiewicz/Pixal3D-mac) by Pawel Mazurkiewicz — first Mac port (PyTorch MPS + Metal kernels)
- [trellis2-apple](https://github.com/pedronaugusto/trellis2-apple) by Pedro Naugusto — MLX backend + Metal GPU packages
- [trellis-mac](https://github.com/shivampkumar/trellis-mac) by Shivam Kumar — proved Mac viability for TRELLIS
- [MLX](https://github.com/ml-explore/mlx) by Apple — the framework

## License

MIT (porting code). Upstream model weights are subject to their own licenses.
