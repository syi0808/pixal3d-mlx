"""NAF (Neural Attention Fields) feature upsampler — MLX port.

Upsamples VFM (DINOv3) patch features to higher resolution using
cross-attention with the input image as guidance.

Original: valeoai/NAF (PyTorch + natten CUDA)
This port replaces neighborhood attention with masked full attention.
For grids up to 64×64 (4096 positions), this is fast enough.

Architecture:
    ImageEncoder: dual Conv2d paths (1×1 + 3×3) → pool → RoPE
    CrossAttention: upsample K/V → neighborhood attention → output
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ── RoPE ──

class NAFRoPE(nn.Module):
    """2D Rotary Position Embeddings for NAF.

    Matches valeoai/NAF src/layers/rope.py with base=100, rescale=2.0,
    normalize_coords="separate".
    """

    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0):
        super().__init__()
        self.num_heads = num_heads
        D_head = embed_dim // num_heads
        self.D_head = D_head
        # Periods: base^(2*i / (D//2)) for i in 0..D//4-1
        periods = base ** (2 * np.arange(D_head // 4, dtype=np.float32) / (D_head // 2))
        self.periods = mx.array(periods)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply 2D RoPE to spatial features.

        Args:
            x: [B, C, H, W] (channels-first)

        Returns:
            [B, C, H, W] with RoPE applied
        """
        B, C, H, W = x.shape
        n = self.num_heads
        d = C // n

        # Reshape to [B, n, H*W, d]
        x = x.reshape(B, n, d, H * W).transpose(0, 1, 3, 2)  # [B, n, HW, d]

        # Create coordinates in [-1, 1]
        coords_h = (mx.arange(H).astype(mx.float32) + 0.5) / H * 2 - 1
        coords_w = (mx.arange(W).astype(mx.float32) + 0.5) / W * 2 - 1
        grid_h, grid_w = mx.meshgrid(coords_h, coords_w, indexing='ij')
        coords = mx.stack([grid_h.reshape(-1), grid_w.reshape(-1)], axis=-1)  # [HW, 2]

        # Compute angles: [HW, 2, D//4] → [HW, D//2] → [HW, D]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.reshape(coords.shape[0], -1)  # [HW, D//2]
        angles = mx.concatenate([angles, angles], axis=-1)  # [HW, D]

        cos = mx.cos(angles)
        sin = mx.sin(angles)

        # Apply RoPE: x * cos + rotate_half(x) * sin
        x1, x2 = x[..., :d//2], x[..., d//2:]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        x = x * cos + rotated * sin

        # Reshape back to [B, C, H, W]
        x = x.transpose(0, 1, 3, 2).reshape(B, C, H, W)
        return x


# ── Convolution blocks ──

def _reflect_pad(x: mx.array, pad: int) -> mx.array:
    """Reflect-pad spatial dims of NHWC tensor.

    MLX doesn't have mx.flip or mode='reflect', so we use reverse slicing.
    For pad=1 on a [B, H, W, C] tensor with H=4:
        top = x[:, 1:2, :, :]  (row 1, reversed = row 1)
        bottom = x[:, -2:-1, :, :]  (row H-2, reversed = row H-2)
    """
    if pad == 0:
        return x
    # Reflect-pad H: mirror rows [1..pad] at top, [H-pad-1..H-2] at bottom
    top = x[:, pad:0:-1, :, :]        # rows pad, pad-1, ..., 1 (reversed)
    bottom = x[:, -2:-(pad+2):-1, :, :]  # rows H-2, H-3, ..., H-pad-1
    x = mx.concatenate([top, x, bottom], axis=1)
    # Reflect-pad W: same for columns
    left = x[:, :, pad:0:-1, :]
    right = x[:, :, -2:-(pad+2):-1, :]
    x = mx.concatenate([left, x, right], axis=2)
    return x


def _bounded_conv(image, convolution, padding, tile_rows):
    if not tile_rows or image.shape[1] <= tile_rows:
        return convolution(_reflect_pad(image, padding))
    height = image.shape[1]
    padded = _reflect_pad(image, padding)
    mx.eval(padded)
    outputs = []
    for start in range(0, height, tile_rows):
        end = min(height, start + tile_rows)
        output = convolution(padded[:, start:end + 2 * padding])
        mx.eval(output)
        outputs.append(output)
    return mx.concatenate(outputs, axis=1)


class EncBlock(nn.Module):
    """Residual conv block with GroupNorm + SiLU + reflect padding."""

    def __init__(self, channels: int, kernel_size: int = 3, num_groups: int = 8):
        super().__init__()
        self.pad = kernel_size // 2
        self.tile_rows = 0
        self.norm1 = nn.GroupNorm(num_groups, channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=0)
        self.norm2 = nn.GroupNorm(num_groups, channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.silu(self.norm1(x))
        h = _bounded_conv(h, self.conv1, self.pad, self.tile_rows)
        if self.tile_rows:
            mx.eval(h)
        h = nn.silu(self.norm2(h))
        h = _bounded_conv(h, self.conv2, self.pad, self.tile_rows)
        return h


class ReflectConv2d(nn.Module):
    """Conv2d with reflect padding (matching upstream padding_mode='reflect')."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.pad = kernel_size // 2
        self.tile_rows = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def __call__(self, x):
        return _bounded_conv(x, self.conv, self.pad, self.tile_rows)


def make_encoder(in_channels: int, out_channels: int, kernel_size: int = 1,
                 ks_res: int = 1, num_layers: int = 2):
    """Build encoder: ReflectConv2d → EncBlock × num_layers."""
    layers = [ReflectConv2d(in_channels, out_channels, kernel_size)]
    for _ in range(num_layers):
        layers.append(EncBlock(out_channels, kernel_size=ks_res))
    return layers


# ── Neighborhood attention as masked full attention ──

def _natten_window_indices(length: int, kernel: int, dilation: int) -> np.ndarray:
    """Per-position neighbor indices with NATTEN border semantics.

    Dilated NA is standard NA within each interleaved dilation group
    (natten token_permute); standard NA shifts windows inward at borders
    (natten na_utils.cuh get_window_start, stride=1). Returns [length, kernel].
    """
    i = np.arange(length)
    group = i % dilation
    a = i // dilation
    group_len = (length - group + dilation - 1) // dilation
    if np.any(group_len < kernel):
        raise ValueError(
            f"neighborhood kernel {kernel} with dilation {dilation} needs "
            f"length >= {kernel * dilation}, got {length}"
        )
    start = np.clip(a - kernel // 2, 0, group_len - kernel)
    return (group[:, None]
            + (start[:, None] + np.arange(kernel)[None, :]) * dilation).astype(np.int32)


def neighborhood_attention(q, k, v, kernel_size, scale, dilation=(1, 1)):
    """Neighborhood attention via gather-based windowed attention.

    Each query attends only to its K×K spatial neighbors, with optional
    dilation. With dilation=d, neighbors are spaced d apart, giving a
    receptive field of (K*d - d + 1) × (K*d - d + 1).

    O(N × K²) instead of O(N²). Works for any grid size.

    Args:
        q: [B, H, X, Y, Dq] queries
        k: [B, H, X, Y, Dq] keys
        v: [B, H, X, Y, Dv] values (may differ from Dq)
        kernel_size: (Kh, Kw) or int neighborhood size
        scale: attention scale factor
        dilation: (dh, dw) dilation factor for the neighborhood window

    Returns:
        [B, H, X, Y, Dv] output
    """
    B, nH, X, Y, Dq = q.shape
    Dv = v.shape[-1]
    Kh, Kw = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)

    # NATTEN semantics: no padding. Windows shift inward at borders so every
    # query sees K distinct in-bounds keys within its dilation group.
    idx_h = _natten_window_indices(X, Kh, dh)  # [X, Kh]
    idx_w = mx.array(_natten_window_indices(Y, Kw, dw))  # [Y, Kw]

    # Process in row chunks to control peak memory.
    # Each chunk allocates [B, H, chunk, Y, K², D] window tensors.
    # At chunk=64, Y=512, K²=81, D=1024, H=4: ~27GB per chunk.
    # At chunk=8: ~3.4GB. At chunk=4: ~1.7GB.
    CHUNK_ROWS = max(1, min(X, 16))

    out_chunks = []
    for row_start in range(0, X, CHUNK_ROWS):
        row_end = min(row_start + CHUNK_ROWS, X)
        chunk_x = row_end - row_start

        # Gather neighbor windows: rows/cols come from NATTEN index tables.
        k_windows = []
        v_windows = []
        for di in range(Kh):
            rows = mx.array(idx_h[row_start:row_end, di])
            k_rows = mx.take(k, rows, axis=2)
            v_rows = mx.take(v, rows, axis=2)
            for dj in range(Kw):
                cols = idx_w[:, dj]
                k_windows.append(mx.take(k_rows, cols, axis=3))
                v_windows.append(mx.take(v_rows, cols, axis=3))

        k_nbr = mx.stack(k_windows, axis=4)
        v_nbr = mx.stack(v_windows, axis=4)

        q_chunk = q[:, :, row_start:row_end, :, :]
        q_exp = q_chunk[:, :, :, :, None, :]

        attn = (q_exp @ k_nbr.transpose(0, 1, 2, 3, 5, 4)) * scale
        attn = mx.softmax(attn, axis=-1)
        chunk_out = (attn @ v_nbr).squeeze(4)
        out_chunks.append(chunk_out)
        mx.eval(chunk_out)

    return mx.concatenate(out_chunks, axis=2)


def fused_neighborhood_attention(queries, keys, values, kernel_size, scale,
                                 dilation=(1, 1), upsample=(1, 1), threadgroup_size=128):
    from ..metal_kernels import neighborhood_attention_metal

    return neighborhood_attention_metal(
        queries, keys, values, kernel_size, scale, dilation=dilation, upsample=upsample,
        threadgroup_size=threadgroup_size)


# ── Cross Attention ──

class NAFCrossAttention(nn.Module):
    """Cross-attention with neighborhood attention.

    Q comes from upsampled image features, K/V from DINOv3 patch features.
    K and V are nearest-upsampled to Q resolution, then neighborhood
    attention is applied.
    """

    def __init__(self, dim: int, num_heads: int, kernel_size=(9, 9), backend="auto"):
        super().__init__()
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.scale = (dim // num_heads) ** -0.5
        self.backend = backend
        self.threadgroup_size = 128

    def __call__(self, q, k, v, image=None):
        """
        Args:
            q: [B, C, Hq, Wq] image-encoded features (query resolution)
            k: [B, C, Hk, Wk] image-encoded features (key resolution, ≤ Hq)
            v: [B, Cv, Hk, Wk] DINOv3 features (value, same res as k)

        Returns:
            [B, Cv, Hq, Wq] upsampled features
        """
        B, C, Hq, Wq = q.shape
        Hk, Wk = k.shape[2], k.shape[3]
        n = self.num_heads

        # Reshape Q to [B, n, Hq, Wq, d]
        dq = C // n
        q = q.reshape(B, n, dq, Hq, Wq).transpose(0, 1, 3, 4, 2)  # [B, n, Hq, Wq, dq]

        if self.backend == "metal" or (self.backend == "auto" and mx.default_device() == mx.gpu):
            value_channels = v.shape[1]
            keys = k.reshape(B, n, dq, Hk, Wk).transpose(0, 1, 3, 4, 2)
            values = v.reshape(B, n, value_channels // n, Hk, Wk).transpose(0, 1, 3, 4, 2)
            upsample = (Hq // Hk, Wq // Wk)
            output = fused_neighborhood_attention(
                q, keys, values, self.kernel_size, self.scale,
                dilation=upsample, upsample=upsample, threadgroup_size=self.threadgroup_size)
            return output.transpose(0, 1, 4, 2, 3).reshape(B, value_channels, Hq, Wq)
        if self.backend not in ("auto", "reference"):
            raise ValueError(f"Unknown NAF backend: {self.backend}")

        # Upsample K to query resolution via nearest
        # K: [B, C, Hk, Wk] → [B, C, Hq, Wq]
        k_up = mx.repeat(mx.repeat(k, Hq // Hk, axis=2), Wq // Wk, axis=3)
        k_up = k_up.reshape(B, n, dq, Hq, Wq).transpose(0, 1, 3, 4, 2)

        # Upsample V to query resolution via nearest
        # V may have different channel dim than Q/K (e.g. 1024 vs 256)
        Cv = v.shape[1]
        v_up = mx.repeat(mx.repeat(v, Hq // Hk, axis=2), Wq // Wk, axis=3)
        # For neighborhood attention, V needs same head structure as Q
        # Split V into same number of heads, each with Cv//n channels
        dv = Cv // n
        v_up = v_up.reshape(B, n, dv, Hq, Wq).transpose(0, 1, 3, 4, 2)

        # Dilation from resolution ratio — each query's K×K window is spaced
        # dilation apart, covering the full original-resolution neighborhood
        dilation = (Hq // Hk, Wq // Wk)

        # Neighborhood attention with dilation
        out = neighborhood_attention(q, k_up, v_up, self.kernel_size, self.scale, dilation=dilation)

        # Reshape back to [B, Cv, Hq, Wq]
        out = out.transpose(0, 1, 4, 2, 3).reshape(B, Cv, Hq, Wq)
        return out


# ── NAF Model ──

class NAF(nn.Module):
    """Neural Attention Fields feature upsampler.

    Takes a guide image and low-res VFM features, produces high-res features
    via cross-attention with neighborhood masking.

    Config (from pretrained):
        dim: 256
        heads_attn: 4
        heads_rope: 4
        kernel_size: 9
        img_layers: 2
        rope_base: 100.0
        rope_rescale: 2.0
    """

    def __init__(
        self,
        dim: int = 256,
        heads_attn: int = 4,
        heads_rope: int = 4,
        kernel_size: int = 9,
        img_layers: int = 2,
        rope_base: float = 100.0,
    ):
        super().__init__()
        self.dim = dim
        self.encoder_tile_rows = 128

        # Image encoder: dual path (1×1 + 3×3) → concat → pool → RoPE
        self.encoder_1x1 = make_encoder(3, dim // 2, kernel_size=1, ks_res=1, num_layers=img_layers)
        self.encoder_3x3 = make_encoder(3, dim // 2, kernel_size=3, ks_res=3, num_layers=img_layers)
        self.rope = NAFRoPE(dim, num_heads=heads_rope, base=rope_base)

        # Cross-attention upsampler
        self.upsampler = NAFCrossAttention(dim=dim, num_heads=heads_attn, kernel_size=(kernel_size, kernel_size))

    def _run_encoder(self, layers, x):
        for layer in layers:
            layer.tile_rows = self.encoder_tile_rows
            x = layer(x)
            if self.encoder_tile_rows:
                mx.eval(x)
        return x

    def _encode_image(self, image, output_size):
        """Encode guide image to feature map at output_size resolution.

        Args:
            image: [B, H, W, 3] in [0, 1] (MLX channels-last)
            output_size: (H', W') target spatial size

        Returns:
            [B, dim, H', W'] encoded features with RoPE
        """
        # Pre-downsample if image is >4x target (matches upstream guard)
        oh, ow = output_size
        if image.shape[1] > 4 * oh or image.shape[2] > 4 * ow:
            target_h = min(image.shape[1], 4 * oh, 4 * ow)
            target_w = min(image.shape[2], 4 * ow, 4 * oh)
            image = nn.Upsample(scale_factor=target_h / image.shape[1],
                                mode="linear", align_corners=False)(image)

        if self.encoder_tile_rows:
            pooled = []
            for layers in (self.encoder_1x1, self.encoder_3x3):
                encoded = self._run_encoder(layers, image)
                batch, height, width, channels = encoded.shape
                branch = encoded.reshape(batch, oh, height // oh, ow, width // ow, channels).mean(axis=(2, 4))
                mx.eval(branch)
                pooled.append(branch)
                del encoded
            combined = mx.concatenate(pooled, axis=-1).transpose(0, 3, 1, 2)
            mx.eval(combined)
            return self.rope(combined)

        # Run encoders in NHWC
        enc_1x1 = self._run_encoder(self.encoder_1x1, image)  # [B, H, W, dim//2]
        enc_3x3 = self._run_encoder(self.encoder_3x3, image)  # [B, H, W, dim//2]
        x = mx.concatenate([enc_1x1, enc_3x3], axis=-1)  # [B, H, W, dim]

        # Adaptive avg pool: convert to BCHW, pool, convert back
        x_bchw = x.transpose(0, 3, 1, 2)  # [B, dim, H, W]
        # Simple pooling via reshape+mean for exact output_size
        B, C, H, W = x_bchw.shape
        oh, ow = output_size
        x_pooled = x_bchw.reshape(B, C, oh, H // oh, ow, W // ow).mean(axis=(3, 5))

        # Apply RoPE (operates on BCHW)
        x_rope = self.rope(x_pooled)  # [B, dim, oh, ow]

        return x_rope

    def _prepare_attention(self, image, features, output_size):
        """Upsample VFM features using image guidance.

        Args:
            image: [B, 3, H, W] guide image in [0, 1] (BCHW for compat with upstream)
            features: [B, C, h, w] low-res VFM features (BCHW)
            output_size: (H', W') target resolution

        Returns:
            [B, C, H', W'] upsampled features (BCHW)
        """
        # Convert image from BCHW to NHWC for MLX Conv2d
        image_nhwc = image.transpose(0, 2, 3, 1)  # [B, H, W, 3]

        # Encode image at output resolution
        x = self._encode_image(image_nhwc, output_size)  # [B, dim, H', W'] BCHW

        # Q = encoded image features at output resolution
        queries = x

        # K = encoded image features pooled to feature resolution
        # adaptive_avg_pool2d to features spatial size
        Hf, Wf = features.shape[2], features.shape[3]
        Hq, Wq = output_size
        B, C = x.shape[0], x.shape[1]
        keys = x.reshape(B, C, Hf, Hq // Hf, Wf, Wq // Wf).mean(axis=(3, 5))

        # V = input features (DINOv3 patch features)
        values = features

        # Cross-attention with neighborhood masking
        return queries, keys, values

    def __call__(self, image, features, output_size):
        return self.upsampler(*self._prepare_attention(image, features, output_size))

    def project(self, image, features, output_size, queries_ndc, chunk_size=1024):
        from ..metal_kernels import neighborhood_attention_metal
        from ..modules.proj_grid import _grid_sample_bilinear

        if chunk_size < 1:
            raise ValueError("NAF projection chunk size must be positive")
        if queries_ndc.shape[0] != 1 or image.shape[0] != 1:
            raise ValueError("Sparse NAF projection currently requires one image")
        if self.upsampler.backend == "reference" or mx.default_device() != mx.gpu:
            return _grid_sample_bilinear(self(image, features, output_size), queries_ndc).transpose(0, 2, 1)
        if queries_ndc.shape[1] == 0:
            return mx.zeros((1, 0, features.shape[1]), dtype=features.dtype)
        queries, keys, values = self._prepare_attention(image, features, output_size)
        heads = self.upsampler.num_heads
        height, width = output_size
        source_height, source_width = features.shape[2:]
        query_channels = queries.shape[1] // heads
        value_channels = values.shape[1] // heads
        queries = mx.contiguous(queries.reshape(1, heads, query_channels, height, width).transpose(0, 1, 3, 4, 2))
        keys = mx.contiguous(keys.reshape(1, heads, query_channels, source_height, source_width).transpose(0, 1, 3, 4, 2))
        values = mx.contiguous(values.reshape(1, heads, value_channels, source_height, source_width).transpose(0, 1, 3, 4, 2))
        mx.eval(queries, keys, values)
        outputs = []
        upsample = (height // source_height, width // source_width)
        for start in range(0, queries_ndc.shape[1], chunk_size):
            grid = queries_ndc[0, start:start + chunk_size]
            horizontal = (grid[:, 0] + 1) * 0.5 * width - 0.5
            vertical = (grid[:, 1] + 1) * 0.5 * height - 0.5
            left = mx.floor(horizontal).astype(mx.int32)
            top = mx.floor(vertical).astype(mx.int32)
            horizontal_weight = horizontal - left.astype(mx.float32)
            vertical_weight = vertical - top.astype(mx.float32)
            columns = mx.clip(mx.stack([left, left + 1, left, left + 1], axis=1), 0, width - 1)
            rows = mx.clip(mx.stack([top, top, top + 1, top + 1], axis=1), 0, height - 1)
            positions = (rows * width + columns).reshape(-1)
            corners = neighborhood_attention_metal(
                queries, keys, values, self.upsampler.kernel_size, self.upsampler.scale,
                dilation=upsample, upsample=upsample, positions=positions,
                threadgroup_size=self.upsampler.threadgroup_size,
            ).reshape(1, heads, grid.shape[0], 4, value_channels)
            first = corners[:, :, :, 0] * (1 - horizontal_weight)[None, None, :, None] * (1 - vertical_weight)[None, None, :, None]
            second = corners[:, :, :, 1] * horizontal_weight[None, None, :, None] * (1 - vertical_weight)[None, None, :, None]
            third = corners[:, :, :, 2] * (1 - horizontal_weight)[None, None, :, None] * vertical_weight[None, None, :, None]
            fourth = corners[:, :, :, 3] * horizontal_weight[None, None, :, None] * vertical_weight[None, None, :, None]
            output = (first + second + third + fourth).transpose(0, 2, 1, 3).reshape(1, grid.shape[0], heads * value_channels)
            mx.eval(output)
            outputs.append(output)
        return mx.concatenate(outputs, axis=1)
