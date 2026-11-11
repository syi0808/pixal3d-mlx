"""Compare exact runtime paths without requiring gated DINOv3 weights."""

import argparse
import gc
import json
from pathlib import Path
import subprocess

import mlx.core as mx
import numpy as np

from scripts.benchmark_runtime import measure
from pixalmlx.models.naf import NAF
from pixalmlx.models.naf_loader import load_naf_weights
from pixalmlx.conditioning import ZeroConditioning
from pixalmlx.metal_kernels import neighborhood_attention_metal
from pixalmlx.models.pixal3d_flow import Pixal3DSLatFlowModel
from pixalmlx.modules.proj_grid import _grid_sample_bilinear
from pixalmlx.modules.sparse_conv import build_neighbor_map
from pixalmlx.runtime import InferenceRuntime
from pixalmlx.samplers import flow_euler_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--guide-size", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=6000)
    parser.add_argument("--only", choices=["sparse", "naf", "flow", "naf-sparse", "kernel"])
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.tokens < 1:
        parser.error("Repeats and tokens must be positive")
    report = {"kind": "synthetic-runtime-architecture", "mlx_version": mx.__version__,
              "device": mx.device_info(), "guide_size": arguments.guide_size, "tokens": arguments.tokens,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
              "results": {}}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    def record(name, operation):
        gc.collect()
        mx.clear_cache()
        report["results"][name] = measure(operation, arguments.repeats)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, report["results"][name], flush=True)

    with InferenceRuntime(model_cache_bytes=0, feature_cache_bytes=0):
        mx.random.seed(42)
        if arguments.only == "kernel":
            queries = mx.random.normal((1, 4, 128, 128, 64))
            keys = mx.random.normal(queries.shape)
            values = mx.random.normal((1, 4, 128, 128, 256))
            mx.eval(queries, keys, values)
            for threads in (64, 128, 256):
                record(f"naf_threads_{threads}", lambda: neighborhood_attention_metal(
                    queries, keys, values, (9, 9), 64 ** -0.5, threadgroup_size=threads))
        if arguments.only in (None, "sparse"):
            random = np.random.default_rng(42)
            flat = random.choice(64 ** 3, size=12000, replace=False)
            coords = mx.array(np.column_stack([np.zeros(12000), np.array(np.unravel_index(flat, (64, 64, 64))).T]).astype(np.int32))
            record("neighbors_cpu", lambda: tuple(build_neighbor_map(coords, backend="cpu")))
            record("neighbors_gpu", lambda: tuple(build_neighbor_map(coords, backend="gpu")))
            del coords
        if arguments.only in (None, "naf", "naf-sparse"):
            model = NAF()
            load_naf_weights(model, "weights/naf_release.safetensors", verbose=False)
            guide = mx.random.uniform(shape=(1, 3, arguments.guide_size, arguments.guide_size))
            features = mx.random.normal((1, 1024, arguments.guide_size // 16, arguments.guide_size // 16))
            points = mx.random.uniform(low=-1.25, high=1.25, shape=(1, arguments.tokens, 2))
            mx.eval(model.parameters(), guide, features, points)

            def dense_projection():
                return _grid_sample_bilinear(model(guide, features, (512, 512)), points).transpose(0, 2, 1)

            if arguments.only != "naf-sparse":
                if arguments.guide_size > 512:
                    raise ValueError("Use --only naf-sparse for large guides to avoid an unsafe dense reference")
                model.encoder_tile_rows = 0
                expected = dense_projection()
                mx.eval(expected)
                record("naf_legacy_encoder_dense_output", dense_projection)
                model.encoder_tile_rows = 128
                actual = model.project(guide, features, (512, 512), points)
                np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=4e-4, atol=3e-5)
                report["naf_max_absolute_difference"] = float(mx.max(mx.abs(actual - expected)).item())
                del expected, actual
                record("naf_tiled_encoder_dense_output", dense_projection)
            model.encoder_tile_rows = 128
            record("naf_tiled_encoder_sparse_output", lambda: model.project(guide, features, (512, 512), points))
            del model, guide, features, points
        if arguments.only in (None, "flow"):
            model = Pixal3DSLatFlowModel(in_channels=32, out_channels=32, model_channels=256,
                                         num_heads=8, num_blocks=4, mlp_hidden=1024,
                                         context_channels=128, proj_in_channels=256)
            noise = mx.random.normal((2048, 32))
            coordinates = mx.arange(2048 * 3).reshape(2048, 3)
            cond = {"global": mx.random.normal((1, 5, 128)), "proj": mx.random.normal((1, 2048, 256))}
            neg = ZeroConditioning({name: mx.zeros_like(value) for name, value in cond.items()})
            mx.eval(model.parameters(), noise, coordinates, cond, neg)

            def sample(cache, chunk):
                return flow_euler_sample(model, noise, cond, neg, steps=6, guidance_strength=7.5,
                                          verbose=False, coords=coordinates,
                                          conditioning_cache_bytes=cache, ffn_chunk_size=chunk)

            model.eval_every = 6
            expected = sample(0, 0)
            mx.eval(expected)
            record("flow_uncached", lambda: sample(0, 0))
            model.eval_every = 2
            record("flow_cached_unconditional_folding", lambda: sample(64 * 1024 ** 2, 0))
            actual = sample(64 * 1024 ** 2, 512)
            np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-3, atol=5e-5)
            report["flow_max_absolute_difference"] = float(mx.max(mx.abs(actual - expected)).item())
            del expected, actual
            record("flow_bounded_static_conditioning", lambda: sample(64 * 1024 ** 2, 512))
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
