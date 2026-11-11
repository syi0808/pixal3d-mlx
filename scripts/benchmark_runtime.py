"""Measure isolated runtime stages; these are not end-to-end generation timings."""

import argparse
import gc
import json
from pathlib import Path
import platform
import subprocess
import time

import mlx.core as mx
import numpy as np

from pixalmlx.models.naf import fused_neighborhood_attention, neighborhood_attention
from pixalmlx.models.shape_slat_decoder import SparseChannel2Spatial
from pixalmlx.modules.proj_grid import ProjGrid
from pixalmlx.modules.sparse_conv import SparseConv3d, build_neighbor_map


def measure(function, repeats):
    result = function()
    mx.eval(result)
    mx.synchronize()
    del result
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        mx.eval(result)
        mx.synchronize()
        samples.append(time.perf_counter() - started)
        del result
    return {
        "median_seconds": float(np.median(samples)),
        "samples_seconds": samples,
        "peak_mlx_bytes": mx.get_peak_memory(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--naf-size", type=int, default=64)
    parser.add_argument("--only", help="Comma-separated benchmark names")
    arguments = parser.parse_args()
    rng = np.random.default_rng(42)
    mx.random.seed(42)
    flat = rng.choice(64 ** 3, size=12000, replace=False)
    coordinates = np.column_stack([np.zeros(len(flat), dtype=np.int32),
                                   np.array(np.unravel_index(flat, (64, 64, 64))).T]).astype(np.int32)
    coords = mx.array(coordinates)
    features = mx.array(rng.standard_normal((len(flat), 64)).astype(np.float32))
    mask = mx.array(rng.random((len(flat), 8)) > 0.5)
    convolution = SparseConv3d(32, 32)
    conv_coords = coords[:2000]
    conv_features = features[:2000, :32]
    neighbor_map = build_neighbor_map(conv_coords)
    image_features = mx.random.normal((1, 32, 32, 32))
    projector = ProjGrid(grid_resolution=32, image_resolution=512)
    naf_size = arguments.naf_size
    query = mx.random.normal((1, 4, naf_size, naf_size, 32))
    keys = mx.random.normal(query.shape)
    values = mx.random.normal((1, 4, naf_size, naf_size, 64))
    mx.eval(coords, features, mask, conv_features, image_features, query, keys, values)
    cases = {
        "neighbor_map_12000": lambda: build_neighbor_map(coords),
        "subdivision_12000": lambda: SparseChannel2Spatial.upsample(features, coords, mask),
        "sparse_conv_2000": lambda: convolution(conv_features, neighbor_map),
        "dense_projection_32": lambda: projector(image_features, mx.array([0.7]), mx.array([2.0]), mx.array([1.0])),
        f"naf_{naf_size}": lambda: neighborhood_attention(query, keys, values, (9, 9), 32 ** -0.5),
        f"naf_fused_{naf_size}": lambda: fused_neighborhood_attention(query, keys, values, (9, 9), 32 ** -0.5),
    }
    report = {
        "kind": "synthetic-component-benchmark",
        "platform": platform.platform(),
        "mlx_version": mx.__version__,
        "device": mx.device_info(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "repeats": arguments.repeats,
        "results": {},
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for name, function in cases.items():
        if arguments.only and name not in arguments.only.split(","):
            continue
        report["results"][name] = measure(function, arguments.repeats)
        print(name, report["results"][name], flush=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
