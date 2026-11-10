"""Convert a checksum-pinned MoGe checkpoint once, outside the MLX environment."""

import argparse
import hashlib
import json
from pathlib import Path


def convert(source, output, expected_sha256):
    import torch
    from safetensors.torch import save_file

    digest = hashlib.sha256()
    with source.open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(8 * 1024 ** 2), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Official MoGe checkpoint checksum mismatch")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not weights or not all(isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError("MoGe checkpoint does not contain a tensor state dictionary")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.safetensors")
    save_file({name: value.detach().contiguous() for name, value in weights.items()}, str(temporary))
    temporary.replace(output)
    manifest = {"source": str(source.resolve()), "source_sha256": expected_sha256,
                "output": str(output.resolve()), "tensors": len(weights), "weights_only": True}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "weights/moge_release.safetensors")
    parser.add_argument("--expected-sha256", required=True)
    arguments = parser.parse_args()
    convert(arguments.input, arguments.output, arguments.expected_sha256)


if __name__ == "__main__":
    main()
