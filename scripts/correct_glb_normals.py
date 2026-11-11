"""Correct static GLB normals without changing triangle geometry or embedded PBR textures."""

import argparse
import json
from pathlib import Path

from pixalmlx.glb_normals import accessor_array, correct_glb, image_hashes, read_glb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, default=0, help="Validation guard; 0 disables the cap")
    parser.add_argument("--crease-degrees", type=float, default=75.0)
    arguments = parser.parse_args()
    if arguments.max_faces < 0 or arguments.report.exists():
        parser.error("Use a nonnegative triangle limit and a new report path")
    report = correct_glb(arguments.input, arguments.output, arguments.max_faces, arguments.crease_degrees)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
