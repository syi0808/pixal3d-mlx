"""Download pinned official Pixal3D dependencies without copying the HF cache."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


PIXAL_REPO = "TencentARC/Pixal3D"
PIXAL_REVISION = "0b31f9160aa400719af409098bff7936a932f726"
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
MOGE_REPO = "Ruicheng/moge-2-vitl"
MOGE_REVISION = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
NAF_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
NAF_SHA256 = "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_models(skip_dino=False, verify=False):
    root = Path(__file__).resolve().parents[1]
    weights = root / "weights"
    weights.mkdir(exist_ok=True)
    report = {"models": {}, "dino_requested": not skip_dino}
    manifest_path = weights / "download-manifest.json"

    def record(name, value):
        report["models"][name] = value
        manifest_path.write_text(json.dumps(report, indent=2) + "\n")

    print("Downloading seven single-view Pixal3D checkpoints (~24 GB).", flush=True)
    pixal_path = snapshot_download(
        PIXAL_REPO, revision=PIXAL_REVISION,
        allow_patterns=["ckpts/*.safetensors", "ckpts/*.json", "pipeline.json"],
        max_workers=2,
    )
    record("pixal3d", {"repo": PIXAL_REPO, "revision": PIXAL_REVISION, "path": pixal_path})
    if verify:
        metadata = HfApi().model_info(PIXAL_REPO, revision=PIXAL_REVISION, files_metadata=True)
        verified = []
        for entry in metadata.siblings:
            if entry.lfs and entry.rfilename.startswith("ckpts/"):
                path = Path(pixal_path) / entry.rfilename
                actual = sha256_file(path)
                if actual != entry.lfs.sha256:
                    raise ValueError(f"Checksum mismatch: {entry.rfilename}")
                verified.append(entry.rfilename)
        report["models"]["pixal3d"]["sha256_verified"] = verified
        record("pixal3d", report["models"]["pixal3d"])

    print("Downloading the official MoGe camera checkpoint (~1.3 GB).", flush=True)
    moge_path = hf_hub_download(MOGE_REPO, "model.pt", revision=MOGE_REVISION)
    record("moge", {"repo": MOGE_REPO, "revision": MOGE_REVISION, "path": moge_path})

    naf_path = weights / "naf_release.pth"
    if not naf_path.exists():
        temporary = naf_path.with_suffix(".download")
        with urllib.request.urlopen(NAF_URL) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        temporary.replace(naf_path)
    if sha256_file(naf_path) != NAF_SHA256:
        raise ValueError("NAF checksum mismatch")
    record("naf", {"url": NAF_URL, "sha256": NAF_SHA256, "path": str(naf_path)})

    if not skip_dino:
        print("Downloading gated DINOv3 with the local HF credential.", flush=True)
        dino_path = snapshot_download(
            DINO_REPO, revision=DINO_REVISION,
            allow_patterns=["*.safetensors", "config.json"], max_workers=1,
        )
        record("dinov3", {"repo": DINO_REPO, "revision": DINO_REVISION, "path": dino_path})
    print(f"Manifest: {manifest_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    download_models(skip_dino=arguments.skip_dino, verify=arguments.verify)


if __name__ == "__main__":
    main()
