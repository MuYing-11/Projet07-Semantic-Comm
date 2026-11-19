"""Download the public datasets used by the project into data/ (about 300 MB in total).

    python scripts/download_data.py

- CIFAR-10 (training and test of every learned model), from the Hugging Face mirror uoft-cs/cifar10 (parquet,
  about 145 MB; the original Toronto server is much slower), converted once to data/cifar10/cifar10.npz.
- Kodak Lossless True Color Image Suite (24 images, 768x512): high-resolution image test set.
- Three CIF (352x288) test sequences from the Xiph.org "derf" collection (akiyo: static scene, foreman: moderate
  motion, bus: fast pan). They play the role of the short UVG clips suggested by the brief: the 1080p UVG raw files
  are several GB per sequence, too heavy for a CPU-only study. Only the first 40 frames are downloaded (HTTP range
  request) and the first 30 are used (see semcom/data.py).
- ImageNet ResNet-18 weights (torchvision), used only as an independent semantic evaluator on Kodak and video.
Files already present are skipped.
"""
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")

KODAK = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
CIFAR_HF = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"
XIPH = "https://media.xiph.org/video/derf/y4m/{}_cif.y4m"
CLIPS = ["akiyo", "foreman", "bus"]


def fetch(url, path, n_bytes=None):
    """Download url to path; with n_bytes, only the first n_bytes (HTTP range request)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("downloading", url)
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n_bytes - 1}"} if n_bytes else {})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, path)


def cifar():
    import io

    import numpy as np
    import pandas as pd
    from PIL import Image

    out = os.path.join(DATA, "cifar10", "cifar10.npz")
    if os.path.exists(out):
        return
    arrays = {}
    for split in ("train", "test"):
        path = os.path.join(DATA, "cifar10", f"{split}.parquet")
        fetch(CIFAR_HF.format(split), path)
        df = pd.read_parquet(path)
        arrays[f"x_{split}"] = np.stack([np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["img"]])
        arrays[f"y_{split}"] = df["label"].to_numpy().astype(np.int64)
    np.savez(out, **arrays)


def main():
    import torchvision

    cifar()
    for k in range(1, 25):
        fetch(KODAK.format(k), os.path.join(DATA, "kodak", f"kodim{k:02d}.png"))
    for c in CLIPS:
        # only the first 40 CIF frames (6 + 152064 bytes each) are needed: ranged download of ~6 MB per clip
        fetch(XIPH.format(c), os.path.join(DATA, "video", f"{c}_cif.y4m"), n_bytes=40 * 152070 + 200)
    os.environ.setdefault("TORCH_HOME", os.path.join(DATA, "torch_home"))
    torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    print("data ready in", DATA)


if __name__ == "__main__":
    main()
