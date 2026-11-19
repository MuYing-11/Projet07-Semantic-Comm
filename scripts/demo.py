"""Visual demonstration: send one Kodak image over the simulated AWGN channel with the three systems.

    python scripts/demo.py                                  # kodim23, SNR 2 / 10 / 18 dB -> figures/visual.png
    python scripts/demo.py --image kodim05 --snr 0 6 12 --mcs "16QAM 1/2" --out figures/demo.png
The separated chain uses one fixed MCS (designed for 10 dB by default), which makes its cliff visible.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semcom import chdir_project_root  # noqa: E402

chdir_project_root()
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from semcom import digital as dg  # noqa: E402
from semcom.channel import ratio_from_channels  # noqa: E402
from semcom.data import kodak  # noqa: E402
from semcom.metrics import psnr, semantic_similarity  # noqa: E402
from semcom.train import load  # noqa: E402


def separated(img, mcs, snr, ratio, seed=0):
    _, H, W = img.shape
    budget = dg.max_bits(int(ratio * 3 * H * W), mcs)
    data, q = dg.jpeg_fit(img.numpy(), budget)
    if data is None:
        return torch.full_like(img, 0.5), q
    bits = dg.bytes_to_bits(data)
    rec = dg.transmit_bits([bits], mcs, snr, np.random.default_rng(seed))[0]
    im, _ = dg.jpeg_decode(dg.bits_to_bytes(rec) if not np.array_equal(bits, rec) else data, (H, W))
    return torch.from_numpy(im), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="kodim23")
    ap.add_argument("--snr", type=float, nargs="+", default=[2, 10, 18])
    ap.add_argument("--c", type=int, default=8, help="active latent channels (8 -> k/n = 1/12)")
    ap.add_argument("--mcs", default="16QAM 1/2")
    ap.add_argument("--out", default=os.path.join("figures", "visual.png"))
    a = ap.parse_args()
    x, names = kodak()
    img = x[names.index(a.image)]
    mcs = [m for m in dg.MCS if dg.mcs_name(m) == a.mcs][0]
    ratio = ratio_from_channels(a.c)
    models = {"Syntactic JSCC": load("jscc_mse"), "Semantic JSCC": load("jscc_sem")}
    rows = [f"Separated JPEG + {a.mcs}"] + list(models)
    fig, axs = plt.subplots(len(rows), len(a.snr) + 1, figsize=(3.2 * (len(a.snr) + 1), 2.3 * len(rows)))
    for i, r in enumerate(rows):
        axs[i, 0].imshow(img.permute(1, 2, 0).numpy())
        axs[i, 0].set_title("original" if i == 0 else "", fontsize=8)
        axs[i, 0].set_ylabel(r, fontsize=8)
        for j, snr in enumerate(a.snr):
            with torch.no_grad():
                if i == 0:
                    out, q = separated(img, mcs, snr, ratio)
                else:
                    torch.manual_seed(int(snr))
                    out = models[r](img[None], snr, a.c)[0]
            p = psnr(out[None], img[None]).item()
            cos, _, _ = semantic_similarity(img[None], out[None])
            axs[i, j + 1].imshow(out.clamp(0, 1).permute(1, 2, 0).numpy())
            axs[i, j + 1].set_title(f"SNR {snr:g} dB — PSNR {p:.1f} dB, sem. cos {cos.item():.2f}", fontsize=7)
            print(f"{r:32s} SNR {snr:5.1f} dB  PSNR {p:5.2f} dB  semantic cosine {cos.item():.3f}")
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"{a.image}, bandwidth ratio k/n = 1/{round(1 / ratio)}, simulated AWGN channel", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
