"""Evaluate all schemes on the simulated channel. Results go to results/eval/*.csv (one row per scheme/setting).

    python scripts/evaluate.py cifar      # CIFAR-10 test images: analog JSCC, semantic JSCC, digital AE+code, task
    python scripts/evaluate.py kodak      # Kodak 768x512: JSCC, semantic JSCC, JPEG + conv. code + QAM
    python scripts/evaluate.py video      # 3 CIF clips: frame-by-frame schemes + conditional replenishment
    python scripts/evaluate.py all
    python scripts/evaluate.py kodak --part separated      # only the digital baselines (or --part learned)
Digital points are simulated bit-exactly (packets coded, modulated, sent over AWGN and Viterbi-decoded); the MCS
loops run in parallel processes.
"""
import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from semcom import chdir_project_root  # noqa: E402

chdir_project_root()
from semcom import digital as dg  # noqa: E402
from semcom.channel import ratio_from_channels  # noqa: E402
from semcom.data import cifar10, kodak, video_clips  # noqa: E402
from semcom.metrics import psnr, semantic_similarity, ssim  # noqa: E402
from semcom.train import JSCC_WIDTHS, TASK_WIDTHS, load, pick_device  # noqa: E402
from semcom.video import conditional_replenishment, frame_by_frame  # noqa: E402

DEV = pick_device()  # learned models run here; digital baselines and all metrics stay on CPU
OUT = os.path.join("results", "eval")
SNR_FINE = list(range(0, 21))
SNR_COARSE = list(range(0, 21, 2))
N_CIFAR = 5000
MAIN_C = 8  # 1/12, main operating point


def save(rows, name):
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, name), index=False)
    print("wrote", os.path.join(OUT, name), len(df), "rows", flush=True)
    return df


@torch.no_grad()
def batched(fn, x, bs=500, dev=None):
    """Apply fn in chunks. With dev set, chunks go to that device and results come back on the CPU,
    so every metric downstream still sees CPU tensors."""
    out = []
    for i in range(0, len(x), bs):
        t = x[i:i + bs]
        out.append(fn(t).cpu() if dev is None else fn(t.to(dev)).cpu())
    return torch.cat(out)


# =============================================================================================== CIFAR
def _cifar_metrics(x, xh, y, judge):
    # take the device from the judge itself: the digital worker processes keep theirs on the CPU
    jdev = next(judge.parameters()).device
    return dict(psnr=psnr(xh, x).mean().item(), ssim=ssim(xh, x).mean().item(),
                acc=(batched(judge, xh, dev=None if jdev.type == "cpu" else jdev).argmax(1) == y)
                .float().mean().item())


def _dae_point(args):
    """One MCS, all SNRs, one bandwidth, CIFAR digital chain (runs in a worker process)."""
    torch.set_num_threads(1)
    mcs, c_ratio, snrs, kind = args
    x, y = cifar10(False)
    x, y = x[:N_CIFAR], y[:N_CIFAR]
    dae, judge = load("dae"), load("clsB")
    k = int(c_ratio * 3 * 32 * 32 / 96)  # complex symbols available for the same ratio as c_ratio channels
    nb = dg.max_bits(k, mcs)
    c = min(dae.c, nb // (64 * dae.bits))
    rows = []
    with torch.no_grad():
        lev = dae.to_levels(x, c) if c else None
    for snr in snrs:
        rng = np.random.default_rng(1000 + int(snr))
        if c:
            bits = ((lev.numpy().reshape(len(x), -1)[:, :, None] >> np.arange(dae.bits - 1, -1, -1)) & 1)
            streams = [b.reshape(-1).astype(np.uint8) for b in bits]
            gain = rng.exponential(1.0, len(x)) if kind == "rayleigh" else None
            rec = dg.transmit_bits(streams, mcs, snr, rng, gain)
            ber = float(np.mean([np.mean(a != b) for a, b in zip(streams, rec)]))
            rl = np.stack([r.reshape(-1, dae.bits) @ (1 << np.arange(dae.bits - 1, -1, -1)) for r in rec])
            q = torch.from_numpy(rl).view(len(x), c, 8, 8)
            with torch.no_grad():
                xh = batched(lambda t: dae.from_levels(t, c), q)
        else:
            ber = float("nan")
            with torch.no_grad():
                xh = batched(lambda t: dae.from_levels(torch.zeros(len(t), 0, 8, 8, dtype=torch.int64), 0), x)
        rows.append(dict(dataset="cifar", scheme="sep_ae", mcs=dg.mcs_name(mcs), ratio=ratio_from_channels(c_ratio),
                         snr=snr, channel=kind, c=c, ber=ber, **_cifar_metrics(x, xh, y, judge)))
    return rows


def eval_cifar(workers, part="all"):
    torch.set_num_threads(8)
    x, y = cifar10(False)
    x, y = x[:N_CIFAR], y[:N_CIFAR]
    judge, clsA = load("clsB", DEV), load("clsA", DEV)
    print("learned models on", DEV, flush=True)
    if part in ("all", "separated"):
        eval_cifar_separated(workers, x, y, judge)
    if part == "separated":
        return
    rows = [dict(dataset="cifar", scheme="original",
                 acc=(batched(judge, x, dev=DEV).argmax(1) == y).float().mean().item(),
                 acc_clsA=(batched(clsA, x, dev=DEV).argmax(1) == y).float().mean().item(),
                 psnr=float("inf"), ssim=1.0)]
    # analog JSCC (syntactic and semantic): every bandwidth, every SNR, AWGN + block Rayleigh
    for name in ["jscc_mse", "jscc_sem"]:
        m = load(name, DEV)
        for kind, snrs in (("awgn", SNR_FINE), ("rayleigh", SNR_COARSE)):
            for c in (JSCC_WIDTHS if kind == "awgn" else [MAIN_C]):
                for snr in snrs:
                    torch.manual_seed(snr)
                    xh = batched(lambda t: m(t, float(snr), c, kind), x, dev=DEV)
                    rows.append(dict(dataset="cifar", scheme=name, ratio=ratio_from_channels(c), snr=snr, channel=kind,
                                     c=c, acc_clsA=(batched(clsA, xh, dev=DEV).argmax(1) == y).float().mean().item(),
                                     **_cifar_metrics(x, xh, y, judge)))
        print(name, "done", flush=True)
    # train/test SNR mismatch
    for name in ["jscc_mse_snr0", "jscc_mse_snr10", "jscc_mse_snr20"]:
        m = load(name, DEV)
        for snr in SNR_FINE:
            torch.manual_seed(snr)
            xh = batched(lambda t: m(t, float(snr), MAIN_C), x, dev=DEV)
            rows.append(dict(dataset="cifar", scheme=name, ratio=ratio_from_channels(MAIN_C), snr=snr, channel="awgn",
                             c=MAIN_C, **_cifar_metrics(x, xh, y, judge)))
    # task-oriented transmission (extension): accuracy only
    task = load("task", DEV)
    for kind in ("awgn", "rayleigh"):
        for c in TASK_WIDTHS:
            for snr in SNR_COARSE:
                torch.manual_seed(snr)
                acc = (batched(lambda t: task(t, float(snr), c, kind), x,
                               dev=DEV).argmax(1) == y).float().mean().item()
                rows.append(dict(dataset="cifar", scheme="task", ratio=c / 384, snr=snr, channel=kind, c=c, acc=acc))
    save(rows, "cifar_analog.csv")


def eval_cifar_separated(workers, x, y, judge):
    # capacity bound of the digital AE: as many 4-bit channels as k*log2(1+SNR) bits allow, error-free
    rows = []
    dae = load("dae")
    for c_ratio in (MAIN_C, 16):
        k = c_ratio * 3 * 32 * 32 / 96
        for snr in SNR_FINE:
            c = min(16, int(k * math.log2(1 + 10 ** (snr / 10)) // (64 * dae.bits)))
            xh = batched(lambda t: dae(t, c) if c else dae.from_levels(torch.zeros(len(t), 0, 8, 8,
                                                                                      dtype=torch.int64), 0), x)
            rows.append(dict(dataset="cifar", scheme="sep_ae_capacity", ratio=ratio_from_channels(c_ratio), snr=snr,
                             channel="awgn", c=c, **_cifar_metrics(x, xh, y, judge)))
    save(rows, "cifar_capacity.csv")
    # separated digital chain, bit-exact, one process per (MCS, bandwidth, channel)
    jobs = [(mcs, cr, SNR_FINE, "awgn") for mcs in dg.MCS for cr in (MAIN_C, 16)]
    jobs += [(mcs, MAIN_C, SNR_COARSE, "rayleigh") for mcs in dg.MCS]
    drows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_dae_point, jobs):
            drows += r
            print("digital", r[0]["mcs"], r[0]["ratio"], r[0]["channel"], flush=True)
    save(drows, "cifar_digital.csv")


# =============================================================================================== Kodak
def _nat_metrics(x, xh):
    cos, agree, top5 = semantic_similarity(x, xh)
    return dict(psnr=psnr(xh, x).mean().item(), ssim=ssim(xh, x).mean().item(), sem_cos=cos.mean().item(),
                top1_agree=agree.mean().item(), top5_agree=top5.mean().item())


def _jpeg_point(args):
    """One MCS over all SNRs for a stack of natural images (Kodak or video frames), frame-by-frame JPEG."""
    torch.set_num_threads(2)
    mcs, imgs, snrs, kind, ratio, tag = args
    N, _, H, W = imgs.shape
    k = int(ratio * 3 * H * W)
    budget = dg.max_bits(k, mcs)
    enc = [dg.jpeg_fit(im.numpy(), budget) for im in imgs]
    rows = []
    for snr in snrs:
        rng = np.random.default_rng(2000 + snr)
        streams = [dg.bytes_to_bits(e[0]) if e[0] else np.zeros(0, np.uint8) for e in enc]
        gain = rng.exponential(1.0, N) if kind == "rayleigh" else None
        rec = dg.transmit_bits(streams, mcs, snr, rng, gain)
        out, ok = [], []
        for e, s, r in zip(enc, streams, rec):
            if e[0] is None:
                im, good = np.full((3, H, W), 0.5, np.float32), False
            elif np.array_equal(s, r):
                im, good = dg.jpeg_decode(e[0], (H, W))
            else:
                im, _ = dg.jpeg_decode(dg.bits_to_bytes(r), (H, W))
                good = False
            out.append(torch.from_numpy(im))
            ok.append(good)
        xh = torch.stack(out)
        rows.append(dict(tag=tag, scheme="sep_jpeg", mcs=dg.mcs_name(mcs), ratio=ratio, snr=snr, channel=kind,
                         jpeg_q=float(np.mean([e[1] for e in enc])), ok=float(np.mean(ok)), **_nat_metrics(imgs, xh)))
    return rows


def _jpeg_capacity(imgs, snrs, ratio, tag):
    N, _, H, W = imgs.shape
    k = ratio * 3 * H * W
    rows = []
    for snr in snrs:
        budget = int(k * math.log2(1 + 10 ** (snr / 10)))
        enc = [dg.jpeg_fit(im.numpy(), budget) for im in imgs]
        xh = torch.stack([torch.from_numpy(dg.jpeg_decode(e[0], (H, W))[0]) if e[0] else
                          torch.full((3, H, W), 0.5) for e in enc])
        rows.append(dict(tag=tag, scheme="sep_jpeg_capacity", ratio=ratio, snr=snr, channel="awgn",
                         jpeg_q=float(np.mean([e[1] for e in enc])), ok=float(np.mean([e[0] is not None for e in enc])),
                         **_nat_metrics(imgs, xh)))
    return rows


def eval_kodak(workers, part="all"):
    torch.set_num_threads(8)
    x, _ = kodak()
    rows = []
    for name in ([] if part == "separated" else ["jscc_mse", "jscc_sem"]):
        m = load(name, DEV)
        for kind in ("awgn", "rayleigh"):
            for c in ([4, 8, 16] if kind == "awgn" else [MAIN_C]):
                for snr in SNR_COARSE:
                    torch.manual_seed(snr)
                    xh = batched(lambda t: m(t, float(snr), c, kind), x, bs=4, dev=DEV)
                    rows.append(dict(tag="kodak", scheme=name, ratio=ratio_from_channels(c), snr=snr, channel=kind,
                                     **_nat_metrics(x, xh)))
        print("kodak", name, flush=True)
    if part != "separated":
        save(rows, "kodak_jscc.csv")
    if part == "learned":
        return
    rows = []
    for ratio in (ratio_from_channels(MAIN_C), ratio_from_channels(16)):
        rows += _jpeg_capacity(x, SNR_COARSE, ratio, "kodak")
    jobs = [(mcs, x, SNR_COARSE, "awgn", ratio_from_channels(MAIN_C), "kodak") for mcs in dg.MCS]
    jobs += [(mcs, x, SNR_COARSE, "awgn", ratio_from_channels(16), "kodak") for mcs in dg.MCS]
    jobs += [(mcs, x, SNR_COARSE, "rayleigh", ratio_from_channels(MAIN_C), "kodak") for mcs in dg.MCS]
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_jpeg_point, jobs):
            rows += r
            print("kodak jpeg", r[0]["mcs"], r[0]["ratio"], r[0]["channel"], flush=True)
    save(rows, "kodak_sep.csv")


# =============================================================================================== video
def eval_video(workers, part="all"):
    torch.set_num_threads(8)
    clips = video_clips(30)
    ratio = ratio_from_channels(MAIN_C)
    rows, per_frame = [], []
    for name in ([] if part == "separated" else ["jscc_mse", "jscc_sem"]):
        m = load(name, DEV)
        for clip, fr in clips.items():
            for snr in SNR_COARSE:
                xh = frame_by_frame(m, fr, float(snr), MAIN_C, seed=snr)
                rows.append(dict(tag=clip, scheme=name, ratio=ratio, snr=snr, channel="awgn", **_nat_metrics(fr, xh)))
            # extension: rate/quality trade-off at 10 dB, frame-by-frame vs conditional replenishment
            for c in JSCC_WIDTHS:
                xh = frame_by_frame(m, fr, 10.0, c, seed=7)
                rows.append(dict(tag=clip, scheme=name + "_fbf", ratio=ratio_from_channels(c), snr=10, channel="awgn",
                                 **_nat_metrics(fr, xh)))
            for ck in (8, 16):
                for f in (0.05, 0.1, 0.2, 0.4):
                    xh, r, side = conditional_replenishment(m, fr, 10.0, ck, f, gop=10, seed=7)
                    rows.append(dict(tag=clip, scheme=name + "_cr", ratio=r, snr=10, channel="awgn", c=ck, frac=f,
                                     side_share=side, **_nat_metrics(fr, xh)))
                    if name == "jscc_sem" and ck == 16 and f == 0.2:
                        p, s = psnr(xh, fr), ssim(xh, fr)
                        for t in range(len(fr)):
                            per_frame.append(dict(tag=clip, scheme="cr", t=t, psnr=p[t].item(), ssim=s[t].item(),
                                                  ratio=r))
            if name == "jscc_sem":
                xh = frame_by_frame(m, fr, 10.0, MAIN_C, seed=7)
                p = psnr(xh, fr)
                for t in range(len(fr)):
                    per_frame.append(dict(tag=clip, scheme="fbf", t=t, psnr=p[t].item(), ssim=ssim(xh, fr)[t].item(),
                                          ratio=ratio))
            print("video", name, clip, flush=True)
    if part != "separated":
        save(rows, "video_jscc.csv")
        save(per_frame, "video_frames.csv")
    if part == "learned":
        return
    rows = []
    for clip, fr in clips.items():
        rows += _jpeg_capacity(fr, SNR_COARSE, ratio, clip)
    jobs = [(mcs, fr, SNR_COARSE, "awgn", ratio, clip) for mcs in dg.MCS for clip, fr in clips.items()]
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_jpeg_point, jobs):
            rows += r
            print("video jpeg", r[0]["tag"], r[0]["mcs"], flush=True)
    save(rows, "video_sep.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cifar", "kodak", "video", "all"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--part", choices=["all", "learned", "separated"], default="all",
                    help="learned = JSCC/task models only, separated = digital baselines only")
    a = ap.parse_args()
    t0 = time.time()
    for w in (["cifar", "kodak", "video"] if a.what == "all" else [a.what]):
        {"cifar": eval_cifar, "kodak": eval_kodak, "video": eval_video}[w](a.workers, a.part)
    print(f"evaluation finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
