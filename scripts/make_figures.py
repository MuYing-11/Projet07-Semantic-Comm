"""Aggregate the evaluation CSVs into tables (results/tables/) and article figures (figures/*.png).

    python scripts/make_figures.py
All curves come from simulations of the AWGN / block-Rayleigh channel (no real radio link).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semcom import chdir_project_root  # noqa: E402

chdir_project_root()
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

FIG, TAB, EV = "figures", os.path.join("results", "tables"), os.path.join("results", "eval")
# categorical slots of the validated reference palette (same as project 1); baselines are neutral references
COLOR = {"jscc_mse": "#2a78d6", "jscc_sem": "#eb6834", "task": "#1baf7a", "sep": "#52514e", "cap": "#a3a29c",
         "cr": "#eb6834", "snr0": "#1baf7a", "snr10": "#eda100", "snr20": "#e87ba4"}
LABEL = {"jscc_mse": "Syntactic JSCC (MSE)", "jscc_sem": "Semantic JSCC (ours)", "sep": "Separated: best MCS",
         "cap": "Separated: capacity bound", "task": "Task-oriented (ext.)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MAIN_RATIO = 1 / 12
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "axes.titlesize": 10, "axes.titlecolor": INK,
    "savefig.dpi": 220, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})
MARK = {"jscc_mse": "o", "jscc_sem": "s", "sep": "^", "cap": None, "task": "D"}


def read(name):
    p = os.path.join(EV, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def close(a, b):
    return np.isclose(a, b, rtol=1e-3)


def envelope(dig, key):
    """Adaptive MCS with a genie transmitter that knows the SNR: best MCS per SNR (by `key`)."""
    return dig.loc[dig.groupby("snr")[key].idxmax()].sort_values("snr")


def curve(ax, df, key, who, **kw):
    df = df.sort_values("snr")
    ax.plot(df.snr, df[key], color=COLOR[who], marker=MARK.get(who), ms=4, label=LABEL.get(who, who), **kw)


# ---------------------------------------------------------------------------------------------- architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, txt, fc="#ffffff", ec=INK2, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK)

    def arrow(x0, x1, y):
        ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))

    rows = [(36, "(a) Separated digital chain", ["Source coder\nJPEG or 4-bit AE", "Conv. code K=7\nrate 1/2–3/4",
                                                  "Gray QAM\nQPSK/16/64", "AWGN /\nRayleigh", "Soft demod.\n+ Viterbi",
                                                  "Source\ndecoder"], "#f1f1ef"),
            (21, "(b) Syntactic deep JSCC", ["CNN encoder\n(stride 4)", "Truncate to c\nchannels",
                                            "Power norm.\n‖x‖² = k", "AWGN /\nRayleigh", "CNN decoder", "x̂"], "#e6f0fb"),
            (6, "(c) Semantic deep JSCC", ["CNN encoder\n(stride 4)", "Truncate to c\nchannels", "Power norm.\n‖x‖² = k",
                                          "AWGN /\nRayleigh", "CNN decoder", "x̂ → frozen\nclassifier A"], "#fdeee6")]
    for y, title, blocks, fc in rows:
        ax.text(0.5, y + 8.4, title, fontsize=9, color=INK, weight="bold")
        ax.text(0.5, y + 3, "x", fontsize=11, color=INK)
        arrow(2.5, 5.5, y + 3.2)
        for i, b in enumerate(blocks):
            x = 6 + i * 16.2
            box(x, y, 13.2, 6.4, b, fc=fc if i not in (3,) else "#ffffff")
            if i < len(blocks) - 1:
                arrow(x + 13.9, x + 16.1, y + 3.2)
    ax.text(6, 2.2, "Training losses:  (b) L = MSE(x, x̂)      (c) L = MSE + λ_f · Σ_l ‖φ_l(x̂) − φ_l(x)‖²/‖φ_l(x)‖² "
                    "+ λ_t · CE(A(x̂), y)       channel simulated inside the network, SNR ~ U[0, 20] dB",
            fontsize=8, color=INK2)
    fig.savefig(os.path.join(FIG, "architecture.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- CIFAR
def fig_cifar(an, dg):
    a = an[(an.channel == "awgn") & close(an.ratio, MAIN_RATIO)]
    d = dg[(dg.channel == "awgn") & close(dg.ratio, MAIN_RATIO)]
    acc0 = an[an.scheme == "original"].acc.iloc[0]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, key, yl in zip(axs, ["psnr", "ssim", "acc"], ["PSNR (dB)", "SSIM", "Accuracy of judge B"]):
        for m, name in d.groupby("mcs"):
            ax.plot(name.snr, name[key], color=COLOR["cap"], lw=0.8, alpha=0.8)
        curve(ax, envelope(d, key), key, "sep")
        curve(ax, a[a.scheme == "sep_ae_capacity"], key, "cap", ls="--", lw=1.5)
        curve(ax, a[a.scheme == "jscc_mse"], key, "jscc_mse")
        curve(ax, a[a.scheme == "jscc_sem"], key, "jscc_sem")
        if key == "acc":
            ax.axhline(acc0, color=INK2, lw=1, ls=":")
            ax.text(20, acc0 + 0.01, "clean images", ha="right", fontsize=7, color=INK2)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(yl)
    axs[0].legend(fontsize=7, loc="lower right")
    axs[0].text(0.5, 0.97, "thin grey: individual MCS", transform=axs[0].transAxes, fontsize=7, color=INK2, va="top")
    fig.suptitle("CIFAR-10, bandwidth ratio k/n = 1/12, AWGN (simulation)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cifar_snr.png"))
    plt.close(fig)


def fig_bandwidth(an, dg):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, key, yl in zip(axs, ["psnr", "acc"], ["PSNR (dB)", "Accuracy of judge B"]):
        for snr, ls in ((4, ":"), (10, "--"), (18, "-")):
            for m in ("jscc_mse", "jscc_sem"):
                s = an[(an.scheme == m) & (an.channel == "awgn") & (an.snr == snr)].sort_values("ratio")
                ax.plot(s.ratio, s[key], color=COLOR[m], ls=ls, marker=MARK[m], ms=4,
                        label=f"{LABEL[m]}, {snr} dB")
            e = dg[(dg.channel == "awgn") & (dg.snr == snr)]
            pts = [envelope(e[close(e.ratio, r)], key) for r in sorted(e.ratio.unique())]
            pts = pd.concat(pts) if pts else pd.DataFrame()
            if len(pts):
                ax.plot(pts.ratio, pts[key], color=COLOR["sep"], ls=ls, marker="^", ms=5, label=f"Separated, {snr} dB")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1 / 48, 1 / 24, 1 / 12, 1 / 8, 1 / 6])
        ax.set_xticklabels(["1/48", "1/24", "1/12", "1/8", "1/6"])
        ax.set_xlabel("Bandwidth ratio k/n")
        ax.set_ylabel(yl)
    axs[1].legend(fontsize=6, ncol=1, loc="lower right")
    fig.suptitle("CIFAR-10: quality versus compression (bandwidth) ratio at three SNRs (simulation)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cifar_bandwidth.png"))
    plt.close(fig)


def fig_mismatch(an):
    fig, ax = plt.subplots(figsize=(5, 3.3))
    for name, col, lab in (("jscc_mse_snr0", "snr0", "trained at 0 dB"), ("jscc_mse_snr10", "snr10", "trained at 10 dB"),
                           ("jscc_mse_snr20", "snr20", "trained at 20 dB")):
        s = an[an.scheme == name].sort_values("snr")
        ax.plot(s.snr, s.psnr, color=COLOR[col], label=lab, marker="o", ms=3)
        tr = int(name.split("snr")[1])
        r = s[s.snr == tr]
        ax.plot(r.snr, r.psnr, "o", ms=9, mfc="none", mec=COLOR[col], mew=1.5)
    s = an[(an.scheme == "jscc_mse") & (an.channel == "awgn") & close(an.ratio, MAIN_RATIO)].sort_values("snr")
    ax.plot(s.snr, s.psnr, color=COLOR["jscc_mse"], ls="--", label="trained on U[0,20] dB")
    ax.set_xlabel("Test SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Train/test SNR mismatch, syntactic JSCC, k/n = 1/12")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "snr_mismatch.png"))
    plt.close(fig)


def fig_task(an):
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    t = an[(an.scheme == "task") & (an.channel == "awgn")]
    acc0 = an[an.scheme == "original"].acc.iloc[0]
    for snr, ls in ((0, ":"), (10, "-")):
        s = t[t.snr == snr].sort_values("ratio")
        ax.plot(s.ratio * 3072, s.acc, color=COLOR["task"], ls=ls, marker="D", ms=4, label=f"Task-oriented, {snr} dB")
        for m in ("jscc_mse", "jscc_sem"):
            s = an[(an.scheme == m) & (an.channel == "awgn") & (an.snr == snr)].sort_values("ratio")
            ax.plot(s.ratio * 3072, s.acc, color=COLOR[m], ls=ls, marker=MARK[m], ms=4,
                    label=f"{LABEL[m]} + judge, {snr} dB")
    ax.axhline(acc0, color=INK2, lw=1, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Complex channel symbols per image")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Extension: task-oriented vs reconstruct-then-classify")
    ax.legend(fontsize=6)
    fig.savefig(os.path.join(FIG, "task_oriented.png"))
    plt.close(fig)


def fig_rayleigh(an, dg, kd):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    a = an[(an.channel == "rayleigh") & close(an.ratio, MAIN_RATIO)]
    d = dg[(dg.channel == "rayleigh")]
    ax = axs[0]
    for m in ("jscc_mse", "jscc_sem"):
        curve(ax, a[a.scheme == m], "psnr", m)
        aw = an[(an.channel == "awgn") & close(an.ratio, MAIN_RATIO) & (an.scheme == m)]
        ax.plot(aw.snr, aw.psnr, color=COLOR[m], lw=0.8, ls=":")
    if len(d):
        curve(ax, envelope(d, "psnr"), "psnr", "sep")
    ax.set_title("CIFAR-10 (dotted: AWGN)")
    ax.set_xlabel("Average SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=7)
    ax = axs[1]
    k = kd[(kd.channel == "rayleigh") & close(kd.ratio, MAIN_RATIO)]
    for m in ("jscc_mse", "jscc_sem"):
        curve(ax, k[k.scheme == m], "sem_cos", m)
    ks = k[k.scheme == "sep_jpeg"]
    if len(ks):
        curve(ax, envelope(ks, "sem_cos"), "sem_cos", "sep")
    ax.set_title("Kodak")
    ax.set_xlabel("Average SNR (dB)")
    ax.set_ylabel("ImageNet feature cosine similarity")
    fig.suptitle("Extension: block Rayleigh fading, k/n = 1/12, perfect CSI at the receiver (simulation). "
                 "Separated: MCS chosen on the average SNR", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rayleigh.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- Kodak / video
def fig_natural(df, title, path, tags=None):
    d = df[(df.channel == "awgn") & close(df.ratio, MAIN_RATIO)]
    if tags is not None:
        d = d[d.tag.isin(tags)]
    num = ["psnr", "ssim", "sem_cos", "top1_agree"]
    d = d.groupby(["scheme", "mcs", "snr"], dropna=False)[num].mean().reset_index()
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, key, yl in zip(axs, num, ["PSNR (dB)", "SSIM", "ImageNet feature cosine", "Top-1 label agreement"]):
        s = d[d.scheme == "sep_jpeg"]
        for _, g in s.groupby("mcs"):
            ax.plot(g.snr, g[key], color=COLOR["cap"], lw=0.8, alpha=0.8)
        if len(s):
            curve(ax, envelope(s, key), key, "sep")
        curve(ax, d[d.scheme == "sep_jpeg_capacity"], key, "cap", ls="--", lw=1.5)
        for m in ("jscc_mse", "jscc_sem"):
            curve(ax, d[d.scheme == m], key, m)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(yl)
    axs[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_video_rate(v, vf):
    clips = list(v.tag.unique()) if len(v) else []
    clips = [c for c in ["akiyo", "foreman", "bus"] if c in clips]
    fig, axs = plt.subplots(1, len(clips) + 1, figsize=(4 * (len(clips) + 1), 3.3))
    for ax, clip in zip(axs, clips):
        for m in ("jscc_mse", "jscc_sem"):
            f = v[(v.tag == clip) & (v.scheme == m + "_fbf")].sort_values("ratio")
            ax.plot(f.ratio, f.psnr, color=COLOR[m], marker=MARK[m], ms=4, label=f"{LABEL[m]}, frame-by-frame")
            for ck, ls in ((8, "--"), (16, ":")):
                c = v[(v.tag == clip) & (v.scheme == m + "_cr") & (v.c == ck)].sort_values("ratio")
                ax.plot(c.ratio, c.psnr, color=COLOR[m], ls=ls, marker="x", ms=5,
                        label=f"{LABEL[m]}, CR key 1/{int(96 / ck)}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Average bandwidth ratio k/n")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(clip)
    axs[0].legend(fontsize=6)
    ax = axs[-1]
    for clip, col in zip(clips, ["#2a78d6", "#eb6834", "#1baf7a"]):
        for sch, ls in (("fbf", "-"), ("cr", "--")):
            s = vf[(vf.tag == clip) & (vf.scheme == sch)]
            ax.plot(s.t, s.psnr, color=col, ls=ls, lw=1.5,
                    label=f"{clip} {'frame-by-frame 1/12' if sch == 'fbf' else 'CR (key 1/6, f=0.2)'}")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Semantic JSCC, 10 dB: per-frame quality")
    ax.legend(fontsize=6)
    fig.suptitle("Extension: temporal redundancy by latent conditional replenishment (CR), SNR 10 dB (simulation)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "video_rate.png"))
    plt.close(fig)


def fig_learning():
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    groups = [("Classifiers (val. accuracy)", ["clsA", "clsB"]),
              ("Image JSCC (val. PSNR, 10 dB, 1/12)", ["jscc_mse", "jscc_sem", "jscc_mse_snr0", "jscc_mse_snr10",
                                                         "jscc_mse_snr20", "dae"]),
              ("Task-oriented (val. accuracy, 10 dB)", ["task"])]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#52514e"]
    for ax, (t, names) in zip(axs, groups):
        for n, c in zip(names, cols):
            p = os.path.join("results", "models", n, "train_log.csv")
            if os.path.exists(p) and os.path.getsize(p) > 30:
                d = pd.read_csv(p)
                ax.plot(d.epoch, d.val_metric, color=c, label=n, lw=1.5)
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "learning_curves.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- tables
def tables(an, dg, kd, v):
    os.makedirs(TAB, exist_ok=True)
    rows = []
    snrs = [0, 4, 10, 16, 20]
    a = an[(an.channel == "awgn") & close(an.ratio, MAIN_RATIO)]
    d = dg[(dg.channel == "awgn") & close(dg.ratio, MAIN_RATIO)]
    for snr in snrs:
        for sch in ("jscc_mse", "jscc_sem", "sep_ae_capacity"):
            r = a[(a.scheme == sch) & (a.snr == snr)]
            if len(r):
                rows.append(dict(dataset="cifar", scheme=sch, snr=snr, **r[["psnr", "ssim", "acc"]].iloc[0].to_dict()))
        e = d[d.snr == snr]
        if len(e):
            b = e.loc[e.psnr.idxmax()]
            rows.append(dict(dataset="cifar", scheme="sep_ae_best", snr=snr, mcs=b.mcs, psnr=b.psnr, ssim=b.ssim,
                             acc=b.acc))
    for name, df in (("kodak", kd), ("video", v)):
        if not len(df):
            continue
        x = df[(df.channel == "awgn") & close(df.ratio, MAIN_RATIO)]
        x = x.groupby(["scheme", "mcs", "snr"], dropna=False)[["psnr", "ssim", "sem_cos", "top1_agree"]].mean()
        x = x.reset_index()
        for snr in snrs:
            for sch in ("jscc_mse", "jscc_sem", "sep_jpeg_capacity"):
                r = x[(x.scheme == sch) & (x.snr == snr)]
                if len(r):
                    rows.append(dict(dataset=name, scheme=sch, snr=snr, **r[["psnr", "ssim", "sem_cos",
                                                                           "top1_agree"]].iloc[0].to_dict()))
            e = x[(x.scheme == "sep_jpeg") & (x.snr == snr)]
            if len(e):
                b = e.loc[e.psnr.idxmax()]
                rows.append(dict(dataset=name, scheme="sep_jpeg_best", snr=snr, mcs=b.mcs, psnr=b.psnr, ssim=b.ssim,
                                 sem_cos=b.sem_cos, top1_agree=b.top1_agree))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB, "summary.csv"), index=False)
    with open(os.path.join(TAB, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("# Summary at k/n = 1/12, AWGN (simulation)\n\n" + t.round(4).to_markdown(index=False) + "\n")
    info = []
    for p in glob.glob(os.path.join("results", "models", "*", "info.json")):
        j = json.load(open(p))
        info.append({k: j.get(k) for k in ("name", "epochs", "params", "train_s", "val_metric")})
    pd.DataFrame(info).sort_values("name").to_csv(os.path.join(TAB, "models.csv"), index=False)
    return t


def main():
    os.makedirs(FIG, exist_ok=True)
    an = pd.concat([read("cifar_analog.csv"), read("cifar_capacity.csv")], ignore_index=True)
    dg = read("cifar_digital.csv")
    kd = pd.concat([read("kodak_jscc.csv"), read("kodak_sep.csv")], ignore_index=True)
    v = pd.concat([read("video_jscc.csv"), read("video_sep.csv")], ignore_index=True)
    vf = read("video_frames.csv")
    fig_architecture()
    fig_learning()
    if len(an) and len(dg):
        fig_cifar(an, dg)
        fig_bandwidth(an, dg)
        fig_mismatch(an)
        fig_task(an)
    if len(kd):
        fig_natural(kd, "Kodak (24 images, 768×512), k/n = 1/12, AWGN (simulation, models trained on CIFAR-10)",
                    os.path.join(FIG, "kodak_snr.png"))
        if len(an):
            fig_rayleigh(an, dg, kd)
    if len(v):
        fig_natural(v, "Video: 3 CIF clips × 30 frames, frame-by-frame, k/n = 1/12, AWGN (simulation)",
                    os.path.join(FIG, "video_snr.png"))
        fig_video_rate(v, vf)
    t = tables(an, dg, kd, v) if len(an) else None
    print("figures in figures/, tables in results/tables/" if t is not None else "figures (partial)")


if __name__ == "__main__":
    main()
