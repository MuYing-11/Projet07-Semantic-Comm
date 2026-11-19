"""Training loops for every learned component. One job = one model, saved to results/models/<name>/.

Jobs (see JOBS):
    clsA        ResNetSmall CIFAR-10 classifier, used inside the semantic loss
    clsB        VGGSmall CIFAR-10 classifier, independent semantic judge (never used in a loss)
    jscc_mse    syntactic deep JSCC, pixel MSE loss, SNR ~ U[0,20] dB, random bandwidth
    jscc_sem    semantic deep JSCC, MSE + feature + task loss through clsA, SNR ~ U[0,20] dB, random bandwidth
    jscc_mse_snr{0,10,20}   syntactic JSCC trained at one SNR (train/test SNR mismatch study), ratio 1/12
    dae         source auto-encoder of the separated digital chain (4-bit quantised latent, no channel)
    task        task-oriented transmitter/receiver (extension), CE loss, SNR ~ U[0,20] dB, random bandwidth
"""
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import ROOT
from .data import augment, cifar10
from .metrics import psnr
from .models import C_LATENT, DeepJSCC, DigitalAE, ResNetSmall, TaskJSCC, VGGSmall, n_params

MODELS = os.path.join(ROOT, "results", "models")
JSCC_WIDTHS = [2, 4, 8, 12, 16]  # active latent channels -> ratios 1/48, 1/24, 1/12, 1/8, 1/6
TASK_WIDTHS = [1, 2, 4, 8, 16]  # stride-8 task model -> 8..128 symbols
LAMBDA_FEAT, LAMBDA_TASK = 0.01, 0.003  # semantic loss weights (see article, Sec. III)

JOBS = {
    "clsA": dict(kind="cls", arch="A", epochs=10),
    "clsB": dict(kind="cls", arch="B", epochs=15),
    "jscc_mse": dict(kind="jscc", loss="mse", snr=(0, 20), epochs=25),
    "jscc_sem": dict(kind="jscc", loss="sem", snr=(0, 20), epochs=25),
    "jscc_mse_snr0": dict(kind="jscc", loss="mse", snr=(0, 0), widths=[8], epochs=15),
    "jscc_mse_snr10": dict(kind="jscc", loss="mse", snr=(10, 10), widths=[8], epochs=15),
    "jscc_mse_snr20": dict(kind="jscc", loss="mse", snr=(20, 20), widths=[8], epochs=15),
    "dae": dict(kind="dae", epochs=25),
    "task": dict(kind="task", snr=(0, 20), epochs=20),
}


def build(name):
    cfg = JOBS[name]
    if cfg["kind"] == "cls":
        return ResNetSmall(w=32) if cfg["arch"] == "A" else VGGSmall()
    if cfg["kind"] == "jscc":
        return DeepJSCC(C_LATENT)
    if cfg["kind"] == "dae":
        return DigitalAE(C_LATENT, bits=4)
    return TaskJSCC(16)


def pick_device(device=None):
    """'cuda' when available (override with SEMCOM_DEVICE=cpu), else 'cpu'."""
    if device is None:
        device = os.environ.get("SEMCOM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load(name, device="cpu"):
    m = build(name)
    m.load_state_dict(torch.load(os.path.join(MODELS, name, "model.pt"), map_location="cpu"))
    return m.eval().to(device)


def done(name):
    return os.path.exists(os.path.join(MODELS, name, "info.json"))


def _sem_loss(clsA, x, xh, y):
    with torch.no_grad():
        fx = clsA.features(x)
    fh = clsA.features(xh)
    feat = sum(F.mse_loss(a, b) / b.pow(2).mean().clamp_min(1e-6) for a, b in zip(fh, fx)) / len(fx)
    logits = clsA.fc(F.adaptive_avg_pool2d(fh[-1], 1).flatten(1))
    return feat, F.cross_entropy(logits, y)


def train(name, batch=128, lr=1e-3, seed=0, threads=None, epochs=None, device=None):
    if threads:
        torch.set_num_threads(threads)
    dev = pick_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = JOBS[name]
    epochs = epochs or cfg["epochs"]
    out = os.path.join(MODELS, name)
    os.makedirs(out, exist_ok=True)
    xtr, ytr = cifar10(True)
    xte, yte = cifar10(False)
    xte, yte = xte[:2000], yte[:2000]
    xte, yte = xte.to(dev), yte.to(dev)
    model = build(name).to(dev)
    clsA = load("clsA", dev) if cfg.get("loss") == "sem" else None
    if clsA is not None:
        for p in clsA.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4 if cfg["kind"] == "cls" else 0)
    steps = epochs * (len(xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * (5 if cfg["kind"] == "cls" else 1),
                                                total_steps=steps, pct_start=0.15)
    widths = cfg.get("widths", TASK_WIDTHS if cfg["kind"] == "task" else JSCC_WIDTHS)
    snr_lo, snr_hi = cfg.get("snr", (0, 20))
    ck_path = os.path.join(out, "checkpoint.pt")
    ep0, elapsed = 0, 0.0
    if os.path.exists(ck_path):  # resume an interrupted run
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["rng"])
        ep0, elapsed = ck["epoch"], ck["elapsed"]
        print(f"[{name}] resuming at epoch {ep0 + 1}/{epochs}", flush=True)
    log = open(os.path.join(out, "train_log.csv"), "a" if ep0 else "w", newline="")
    wr = csv.writer(log)
    if not ep0:
        wr.writerow(["epoch", "loss", "val_metric", "time_s"])
    t0 = time.time() - elapsed
    for ep in range(ep0, epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        tot, nb = 0.0, 0
        for b in range(len(xtr) // batch):
            idx = perm[b * batch:(b + 1) * batch]
            x, y = augment(xtr[idx]), ytr[idx]          # indexing/augment stay on CPU:
            snr = torch.empty(len(x)).uniform_(snr_lo, snr_hi)  # same data order as the CPU runs
            x, y, snr = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), snr.to(dev)
            c = int(np.random.choice(widths))
            if cfg["kind"] == "cls":
                loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
            elif cfg["kind"] == "jscc":
                xh = model(x, snr, c)
                loss = F.mse_loss(xh, x)
                if clsA is not None:
                    feat, ce = _sem_loss(clsA, x, xh, y)
                    loss = loss + LAMBDA_FEAT * feat + LAMBDA_TASK * ce
            elif cfg["kind"] == "dae":
                c = int(np.random.randint(1, C_LATENT + 1))
                loss = F.mse_loss(model(x, c, train=True), x)
            else:
                loss = F.cross_entropy(model(x, snr, c), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        val = validate(model, cfg, xte, yte)
        wr.writerow([ep + 1, tot / nb, val, round(time.time() - t0, 1)])
        log.flush()
        print(f"[{name}] epoch {ep + 1}/{epochs} loss {tot / nb:.5f} val {val:.4f} ({time.time() - t0:.0f} s)",
              flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": torch.get_rng_state(), "epoch": ep + 1, "elapsed": time.time() - t0}, ck_path)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, os.path.join(out, "model.pt"))
    json.dump({"name": name, **{k: v for k, v in cfg.items()}, "epochs": epochs, "params": n_params(model),
               "train_s": time.time() - t0, "val_metric": val, "device": str(dev)},
              open(os.path.join(out, "info.json"), "w"), indent=1)
    if os.path.exists(ck_path):
        os.remove(ck_path)
    return model


@torch.no_grad()
def _chunks(x, y, n=250):
    for i in range(0, len(x), n):
        yield x[i:i + n], y[i:i + n]


@torch.no_grad()
def validate(model, cfg, x, y, chunk=250):
    """Accuracy for classifiers/task model (10 dB, max width), PSNR at 10 dB / 1/12 for JSCC, noiseless for DAE."""
    model.eval()
    torch.manual_seed(123)
    snr = None
    if cfg["kind"] == "jscc":
        snr = cfg["snr"][0] if cfg["snr"][0] == cfg["snr"][1] else 10.0
    vals = []
    for xb, yb in _chunks(x, y, chunk):
        if cfg["kind"] == "cls":
            v = (model(xb).argmax(1) == yb).float()
        elif cfg["kind"] == "task":
            v = (model(xb, 10.0, 4).argmax(1) == yb).float()
        elif cfg["kind"] == "dae":
            v = psnr(model(xb, 8), xb)
        else:
            v = psnr(model(xb, snr, 8), xb)
        vals.append(v.flatten().cpu())
    return torch.cat(vals).mean().item()
