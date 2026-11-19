"""Interactive Protocol-Explorer-style web simulator for Project 07 (Semantic Communication).

Usage:
    python scripts/simulator.py
    python scripts/simulator.py --port 8765 --no-browser
"""
import argparse
import base64
import csv
import glob
import io
import json
import math
import os
import socket
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semcom import ROOT, chdir_project_root  # noqa: E402

chdir_project_root()

from semcom import channel as ch  # noqa: E402
from semcom import digital as dg  # noqa: E402
from semcom.channel import ratio_from_channels  # noqa: E402
from semcom.data import cifar10, kodak, y4m  # noqa: E402
from semcom.metrics import imagenet_embed, psnr, ssim  # noqa: E402
from semcom.train import JOBS, load  # noqa: E402
from semcom.video import conditional_replenishment, frame_by_frame  # noqa: E402

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulator_ui.html")


def _tensor_to_b64_png(t):
    """Convert (3, H, W) float tensor in [0, 1] to data:image/png;base64,..."""
    arr = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _latent_to_b64_png(z, c_active):
    """Render up to 8 active latent channels of z (1, C, h, w) as a crisp false-color strip."""
    n_show = min(int(c_active), 8, z.shape[1])
    if n_show <= 0:
        return None
    maps = z[0, :n_show].detach().cpu().numpy()
    h, w = maps.shape[1], maps.shape[2]
    # Robust normalization across displayed channels
    vmin = float(np.percentile(maps, 2))
    vmax = float(np.percentile(maps, 98))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    norm = np.clip((maps - vmin) / (vmax - vmin), 0.0, 1.0)
    # Cool cyber colormap: dark navy -> cyan -> mint -> bright white-yellow
    r = np.clip(1.8 * norm - 0.8, 0, 1)
    g = np.clip(1.4 * norm - 0.1, 0, 1)
    b = np.clip(0.25 + 0.9 * np.sin(norm * math.pi * 0.75), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)  # (n_show, h, w, 3)

    pad = 2
    cols = min(4, n_show)
    rows = math.ceil(n_show / cols)
    canvas = np.full((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, 3), 18, dtype=np.uint8)
    for idx in range(n_show):
        rr, cc = divmod(idx, cols)
        y0 = rr * (h + pad)
        x0 = cc * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = (rgb[idx] * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _error_map_b64(orig, recon):
    """Render amplified pixel error |orig - recon| as a bright heat-map PNG."""
    diff = (orig - recon).abs().mean(0).detach().cpu().numpy()
    val = np.clip(diff * 4.5, 0.0, 1.0)
    # Inferno-inspired palette: dark -> violet -> orange -> yellow
    r = np.clip(1.6 * val, 0, 1)
    g = np.clip(2.0 * val - 0.7, 0, 1)
    b = np.clip(0.8 * np.sin(val * math.pi) + 0.15 * (1 - val), 0, 1) * (1 - 0.6 * val)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _llr_hist(llr_arr, n_bins=16):
    """Compute a 16-bin histogram of soft LLRs in [-16, 16] for digital step visualization."""
    if llr_arr.size == 0:
        return [0.0] * n_bins
    hist, _ = np.histogram(np.clip(llr_arr, -16.0, 16.0), bins=n_bins, range=(-16.0, 16.0))
    mx = float(hist.max()) if hist.max() > 0 else 1.0
    return [round(float(v / mx), 3) for v in hist]


def _sample_iq(tx_c, rx_c, max_pts=220, seed=0):
    """Subsample complex symbol pairs (tx, rx) for the live I/Q constellation canvas."""
    n = len(tx_c)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(max_pts, n), replace=False) if n > max_pts else np.arange(n)
    return [
        [round(float(tx_c[i].real), 4), round(float(tx_c[i].imag), 4),
         round(float(rx_c[i].real), 4), round(float(rx_c[i].imag), 4)]
        for i in idx
    ]


class SimEngine:
    """Thread-safe simulation engine wrapping the trained models and datasets."""

    def __init__(self):
        self.lock = threading.Lock()
        self.device = torch.device("cpu")
        self.models = {}
        self._kodak = None
        self._cifar = None
        self._cifar_picks = None
        self._custom = {}
        self._videos = {}
        self._sweep_cache = {}
        self._bw_cache = {}
        self._video_cache = {}
        self._project_data = None
        self._proj_lock = threading.Lock()
        self._imnet_labels = self._load_imagenet_labels()

    @staticmethod
    def _load_imagenet_labels():
        try:
            from torchvision.models import ResNet18_Weights
            return ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
        except Exception:
            return None

    def _get_model(self, name):
        if name not in self.models:
            self.models[name] = load(name, self.device)
        return self.models[name]

    def _get_kodak(self):
        if self._kodak is None:
            self._kodak = kodak()
        return self._kodak

    def _get_cifar(self):
        if self._cifar is None:
            x, y = cifar10(train=False)
            self._cifar = (x, y)
            # Pick 2 clean, recognizable examples per class
            picks = []
            for cls_idx in range(10):
                idxs = (y == cls_idx).nonzero(as_tuple=True)[0][:2].tolist()
                for idx in idxs:
                    picks.append((int(idx), int(cls_idx), CIFAR10_CLASSES[cls_idx]))
            self._cifar_picks = picks
        return self._cifar

    def get_catalog(self):
        with self.lock:
            kx, knames = self._get_kodak()
            cx, _ = self._get_cifar()
            kodak_items = []
            for i, name in enumerate(knames):
                # Small thumbnail for fast catalog load
                arr = (kx[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                im = Image.fromarray(arr).resize((96, 64), Image.Resampling.BILINEAR)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=75)
                thumb = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
                kodak_items.append({"id": name, "label": name, "thumb": thumb})

            cifar_items = []
            for idx, cls_idx, cls_name in self._cifar_picks:
                cifar_items.append({
                    "id": str(idx),
                    "label": f"#{idx} · {cls_name}",
                    "class_name": cls_name,
                    "thumb": _tensor_to_b64_png(cx[idx]),
                })

            return {
                "kodak": kodak_items,
                "cifar10": cifar_items,
                "custom": [
                    {"id": cid, "label": info["label"], "thumb": info["thumb"]}
                    for cid, info in self._custom.items()
                ],
                "videos": [c for c in ["akiyo", "foreman", "bus"] if os.path.exists(os.path.join(ROOT, "data", "video", f"{c}_cif.y4m"))],
                "mcs_list": [dg.mcs_name(m) for m in dg.MCS],
                "widths": [
                    {"c": c, "ratio": f"1/{round(1 / ratio_from_channels(c))}"}
                    for c in [2, 4, 8, 12, 16]
                ],
            }

    def upload_image(self, data_url, name="custom"):
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url)
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        # Resize so longest side <= 512 and both dims are multiples of 4
        w, h = im.size
        scale = min(1.0, 512.0 / max(w, h))
        nw = max(32, (int(round(w * scale)) // 4) * 4)
        nh = max(32, (int(round(h * scale)) // 4) * 4)
        im = im.resize((nw, nh), Image.Resampling.BICUBIC)
        t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0

        thumb_im = im.resize((96, 64), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        thumb_im.save(buf, format="JPEG", quality=80)
        thumb = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        cid = f"custom_{len(self._custom) + 1}"
        with self.lock:
            self._custom[cid] = {"tensor": t, "label": name[:24], "thumb": thumb}
        return {"id": cid, "label": name[:24], "thumb": thumb, "W": nw, "H": nh}

    def _get_sample(self, dataset, sample_id):
        if dataset == "custom" and sample_id in self._custom:
            item = self._custom[sample_id]
            return item["tensor"], item["label"], None
        if dataset == "kodak":
            kx, knames = self._get_kodak()
            idx = knames.index(sample_id) if sample_id in knames else 0
            return kx[idx], knames[idx], None
        else:
            cx, cy = self._get_cifar()
            idx = int(sample_id) if str(sample_id).isdigit() else 0
            idx = max(0, min(len(cx) - 1, idx))
            cls_idx = int(cy[idx].item())
            return cx[idx], f"cifar_{idx}", cls_idx

    def _eval_semantic(self, dataset, orig, recon, orig_embed=None, orig_cls=None):
        p = float(psnr(recon[None], orig[None]).item())
        s = float(ssim(recon[None], orig[None]).item())
        if dataset in ("kodak", "custom"):
            fx, lx = orig_embed
            fy, ly = imagenet_embed(recon[None])
            cos = float(F.cosine_similarity(fx, fy, dim=1).item())
            top1_orig = int(lx.argmax(1).item())
            top1_rec = int(ly.argmax(1).item())
            probs = F.softmax(ly, dim=1)[0]
            conf = float(probs[top1_rec].item())
            pred_name = self._imnet_labels[top1_rec] if self._imnet_labels else f"class {top1_rec}"
            return {
                "psnr": round(p, 2),
                "ssim": round(s, 4),
                "sem_score": round(cos, 4),
                "sem_metric": "ImageNet Cosine",
                "pred_label": pred_name,
                "pred_conf": round(conf * 100, 1),
                "label_match": bool(top1_orig == top1_rec),
            }
        else:
            judge = self._get_model("clsB")
            logits = judge(recon[None])
            probs = F.softmax(logits, dim=1)[0]
            top1 = int(probs.argmax().item())
            conf = float(probs[top1].item())
            return {
                "psnr": round(p, 2),
                "ssim": round(s, 4),
                "sem_score": round(conf, 4),
                "sem_metric": "Judge-B Confidence",
                "pred_label": CIFAR10_CLASSES[top1],
                "pred_conf": round(conf * 100, 1),
                "label_match": bool(top1 == orig_cls),
            }

    def _run_digital(self, img, dataset, mcs, snr_db, c_active, channel_kind, fading_gain, seed, orig_embed, orig_cls, include_extras=True):
        t0 = time.perf_counter()
        _, H, W = img.shape
        ratio = ratio_from_channels(c_active)
        k_symbols = int(round(ratio * 3 * H * W))
        mod_name, bps, rate_str = mcs
        rng = np.random.default_rng(seed + 100)

        if dataset in ("kodak", "custom"):
            budget = dg.max_bits(k_symbols, mcs)
            data, jpeg_q = dg.jpeg_fit(img.numpy(), budget)
            if data is None:
                recon = torch.full_like(img, 0.5)
                metrics = self._eval_semantic(dataset, img, recon, orig_embed, orig_cls)
                return {
                    "source_desc": "JPEG (budget too small)",
                    "source_param": "q=0",
                    "n_info_bits": 0,
                    "n_packets": 0,
                    "n_coded_bits": 0,
                    "n_symbols": 0,
                    "mcs": dg.mcs_name(mcs),
                    "ber": 1.0,
                    "bit_errors": 0,
                    "decode_ok": False,
                    "cliff_hit": True,
                    "iq_points": [],
                    "qam_levels": [],
                    "llr_hist": [0.0] * 16,
                    "sample_info_bits": [],
                    "sample_rec_bits": [],
                    "time_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "recon_b64": _tensor_to_b64_png(recon),
                    "err_b64": _error_map_b64(img, recon) if include_extras else None,
                    **metrics,
                }
            bits_1d = dg.bytes_to_bits(data)
            source_desc = f"JPEG (q={jpeg_q}, {len(data):,} B)"
            source_param = f"q={jpeg_q}"
        else:
            dae = self._get_model("dae")
            nb = dg.max_bits(k_symbols, mcs)
            c_dae = min(dae.c, nb // (64 * dae.bits))
            if c_dae == 0:
                recon = dae.from_levels(torch.zeros(1, 0, 8, 8, dtype=torch.int64), 0)[0]
                metrics = self._eval_semantic(dataset, img, recon, orig_embed, orig_cls)
                return {
                    "source_desc": "4-bit DAE (0 ch fit)",
                    "source_param": "c=0",
                    "n_info_bits": 0,
                    "n_packets": 0,
                    "n_coded_bits": 0,
                    "n_symbols": 0,
                    "mcs": dg.mcs_name(mcs),
                    "ber": 1.0,
                    "bit_errors": 0,
                    "decode_ok": False,
                    "cliff_hit": True,
                    "iq_points": [],
                    "qam_levels": [],
                    "sample_info_bits": [],
                    "sample_rec_bits": [],
                    "recon_b64": _tensor_to_b64_png(recon),
                    **metrics,
                }
            lev = dae.to_levels(img[None], c_dae)
            b_arr = ((lev.numpy().reshape(-1)[:, None] >> np.arange(dae.bits - 1, -1, -1)) & 1)
            bits_1d = b_arr.reshape(-1).astype(np.uint8)
            source_desc = f"4-bit DAE ({c_dae} ch, {len(bits_1d)} bits)"
            source_param = f"c={c_dae}"

        # Packetize, encode, modulate, channel, demodulate, Viterbi (capturing I/Q constellation!)
        n_info_bits = int(len(bits_1d))
        n_pk, L = dg._packets(n_info_bits)
        pad = np.zeros(n_pk * L, np.uint8)
        pad[:n_info_bits] = bits_1d
        pk_bits = pad.reshape(n_pk, L)

        coded, mask = dg.puncture(dg.conv_encode(pk_bits), rate_str)
        n_c = coded.shape[1]
        coded_pad = np.concatenate([coded, np.zeros((n_pk, (-n_c) % bps), np.uint8)], 1)
        tx_sym = dg.qam_mod(coded_pad.reshape(-1), bps).reshape(n_pk, -1)

        eff_snr = snr_db + (10.0 * math.log10(max(fading_gain, 1e-12)) if channel_kind == "rayleigh" else 0.0)
        sigma2 = 10.0 ** (-eff_snr / 10.0)
        noise = math.sqrt(sigma2 / 2.0) * (rng.standard_normal(tx_sym.shape) + 1j * rng.standard_normal(tx_sym.shape))
        rx_sym = tx_sym + noise

        llr = dg.qam_llr(rx_sym.reshape(-1), bps, np.full(rx_sym.size, sigma2)).reshape(n_pk, -1)[:, :n_c]
        dec = dg.viterbi(dg.depuncture(llr, mask), L)
        rec_bits = dec.reshape(-1)[:n_info_bits]

        bit_errors = int(np.sum(bits_1d != rec_bits))
        ber = float(bit_errors / max(1, n_info_bits))

        if dataset in ("kodak", "custom"):
            rec_bytes = data if bit_errors == 0 else dg.bits_to_bytes(rec_bits)
            im_arr, decode_ok = dg.jpeg_decode(rec_bytes, (H, W))
            recon = torch.from_numpy(im_arr)
            cliff_hit = not decode_ok
        else:
            rl = rec_bits.reshape(-1, dae.bits) @ (1 << np.arange(dae.bits - 1, -1, -1))
            q = torch.from_numpy(rl).view(1, c_dae, 8, 8)
            recon = dae.from_levels(q, c_dae)[0]
            decode_ok = (ber < 0.05)
            cliff_hit = (ber >= 0.05)

        levels, _ = dg._pam(bps // 2)
        qam_norm = math.sqrt(2 * float(np.mean(levels ** 2)))
        qam_levels = [round(float(lv / qam_norm), 4) for lv in levels]

        metrics = self._eval_semantic(dataset, img, recon, orig_embed, orig_cls)
        return {
            "source_desc": source_desc,
            "source_param": source_param,
            "n_info_bits": n_info_bits,
            "n_packets": n_pk,
            "n_coded_bits": int(n_pk * n_c),
            "n_symbols": int(tx_sym.size),
            "mcs": dg.mcs_name(mcs),
            "bps": int(bps),
            "rate": rate_str,
            "ber": round(ber, 5),
            "bit_errors": bit_errors,
            "decode_ok": bool(decode_ok),
            "cliff_hit": bool(cliff_hit),
            "iq_points": _sample_iq(tx_sym.reshape(-1), rx_sym.reshape(-1), max_pts=220, seed=seed) if include_extras else [],
            "qam_levels": qam_levels if include_extras else [],
            "llr_hist": _llr_hist(llr.reshape(-1)) if include_extras else [],
            "sample_info_bits": [int(b) for b in bits_1d[:40]] if include_extras else [],
            "sample_rec_bits": [int(b) for b in rec_bits[:40]] if include_extras else [],
            "time_ms": round((time.perf_counter() - t0) * 1000, 1),
            "recon_b64": _tensor_to_b64_png(recon) if include_extras else None,
            "err_b64": _error_map_b64(img, recon) if include_extras else None,
            **metrics,
        }

    def _run_jscc(self, model_name, img, dataset, snr_db, c_active, channel_kind, fading_gain, seed, orig_embed, orig_cls, include_layers=False):
        t0 = time.perf_counter()
        model = self._get_model(model_name)
        layers_info = {}
        if include_layers:
            h_in = img[None] - 0.5
            h1 = model.enc.net[1](model.enc.net[0](h_in))
            h2 = model.enc.net[3](model.enc.net[2](h1))
            h3 = model.enc.net[5](model.enc.net[4](h2))
            z = model.enc.net[6](h3)
            zt = z[:, :c_active]
            k_sym = zt[0].numel() / 2
            energy = float(zt.pow(2).sum().clamp_min(1e-9).item())
            scale = math.sqrt(k_sym / energy)
            ch_e = z[0].pow(2).flatten(1).mean(1).detach().cpu().numpy()
            ch_max = float(np.max(ch_e)) if np.max(ch_e) > 1e-9 else 1.0
            channel_powers = [round(float(v / ch_max), 3) for v in ch_e]
            # First Conv1 5x5 filter averaged across RGB for live dot-product animation
            w0 = model.enc.net[0].weight[0].mean(0).detach().cpu().numpy()
            conv1_kernel = [[round(float(w0[r, c]), 3) for c in range(w0.shape[1])] for r in range(w0.shape[0])]
            layers_info = {
                "conv1_b64": _latent_to_b64_png(h1, 4),
                "conv2_b64": _latent_to_b64_png(h2, 4),
                "res_b64": _latent_to_b64_png(h3, 4),
                "conv1_shape": [int(h1.shape[1]), int(h1.shape[2]), int(h1.shape[3])],
                "conv2_shape": [int(h2.shape[1]), int(h2.shape[2]), int(h2.shape[3])],
                "channel_powers": channel_powers,
                "conv1_kernel": conv1_kernel,
                "norm_stats": {
                    "raw_power": round(energy / max(1.0, k_sym), 3),
                    "scale": round(scale, 4),
                    "norm_power": 1.0,
                },
            }
        else:
            z = model.enc(img[None])
        x_norm = ch.power_normalize(z, c_active)

        eff_snr = snr_db + (10.0 * math.log10(max(fading_gain, 1e-12)) if channel_kind == "rayleigh" else 0.0)
        torch.manual_seed(seed + (17 if model_name.startswith("jscc_mse") else 31))
        y_noisy = ch.truncate(ch.awgn(x_norm, eff_snr), c_active)
        if include_layers:
            d0 = model.dec.net[:4](y_noisy)
            d1 = model.dec.net[5](model.dec.net[4](d0))
            layers_info["deconv1_b64"] = _latent_to_b64_png(d1, 4)
            recon = model.dec.net[7](model.dec.net[6](d1))[0]
        else:
            recon = model.dec(y_noisy)[0]

        # Extract complex symbols from pairs of active latent channels
        tx_flat = x_norm[0, :c_active].detach().cpu().numpy().reshape(-1)
        rx_flat = y_noisy[0, :c_active].detach().cpu().numpy().reshape(-1)
        half = (len(tx_flat) // 2) * 2
        tx_c = tx_flat[:half:2] + 1j * tx_flat[1:half:2]
        rx_c = rx_flat[:half:2] + 1j * rx_flat[1:half:2]

        metrics = self._eval_semantic(dataset, img, recon, orig_embed, orig_cls)
        return {
            "model_name": model_name,
            "latent_shape": [int(c_active), int(z.shape[2]), int(z.shape[3])],
            "k_symbols": int(half // 2),
            "latent_tx_b64": _latent_to_b64_png(x_norm, c_active) if include_layers else None,
            "latent_rx_b64": _latent_to_b64_png(y_noisy, c_active) if include_layers else None,
            "iq_points": _sample_iq(tx_c, rx_c, max_pts=220, seed=seed) if include_layers else [],
            "time_ms": round((time.perf_counter() - t0) * 1000, 1),
            "recon_b64": _tensor_to_b64_png(recon) if include_layers else None,
            "err_b64": _error_map_b64(img, recon) if include_layers else None,
            **layers_info,
            **metrics,
        }

    def _run_capacity_point(self, img, dataset, snr_db, c_active, channel_kind, fading_gain, orig_embed, orig_cls):
        _, H, W = img.shape
        ratio = ratio_from_channels(c_active)
        k_symbols = int(round(ratio * 3 * H * W))
        eff_snr = snr_db + (10.0 * math.log10(max(fading_gain, 1e-12)) if channel_kind == "rayleigh" else 0.0)
        cap_bits = int(k_symbols * math.log2(1.0 + 10.0 ** (eff_snr / 10.0)))
        if dataset in ("kodak", "custom"):
            data, _ = dg.jpeg_fit(img.numpy(), cap_bits)
            if data is None:
                recon = torch.full_like(img, 0.5)
            else:
                im_arr, _ = dg.jpeg_decode(data, (H, W))
                recon = torch.from_numpy(im_arr)
        else:
            dae = self._get_model("dae")
            c_dae = min(dae.c, cap_bits // (64 * dae.bits))
            if c_dae == 0:
                recon = dae.from_levels(torch.zeros(1, 0, 8, 8, dtype=torch.int64), 0)[0]
            else:
                recon = dae.from_levels(dae.to_levels(img[None], c_dae), c_dae)[0]
        return self._eval_semantic(dataset, img, recon, orig_embed, orig_cls)

    def _run_task(self, img, snr_db, c_active, channel_kind, fading_gain, seed, orig_cls):
        t0 = time.perf_counter()
        m = self._get_model("task")
        eff_snr = snr_db + (10.0 * math.log10(max(fading_gain, 1e-12)) if channel_kind == "rayleigh" else 0.0)
        torch.manual_seed(seed + 77)
        logits = m(img[None], eff_snr, c_active=c_active, kind="awgn")
        probs = F.softmax(logits, dim=1)[0]
        top1 = int(probs.argmax().item())
        # TaskJSCC uses stride-8 (4x4 spatial positions on 32x32 CIFAR-10)
        k_sym = int(c_active * 4 * 4 / 2)
        return {
            "k_symbols": k_sym,
            "ratio_str": f"{k_sym}/{3 * 32 * 32}",
            "probs": [round(float(p * 100), 1) for p in probs.tolist()],
            "pred_label": CIFAR10_CLASSES[top1],
            "pred_conf": round(float(probs[top1].item() * 100), 1),
            "label_match": bool(top1 == orig_cls),
            "time_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    @torch.no_grad()
    def simulate(self, params):
        dataset = params.get("dataset", "kodak")
        sample_id = params.get("sample_id", "kodim23" if dataset == "kodak" else "0")
        snr_db = float(params.get("snr_db", 6.0))
        channel_kind = params.get("channel_kind", "awgn")
        c_active = int(params.get("c_active", 8))
        mcs_str = params.get("mcs", "16QAM 1/2")
        mse_variant = params.get("mse_variant", "jscc_mse")
        if mse_variant not in ("jscc_mse", "jscc_mse_snr0", "jscc_mse_snr10", "jscc_mse_snr20"):
            mse_variant = "jscc_mse"
        seed = int(params.get("seed", 42))

        mcs = next((m for m in dg.MCS if dg.mcs_name(m) == mcs_str), dg.MCS[2])

        with self.lock:
            img, label_id, orig_cls = self._get_sample(dataset, sample_id)
            _, H, W = img.shape
            n_real = 3 * H * W
            ratio = ratio_from_channels(c_active)
            k_symbols = int(round(ratio * n_real))

            # Rayleigh block fading complex phasor h ~ CN(0,1)
            rng_fade = np.random.default_rng(seed)
            if channel_kind == "rayleigh":
                h_real = float(rng_fade.normal(0.0, 1.0 / math.sqrt(2.0)))
                h_imag = float(rng_fade.normal(0.0, 1.0 / math.sqrt(2.0)))
                fading_gain = max(1e-6, h_real * h_real + h_imag * h_imag)
            else:
                h_real, h_imag, fading_gain = 1.0, 0.0, 1.0
            fading_db = 10.0 * math.log10(max(fading_gain, 1e-12))
            inst_snr = snr_db + fading_db
            sigma2 = 10.0 ** (-inst_snr / 10.0)

            orig_embed = imagenet_embed(img[None]) if dataset in ("kodak", "custom") else None
            if dataset in ("kodak", "custom"):
                top1 = int(orig_embed[1].argmax(1).item())
                orig_label = self._imnet_labels[top1] if self._imnet_labels else f"class {top1}"
            else:
                orig_label = CIFAR10_CLASSES[orig_cls]

            digital = self._run_digital(
                img, dataset, mcs, snr_db, c_active, channel_kind, fading_gain, seed, orig_embed, orig_cls, include_extras=True
            )
            jscc_mse = self._run_jscc(
                mse_variant, img, dataset, snr_db, c_active, channel_kind, fading_gain, seed, orig_embed, orig_cls, include_layers=True
            )
            jscc_sem = self._run_jscc(
                "jscc_sem", img, dataset, snr_db, c_active, channel_kind, fading_gain, seed, orig_embed, orig_cls, include_layers=True
            )
            task_res = self._run_task(img, snr_db, c_active, channel_kind, fading_gain, seed, orig_cls) if dataset == "cifar10" else None

            # 8x8 center RGB patch for pixel-level animation
            cy0, cx0 = max(0, H // 2 - 4), max(0, W // 2 - 4)
            patch = img[:, cy0:cy0 + 8, cx0:cx0 + 8].permute(1, 2, 0).numpy()
            pixel_patch = [
                [[round(float(patch[r, c, 0]), 2), round(float(patch[r, c, 1]), 2), round(float(patch[r, c, 2]), 2)]
                 for c in range(patch.shape[1])]
                for r in range(patch.shape[0])
            ]

            return {
                "meta": {
                    "dataset": dataset,
                    "sample_id": label_id,
                    "H": H,
                    "W": W,
                    "n_real": n_real,
                    "k_symbols": k_symbols,
                    "c_active": c_active,
                    "ratio_str": f"1/{round(1 / ratio)}",
                    "channel_kind": channel_kind,
                    "snr_db": round(snr_db, 1),
                    "h_real": round(h_real, 3),
                    "h_imag": round(h_imag, 3),
                    "fading_gain": round(fading_gain, 3),
                    "fading_db": round(fading_db, 2),
                    "inst_snr_db": round(inst_snr, 2),
                    "sigma2": round(sigma2, 4),
                    "mse_variant": mse_variant,
                    "orig_b64": _tensor_to_b64_png(img),
                    "orig_label": orig_label,
                    "pixel_patch": pixel_patch,
                },
                "digital": digital,
                "jscc_mse": jscc_mse,
                "jscc_sem": jscc_sem,
                "task": task_res,
            }

    @torch.no_grad()
    def sweep(self, params):
        dataset = params.get("dataset", "kodak")
        sample_id = params.get("sample_id", "kodim23" if dataset == "kodak" else "0")
        channel_kind = params.get("channel_kind", "awgn")
        c_active = int(params.get("c_active", 8))
        mcs_str = params.get("mcs", "16QAM 1/2")
        mse_variant = params.get("mse_variant", "jscc_mse")
        seed = int(params.get("seed", 42))

        key = (dataset, str(sample_id), channel_kind, c_active, mcs_str, mse_variant, seed)
        if key in self._sweep_cache:
            return self._sweep_cache[key]

        snrs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        mcs = next((m for m in dg.MCS if dg.mcs_name(m) == mcs_str), dg.MCS[2])

        with self.lock:
            img, _, orig_cls = self._get_sample(dataset, sample_id)
            orig_embed = imagenet_embed(img[None]) if dataset in ("kodak", "custom") else None
            rng_fade = np.random.default_rng(seed)
            if channel_kind == "rayleigh":
                hr = float(rng_fade.normal(0.0, 1.0 / math.sqrt(2.0)))
                hi = float(rng_fade.normal(0.0, 1.0 / math.sqrt(2.0)))
                fading_gain = max(1e-6, hr * hr + hi * hi)
            else:
                fading_gain = 1.0

            points = []
            for snr_db in snrs:
                d = self._run_digital(img, dataset, mcs, snr_db, c_active, channel_kind, fading_gain, seed, orig_embed, orig_cls, include_extras=False)
                cap = self._run_capacity_point(img, dataset, snr_db, c_active, channel_kind, fading_gain, orig_embed, orig_cls)
                m = self._run_jscc(mse_variant, img, dataset, snr_db, c_active, channel_kind, fading_gain, seed, orig_embed, orig_cls, include_layers=False)
                s = self._run_jscc("jscc_sem", img, dataset, snr_db, c_active, channel_kind, fading_gain, seed, orig_embed, orig_cls, include_layers=False)
                points.append({
                    "snr": snr_db,
                    "digital_psnr": d["psnr"],
                    "digital_sem": d["sem_score"],
                    "digital_ber": d["ber"],
                    "digital_cliff": d["cliff_hit"],
                    "cap_psnr": cap["psnr"],
                    "cap_sem": cap["sem_score"],
                    "mse_psnr": m["psnr"],
                    "mse_sem": m["sem_score"],
                    "sem_psnr": s["psnr"],
                    "sem_sem": s["sem_score"],
                })

            res = {"snrs": snrs, "points": points, "sem_metric": (
                "ImageNet Cosine" if dataset in ("kodak", "custom") else "Judge-B Confidence"
            )}
            self._sweep_cache[key] = res
            return res

    @torch.no_grad()
    def bw_sweep(self, params):
        dataset = params.get("dataset", "kodak")
        sample_id = params.get("sample_id", "kodim23" if dataset == "kodak" else "0")
        snr_db = float(params.get("snr_db", 6.0))
        channel_kind = params.get("channel_kind", "awgn")
        mcs_str = params.get("mcs", "16QAM 1/2")
        mse_variant = params.get("mse_variant", "jscc_mse")
        seed = int(params.get("seed", 42))

        key = (dataset, str(sample_id), snr_db, channel_kind, mcs_str, mse_variant, seed)
        if key in self._bw_cache:
            return self._bw_cache[key]

        widths = [2, 4, 8, 12, 16]
        mcs = next((m for m in dg.MCS if dg.mcs_name(m) == mcs_str), dg.MCS[2])

        with self.lock:
            img, _, orig_cls = self._get_sample(dataset, sample_id)
            orig_embed = imagenet_embed(img[None]) if dataset in ("kodak", "custom") else None
            rng_fade = np.random.default_rng(seed)
            if channel_kind == "rayleigh":
                hr = float(rng_fade.normal(0.0, 1.0 / math.sqrt(2.0)))
                hi = float(rng_fade.normal(0.0, 1.0 / math.sqrt(2.0)))
                fading_gain = max(1e-6, hr * hr + hi * hi)
            else:
                fading_gain = 1.0

            points = []
            for c_act in widths:
                d = self._run_digital(img, dataset, mcs, snr_db, c_act, channel_kind, fading_gain, seed, orig_embed, orig_cls, include_extras=False)
                cap = self._run_capacity_point(img, dataset, snr_db, c_act, channel_kind, fading_gain, orig_embed, orig_cls)
                m = self._run_jscc(mse_variant, img, dataset, snr_db, c_act, channel_kind, fading_gain, seed, orig_embed, orig_cls, include_layers=False)
                s = self._run_jscc("jscc_sem", img, dataset, snr_db, c_act, channel_kind, fading_gain, seed, orig_embed, orig_cls, include_layers=False)
                points.append({
                    "c": c_act,
                    "ratio_label": f"1/{round(1 / ratio_from_channels(c_act))}",
                    "digital_psnr": d["psnr"],
                    "digital_sem": d["sem_score"],
                    "cap_psnr": cap["psnr"],
                    "cap_sem": cap["sem_score"],
                    "mse_psnr": m["psnr"],
                    "mse_sem": m["sem_score"],
                    "sem_psnr": s["psnr"],
                    "sem_sem": s["sem_score"],
                })
            res = {"points": points, "sem_metric": (
                "ImageNet Cosine" if dataset in ("kodak", "custom") else "Judge-B Confidence"
            )}
            self._bw_cache[key] = res
            return res

    @torch.no_grad()
    def simulate_video(self, params):
        clip = params.get("clip", "foreman")
        if clip not in ("akiyo", "foreman", "bus"):
            clip = "foreman"
        snr_db = float(params.get("snr_db", 8.0))
        c_active = int(params.get("c_active", 8))
        frac = float(params.get("frac", 0.35))
        gop = int(params.get("gop", 10))
        seed = int(params.get("seed", 42))

        key = (clip, round(snr_db, 2), c_active, round(frac, 3), gop, seed)
        if key in self._video_cache:
            return self._video_cache[key]

        with self.lock:
            if clip not in self._videos:
                path = os.path.join(ROOT, "data", "video", f"{clip}_cif.y4m")
                self._videos[clip] = y4m(path, n_frames=12)
            frames = self._videos[clip]
            model = self._get_model("jscc_sem")

            fbf = frame_by_frame(model, frames, snr_db, c_active, seed=seed)
            cr, cr_ratio, side_share = conditional_replenishment(model, frames, snr_db, c_active, frac=frac, gop=gop, block=2, seed=seed)

            T, _, H, W = frames.shape
            fbf_ratio = ratio_from_channels(c_active)

            # Build per-frame change heatmaps & selected block masks
            ref = None
            frame_items = []
            for t in range(T):
                z = model.enc(frames[t:t + 1])[:, :c_active]
                is_key = (t % gop == 0)
                if is_key:
                    ref = z.clone()
                    diff_b64 = _latent_to_b64_png(z.abs(), 4)
                    pct_sent = 100.0
                else:
                    d = F.avg_pool2d((z - ref).pow(2).sum(1, keepdim=True), 2)
                    nb = d.numel()
                    n_sel = max(1, int(round(frac * nb)))
                    sel = torch.zeros(nb, dtype=torch.bool)
                    sel[d.flatten().topk(n_sel).indices] = True
                    mask2d = F.interpolate(sel.view(1, 1, *d.shape[-2:]).float(), scale_factor=2, mode="nearest")[0, 0].numpy()
                    heat = d[0, 0].numpy()
                    heat = heat / max(1e-6, float(np.percentile(heat, 98)))
                    heat_up = np.kron(np.clip(heat, 0, 1), np.ones((2, 2)))
                    # Highlight selected macroblocks in bright emerald, skipped in dark blue
                    rgb = np.zeros((*heat_up.shape, 3), dtype=np.uint8)
                    rgb[..., 0] = (np.clip(heat_up * 220, 0, 255) * (1 - 0.5 * mask2d)).astype(np.uint8)
                    rgb[..., 1] = np.clip(heat_up * 140 + mask2d * 115, 0, 255).astype(np.uint8)
                    rgb[..., 2] = np.clip(35 + (1 - mask2d) * 70, 0, 255).astype(np.uint8)
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, format="PNG")
                    diff_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
                    pct_sent = round(100.0 * n_sel / nb, 1)
                    # Update reference on selected blocks
                    mask_full = F.interpolate(sel.view(1, 1, *d.shape[-2:]).float(), scale_factor=2, mode="nearest").bool().expand_as(z)
                    ref[mask_full] = z[mask_full]

                p_fbf = float(psnr(fbf[t:t + 1], frames[t:t + 1]).item())
                p_cr = float(psnr(cr[t:t + 1], frames[t:t + 1]).item())
                frame_items.append({
                    "t": t,
                    "is_key": is_key,
                    "pct_sent": pct_sent,
                    "psnr_fbf": round(p_fbf, 2),
                    "psnr_cr": round(p_cr, 2),
                    "orig_b64": _tensor_to_b64_png(frames[t]),
                    "fbf_b64": _tensor_to_b64_png(fbf[t]),
                    "cr_b64": _tensor_to_b64_png(cr[t]),
                    "diff_b64": diff_b64,
                })

            res = {
                "clip": clip,
                "H": H,
                "W": W,
                "T": T,
                "snr_db": snr_db,
                "c_active": c_active,
                "frac": frac,
                "gop": gop,
                "fbf_ratio": round(fbf_ratio, 5),
                "cr_ratio": round(float(cr_ratio), 5),
                "bw_saving_pct": round((1.0 - float(cr_ratio) / fbf_ratio) * 100.0, 1),
                "side_share_pct": round(float(side_share) * 100.0, 1),
                "frames": frame_items,
            }
            self._video_cache[key] = res
            return res

    def get_project_data(self):
        with self._proj_lock:
            if self._project_data is not None:
                return self._project_data

            models_dir = os.path.join(ROOT, "results", "models")
            models_info = []
            for name, cfg in JOBS.items():
                mdir = os.path.join(models_dir, name)
                info_p = os.path.join(mdir, "info.json")
                log_p = os.path.join(mdir, "train_log.csv")
                info = {}
                if os.path.exists(info_p):
                    with open(info_p, "r", encoding="utf-8") as f:
                        info = json.load(f)
                logs = []
                if os.path.exists(log_p):
                    with open(log_p, "r", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            logs.append({
                                "epoch": int(row["epoch"]),
                                "loss": round(float(row["loss"]), 5),
                                "val_metric": round(float(row["val_metric"]), 4),
                                "time_s": round(float(row["time_s"]), 1),
                            })
                role_map = {
                    "clsA": "ResNetSmall CIFAR-10 classifier (frozen inside Semantic JSCC loss)",
                    "clsB": "VGGSmall CIFAR-10 classifier (independent semantic judge, never in loss)",
                    "dae": "4-bit quantized digital source auto-encoder (CIFAR-10 separated baseline)",
                    "jscc_mse": "Syntactic Deep JSCC trained with pixel MSE across SNR ~ U[0,20] dB",
                    "jscc_sem": "Semantic Deep JSCC trained with MSE + 0.01·Feature + 0.003·CE (ClsA)",
                    "jscc_mse_snr0": "Syntactic Deep JSCC fixed at 0 dB (SNR mismatch study, k/n=1/12)",
                    "jscc_mse_snr10": "Syntactic Deep JSCC fixed at 10 dB (SNR mismatch study, k/n=1/12)",
                    "jscc_mse_snr20": "Syntactic Deep JSCC fixed at 20 dB (SNR mismatch study, k/n=1/12)",
                    "task": "Task-oriented stride-8 JSCC (transmits features directly for classification)",
                }
                models_info.append({
                    "name": name,
                    "kind": cfg["kind"],
                    "role": role_map.get(name, ""),
                    "epochs": info.get("epochs", cfg["epochs"]),
                    "params": info.get("params", 0),
                    "train_s": round(float(info.get("train_s", 0.0)), 1),
                    "val_metric": round(float(info.get("val_metric", 0.0)), 4),
                    "metric_label": "Accuracy" if cfg["kind"] in ("cls", "task") else "Val PSNR (dB)",
                    "device": info.get("device", "cpu"),
                    "logs": logs,
                })

            summary_rows = []
            sum_p = os.path.join(ROOT, "results", "tables", "summary.csv")
            if os.path.exists(sum_p):
                df = pd.read_csv(sum_p)
                for _, r in df.iterrows():
                    summary_rows.append({
                        "dataset": str(r["dataset"]),
                        "scheme": str(r["scheme"]),
                        "snr": int(r["snr"]),
                        "psnr": None if pd.isna(r["psnr"]) else round(float(r["psnr"]), 2),
                        "ssim": None if pd.isna(r["ssim"]) else round(float(r["ssim"]), 4),
                        "acc": None if pd.isna(r["acc"]) else round(float(r["acc"]) * 100, 1),
                        "mcs": "" if pd.isna(r["mcs"]) else str(r["mcs"]),
                        "sem_cos": None if pd.isna(r["sem_cos"]) else round(float(r["sem_cos"]), 4),
                        "top1_agree": None if pd.isna(r["top1_agree"]) else round(float(r["top1_agree"]) * 100, 1),
                    })

            self._project_data = {"models": models_info, "summary": summary_rows}
            return self._project_data


ENGINE = SimEngine()


class SimulatorHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Keep console clean

    def _send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(UI_PATH, "rb") as f:
                data = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/api/catalog":
            self._send_json(ENGINE.get_catalog())
        elif self.path == "/api/project_data":
            self._send_json(ENGINE.get_project_data())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(raw.decode("utf-8"))
        except Exception:
            params = {}

        try:
            if self.path == "/api/simulate":
                self._send_json(ENGINE.simulate(params))
            elif self.path == "/api/sweep":
                self._send_json(ENGINE.sweep(params))
            elif self.path == "/api/bw_sweep":
                self._send_json(ENGINE.bw_sweep(params))
            elif self.path == "/api/video_sim":
                self._send_json(ENGINE.simulate_video(params))
            elif self.path == "/api/upload":
                self._send_json(ENGINE.upload_image(params.get("data_url", ""), params.get("name", "custom")))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def _pick_port(preferred):
    for p in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return preferred


def main():
    ap = argparse.ArgumentParser(description="Interactive Semantic Communication Simulator")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    port = _pick_port(args.port)
    url = f"http://127.0.0.1:{port}"
    server = ThreadingHTTPServer(("127.0.0.1", port), SimulatorHandler)
    print(f"Semantic Communication Explorer running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down simulator.")
        server.server_close()


if __name__ == "__main__":
    main()
