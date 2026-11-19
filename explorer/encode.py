"""Turn tensors and arrays into JSON-friendly things a browser can draw."""
import base64
import io

import numpy as np
import torch

MAX_PREVIEW = 420  # longest side of an image sent to the browser (metrics stay full resolution)
MAX_POINTS = 1400  # symbols drawn in a constellation
MAX_BITS = 1024  # bits shown in the bit dump
MAX_LLR = 192  # LLR values shown


def png(x, max_side=MAX_PREVIEW, zoom_small=True):
    """x: (3,H,W) or (1,3,H,W) float tensor/array in [0,1] -> data: URI of a PNG."""
    from PIL import Image

    a = np.asarray(x.detach().cpu() if torch.is_tensor(x) else x, dtype=np.float32)
    if a.ndim == 4:
        a = a[0]
    a = (np.clip(a.transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(a)
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    elif zoom_small and max(w, h) < 160:  # CIFAR: nearest-neighbour zoom, so pixels stay visible
        s = 160 // max(w, h)
        im = im.resize((w * s, h * s), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def heat(a, size=28):
    """One channel map (h,w) -> {w,h,v:[0..255]} after pooling to at most `size` on each side."""
    t = torch.as_tensor(np.asarray(a, dtype=np.float32))[None, None]
    if max(t.shape[-2:]) > size:
        oh = max(1, min(size, round(size * t.shape[-2] / max(t.shape[-2:]))))
        ow = max(1, min(size, round(size * t.shape[-1] / max(t.shape[-2:]))))
        t = torch.nn.functional.adaptive_avg_pool2d(t, (oh, ow))
    v = t[0, 0].numpy()
    lo, hi = float(v.min()), float(v.max())
    u = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    return dict(w=v.shape[1], h=v.shape[0], lo=round(lo, 3), hi=round(hi, 3),
                v=(u * 255 + 0.5).astype(np.uint8).reshape(-1).tolist())


def latent_stack(z, c_active, size=28):
    """z: (1,C,h,w) tensor -> per-channel maps plus which channels are actually transmitted."""
    z = z[0].detach().cpu()
    maps = [heat(z[i].numpy(), size) for i in range(z.shape[0])]
    energy = z.pow(2).mean((1, 2))
    return dict(maps=maps, c_active=int(c_active), c_total=int(z.shape[0]),
                energy=[round(float(e), 4) for e in energy])


def cloud(sym, n=MAX_POINTS, seed=0):
    """Complex symbols -> at most n (I,Q) pairs, rounded, plus the real count."""
    s = np.asarray(sym).reshape(-1)
    total = int(s.size)
    if total > n:
        idx = np.random.default_rng(seed).choice(total, n, replace=False)
        s = s[idx]
    pts = np.stack([s.real, s.imag], 1).astype(np.float32)
    return dict(points=np.round(pts, 4).tolist(), total=total,
                power=round(float(np.mean(np.abs(np.asarray(sym).reshape(-1)) ** 2)), 4))


def bitdump(bits, errors=None, n=MAX_BITS):
    """Bit array -> a string of 0/1 (truncated) and the positions that differ from the sent bits."""
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    shown = b[:n]
    out = dict(bits="".join(map(str, shown.tolist())), total=int(b.size), shown=int(shown.size))
    if errors is not None:
        e = np.asarray(errors, dtype=bool).reshape(-1)
        out["errors"] = np.flatnonzero(e[:n]).tolist()
        out["n_errors"] = int(e.sum())
        out["ber"] = round(float(e.mean()), 6) if e.size else None
    return out


def llr_dump(llr, n=MAX_LLR):
    v = np.asarray(llr, dtype=np.float32).reshape(-1)[:n]
    return dict(values=np.round(v, 2).tolist(), shown=int(v.size))


def bars(probs, labels, truth=None, top=5):
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    order = np.argsort(-p)[:top]
    return [dict(label=labels[i], value=round(float(p[i]), 4), truth=(truth is not None and int(i) == int(truth)))
            for i in order]
