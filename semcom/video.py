"""Video transmission.

Socle: frame-by-frame transmission (every frame sent independently by the image system).
Extension: latent conditional replenishment (CR). Frame 0 of every group of pictures (GOP) is a key frame sent in
full; for the next frames the transmitter only sends the latent blocks whose semantic content changed most since
they were last sent (a fraction f of the 2x2 latent blocks, i.e. 8x8-pixel macroblocks). The receiver keeps its last
latent and overwrites the received blocks. The block map is side information, counted at its binary-entropy cost
(N_blocks * H(f) bits) and sent at 1 bit per channel symbol (QPSK, rate 1/2), assumed error-free.
"""
import math

import torch
import torch.nn.functional as F

from . import channel as ch


def h2(f):
    return 0.0 if f <= 0 or f >= 1 else -(f * math.log2(f) + (1 - f) * math.log2(1 - f))


@torch.no_grad()
def frame_by_frame(model, frames, snr_db, c_active, seed=0):
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    frames = frames.to(dev)
    return torch.cat([model(frames[i:i + 1], snr_db, c_active).cpu() for i in range(len(frames))])


@torch.no_grad()
def conditional_replenishment(model, frames, snr_db, c_active, frac, gop=10, block=2, seed=0):
    """Returns (received frames, average bandwidth ratio k/n actually used, side-information share)."""
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    frames = frames.to(dev)
    T, _, H, W = frames.shape
    n_src = 3 * H * W
    out, symbols, side = [], 0.0, 0.0
    ref = rx = None
    for t in range(T):
        z = model.enc(frames[t:t + 1])[:, :c_active]
        if t % gop == 0:  # key frame: everything
            y = ch.awgn(ch.power_normalize(z, c_active), snr_db)
            ref, rx = z.clone(), y.clone()
            symbols += z.numel() / 2
        else:
            # semantic change per latent block, measured on the transmitter's copy of what was last sent
            d = F.avg_pool2d((z - ref).pow(2).sum(1, keepdim=True), block)
            nb = d.numel()
            n_sel = max(1, int(round(frac * nb)))
            sel = torch.zeros(nb, dtype=torch.bool, device=d.device)
            sel[d.flatten().topk(n_sel).indices] = True
            mask = F.interpolate(sel.view(1, 1, *d.shape[-2:]).float(), scale_factor=block, mode="nearest").bool()
            mask = mask.expand_as(z)
            zs = z[mask]  # symbols of the selected blocks only
            k = zs.numel() / 2
            xs = zs * math.sqrt(k / zs.pow(2).sum().clamp_min(1e-9))  # unit power per complex symbol
            ys = ch.awgn(xs.view(1, -1), snr_db).view(-1) / math.sqrt(k / zs.pow(2).sum().clamp_min(1e-9))
            # the receiver rescales with the power factor of the key frame statistics (sent as 1 scalar)
            rx_un = rx.clone()
            scale_key = math.sqrt((ref.numel() / 2) / ref.pow(2).sum().clamp_min(1e-9).item())
            rx_un[mask] = ys * scale_key
            rx = rx_un
            ref[mask] = zs
            sbits = nb * h2(frac)
            symbols += k + sbits
            side += sbits
        full = torch.zeros(1, model.c, H // 4, W // 4, device=rx.device, dtype=rx.dtype)
        full[:, :c_active] = rx
        out.append(model.dec(full).cpu())
    ratio = symbols / (T * n_src)
    return torch.cat(out), ratio, side / max(symbols, 1)
