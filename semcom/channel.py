"""Simulated wireless channel layer, inserted between the neural encoder and decoder.

The encoder outputs a real tensor z of shape (B, C, h, w). Pairs of real values form complex channel symbols, so an
image of n = 3HW real source values is sent with k = C*h*w/2 complex symbols (bandwidth ratio k/n).
Only the first `c_active` latent channels are transmitted (bandwidth-agile transmission): the others are zeroed
at the receiver.
"""
import math

import torch


def ratio_from_channels(c_active, stride=4):
    """Bandwidth ratio k/n for c_active latent channels at the given spatial stride (RGB input)."""
    return c_active / (2 * 3 * stride * stride)


def truncate(z, c_active):
    if c_active >= z.shape[1]:
        return z
    mask = torch.zeros(1, z.shape[1], 1, 1, device=z.device, dtype=z.dtype)
    mask[:, :c_active] = 1
    return z * mask


def power_normalize(z, c_active):
    """Average power 1 per complex symbol over the transmitted symbols of each image (P = 1)."""
    zt = z[:, :c_active]
    k = zt[0].numel() / 2  # complex symbols per image
    energy = zt.pow(2).flatten(1).sum(1).clamp_min(1e-9)
    scale = torch.sqrt(k / energy).view(-1, 1, 1, 1)
    return truncate(z * scale, c_active)


def awgn(x, snr_db):
    """x: power-normalised real tensor (pairs = complex symbols). snr_db: float or (B,) tensor."""
    snr = torch.as_tensor(snr_db, dtype=x.dtype, device=x.device)
    if snr.dim() == 0:
        snr = snr.expand(x.shape[0])
    sigma2 = 10 ** (-snr / 10)  # complex noise variance, i.e. sigma2/2 per real dimension
    std = torch.sqrt(sigma2 / 2).view(-1, *([1] * (x.dim() - 1)))
    return x + std * torch.randn_like(x)


def rayleigh_block(x, snr_db, generator=None):
    """Slow (block) Rayleigh fading: one h ~ CN(0,1) per image, perfect CSI and zero-forcing at the receiver.
    Returns the equalised symbols and the per-image instantaneous SNR (dB)."""
    B = x.shape[0]
    hr = torch.randn(B, generator=generator) / math.sqrt(2)
    hi = torch.randn(B, generator=generator) / math.sqrt(2)
    g = (hr ** 2 + hi ** 2).to(x)  # |h|^2, exponential(1)
    snr_inst = torch.as_tensor(snr_db, dtype=x.dtype, device=x.device) + 10 * torch.log10(g.clamp_min(1e-12))
    # after ZF equalisation with perfect CSI: y/h = x + n/h -> AWGN with SNR * |h|^2
    return awgn(x, snr_inst), snr_inst


def channel(z, c_active, snr_db, kind="awgn"):
    x = power_normalize(z, c_active)
    if kind == "awgn":
        y = awgn(x, snr_db)
    elif kind == "rayleigh":
        y, _ = rayleigh_block(x, snr_db)
    elif kind == "none":
        y = x
    else:
        raise ValueError(kind)
    return truncate(y, c_active)
