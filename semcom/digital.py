"""Separated (Shannon-style) digital chain: source code -> channel code -> modulation -> AWGN -> soft demodulation
-> Viterbi decoding -> source decoding.

Channel code: rate-1/2 convolutional code, constraint length 7, generators (133, 171) octal, punctured to 2/3 and
3/4 with the IEEE 802.11a patterns; packets of at most PACKET information bits, each terminated with 6 tail bits.
Modulation: Gray-mapped square QAM (QPSK, 16-QAM, 64-QAM), unit average energy, max-log LLRs.
All packets of all images of one (MCS, SNR) point are decoded in parallel (batched numpy Viterbi).
"""
import io
import math

import numpy as np

K = 7
G = (0o133, 0o171)
NS = 1 << (K - 1)
PACKET = 1024
PUNCT = {"1/2": np.array([[1], [1]]), "2/3": np.array([[1, 1], [1, 0]]), "3/4": np.array([[1, 1, 0], [1, 0, 1]])}
RATE = {"1/2": 0.5, "2/3": 2 / 3, "3/4": 0.75}
# modulation and coding schemes (spectral efficiency in information bits per complex symbol)
MCS = [("QPSK", 2, "1/2"), ("QPSK", 2, "3/4"), ("16QAM", 4, "1/2"), ("16QAM", 4, "3/4"), ("64QAM", 6, "2/3"),
       ("64QAM", 6, "3/4")]


def mcs_name(m):
    return f"{m[0]} {m[2]}"


def mcs_eff(m):
    return m[1] * RATE[m[2]]


# ----------------------------------------------------------------------------------------- trellis
def _parity(v):
    return bin(v).count("1") & 1


# transition into next state ns from predecessor p (state = last 6 input bits, newest = LSB): word = (p<<1)|u
_PREV = np.zeros((NS, 2), np.int64)
_OUT = np.zeros((NS, 2, 2), np.int8)  # [ns, which predecessor, output bit index]
for ns in range(NS):
    u = ns & 1
    for j, msb in enumerate((0, 1)):
        p = (ns >> 1) | (msb << (K - 2))
        word = (p << 1) | u
        _PREV[ns, j] = p
        _OUT[ns, j] = [_parity(word & g) for g in G]


def conv_encode(bits):
    """bits: (B, L) uint8 -> coded (B, 2(L+6)) interleaved [c0 c1 c0 c1 ...], zero-terminated."""
    B, L = bits.shape
    u = np.concatenate([bits, np.zeros((B, K - 1), np.uint8)], 1)
    padded = np.concatenate([np.zeros((B, K - 1), np.uint8), u], 1)
    out = []
    for g in G:
        acc = np.zeros_like(u)
        for k in range(K):  # tap k multiplies u[t-k]; bit k of g (LSB = current input)
            if (g >> k) & 1:
                acc ^= padded[:, K - 1 - k:K - 1 - k + u.shape[1]]
        out.append(acc)
    return np.stack(out, 2).reshape(B, -1)


def puncture(coded, rate):
    P = PUNCT[rate]
    B, n = coded.shape
    T = n // 2
    period = P.shape[1]
    mask = np.tile(P, (1, math.ceil(T / period)))[:, :T].T.reshape(-1).astype(bool)
    return coded[:, mask], mask


def depuncture(llr, mask):
    out = np.zeros((llr.shape[0], mask.size), np.float32)
    out[:, mask] = llr
    return out


def viterbi(llr, L):
    """Soft-input Viterbi. llr: (B, 2(L+6)) with LLR = log P(0)/P(1); returns (B, L) decoded bits."""
    B = llr.shape[0]
    T = L + K - 1
    llr = llr.reshape(B, T, 2)
    sgn = 1 - 2 * _OUT.astype(np.float32)  # +1 for bit 0, -1 for bit 1 -> metric = sum(sgn * llr)
    metric = np.full((B, NS), -1e9, np.float32)
    metric[:, 0] = 0
    dec = np.zeros((T, B, NS), np.uint8)
    p0, p1 = _PREV[:, 0], _PREV[:, 1]
    s0, s1 = sgn[:, 0], sgn[:, 1]  # (NS, 2)
    for t in range(T):
        l = llr[:, t]  # (B,2)
        c0 = metric[:, p0] + l @ s0.T
        c1 = metric[:, p1] + l @ s1.T
        choose1 = c1 > c0
        metric = np.where(choose1, c1, c0)
        dec[t] = choose1
    state = np.zeros(B, np.int64)  # terminated trellis ends in state 0
    bits = np.zeros((B, T), np.uint8)
    ar = np.arange(B)
    for t in range(T - 1, -1, -1):
        bits[:, t] = state & 1
        j = dec[t, ar, state]
        state = _PREV[state, j]
    return bits[:, :L]


# ---------------------------------------------------------------------------------------- modulation
def _pam(m):
    """Gray-coded PAM with m bits: levels and the bit labels of each level."""
    n = 1 << m
    levels = np.arange(-(n - 1), n, 2, dtype=np.float32)
    gray = np.array([i ^ (i >> 1) for i in range(n)])
    labels = ((gray[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.uint8)
    return levels, labels


def qam_mod(bits, bps):
    """bits (N*bps,) -> complex symbols (N,), unit average energy (square Gray QAM, I and Q independent PAM)."""
    m = bps // 2
    levels, labels = _pam(m)
    norm = math.sqrt(2 * np.mean(levels ** 2))
    b = bits.reshape(-1, 2, m)
    w = 1 << np.arange(m - 1, -1, -1)
    lut = np.zeros(1 << m, np.int64)
    lut[(labels * w).sum(1)] = np.arange(1 << m)
    i = levels[lut[(b[:, 0] * w).sum(1)]]
    q = levels[lut[(b[:, 1] * w).sum(1)]]
    return (i + 1j * q) / norm


def qam_llr(y, bps, sigma2):
    """Max-log LLRs (log P(0)/P(1)) for each bit; sigma2 = complex noise variance."""
    m = bps // 2
    levels, labels = _pam(m)
    norm = math.sqrt(2 * np.mean(levels ** 2))
    out = []
    for comp in (y.real * norm, y.imag * norm):
        d = (comp[:, None] - levels[None, :]) ** 2 / (np.asarray(sigma2).reshape(-1, 1) * norm ** 2)  # per-dimension variance sigma2/2
        for k in range(m):
            d0 = np.where(labels[None, :, k] == 0, d, np.inf).min(1)
            d1 = np.where(labels[None, :, k] == 1, d, np.inf).min(1)
            out.append(d1 - d0)
    return np.stack(out, 1).reshape(-1).astype(np.float32)


# --------------------------------------------------------------------------------------- full link
def _packets(n_bits):
    """Split n_bits into ceil(n/PACKET) packets of (almost) equal length -> (n_packets, packet_length)."""
    if n_bits == 0:
        return 0, 0
    n_pk = math.ceil(n_bits / PACKET)
    return n_pk, math.ceil(n_bits / n_pk)


def _coded_len(pk_len, rate):
    T = pk_len + K - 1
    P = PUNCT[rate]
    return int(np.tile(P, (1, math.ceil(T / P.shape[1])))[:, :T].sum())


def transmit_bits(streams, mcs, snr_db, rng, fading_gain=None):
    """Send a list of bit streams (one per image) with one MCS over AWGN (optionally with a per-stream block fading
    power gain |h|^2, perfect CSI). Each packet is coded, modulated and transmitted separately.
    Returns the received (decoded) streams."""
    mod, bps, rate = mcs
    snr = np.full(len(streams), snr_db, np.float64)
    if fading_gain is not None:
        snr = snr + 10 * np.log10(np.maximum(fading_gain, 1e-12))
    groups = {}
    for s_i, s in enumerate(streams):
        n_pk, L = _packets(len(s))
        if n_pk == 0:
            continue
        pad = np.zeros(n_pk * L, np.uint8)
        pad[:len(s)] = s
        groups.setdefault(L, []).append((s_i, pad.reshape(n_pk, L)))
    dec_streams = [s.copy() for s in streams]
    for L, items in groups.items():
        bits = np.concatenate([b for _, b in items])
        owner = np.concatenate([[i] * len(b) for i, b in items])
        coded, mask = puncture(conv_encode(bits), rate)
        n_c = coded.shape[1]
        coded = np.concatenate([coded, np.zeros((coded.shape[0], (-n_c) % bps), np.uint8)], 1)
        sym = qam_mod(coded.reshape(-1), bps).reshape(coded.shape[0], -1)
        sigma2 = 10 ** (-snr[owner] / 10)[:, None]
        y = sym + np.sqrt(sigma2 / 2) * (rng.standard_normal(sym.shape) + 1j * rng.standard_normal(sym.shape))
        llr = qam_llr(y.reshape(-1), bps, np.repeat(sigma2[:, 0], sym.shape[1])).reshape(len(y), -1)[:, :n_c]
        dec = np.zeros_like(bits)
        for a in range(0, len(bits), 512):
            dec[a:a + 512] = viterbi(depuncture(llr[a:a + 512], mask), L)
        for i, _ in items:
            dec_streams[i] = dec[owner == i].reshape(-1)[:len(streams[i])]
    return dec_streams


def symbols_needed(n_bits, mcs):
    """Complex channel symbols used by n_bits information bits (packetisation and tail bits included)."""
    mod, bps, rate = mcs
    n_pk, L = _packets(n_bits)
    return n_pk * math.ceil(_coded_len(L, rate) / bps) if n_pk else 0


def max_bits(k_symbols, mcs):
    """Largest number of information bits (multiple of 8) whose transmission fits in k_symbols."""
    lo, hi = 0, int(k_symbols * mcs_eff(mcs)) + 8
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if symbols_needed(mid, mcs) <= k_symbols:
            lo = mid
        else:
            hi = mid - 1
    return lo - lo % 8


# ---------------------------------------------------------------------------------------- JPEG codec
def jpeg_fit(img, budget_bits):
    """img (3,H,W) float tensor/array. Highest JPEG quality (1..95) whose file fits in budget_bits.
    Returns (bytes, quality) or (None, 0) if even quality 1 does not fit."""
    from PIL import Image

    arr = (np.clip(np.asarray(img), 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(arr)
    best = (None, 0)
    lo, hi = 1, 95
    while lo <= hi:
        q = (lo + hi) // 2
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        if len(data) * 8 <= budget_bits:
            best, lo = (data, q), q + 1
        else:
            hi = q - 1
    return best


def jpeg_decode(data, shape):
    """Decode (possibly corrupted) JPEG bytes. A file the decoder rejects is replaced by a mid-grey image
    (the receiver has nothing else to show)."""
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    H, W = shape
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.load()
        if im.size != (W, H):
            raise ValueError("size")
        return np.asarray(im).astype(np.float32).transpose(2, 0, 1) / 255, True
    except Exception:
        return np.full((3, H, W), 0.5, np.float32), False


def bytes_to_bits(b):
    return np.unpackbits(np.frombuffer(b, np.uint8))


def bits_to_bytes(bits):
    return np.packbits(bits).tobytes()
