"""Dataset loaders. Every image is a float tensor in [0,1] with shape (3,H,W)."""
import glob
import os

import numpy as np
import torch

from . import ROOT

DATA = os.path.join(ROOT, "data")


def cifar10(train=True):
    """CIFAR-10 as in-memory tensors: images (N,3,32,32) float in [0,1], labels (N,)."""
    d = np.load(os.path.join(DATA, "cifar10", "cifar10.npz"))
    split = "train" if train else "test"
    x = torch.from_numpy(d[f"x_{split}"]).permute(0, 3, 1, 2).float() / 255
    return x, torch.from_numpy(d[f"y_{split}"])


def augment(x):
    """Random horizontal flip + random crop with 4-pixel reflection padding (training only)."""
    B = x.shape[0]
    flip = torch.rand(B) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    p = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = np.random.randint(0, 9, 2)
    return p[:, :, i:i + 32, j:j + 32]


def kodak():
    from PIL import Image

    files = sorted(glob.glob(os.path.join(DATA, "kodak", "kodim*.png")))
    imgs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        if im.height > im.width:  # 4 portrait images are rotated to 768x512 so they can be batched
            im = im.transpose(Image.Transpose.ROTATE_90)
        imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255)
    return torch.stack(imgs), [os.path.basename(f)[:-4] for f in files]


def yuv420_to_rgb(y, u, v):
    """BT.601 limited-range YUV 4:2:0 -> RGB in [0,1]."""
    u = u.repeat(2, 0).repeat(2, 1).astype(np.float32)
    v = v.repeat(2, 0).repeat(2, 1).astype(np.float32)
    y = y.astype(np.float32)
    c, d, e = y - 16, u - 128, v - 128
    r = 1.164 * c + 1.596 * e
    g = 1.164 * c - 0.392 * d - 0.813 * e
    b = 1.164 * c + 2.017 * d
    return np.clip(np.stack([r, g, b]) / 255, 0, 1)


def y4m(path, n_frames=30, step=1):
    """Read the first n_frames (every `step` frame) of a YUV 4:2:0 Y4M file -> (T,3,H,W) tensor."""
    with open(path, "rb") as fh:
        header = fh.readline().decode().split()
        W = int([t for t in header if t.startswith("W")][0][1:])
        H = int([t for t in header if t.startswith("H")][0][1:])
        fsize = W * H * 3 // 2
        frames, k = [], 0
        while len(frames) < n_frames:
            line = fh.readline()
            if not line.startswith(b"FRAME"):
                break
            buf = np.frombuffer(fh.read(fsize), np.uint8)
            if k % step == 0:
                y = buf[:W * H].reshape(H, W)
                u = buf[W * H:W * H + W * H // 4].reshape(H // 2, W // 2)
                v = buf[W * H + W * H // 4:].reshape(H // 2, W // 2)
                frames.append(torch.from_numpy(yuv420_to_rgb(y, u, v)))
            k += 1
    return torch.stack(frames)


def video_clips(n_frames=30):
    clips = {}
    for c in ["akiyo", "foreman", "bus"]:
        p = os.path.join(DATA, "video", f"{c}_cif.y4m")
        if os.path.exists(p):
            clips[c] = y4m(p, n_frames)
    return clips
