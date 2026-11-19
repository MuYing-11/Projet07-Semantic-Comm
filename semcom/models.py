"""Neural models. All image models are fully convolutional, so a model trained on 32x32 CIFAR-10 images also
runs on 768x512 Kodak images and 352x288 video frames (any size divisible by 4)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import channel as ch

C_LATENT = 16  # latent channels at stride 4 -> bandwidth ratio up to 16/96 = 1/6
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def conv(i, o, k=3, s=1):
    return nn.Conv2d(i, o, k, s, k // 2)


def deconv(i, o, k=5, s=2):
    return nn.ConvTranspose2d(i, o, k, s, k // 2, output_padding=s - 1)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.c2, self.a = conv(c, c), conv(c, c), nn.PReLU(c)

    def forward(self, x):
        return x + self.c2(self.a(self.c1(x)))


class Encoder(nn.Module):
    """x in [0,1]^(3,H,W) -> z in R^(C, H/4, W/4)."""

    def __init__(self, c_out=C_LATENT, width=64):
        super().__init__()
        self.net = nn.Sequential(
            conv(3, width // 2, 5, 2), nn.PReLU(width // 2),
            conv(width // 2, width, 5, 2), nn.PReLU(width),
            ResBlock(width), ResBlock(width),
            conv(width, c_out, 3))

    def forward(self, x):
        return self.net(x - 0.5)


class Decoder(nn.Module):
    def __init__(self, c_in=C_LATENT, width=64):
        super().__init__()
        self.net = nn.Sequential(
            conv(c_in, width, 3), nn.PReLU(width),
            ResBlock(width), ResBlock(width),
            deconv(width, width // 2), nn.PReLU(width // 2),
            deconv(width // 2, 3), nn.Sigmoid())

    def forward(self, y):
        return self.net(y)


class DeepJSCC(nn.Module):
    """Analog joint source-channel coding: encoder -> power normalisation -> channel -> decoder.
    The same architecture is trained with a pixel loss (syntactic JSCC) or a semantic loss (semantic system)."""

    def __init__(self, c=C_LATENT):
        super().__init__()
        self.c = c
        self.enc, self.dec = Encoder(c), Decoder(c)

    def forward(self, x, snr_db, c_active=None, kind="awgn"):
        c_active = c_active or self.c
        return self.dec(ch.channel(self.enc(x), c_active, snr_db, kind))

    def encode_latent(self, x):
        return self.enc(x)


class DigitalAE(nn.Module):
    """Source code of the separated digital chain: auto-encoder with a bounded latent uniformly quantised on
    `bits` bits per value (additive uniform noise as a differentiable proxy during training). The latent channels
    are ordered by importance through random truncation during training, so the bit rate is chosen at run time."""

    def __init__(self, c=C_LATENT, bits=4):
        super().__init__()
        self.c, self.bits = c, bits
        self.enc, self.dec = Encoder(c), Decoder(c)

    def quantize(self, z, train):
        u = torch.sigmoid(z)
        L = 2 ** self.bits - 1
        if train:
            return u + (torch.rand_like(u) - 0.5) / L
        return torch.round(u * L) / L

    def to_levels(self, x, c_active):
        u = torch.sigmoid(self.enc(x))[:, :c_active]
        return torch.round(u * (2 ** self.bits - 1)).to(torch.int64)

    def from_levels(self, q, c_active):
        u = q.float() / (2 ** self.bits - 1)
        B, _, h, w = u.shape
        full = torch.full((B, self.c, h, w), 0.5)  # missing channels = mid-level, as in training
        full[:, :c_active] = u
        return self.dec(full * 2 - 1)

    def forward(self, x, c_active=None, train=False):
        c_active = c_active or self.c
        u = self.quantize(self.enc(x), train)
        return self.dec(ch.truncate(u * 2 - 1, c_active))


# ------------------------------------------------------------------------------------- CIFAR-10 classifiers
class ResNetSmall(nn.Module):
    """Classifier A: small ResNet. Used inside the semantic loss (feature + task terms)."""

    def __init__(self, n_cls=10, w=64):
        super().__init__()
        self.stem = nn.Sequential(conv(3, w), nn.BatchNorm2d(w), nn.ReLU())
        self.l1 = self._stage(w, w, 1)
        self.l2 = self._stage(w, 2 * w, 2)
        self.l3 = self._stage(2 * w, 4 * w, 2)
        self.fc = nn.Linear(4 * w, n_cls)

    @staticmethod
    def _stage(i, o, s):
        return nn.Sequential(conv(i, o, 3, s), nn.BatchNorm2d(o), nn.ReLU(), conv(o, o), nn.BatchNorm2d(o), nn.ReLU())

    def features(self, x):
        x = (x - CIFAR_MEAN.to(x)) / CIFAR_STD.to(x)
        f1 = self.l1(self.stem(x))
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        return [f1, f2, f3]

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.features(x)[-1], 1).flatten(1))


class VGGSmall(nn.Module):
    """Classifier B: different architecture, never used in any training loss. It is the independent judge of
    semantic preservation on CIFAR-10 (accuracy on received images)."""

    def __init__(self, n_cls=10):
        super().__init__()
        cfg = [32, 32, "M", 64, 64, "M", 128, 128, "M"]
        layers, c = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [conv(c, v), nn.BatchNorm2d(v), nn.ReLU()]
                c = v
        self.body = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 * 16, n_cls))

    def forward(self, x):
        return self.fc(self.body((x - CIFAR_MEAN.to(x)) / CIFAR_STD.to(x)).flatten(1))


# ------------------------------------------------------------------------------- task-oriented (extension)
class TaskJSCC(nn.Module):
    """Task-oriented transmission: only what the receiver needs to classify is sent (no reconstruction).
    Encoder at stride 8 (4x4 positions on CIFAR) -> C_t channels -> channel -> classifier head."""

    def __init__(self, c=16, n_cls=10, w=64):
        super().__init__()
        self.c = c
        self.enc = nn.Sequential(conv(3, w, 3, 2), nn.PReLU(w), conv(w, w, 3, 2), nn.PReLU(w), ResBlock(w),
                                 conv(w, w, 3, 2), nn.PReLU(w), conv(w, c, 3))
        self.head = nn.Sequential(conv(c, 2 * w), nn.PReLU(2 * w), ResBlock(2 * w), nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(), nn.Linear(2 * w, n_cls))

    def forward(self, x, snr_db, c_active=None, kind="awgn"):
        c_active = c_active or self.c
        return self.head(ch.channel(self.enc((x - CIFAR_MEAN.to(x)) / CIFAR_STD.to(x)), c_active, snr_db, kind))


def n_params(m):
    return sum(p.numel() for p in m.parameters())
