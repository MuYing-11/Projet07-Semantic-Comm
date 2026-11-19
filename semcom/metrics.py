"""Quality metrics: bit/pixel-level (PSNR, SSIM) and meaning-level (classifier accuracy, ImageNet feature
similarity, top-1 label agreement)."""
import functools

import torch
import torch.nn.functional as F


def psnr(x, y):
    """Per-image PSNR (dB) for images in [0,1]."""
    mse = (x - y).pow(2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1 / mse)


def _gauss(size=11, sigma=1.5):
    g = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim(x, y):
    """Per-image SSIM (Wang et al. 2004, 11x11 Gaussian window, sigma 1.5), averaged over RGB channels."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    c = x.shape[1]
    size = min(11, x.shape[-1], x.shape[-2])
    w = _gauss(size).to(x).repeat(c, 1, 1, 1)
    mu_x, mu_y = F.conv2d(x, w, groups=c), F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return s.flatten(1).mean(1)


@functools.lru_cache(maxsize=1)
def imagenet_judge():
    """Independent semantic evaluator for natural images: ImageNet-pretrained ResNet-18 (torchvision)."""
    import torchvision

    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def imagenet_embed(x, size=224):
    """Penultimate features (512-d) and logits of the ImageNet judge; images resized to `size` on the short side."""
    m = imagenet_judge()
    h, w = x.shape[-2:]
    s = size / min(h, w)
    x = F.interpolate(x, size=(int(round(h * s)), int(round(w * s))), mode="bilinear", align_corners=False,
                      antialias=True)
    x = (x - IMNET_MEAN) / IMNET_STD
    f = m.avgpool(m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x))))))))).flatten(1)
    return f, m.fc(f)


@torch.no_grad()
def semantic_similarity(x, y):
    """Cosine similarity of ImageNet features and top-1 label agreement between original x and received y."""
    fx, lx = imagenet_embed(x)
    fy, ly = imagenet_embed(y)
    cos = F.cosine_similarity(fx, fy, dim=1)
    agree = (lx.argmax(1) == ly.argmax(1)).float()
    top5 = torch.stack([torch.isin(lx.argmax(1)[i], ly.topk(5, 1).indices[i]) for i in range(len(lx))]).float()
    return cos, agree, top5
