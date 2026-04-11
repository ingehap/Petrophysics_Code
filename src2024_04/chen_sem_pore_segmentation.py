"""
chen_sem_pore_segmentation.py
Implementation of ideas from:
Chen et al., "Pore Identification and Quantitative Characterization of Shale
SEM Images using a Deep-Learning Semantic Segmentation Pore-Net Model",
Petrophysics, Vol. 65, No. 2 (April 2024), pp. 233-245.

The paper applies a U-Net-style ("pore-net") semantic segmentation network to
shale SEM images to label pore vs. matrix pixels and then computes porosity
and pore-size statistics. We provide:
  - synthetic SEM image generator,
  - a small "pore-net" inspired thresholding+morphology baseline,
  - an optional U-Net definition (PyTorch) used only if torch is available,
  - quantitative characterization (porosity, pore size distribution).
"""
import numpy as np

try:
    from scipy import ndimage as ndi
except Exception as e:
    raise SystemExit("scipy is required: " + str(e))


def synthetic_sem_image(size=128, n_pores=40, rng=None):
    rng = rng or np.random.default_rng(4)
    img = 0.6 + 0.05 * rng.standard_normal((size, size))
    mask = np.zeros((size, size), dtype=bool)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(n_pores):
        cy, cx = rng.integers(0, size, size=2)
        r = rng.integers(2, 8)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        mask |= m
    img[mask] = 0.15 + 0.05 * rng.standard_normal(mask.sum())
    return np.clip(img, 0, 1), mask


def porenet_segment(img, threshold=0.35):
    """Lightweight 'pore-net' baseline: threshold + morphological cleanup."""
    seg = img < threshold
    seg = ndi.binary_opening(seg, iterations=1)
    seg = ndi.binary_closing(seg, iterations=1)
    return seg


def porosity(seg):
    return float(seg.mean())


def pore_size_distribution(seg):
    labels, n = ndi.label(seg)
    if n == 0:
        return np.array([])
    sizes = ndi.sum(seg, labels, index=np.arange(1, n + 1))
    return sizes  # pixels per pore


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union else 1.0


def build_unet(in_ch=1, out_ch=2):
    """Optional small U-Net (only built if PyTorch is installed)."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None

    class DoubleConv(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))
        def forward(self, x): return self.net(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1 = DoubleConv(in_ch, 16)
            self.d2 = DoubleConv(16, 32)
            self.u1 = DoubleConv(32 + 16, 16)
            self.out = nn.Conv2d(16, out_ch, 1)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        def forward(self, x):
            x1 = self.d1(x)
            x2 = self.d2(self.pool(x1))
            x3 = self.u1(torch.cat([self.up(x2), x1], dim=1))
            return self.out(x3)
    return UNet()


def test_all():
    img, gt = synthetic_sem_image()
    seg = porenet_segment(img)
    phi = porosity(seg)
    sizes = pore_size_distribution(seg)
    score = iou(seg, gt)
    assert 0 < phi < 0.5
    assert sizes.size > 0
    assert score > 0.3, "IoU too low: %.2f" % score
    _ = build_unet()  # may be None if torch missing
    print("chen_sem_pore_segmentation OK  porosity=%.3f IoU=%.2f n_pores=%d" % (phi, score, len(sizes)))


if __name__ == "__main__":
    test_all()
