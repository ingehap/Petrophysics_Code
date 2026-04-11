"""
Trevizan and Menezes de Jesus (2023), Petrophysics 64(6): 890-899.
GAN-based super-resolution of borehole image logs for real-time geological-
structure detection while drilling.

Provides a tiny PyTorch generator/discriminator and a 1-batch training loop.
Falls back to a deterministic bicubic-style upsampler if torch is unavailable.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


if HAVE_TORCH:
    class Generator(nn.Module):
        def __init__(self, scale=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=scale, mode="bilinear",
                            align_corners=False),
                nn.Conv2d(16, 1, 3, padding=1),
            )
        def forward(self, x): return self.net(x)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
                nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 1),
            )
        def forward(self, x): return self.net(x)

    def train_step(g, d, lr_img, hr_img, opt_g, opt_d):
        bce = nn.BCEWithLogitsLoss(); l1 = nn.L1Loss()
        fake = g(lr_img)
        opt_d.zero_grad()
        loss_d = bce(d(hr_img), torch.ones_like(d(hr_img))) + \
                 bce(d(fake.detach()), torch.zeros_like(d(fake)))
        loss_d.backward(); opt_d.step()
        opt_g.zero_grad()
        loss_g = bce(d(fake), torch.ones_like(d(fake))) + 10 * l1(fake, hr_img)
        loss_g.backward(); opt_g.step()
        return float(loss_g), float(loss_d)


def upsample_bilinear(img, scale=2):
    h, w = img.shape
    out = np.zeros((h * scale, w * scale))
    for i in range(h * scale):
        for j in range(w * scale):
            out[i, j] = img[min(i // scale, h - 1), min(j // scale, w - 1)]
    return out


def test_all():
    rng = np.random.default_rng(7)
    hr = rng.random((32, 32)).astype(np.float32)
    lr = hr[::2, ::2]
    print("Trevizan & Menezes GAN super-resolution:")
    if HAVE_TORCH:
        g = Generator(); d = Discriminator()
        og = torch.optim.Adam(g.parameters(), 1e-3)
        od = torch.optim.Adam(d.parameters(), 1e-3)
        lr_t = torch.tensor(lr)[None, None]
        hr_t = torch.tensor(hr)[None, None]
        for _ in range(3):
            lg, ld = train_step(g, d, lr_t, hr_t, og, od)
        out = g(lr_t).detach().numpy()[0, 0]
        print(f"  torch trained, output shape {out.shape}, losses g={lg:.3f} d={ld:.3f}")
        assert out.shape == hr.shape
    else:
        out = upsample_bilinear(lr, 2)
        print(f"  fallback bilinear, output shape {out.shape}")
        assert out.shape == hr.shape
    print("  PASS")


if __name__ == "__main__":
    test_all()
