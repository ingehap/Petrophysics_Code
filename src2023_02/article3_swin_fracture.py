"""
Article 3: Fracture Extraction From Logging Image Using a Dual Encoder-Decoder
Architecture With Swin Transformer
Wang, Zhou (2023)
DOI: 10.30632/PJV64N1-2023a3

A NumPy-only proof-of-concept of the W-shape dual encoder-decoder used in
the paper to segment sinusoidal fractures on microresistivity borehole
images.  The full paper trains a Swin-Transformer-based deep network; here
we implement the SAME data + evaluation pipeline and substitute the network
with a tractable analytical pipeline that captures the key ideas:

  - Patch partitioning + window-based local feature aggregation (the
    Swin "W-MSA" idea, Eqs. 1-2 of the paper).
  - A second decoder branch that locks onto sinusoidal-fracture priors
    (Hough-like radon accumulation across (amplitude, phase, depth)).
  - Per-pixel Dice, mIoU, Precision, Recall (Eqs. 3-6).

The synthetic dataset reproduces the planar fractures that traverse the
borehole as sinusoids in azimuth-depth (unwrapped) space.
"""

import numpy as np


# ------------------------------------------- complexity formulas (Eqs 1-2) -

def msa_flops(h, w, C):
    """Standard multi-head self-attention complexity (Eq. 1)."""
    return 4 * h * w * C ** 2 + 2 * (h * w) ** 2 * C


def w_msa_flops(h, w, C, M=7):
    """Window-MSA complexity for window size M (Eq. 2)."""
    return 4 * h * w * C ** 2 + 2 * M * M * h * w * C


# ------------------------------------------- synthetic borehole images ----

def synth_fractures(depth=256, azim=128, n_fracs=4,
                    noise=0.28, frac_darkness=0.32, seed=0):
    """Generate a binary fracture mask + greyscale "resistivity" image.

    Fractures appear as sin waves in (depth, azimuth) space because a
    planar fracture cuts a cylindrical borehole in a sinusoid.  We deliberately
    keep the contrast modest and the noise high so that a naive darkness
    threshold misclassifies many background pixels -- the regime where the
    sinusoidal prior of the dual-branch decoder is supposed to help.
    """
    rng = np.random.default_rng(seed)
    image = np.full((depth, azim), 0.55)
    # Add a low-frequency background drift to defeat a global threshold
    image += 0.10 * np.sin(2.0 * np.pi * np.arange(depth)[:, None] / depth)
    mask = np.zeros((depth, azim), dtype=bool)

    a = np.arange(azim)
    for _ in range(n_fracs):
        amp = rng.uniform(8.0, 20.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        centre = rng.uniform(0.15 * depth, 0.85 * depth)
        thickness = rng.uniform(1.5, 3.5)
        z_frac = centre + amp * np.sin(2.0 * np.pi * a / azim + phase)
        for ai, zi in enumerate(z_frac):
            lo = int(np.floor(zi - thickness))
            hi = int(np.ceil(zi + thickness))
            for zz in range(max(0, lo), min(depth, hi)):
                mask[zz, ai] = True
                image[zz, ai] = frac_darkness
    image += noise * rng.standard_normal(image.shape)
    image = np.clip(image, 0.0, 1.0)
    return image, mask


# ------------------------------------------- patch-window encoder ---------

def window_pool(image, M=8):
    """Mean-pool over M x M windows (the patch-partition step).  Returns
    a downsampled feature map of shape (H//M, W//M)."""
    H, W = image.shape
    Hc = (H // M) * M
    Wc = (W // M) * M
    return image[:Hc, :Wc].reshape(Hc // M, M, Wc // M, M).mean(axis=(1, 3))


def window_attention_score(image, M=8):
    """Local-contrast attention proxy: high where window variance is large.

    Returns an upsampled per-pixel attention map (same shape as input).
    """
    H, W = image.shape
    Hc = (H // M) * M
    Wc = (W // M) * M
    blocks = image[:Hc, :Wc].reshape(Hc // M, M, Wc // M, M)
    var = blocks.var(axis=(1, 3))
    a_up = np.repeat(np.repeat(var, M, axis=0), M, axis=1)
    out = np.zeros_like(image)
    out[:Hc, :Wc] = a_up
    return out / (out.max() + 1e-9)


# ------------------------------------------- sinusoidal-prior decoder ----

def sinusoid_radon(image, n_fracs=4, thickness=2, amp_range=(4, 22),
                   n_phases=24, n_amps=10, nms_window=8):
    """Top-K sinusoidal Hough on the image.

    Scores every (centre_depth, amplitude, phase) triple by the mean
    darkness along the corresponding sinusoid, applies depth-axis
    non-maximum-suppression to avoid duplicate detections of the same
    fracture, then stamps the top `n_fracs` matches at the requested
    pixel `thickness`.

    Returns a per-pixel score map in [0, 1].
    """
    H, W = image.shape
    a = np.arange(W)
    amps = np.linspace(*amp_range, n_amps)
    phases = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)

    candidates = []  # (score, centre, amp, phase)
    for amp in amps:
        templ = amp * np.sin(2 * np.pi * a / W + 0.0)
        for phase in phases:
            templ_p = (amp * np.sin(2 * np.pi * a / W + phase)).astype(int)
            for centre in range(int(amp) + thickness,
                                H - int(amp) - thickness):
                idx = centre + templ_p
                m = float((1.0 - image[idx, a]).mean())
                candidates.append((m, centre, amp, phase, templ_p))

    candidates.sort(key=lambda c: c[0], reverse=True)
    chosen, used_centres = [], []
    for c in candidates:
        m, centre, amp, phase, templ_p = c
        if any(abs(centre - uc) < nms_window for uc in used_centres):
            continue
        chosen.append(c)
        used_centres.append(centre)
        if len(chosen) >= n_fracs:
            break

    score = np.zeros_like(image)
    for m, centre, amp, phase, templ_p in chosen:
        for d in range(-thickness, thickness + 1):
            idx = np.clip(centre + templ_p + d, 0, H - 1)
            score[idx, a] = np.maximum(score[idx, a], m)
    return score / (score.max() + 1e-9)


# ------------------------------------------- combined two-branch decoder --

def dual_branch_predict(image, M=8, thresh=0.40, n_fracs=4):
    branch_attn = window_attention_score(image, M=M)
    branch_sin = sinusoid_radon(image, n_fracs=n_fracs)
    combined = 0.7 * branch_sin + 0.3 * branch_attn
    return combined > thresh, combined


def baseline_threshold(image, thresh=0.40):
    """Simple darkness-threshold baseline (Otsu-like fixed cutoff)."""
    return image < thresh


# ------------------------------------------- metrics (Eqs 3-6) -----------

def confusion(pred, truth):
    p = pred.astype(bool)
    t = truth.astype(bool)
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())
    return tp, fp, fn, tn


def metrics(pred, truth):
    tp, fp, fn, tn = confusion(pred, truth)
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return dict(precision=precision, recall=recall, iou=iou, dice=dice)


# ------------------------------------------- tests -----------------------

def test_all():
    print("=" * 60)
    print("Article 3: Swin-Style Borehole-Image Fracture Extraction")
    print("=" * 60)

    # Complexity sanity (Eqs 1-2)
    f_full = msa_flops(64, 64, 96)
    f_win = w_msa_flops(64, 64, 96, M=8)
    print(f"  FLOPs MSA      ({64}x{64}, C=96)        = {f_full:.2e}")
    print(f"  FLOPs W-MSA    (window M=8)             = {f_win:.2e}")
    assert f_win < f_full, "Window MSA must be cheaper than full MSA"

    image, truth = synth_fractures(seed=0)
    pred_base = baseline_threshold(image)
    pred_dual, _ = dual_branch_predict(image)

    m_b = metrics(pred_base, truth)
    m_d = metrics(pred_dual, truth)
    print("  Baseline threshold:")
    for k, v in m_b.items():
        print(f"     {k:10s} = {v:.3f}")
    print("  Dual-branch predictor:")
    for k, v in m_d.items():
        print(f"     {k:10s} = {v:.3f}")

    # The sinusoid prior should beat the baseline on Dice / IoU
    assert m_d["dice"] > m_b["dice"], "Dual-branch must beat baseline on Dice"
    print("  PASS")
    return {"baseline": m_b, "dual": m_d}


if __name__ == "__main__":
    test_all()
