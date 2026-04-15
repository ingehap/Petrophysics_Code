"""
Article 8: Exemplar-Guided Sedimentary Facies Modeling for Bridging Pattern
Controllability Gap
Wu, Hu, Sun, Zhang, Wang, and Zhang (2023)
DOI: 10.30632/PJV64N2-2023a8

Implements a simplified, non-GAN proof-of-concept of the EGFM idea:
content/pattern decoupling.  The full paper trains a GAN end-to-end with
two encoders, a decoder, an Adaptive Feature Fusion Block (AFB) and three
losses.  Here we substitute:
  - Content encoder    --> distance transform of the well-point map
  - Pattern encoder    --> Gabor / orientation features of the exemplar
  - AFB                --> attention-weighted fusion of content & pattern
  - Decoder            --> threshold + smoothing + condition-honouring
The result still demonstrates how a synthesised facies map can be steered
by both well constraints (content) and a chosen pattern exemplar.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor


# -------------------------------------------- content (well-point) encoder --

def content_features(well_map, sigma=3.0):
    """
    well_map: 2D array with 1 = channel-well, -1 = background-well, 0 = unknown.
    Returns a continuous "content field" that interpolates well evidence.
    """
    pos_mask = well_map > 0
    neg_mask = well_map < 0
    pos_dist = ndi.distance_transform_edt(~pos_mask) if pos_mask.any() else np.full_like(well_map, np.inf, dtype=float)
    neg_dist = ndi.distance_transform_edt(~neg_mask) if neg_mask.any() else np.full_like(well_map, np.inf, dtype=float)
    field = np.exp(-pos_dist ** 2 / (2 * sigma ** 2)) - np.exp(-neg_dist ** 2 / (2 * sigma ** 2))
    return field


# ------------------------------------------------- pattern encoder ---

def pattern_features(exemplar, frequencies=(0.1, 0.2)):
    """Bank of Gabor filters at multiple orientations capturing the pattern."""
    feats = []
    for freq in frequencies:
        for theta in np.linspace(0, np.pi, 4, endpoint=False):
            real, _ = gabor(exemplar.astype(float), frequency=freq, theta=theta)
            feats.append(real)
    return np.stack(feats, axis=-1)   # (H, W, n_filters)


def pattern_field(exemplar, scale=1.0):
    """A single scalar 'pattern field' derived as the energy of Gabor features
    multiplied by the local exemplar value (preserves orientation cues)."""
    feats = pattern_features(exemplar)
    energy = np.sqrt((feats ** 2).sum(axis=-1))
    energy = (energy - energy.mean()) / (energy.std() + 1e-9)
    return scale * (energy * (2 * exemplar - 1))


# ------------------------------------------- adaptive feature fusion (AFB) ---

def attention_weight(feature, k=2.0):
    """Sigmoid-like attention emphasizing high-magnitude features."""
    return 1.0 / (1.0 + np.exp(-k * (np.abs(feature) - np.abs(feature).mean())))


def afb_fuse(content, pattern, gamma_c=1.0, gamma_p=1.0):
    """Adaptive Feature Fusion Block (Eq. 17 of the paper, simplified)."""
    ac = attention_weight(content)
    ap = attention_weight(pattern)
    fused = gamma_c * (ac * content) + gamma_p * (ap * pattern)
    return fused


# --------------------------------------------------- decoder ---

def decode(fused, well_map, threshold=0.0):
    """Threshold + honour wells (hard constraint)."""
    img = (fused > threshold).astype(np.uint8)
    img[well_map > 0] = 1
    img[well_map < 0] = 0
    return img


# ---------------------------------------------- end-to-end EGFM model ---

def egfm_generate(well_map, exemplar, sigma=4.0, gamma_c=1.0, gamma_p=0.6):
    content = content_features(well_map, sigma=sigma)
    pattern = pattern_field(exemplar)
    fused = afb_fuse(content, pattern, gamma_c, gamma_p)
    return decode(fused, well_map), content, pattern, fused


# -------------------------------------------------- testing ---

def synthetic_data(seed=0, H=64, W=64, orientation="horizontal"):
    """Synthetic 'fluvial' exemplar + sparse well map.
    orientation: 'horizontal' or 'vertical' produces visibly different patterns."""
    rng = np.random.default_rng(seed)
    y_axis = np.arange(H)
    x_axis = np.arange(W)
    XX, YY = np.meshgrid(x_axis, y_axis)
    if orientation == "horizontal":
        exemplar = ((np.abs(YY - (H / 2 + 8 * np.sin(XX / 6))) < 4) |
                    (np.abs(YY - (H / 2 + 14 * np.sin(XX / 9 + 1))) < 3)).astype(int)
    else:  # vertical channels
        exemplar = ((np.abs(XX - (W / 2 + 8 * np.sin(YY / 6))) < 4) |
                    (np.abs(XX - (W / 2 + 14 * np.sin(YY / 9 + 1))) < 3)).astype(int)
    # well map: scatter wells with known facies
    well_map = np.zeros_like(exemplar)
    for _ in range(12):
        i = rng.integers(0, H)
        j = rng.integers(0, W)
        well_map[i, j] = 1 if exemplar[i, j] == 1 else -1
    return exemplar, well_map


def test_all():
    print("=" * 60)
    print("Article 8: Exemplar-Guided Facies Modeling (simplified)")
    print("=" * 60)
    exemplar, wellmap = synthetic_data()
    gen, content, pattern, fused = egfm_generate(wellmap, exemplar)

    # honour-the-wells check
    pos_idx = wellmap > 0
    neg_idx = wellmap < 0
    pos_honoured = (gen[pos_idx] == 1).mean() if pos_idx.any() else 1.0
    neg_honoured = (gen[neg_idx] == 0).mean() if neg_idx.any() else 1.0
    iou = (np.logical_and(gen, exemplar).sum() /
           max(np.logical_or(gen, exemplar).sum(), 1))
    print(f"  Generated facies map: {gen.shape}, "
          f"channel pixels = {gen.sum()}/{gen.size}")
    print(f"  Wells honoured: pos={pos_honoured:.2f}  neg={neg_honoured:.2f}")
    print(f"  IoU vs exemplar: {iou:.3f}")

    # also test that swapping exemplars changes the result
    alt_exemplar, _ = synthetic_data(seed=2, orientation="vertical")
    gen_alt, *_ = egfm_generate(wellmap, alt_exemplar)
    diff = np.mean(gen_alt != gen)
    print(f"  Pattern controllability (diff w/ vertical exemplar) = {diff:.3f}")

    assert pos_honoured > 0.8 and neg_honoured > 0.8
    assert diff > 0.01
    print("  PASS")
    return {"iou": iou, "pos_honoured": pos_honoured, "neg_honoured": neg_honoured,
            "controllability": diff}


if __name__ == "__main__":
    test_all()
