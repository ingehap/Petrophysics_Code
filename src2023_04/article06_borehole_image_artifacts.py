"""
Article 6: Removal of Artifacts in Borehole Images Using Machine Learning
Guner, Fouda, and Barrett (2023)
DOI: 10.30632/PJV64N2-2023a6

Implements a supervised gradient-boosting (XGBoost) model that maps raw
borehole image patches to traditional-processed (clean) patches.  The model
operates one pad at a time, using a sliding window of pixels around the
target location as features.
"""

import numpy as np
import xgboost as xgb


# ----------------------------------------------- traditional processing ---

def traditional_baseline(image, window=51, axis=0):
    """
    Approximation of the paper's "traditional" method:
    Subtract a moving-window column-wise mean (the per-button baseline
    that captures the depth-invariant geometric factor effect), then
    add back the global mean to keep amplitude.
    """
    img_f = image.astype(float)
    H, W = img_f.shape
    pad = window // 2
    out = np.empty_like(img_f)
    img_pad = np.pad(img_f, ((pad, pad), (0, 0)), mode="edge")
    # rolling per-column mean inside a vertical window (depth-direction)
    cumsum = np.cumsum(img_pad, axis=0)
    win_sum = cumsum[window:] - cumsum[:-window]
    rolling_mean = win_sum / window  # shape (H, W)
    # the per-button (column) low-frequency baseline
    column_baseline = rolling_mean.mean(axis=0)  # depth-invariant per column
    out = img_f - column_baseline + img_f.mean()
    return out


# ----------------------------------------------------- patch extraction ---

def extract_patches(image, depth_window=11, az_window=5):
    """For every pixel, gather a (depth_window x az_window) neighbourhood."""
    H, W = image.shape
    dpad, apad = depth_window // 2, az_window // 2
    img = np.pad(image, ((dpad, dpad), (apad, apad)), mode="edge")
    out = np.zeros((H * W, depth_window * az_window))
    for i in range(H):
        for j in range(W):
            patch = img[i:i + depth_window, j:j + az_window]
            out[i * W + j] = patch.ravel()
    return out


# ---------------------------------------------------- model ---

class ImageArtifactCleaner:
    def __init__(self, depth_window=11, az_window=5,
                 n_estimators=200, max_depth=5, seed=0):
        self.dw = depth_window
        self.aw = az_window
        self.model = xgb.XGBRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      learning_rate=0.1,
                                      random_state=seed,
                                      verbosity=0)

    def fit(self, raw_image, processed_image):
        X = extract_patches(raw_image, self.dw, self.aw)
        y = processed_image.astype(float).ravel()
        self.model.fit(X, y)
        return self

    def predict(self, raw_image):
        X = extract_patches(raw_image, self.dw, self.aw)
        y = self.model.predict(X)
        return y.reshape(raw_image.shape)


# --------------------------------------------------- testing ---

def synthetic_image(seed=0, H=80, W=24, n_features=6):
    """
    Synthetic borehole image: depth-varying formation features + a per-button
    geometric-factor offset that mimics the artefact in Fig. 4 of the paper.
    The geometric factor here is large enough that the raw image is worse
    than a column-baseline-subtracted (traditional) image.
    """
    rng = np.random.default_rng(seed)
    layers = np.zeros((H, W))
    depths = rng.choice(H, size=n_features, replace=False)
    for d in depths:
        thick = rng.integers(2, 5)
        amp = rng.uniform(20, 60)
        layers[d:d + thick, :] = amp
    base = 100 + 30 * np.sin(np.linspace(0, 4 * np.pi, H))[:, None]
    clean = base + layers + rng.normal(0, 2, (H, W))
    # Strong geometric-factor effect: large U-shaped offset across the pad
    cols = np.arange(W)
    geom = 80 * (np.abs(cols - W / 2) / (W / 2)) ** 1.5
    raw = clean + geom[None, :] + rng.normal(0, 3, (H, W))
    return raw, clean


def test_all():
    print("=" * 60)
    print("Article 6: Borehole Image Artifact Removal")
    print("=" * 60)
    # training pair
    raw_tr, clean_tr = synthetic_image(seed=0)
    proc_tr = traditional_baseline(raw_tr)   # surrogate "traditional" output

    cleaner = ImageArtifactCleaner(depth_window=7, az_window=5,
                                   n_estimators=150, max_depth=4).fit(raw_tr, proc_tr)

    # held-out test image
    raw_te, clean_te = synthetic_image(seed=1)
    pred = cleaner.predict(raw_te)
    proc_te = traditional_baseline(raw_te)

    rmse_raw = float(np.sqrt(np.mean((raw_te - clean_te) ** 2)))
    rmse_proc = float(np.sqrt(np.mean((proc_te - clean_te) ** 2)))
    rmse_ml = float(np.sqrt(np.mean((pred - clean_te) ** 2)))

    print(f"  RMSE raw    vs clean = {rmse_raw:.2f}")
    print(f"  RMSE traditional      = {rmse_proc:.2f}")
    print(f"  RMSE ML cleaner       = {rmse_ml:.2f}")
    assert rmse_ml < rmse_raw
    print("  PASS")
    return {"raw": rmse_raw, "trad": rmse_proc, "ml": rmse_ml}


if __name__ == "__main__":
    test_all()
