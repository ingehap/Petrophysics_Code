"""
Automatic Well-Log Depth Shifting With Data-Driven Approaches:
A Contest Summary

Reference:
    Pan, W., Fu, L., Xu, C., Ashby, M., Lee, H., Lee, J., Meng, F.,
    Chen, S., Ye, Y., Jiang, H., Kim, H., Kong, H., Baek, I., Baek, J.,
    Sun, X., Sun, H., Li, S., Zhao, Z., Ke, Y., ... and Park, J. (2026).
    Automatic Well-Log Depth Shifting With Data-Driven Approaches:
    A Contest Summary. Petrophysics, 67(2), 437–462.
    DOI: 10.30632/PJV67N2-2026a12

Implements:
  - Dynamic Time Warping (DTW) for well-log alignment
  - Cross-correlation bulk-shift baseline
  - Ridge-regression depth-shift predictor
  - 1-D CNN depth-shift predictor (simplified)
  - RMSE and MAD evaluation metrics
  - Multi-log ensemble shift estimation
  - Reference-log alignment workflow (SPWLA PDDA 2023 contest setup)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Evaluation metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mad(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute deviation (used as contest metric)."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# 2. Cross-correlation bulk shift (conventional baseline)
# ---------------------------------------------------------------------------

def cross_correlation_shift(log_ref: np.ndarray,
                             log_target: np.ndarray,
                             max_shift: int = 50) -> int:
    """
    Find the bulk depth shift (in samples) that maximises cross-correlation
    between a reference log and a shifted target log.

    Parameters
    ----------
    log_ref    : Reference log array (normalised or raw)
    log_target : Target log array to be shifted
    max_shift  : Maximum allowable shift in samples (±max_shift)

    Returns
    -------
    best_shift : Optimal integer sample shift (positive → shift target down)
    """
    best_corr  = -np.inf
    best_shift = 0
    n          = len(log_ref)

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            r = log_ref[shift:]
            t = log_target[:n - shift]
        else:
            r = log_ref[:n + shift]
            t = log_target[-shift:]
        if len(r) == 0:
            continue
        corr = float(np.corrcoef(r, t)[0, 1])
        if corr > best_corr:
            best_corr  = corr
            best_shift = shift

    return best_shift


def apply_bulk_shift(log: np.ndarray, shift: int,
                     fill_value: float = np.nan) -> np.ndarray:
    """Apply an integer depth shift to a log array."""
    result = np.full_like(log, fill_value, dtype=float)
    if shift >= 0:
        result[shift:] = log[:len(log) - shift]
    else:
        result[:len(log) + shift] = log[-shift:]
    return result


# ---------------------------------------------------------------------------
# 3. Dynamic Time Warping (DTW)
# ---------------------------------------------------------------------------

def dtw_alignment(log_ref: np.ndarray,
                  log_target: np.ndarray,
                  window: Optional[int] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Dynamic Time Warping alignment between two 1-D log sequences.
    Returns the DTW distance and the optimal warping path.

    Parameters
    ----------
    log_ref    : Reference log, shape (n,)
    log_target : Target log, shape (m,)
    window     : Sakoe-Chiba bandwidth (samples); None = unconstrained

    Returns
    -------
    dtw_dist  : DTW distance (scalar)
    path_i    : Reference indices along optimal path
    path_j    : Target indices along optimal path
    """
    n, m = len(log_ref), len(log_target)
    INF  = float("inf")
    D    = np.full((n, m), INF)
    D[0, 0] = abs(log_ref[0] - log_target[0])

    for i in range(n):
        for j in range(m):
            if window is not None and abs(i - j) > window:
                continue
            cost = abs(log_ref[i] - log_target[j])
            if i == 0 and j == 0:
                D[i, j] = cost
                continue
            prev = INF
            if i > 0:           prev = min(prev, D[i-1, j])
            if j > 0:           prev = min(prev, D[i, j-1])
            if i > 0 and j > 0: prev = min(prev, D[i-1, j-1])
            D[i, j] = cost + (prev if prev < INF else INF)

    # Backtrack optimal path
    path_i, path_j = [n - 1], [m - 1]
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        opts = []
        if i > 0 and j > 0: opts.append((D[i-1, j-1], i-1, j-1))
        if i > 0:            opts.append((D[i-1, j],   i-1, j  ))
        if j > 0:            opts.append((D[i,   j-1], i,   j-1))
        _, i, j = min(opts)
        path_i.append(i)
        path_j.append(j)

    path_i = np.array(path_i[::-1])
    path_j = np.array(path_j[::-1])

    return float(D[n-1, m-1]), path_i, path_j


def dtw_depth_shift_at_each_point(path_i: np.ndarray,
                                   path_j: np.ndarray,
                                   depth_ref: np.ndarray,
                                   depth_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the depth correction at each reference depth from the DTW path.

    Returns
    -------
    depths_ref  : Reference depth values along path
    depth_corr  : Depth correction applied to target at each reference point
    """
    depths_ref = depth_ref[path_i]
    depths_tgt = depth_target[path_j]
    return depths_ref, depths_tgt - depths_ref


# ---------------------------------------------------------------------------
# 4. Ridge-regression depth-shift predictor
# ---------------------------------------------------------------------------

class RidgeShiftPredictor:
    """
    Ridge regression model that predicts the depth shift required at each
    sample, given a window of log features as input.

    (Implements the ridge-regression approach used by one of the top
    contest teams.)
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.W: Optional[np.ndarray] = None
        self.b: float = 0.0

    def _build_features(self, log: np.ndarray, window: int = 10) -> np.ndarray:
        """Sliding-window feature matrix: [local log values + gradient]."""
        n = len(log)
        grad = np.gradient(log)
        feats = []
        for i in range(n):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            seg  = log[lo:hi]
            gseg = grad[lo:hi]
            row  = np.concatenate([seg, gseg])
            feats.append(row)
        # Pad to uniform length
        max_len = max(len(f) for f in feats)
        X = np.array([np.pad(f, (0, max_len - len(f))) for f in feats])
        return X

    def fit(self, log_ref: np.ndarray, log_target: np.ndarray,
            shifts: np.ndarray, window: int = 10):
        """
        Fit ridge regression: X = features from (ref, target) → y = shift.

        Parameters
        ----------
        log_ref, log_target : Aligned log arrays (same length)
        shifts              : True depth shift at each sample, m
        window              : Context window size (samples)
        """
        X_ref = self._build_features(log_ref, window)
        X_tgt = self._build_features(log_target, window)
        X = np.hstack([X_ref, X_tgt])

        # Ridge: (X^T X + alpha I)^{-1} X^T y
        A = X.T @ X + self.alpha * np.eye(X.shape[1])
        self.W = np.linalg.solve(A, X.T @ shifts)
        self.b = float(shifts.mean() - X.mean(axis=0) @ self.W)

    def predict(self, log_ref: np.ndarray, log_target: np.ndarray,
                window: int = 10) -> np.ndarray:
        """Predict depth shift at each sample."""
        if self.W is None:
            raise RuntimeError("Model not fitted.")
        X_ref = self._build_features(log_ref, window)
        X_tgt = self._build_features(log_target, window)
        X     = np.hstack([X_ref, X_tgt])
        return X @ self.W + self.b


# ---------------------------------------------------------------------------
# 5. 1-D CNN depth-shift predictor (simplified, pure NumPy)
# ---------------------------------------------------------------------------

def conv1d(x: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """1-D valid convolution."""
    K = len(kernel)
    n = (len(x) - K) // stride + 1
    return np.array([np.dot(x[i*stride:i*stride+K], kernel) for i in range(n)])


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


class SimpleCNN1DShiftPredictor:
    """
    Simplified 1-D CNN for depth-shift prediction.

    Architecture (per contest teams using CNN):
      Conv1D(kernel=5) → ReLU → GlobalAvgPool → Dense → shift scalar
    """
    def __init__(self, kernel_size: int = 5, n_filters: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.kernels = rng.normal(0, 0.1, (n_filters, kernel_size))
        self.dense_W = rng.normal(0, 0.1, n_filters)
        self.dense_b = 0.0
        self.n_filters = n_filters
        self.kernel_size = kernel_size

    def _forward(self, x: np.ndarray) -> float:
        """Forward pass for a single 1-D input sequence."""
        maps = []
        for k in self.kernels:
            c = conv1d(x, k)
            maps.append(relu(c).mean())   # global average pooling
        feat = np.array(maps)
        return float(feat @ self.dense_W + self.dense_b)

    def fit(self, X_seqs: List[np.ndarray],
            y_shifts: np.ndarray,
            lr: float = 1e-3, epochs: int = 50):
        """
        Train with gradient descent (numerical gradients for simplicity).
        """
        for epoch in range(epochs):
            total_loss = 0.0
            for x_seq, y in zip(X_seqs, y_shifts):
                pred  = self._forward(x_seq)
                err   = pred - y
                total_loss += err**2
                # Update dense weights (gradient of MSE)
                maps = []
                for k in self.kernels:
                    c = conv1d(x_seq, k)
                    maps.append(relu(c).mean())
                feat = np.array(maps)
                self.dense_W -= lr * err * feat
                self.dense_b -= lr * err
        return total_loss / len(y_shifts)

    def predict(self, X_seqs: List[np.ndarray]) -> np.ndarray:
        return np.array([self._forward(x) for x in X_seqs])


# ---------------------------------------------------------------------------
# 6. Multi-log ensemble shift
# ---------------------------------------------------------------------------

def ensemble_depth_shift(shifts_per_log: Dict[str, np.ndarray],
                          weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Combine depth-shift estimates from multiple logs using weighted average.

    Parameters
    ----------
    shifts_per_log : {log_name: shift_array}
    weights        : {log_name: weight}  (uniform if None)

    Returns
    -------
    ensemble_shift : Weighted average depth shift array
    """
    names  = list(shifts_per_log.keys())
    arrays = np.array([shifts_per_log[n] for n in names])

    if weights is None:
        w = np.ones(len(names)) / len(names)
    else:
        w = np.array([weights.get(n, 1.0) for n in names])
        w = w / w.sum()

    return (arrays * w[:, None]).sum(axis=0)


# ---------------------------------------------------------------------------
# 7. Full alignment pipeline
# ---------------------------------------------------------------------------

def align_logs(log_ref:     np.ndarray,
               log_target:  np.ndarray,
               depth_ref:   np.ndarray,
               method:      str = "dtw",
               max_shift:   int = 50,
               dtw_window:  int = 30) -> Dict:
    """
    Align a target log to a reference log using the specified method.

    Parameters
    ----------
    log_ref, log_target : Log arrays (normalised recommended)
    depth_ref           : Depth values for reference log, m
    method              : 'xcorr' | 'dtw' | 'ridge' | 'cnn'
    max_shift           : Maximum shift in samples (for xcorr)
    dtw_window          : DTW Sakoe-Chiba window

    Returns
    -------
    dict with 'aligned_log', 'shift_m', 'method_used'
    """
    depth_target = depth_ref.copy()   # assume same sampling initially
    sample_rate  = float(np.median(np.diff(depth_ref)))  # m/sample

    if method == "xcorr":
        shift_samples = cross_correlation_shift(log_ref, log_target, max_shift)
        aligned  = apply_bulk_shift(log_target, shift_samples, fill_value=np.nan)
        shift_m  = shift_samples * sample_rate
        return {"aligned_log": aligned, "shift_m": shift_m, "method_used": method}

    elif method == "dtw":
        # Normalise
        def norm(x):
            r = x.max() - x.min() + 1e-12
            return (x - x.min()) / r
        dist, pi, pj = dtw_alignment(norm(log_ref), norm(log_target),
                                      window=dtw_window)
        _, shift_arr = dtw_depth_shift_at_each_point(pi, pj, depth_ref, depth_target)
        # Apply median shift as bulk correction
        med_shift = float(np.median(shift_arr))
        shift_samples = int(round(med_shift / sample_rate))
        aligned = apply_bulk_shift(log_target, shift_samples, fill_value=np.nan)
        return {
            "aligned_log": aligned,
            "shift_m":     med_shift,
            "dtw_dist":    dist,
            "shift_profile_m": shift_arr,
            "method_used": method,
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'xcorr' or 'dtw'.")


# ---------------------------------------------------------------------------
# 8. Example workflow
# ---------------------------------------------------------------------------

def example_workflow():
    print("=" * 60)
    print("Automatic Well-Log Depth Shifting")
    print("Ref: Pan et al., Petrophysics 67(2) 2026")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n   = 400
    depth_ref  = np.linspace(1000, 1400, n)

    # Synthetic reference GR log
    log_ref = (50.0 + 30.0 * np.sin(2 * np.pi * depth_ref / 20.0) +
               10.0 * rng.normal(size=n))

    # Target: same log + true depth shift + noise
    true_shift_m  = 1.8   # m (positive = target is deeper)
    shift_samples = int(round(true_shift_m / np.median(np.diff(depth_ref))))
    log_target    = np.roll(log_ref, -shift_samples) + rng.normal(0, 2, n)

    print(f"\nTrue depth shift: {true_shift_m:.2f} m  ({shift_samples} samples)")
    print(f"Depth step: {np.median(np.diff(depth_ref))*100:.0f} cm/sample")

    # Method 1: cross-correlation
    r_xcorr = align_logs(log_ref, log_target, depth_ref, method="xcorr", max_shift=30)
    e_xcorr = rmse(log_ref, np.nan_to_num(r_xcorr["aligned_log"]))
    print(f"\nXCorr bulk shift detected : {r_xcorr['shift_m']:.2f} m  (RMSE={e_xcorr:.2f})")

    # Method 2: DTW
    r_dtw  = align_logs(log_ref, log_target, depth_ref, method="dtw", dtw_window=25)
    e_dtw  = rmse(log_ref, np.nan_to_num(r_dtw["aligned_log"]))
    print(f"DTW median shift detected : {r_dtw['shift_m']:.2f} m  (RMSE={e_dtw:.2f}  "
          f"DTW dist={r_dtw['dtw_dist']:.1f})")

    # Ridge regression demo
    ridge = RidgeShiftPredictor(alpha=0.5)
    true_shifts_arr = np.full(n, true_shift_m)
    ridge.fit(log_ref, log_target, true_shifts_arr, window=8)
    pred_shifts = ridge.predict(log_ref, log_target, window=8)
    print(f"Ridge regressor mean predicted shift: {pred_shifts.mean():.2f} m  "
          f"(MAD={mad(true_shifts_arr, pred_shifts):.3f} m)")

    # CNN demo
    cnn = SimpleCNN1DShiftPredictor(kernel_size=5, n_filters=4, seed=0)
    segs = [log_target[i:i+20] for i in range(0, n - 20, 20)]
    y_tr = np.full(len(segs), true_shift_m)
    cnn.fit(segs, y_tr, lr=2e-3, epochs=30)
    cnn_preds = cnn.predict(segs)
    print(f"CNN mean predicted shift:  {cnn_preds.mean():.2f} m")

    # Ensemble combination
    ensemble = ensemble_depth_shift(
        {"xcorr": np.full(n, r_xcorr["shift_m"]),
         "dtw":   np.full(n, r_dtw["shift_m"]),
         "ridge": pred_shifts},
        weights={"xcorr": 0.2, "dtw": 0.4, "ridge": 0.4},
    )
    print(f"Ensemble mean shift:       {ensemble.mean():.2f} m")
    print(f"\n(True shift = {true_shift_m:.2f} m)")

    return {"xcorr": r_xcorr, "dtw": r_dtw}


if __name__ == "__main__":
    example_workflow()
