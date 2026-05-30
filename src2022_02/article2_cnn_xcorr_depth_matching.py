"""
Article 2: Automated Well-Log Depth Matching - 1D Convolutional Neural
           Networks vs. Classic Cross Correlation
Torres Caceres, Duffaut, Yazidi, Westad, Johansen (2022)
DOI: 10.30632/PJV63N1-2022a2

Compares classic cross-correlation depth matching with a supervised 1D
convolutional neural network (CNN) that regresses the depth shift (lag)
needed to align a reference EWL log window onto a test LWD log window.

Implements:

  - Normalized cross correlation, lag = argmax                 (Eq. 1)
  - A compact 1D CNN regressor                                 (Eqs. 2-5)
        conv(2ch) -> ReLU -> average-pool bins -> flatten -> linear,
        trained with MSE on synthetic shifted windows
  - Pearson correlation coefficient                            (Eq. 6)
  - Euclidean distance                                         (Eq. 7)
  - Ind1% / Ind4% improvement indicators                       (Eqs. 8-9)

Constants from the paper: 256-sample windows, +/-20-sample shift label
range, 0.5-ft sampling.  The CNN here is a faithful but deliberately small
numpy implementation (no TensorFlow dependency) so the module stays
runnable with numpy alone.
"""

import numpy as np

WINDOW = 256                 # samples per window (~39 m at 0.5 ft)
MAX_SHIFT = 20               # +/- shift label range (samples)


# ---------------------------------------------- helpers ----------------

def standardize(x):
    """Zero-mean, unit-variance (per-window normalization used in the paper)."""
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


# ---------------------------------------------- Eq. 1: cross correlation -

def cross_correlation_shift(reference, test, max_shift=MAX_SHIFT):
    """Normalized cross-correlation alignment lag (Eq. 1).

    Returns the integer lag L (samples) maximizing the correlation of
    roll(test, L) with reference, i.e. the shift to apply to `test` so it
    aligns onto `reference`.
    """
    r = standardize(reference)
    t = standardize(test)
    lags = np.arange(-max_shift, max_shift + 1)
    best_lag, best_c = 0, -np.inf
    for L in lags:
        c = float(np.dot(r, np.roll(t, L)))
        if c > best_c:
            best_c, best_lag = c, L
    return best_lag


# ---------------------------------------------- Eqs. 6-7: metrics -------

def pearson(x, y):
    """Pearson correlation coefficient r (Eq. 6), range [-1, 1]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    return float(np.sum(xc * yc) / denom) if denom > 1e-12 else 0.0


def euclidean(x, y):
    """Euclidean distance d (Eq. 7)."""
    return float(np.sqrt(np.sum((np.asarray(x, float) - np.asarray(y, float)) ** 2)))


# ---------------------------------------------- Eqs. 8-9: indicators ----

def ind_percent(count, total):
    """Generic indicator  count / total * 100  (Eqs. 8-9)."""
    return 100.0 * count / total if total else 0.0


# ---------------------------------------------- Eqs. 2-5: 1D CNN --------

def _relu(x):
    """ReLU activation h(x) = max(0, x)  (Eq. 5)."""
    return np.maximum(0.0, x)


class CNNShiftRegressor:
    """Compact 1D CNN regressing the alignment lag from a 2-channel window.

    Architecture (Eqs. 2-4): a 1D convolution over the stacked
    [reference, test] channels (Eq. 3), ReLU (Eq. 5), average pooling into
    `n_bins` positional bins, flatten, and a single linear output (Eq. 4).
    Average pooling into positional bins (rather than a single global value)
    preserves enough spatial information to recover the *sign* of the shift.
    Trained with MSE by per-sample gradient descent on labels scaled to
    [-1, 1]; small by design so test_all() runs in well under a second.
    """

    def __init__(self, n_filters=6, kernel=9, n_bins=8, seed=0):
        rng = np.random.default_rng(seed)
        self.k = kernel
        self.F = n_filters
        self.P = n_bins
        self.W = rng.normal(0, 0.2, size=(n_filters, 2, kernel))  # conv kernels
        self.b = np.zeros(n_filters)                              # conv bias
        self.V = rng.normal(0, 0.2, size=n_filters * n_bins)      # dense weights
        self.c = 0.0                                              # dense bias
        self.scale = float(MAX_SHIFT)                            # label scaling

    def _windows(self, ref, test):
        """Sliding windows of the 2-channel standardized input -> (2, n, k)."""
        x = np.stack([standardize(ref), standardize(test)])      # (2, L)
        n = x.shape[1] - self.k + 1
        idx = np.arange(self.k)[None, :] + np.arange(n)[:, None]  # (n, k)
        return x[:, idx]                                          # (2, n, k)

    def _forward(self, ref, test):
        win = self._windows(ref, test)                           # (2, n, k)
        conv = np.einsum("cnk,fck->fn", win, self.W) + self.b[:, None]
        act = _relu(conv)                                        # (F, n)
        n = act.shape[1]
        binsize = n // self.P
        used = binsize * self.P
        pooled = act[:, :used].reshape(self.F, self.P, binsize).mean(axis=2)
        flat = pooled.reshape(-1)                                # (F*P,)
        pred = float(flat @ self.V + self.c)
        cache = (win, conv, act, binsize, used)
        return pred, flat, cache

    def predict_one(self, reference, test):
        pred, _, _ = self._forward(reference, test)
        return pred * self.scale

    def fit(self, refs, tests, labels, epochs=40, lr=0.02, clip=5.0):
        y = np.asarray(labels, float) / self.scale              # scale to ~[-1,1]
        n = len(y)
        for _ in range(epochs):
            for i in range(n):
                pred, flat, (win, conv, act, binsize, used) = \
                    self._forward(refs[i], tests[i])
                err = pred - y[i]                                # dL/dpred (0.5 MSE)
                err = float(np.clip(err, -clip, clip))
                # dense layer
                dV = err * flat
                dflat = err * self.V
                # unpool: spread each bin's gradient over its members
                dpooled = dflat.reshape(self.F, self.P) / binsize
                dact = np.repeat(dpooled, binsize, axis=1)       # (F, used)
                drelu = dact * (conv[:, :used] > 0)
                # conv params
                dW = np.einsum("fn,cnk->fck", drelu, win[:, :used, :])
                db = drelu.sum(axis=1)
                # update
                self.V -= lr * dV
                self.c -= lr * err
                self.W -= lr * dW
                self.b -= lr * db
        return self


def _make_log(n, seed):
    """Synthetic blocky GR-like reference log (random walk + bedding)."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 1, n))
    beds = 30 * np.sin(np.linspace(0, 12 * np.pi, n))
    return walk + beds + rng.normal(0, 2, n)


def _sample(base, rng):
    """Draw a (reference, test, lag) triple from the base log.

    test is base shifted so that roll(test, lag) aligns onto reference.
    """
    start = MAX_SHIFT + int(rng.integers(0, 32))
    ref = base[start:start + WINDOW]
    lag = int(rng.integers(-MAX_SHIFT, MAX_SHIFT + 1))
    test = base[start + lag:start + lag + WINDOW] + rng.normal(0, 1.5, WINDOW)
    return ref, test, lag


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: CNN vs Cross-Correlation Depth Matching")
    print("=" * 60)

    base = _make_log(WINDOW + 4 * MAX_SHIFT + 64, seed=7)

    # ---- build training set: shifted copies with known alignment lags -----
    rng = np.random.default_rng(1)
    refs, tests, labels = [], [], []
    for _ in range(300):
        r, t, lag = _sample(base, rng)
        refs.append(r); tests.append(t); labels.append(lag)

    net = CNNShiftRegressor().fit(refs, tests, labels, epochs=40, lr=0.02)

    # ---- evaluate on held-out shifts -------------------------------------
    c1 = c4 = 0
    cnn_err = xc_err = 0.0
    n_test = 40
    r_before_sum = r_after_sum = 0.0
    for _ in range(n_test):
        ref, test, lag = _sample(base, rng)

        s_cnn = int(round(net.predict_one(ref, test)))
        s_xc = cross_correlation_shift(ref, test)
        cnn_err += abs(s_cnn - lag)
        xc_err += abs(s_xc - lag)

        r_orig = pearson(ref, test)
        r_cnn = pearson(ref, np.roll(standardize(test), s_cnn))
        r_xc = pearson(ref, np.roll(standardize(test), s_xc))
        r_before_sum += r_orig
        r_after_sum += r_cnn
        if r_cnn > r_orig:
            c1 += 1
            if r_cnn > r_xc:
                c4 += 1

    cnn_mae = cnn_err / n_test
    xc_mae = xc_err / n_test
    print(f"  CNN  shift MAE        = {cnn_mae:.2f} samples")
    print(f"  Xcorr shift MAE       = {xc_mae:.2f} samples")
    print(f"  mean Pearson before   = {r_before_sum / n_test:.3f}")
    print(f"  mean Pearson after CNN= {r_after_sum / n_test:.3f}")
    print(f"  Ind1% (CNN improves)  = {ind_percent(c1, n_test):.0f}%")
    print(f"  Ind4% (CNN > Xcorr)   = {ind_percent(c4, n_test):.0f}%")

    assert cnn_mae < 4.0, "CNN should recover shift to within a few samples"
    assert xc_mae < 1.5, "cross correlation should be near-exact on bulk shift"
    assert r_after_sum > r_before_sum, "alignment must raise mean correlation"
    print("  PASS")
    return {"cnn_mae": cnn_mae, "xc_mae": xc_mae,
            "Ind1": ind_percent(c1, n_test), "Ind4": ind_percent(c4, n_test)}


if __name__ == "__main__":
    test_all()
