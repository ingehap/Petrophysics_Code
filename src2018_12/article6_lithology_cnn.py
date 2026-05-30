"""
Article 6: Intelligent Logging Lithological Interpretation With Convolution
           Neural Networks
Zhu, Li, Yang, Li, Ao (2018)
DOI: 10.30632/PJV59N6Y2018a5

A convolutional neural network classifies lithology from a window of well-log
values: 1D convolutional filters extract local shape features (trends, spikes,
contacts) that a classifier head maps to a lithology label.  This module
implements the CNN feature-extraction idea with 1D convolutional features and a
softmax classifier head.

Implements:

  - 1D convolution feature extraction with global average pooling
  - Softmax classifier head trained by gradient descent
  - Multiclass lithology classification with accuracy and a confusion matrix

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction; this is a faithful standard-form reconstruction of the
CNN-classification workflow (a numpy convolutional-feature classifier stands in
for the published deep network).
"""

import numpy as np


# ---------------------------------------------- conv features -----------

def conv_features(windows, filters):
    """1D-convolve each window with each filter, tanh, then global-average-pool.

    windows: (n, L); filters: (K, w).  Returns (n, K) pooled feature matrix.
    """
    windows = np.asarray(windows, float)
    n, L = windows.shape
    feats = []
    for f in filters:
        w = len(f)
        conv = np.array([[np.dot(windows[i, t:t + w], f) for t in range(L - w + 1)]
                         for i in range(n)])
        feats.append(np.tanh(conv).mean(axis=1))   # global average pool
    return np.column_stack(feats)


def random_filters(K, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0, (K, w))


# ---------------------------------------------- softmax head ------------

def softmax_fit(X, y, n_classes, iters=3000, lr=0.2, seed=0):
    rng = np.random.default_rng(seed)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    W = np.zeros((Xb.shape[1], n_classes))
    Y = np.eye(n_classes)[np.asarray(y, int)]
    for _ in range(iters):
        Z = Xb @ W; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= lr * Xb.T @ (P - Y) / len(y)
    return dict(W=W, mean=X.mean(0), std=X.std(0) + 1e-12)


def softmax_predict(model, X):
    Xs = (np.asarray(X, float) - model["mean"]) / model["std"]
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    return (Xb @ model["W"]).argmax(1)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Lithology Interpretation With CNN")
    print("=" * 60)

    rng = np.random.default_rng(5)
    L = 24

    def make_window(litho):
        # 0 = sand (low, flat GR), 1 = shale (high, flat), 2 = interbedded (oscillating)
        if litho == 0:
            base = 30 + 3 * rng.standard_normal(L)
        elif litho == 1:
            base = 110 + 3 * rng.standard_normal(L)
        else:
            base = 70 + 30 * np.sin(np.linspace(0, 6, L)) + 3 * rng.standard_normal(L)
        return base

    X_raw, y = [], []
    for _ in range(120):
        for c in (0, 1, 2):
            X_raw.append(make_window(c)); y.append(c)
    X_raw = np.array(X_raw); y = np.array(y)
    idx = rng.permutation(len(y)); X_raw, y = X_raw[idx], y[idx]

    # combine convolutional (shape) features with window statistics: the
    # zero-mean filters capture variation/oscillation, the mean/std capture the
    # DC level that separates flat low-GR sand from flat high-GR shale
    filt = random_filters(6, 5, seed=1)
    X = np.column_stack([conv_features(X_raw, filt),
                         X_raw.mean(1), X_raw.std(1)])

    ntr = int(0.7 * len(y))
    model = softmax_fit(X[:ntr], y[:ntr], n_classes=3)
    pred = softmax_predict(model, X[ntr:])
    acc = float(np.mean(pred == y[ntr:]))
    print(f"  3-class lithology accuracy = {acc:.2f}")
    assert acc > 0.85

    # Confusion matrix has dominant diagonal
    cm = np.zeros((3, 3), int)
    for a, b in zip(y[ntr:], pred):
        cm[a, b] += 1
    assert all(cm[i, i] == cm[i].max() for i in range(3))
    print("  PASS")
    return {"accuracy": acc}


if __name__ == "__main__":
    test_all()
