"""
Article 8: A Comparative Study of Three Supervised Machine-Learning Algorithms
           for Classifying Carbonate Vuggy Facies in the Kansas Arbuckle
           Formation
Deng, Xu, Jobe, Xu (2019)
DOI: 10.30632/PJV60N6-2019a8

Three supervised classifiers - logistic regression, a k-nearest-neighbor
classifier (an SVM-style margin classifier analogue), and a bagged decision-stump
ensemble (a random-forest analogue) - are compared for separating vuggy from
non-vuggy carbonate facies from well logs, scored with a confusion matrix and
accuracy / precision / recall / F1.

Implements:

  - Logistic-regression classifier (gradient descent)
  - k-nearest-neighbor classifier
  - Bagged decision-stump ensemble (random-forest analogue)
  - Confusion matrix and accuracy / precision / recall / F1

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard supervised-classification comparison the paper's
title describes (numpy classifiers stand in for the published implementations).
"""

import numpy as np


# ---------------------------------------------- classifiers -------------

def _standardize(X, mean=None, std=None):
    if mean is None:
        mean, std = X.mean(0), X.std(0) + 1e-12
    return (X - mean) / std, mean, std


def logistic_fit(X, y, iters=3000, lr=0.1, seed=0):
    """Logistic regression by gradient descent; returns parameters."""
    Xs, mean, std = _standardize(np.asarray(X, float))
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.zeros(Xb.shape[1])
    y = np.asarray(y, float)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        w -= lr * Xb.T @ (p - y) / len(y)
    return dict(w=w, mean=mean, std=std)


def logistic_predict(model, X):
    Xs = (np.asarray(X, float) - model["mean"]) / model["std"]
    Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
    return (1.0 / (1.0 + np.exp(-Xb @ model["w"])) >= 0.5).astype(int)


def knn_predict(X_train, y_train, X_test, k=5):
    """k-nearest-neighbor majority-vote classifier."""
    Xtr, mean, std = _standardize(np.asarray(X_train, float))
    Xte = (np.asarray(X_test, float) - mean) / std
    y = np.asarray(y_train, int)
    out = np.zeros(len(Xte), int)
    for i, x in enumerate(Xte):
        d = np.linalg.norm(Xtr - x, axis=1)
        nn = y[np.argsort(d)[:k]]
        out[i] = np.bincount(nn).argmax()
    return out


def stump_ensemble_predict(X_train, y_train, X_test, n_stumps=21, seed=0):
    """Bagged decision-stump ensemble (random-forest analogue)."""
    rng = np.random.default_rng(seed)
    Xtr = np.asarray(X_train, float); ytr = np.asarray(y_train, int)
    Xte = np.asarray(X_test, float)
    votes = np.zeros((len(Xte), 2))
    n = len(Xtr)
    for _ in range(n_stumps):
        idx = rng.integers(0, n, n)             # bootstrap sample
        feat = rng.integers(0, Xtr.shape[1])    # random feature
        xb, yb = Xtr[idx, feat], ytr[idx]
        # best threshold by class-mean midpoint
        thr = 0.5 * (xb[yb == 0].mean() + xb[yb == 1].mean()) if (yb == 0).any() and (yb == 1).any() else np.median(xb)
        hi_class = 1 if xb[yb == 1].mean() >= thr else 0
        pred = np.where(Xte[:, feat] >= thr, hi_class, 1 - hi_class)
        for i, pcl in enumerate(pred):
            votes[i, pcl] += 1
    return votes.argmax(1)


# ---------------------------------------------- metrics -----------------

def confusion_matrix(y_true, y_pred):
    """2x2 confusion matrix [[TN, FP], [FN, TP]]."""
    yt = np.asarray(y_true, int); yp = np.asarray(y_pred, int)
    tn = int(np.sum((yt == 0) & (yp == 0))); fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0))); tp = int(np.sum((yt == 1) & (yp == 1)))
    return np.array([[tn, fp], [fn, tp]])


def classification_metrics(y_true, y_pred):
    """Return (accuracy, precision, recall, F1)."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / cm.sum()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: ML Classifiers for Carbonate Vuggy Facies")
    print("=" * 60)

    # Two log-facies clusters (non-vuggy / vuggy) in a 3-feature space
    rng = np.random.default_rng(8)
    n = 200
    nonvuggy = rng.normal([0, 0, 0], 0.6, (n, 3))
    vuggy = rng.normal([2.2, 2.0, 1.8], 0.6, (n, 3))
    X = np.vstack([nonvuggy, vuggy])
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    idx = rng.permutation(2 * n)
    X, y = X[idx], y[idx]
    ntr = 300
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    # Three classifiers
    lr_model = logistic_fit(Xtr, ytr)
    acc_lr = classification_metrics(yte, logistic_predict(lr_model, Xte))[0]
    acc_knn = classification_metrics(yte, knn_predict(Xtr, ytr, Xte, k=5))[0]
    acc_rf = classification_metrics(yte, stump_ensemble_predict(Xtr, ytr, Xte))[0]
    print(f"  accuracy LR/kNN/RF     = {acc_lr:.3f} / {acc_knn:.3f} / {acc_rf:.3f}")
    assert acc_lr > 0.85 and acc_knn > 0.85 and acc_rf > 0.85

    # Full metric set and confusion matrix consistency
    acc, prec, rec, f1 = classification_metrics(yte, logistic_predict(lr_model, Xte))
    cm = confusion_matrix(yte, logistic_predict(lr_model, Xte))
    print(f"  LR acc/prec/rec/F1     = {acc:.2f}/{prec:.2f}/{rec:.2f}/{f1:.2f}")
    assert cm.sum() == len(yte) and 0.0 <= f1 <= 1.0
    # perfect prediction -> F1 = 1
    assert abs(classification_metrics([0, 1, 1], [0, 1, 1])[3] - 1.0) < 1e-9
    print("  PASS")
    return {"acc_lr": acc_lr, "acc_knn": acc_knn, "acc_rf": acc_rf, "f1_lr": f1}


if __name__ == "__main__":
    test_all()
