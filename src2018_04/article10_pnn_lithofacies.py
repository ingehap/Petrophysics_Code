"""
Article 10: Complex Lithofacies Identification Using Improved Probabilistic
            Neural Networks
Gu, Bao, Rui (2018)
DOI: 10.30632/PJV59N2-2018a9  (inferred; body beyond extraction)

A probabilistic neural network (PNN) classifies lithofacies from well logs by
estimating each facies' probability density with a Parzen-window (Gaussian
kernel) sum over its training samples and assigning the facies of highest
posterior.  This *methodology proxy* implements that PNN: a per-class Gaussian
Parzen estimator, the Bayes decision over classes (optionally with priors), and
a simple smoothing-parameter selection by leave-one-out accuracy - the
"improved" tuning the paper studies.

Implements:

  - Per-class Parzen-window (Gaussian kernel) density
  - PNN classification (Bayes decision over classes)
  - Leave-one-out accuracy for the smoothing parameter sigma
  - Best-sigma selection over a grid

Note: this article's body was beyond this issue's machine extraction, so - as
with the other methodology proxies in this repository - the PNN below is the
standard Specht (1990) probabilistic-neural-network formulation the paper
improves on, not code transcribed from the paper.  The DOI suffix (a9) is
inferred from the issue's confirmed pattern.
"""

import numpy as np


# ---------------------------------------------- PNN --------------

def parzen_density(x, samples, sigma):
    """Gaussian Parzen-window density estimate at x from a class's samples.

        f(x) = mean_j exp(-||x - x_j||^2 / (2*sigma^2)).
    """
    d2 = ((np.asarray(samples, float) - np.asarray(x, float)) ** 2).sum(axis=1)
    return np.mean(np.exp(-d2 / (2.0 * sigma ** 2)))


def pnn_classify(x_train, y_train, x_test, sigma=1.0, priors=None):
    """Classify each test row by the PNN Bayes decision over classes."""
    x_train = np.asarray(x_train, float)
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    pri = {c: 1.0 for c in classes} if priors is None else priors
    preds = []
    for x in np.atleast_2d(np.asarray(x_test, float)):
        scores = [pri[c] * parzen_density(x, x_train[y_train == c], sigma) for c in classes]
        preds.append(classes[int(np.argmax(scores))])
    return np.array(preds)


def loo_accuracy(x_train, y_train, sigma):
    """Leave-one-out classification accuracy at a given smoothing sigma."""
    x = np.asarray(x_train, float)
    y = np.asarray(y_train)
    correct = 0
    for i in range(len(x)):
        mask = np.arange(len(x)) != i
        pred = pnn_classify(x[mask], y[mask], x[i], sigma)[0]
        correct += int(pred == y[i])
    return correct / len(x)


def best_sigma(x_train, y_train, grid):
    """Pick the smoothing sigma that maximizes leave-one-out accuracy."""
    accs = [loo_accuracy(x_train, y_train, s) for s in grid]
    return grid[int(np.argmax(accs))], max(accs)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 10: PNN Lithofacies Identification (proxy)")
    print("=" * 60)

    rng = np.random.default_rng(10)
    # Three well-separated facies clusters in a 2D log-feature space
    centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 6.0]])
    X = np.vstack([rng.normal(c, 0.6, (30, 2)) for c in centers])
    y = np.repeat([0, 1, 2], 30)

    # The PNN classifies the cluster centers to the right facies
    preds = pnn_classify(X, y, centers, sigma=1.0)
    print(f"  center predictions     = {preds.tolist()}")
    assert list(preds) == [0, 1, 2]

    # Leave-one-out accuracy is high for well-separated facies
    acc = loo_accuracy(X, y, sigma=1.0)
    print(f"  LOO accuracy           = {acc:.3f}")
    assert acc > 0.9

    # Best-sigma selection returns a grid value and its accuracy
    s, a = best_sigma(X, y, grid=[0.3, 1.0, 3.0])
    print(f"  best sigma / accuracy  = {s} / {a:.3f}")
    assert s in (0.3, 1.0, 3.0) and a >= acc - 1e-9
    print("  PASS")
    return {"LOO_accuracy": float(acc), "best_sigma": float(s)}


if __name__ == "__main__":
    test_all()
