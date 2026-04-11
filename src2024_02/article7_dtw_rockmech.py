"""
Article 7: Study on Rock Mechanics Parameter Prediction Method Based on
DTW Similarity and Machine-Learning Algorithms
Cai, Ding, Li, Yin, Feng (Petrophysics, Vol. 65, No. 1, Feb 2024, pp. 128-end)

The authors find analog wells using Dynamic Time Warping (DTW) on log
curves, then train an ML model on the most similar analogs to predict
rock mechanics parameters (e.g., Young's modulus, UCS) at the target
well. This module implements a compact DTW + kNN regressor.
"""
import numpy as np


def dtw_distance(a, b):
    """Standard O(n*m) DTW with Euclidean local cost."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def find_analogs(target_curve, library_curves, k=3):
    dists = [dtw_distance(target_curve, c) for c in library_curves]
    idx = np.argsort(dists)[:k]
    return list(idx), [dists[i] for i in idx]


def predict_property(target_curve, library_curves, library_values, k=3):
    idx, dists = find_analogs(target_curve, library_curves, k)
    w = 1.0 / (np.array(dists) + 1e-9)
    return float(np.average(np.array(library_values)[idx], weights=w))


def test_all():
    rng = np.random.default_rng(2)
    # 20 analog wells: each curve is a noisy sinusoid; UCS depends on amplitude
    library_curves, library_values = [], []
    for _ in range(20):
        amp = rng.uniform(1.0, 5.0)
        x = np.linspace(0, 4 * np.pi, 60)
        library_curves.append(amp * np.sin(x) + rng.normal(0, 0.1, x.size))
        library_values.append(20.0 * amp + 5.0)  # UCS in MPa
    # Target with amp ~ 3 -> expected UCS ~ 65
    x = np.linspace(0, 4 * np.pi, 60)
    target = 3.0 * np.sin(x) + rng.normal(0, 0.1, x.size)
    pred = predict_property(target, library_curves, library_values, k=3)
    assert 55.0 < pred < 75.0, pred
    print(f"article7 OK | predicted UCS = {pred:.1f} MPa (true ≈ 65)")


if __name__ == "__main__":
    test_all()
