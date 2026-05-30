"""
Article 7: Integrated Multiphysics Workflow for Automatic Rock Classification and
           Formation Evaluation Using Multiscale Image Analysis and Conventional
           Well Logs
Gonzalez, Kanyan, Heidari, Lopez (2020)
DOI: 10.30632/PJV61N5-2020a7

Rock-fabric texture features are extracted from core/CT images via the
gray-level co-occurrence matrix (GLCM), combined with conventional logs, and
clustered with k-means into rock classes.  The optimum number of classes is
chosen by a permeability-based cost function (and validated with the silhouette
coefficient); the classes are then propagated to noncored wells.

Implements:

  - Mean gray level per depth                                     (Eq. 1)
  - GLCM contrast and energy textural features                    (Eqs. 2-3)
  - Experimental variogram for GLCM window selection              (Eq. 5)
  - Silhouette coefficient for cluster validation                 (Eq. 6)
  - k-means rock classification + permeability cost function       (Eq. 7)

Note: this issue's PDF text layer kept the equation numbers / variable
definitions but dropped the typeset glyphs, so the GLCM contrast/energy,
silhouette, and variogram closed forms are the standard (Haralick / Rousseeuw)
expressions.  The paper's anchor is reproduced: the permeability cost drops most
when the class count matches the three formations present.
"""

import numpy as np


# ---------------------------------------------- image features ----------

def mean_gray_level(row):
    """Average gray level over the pixels at one depth  (Eq. 1)."""
    return float(np.mean(np.asarray(row, float)))


def glcm(image, levels, offset=(0, 1)):
    """Normalized, symmetric gray-level co-occurrence matrix.

    image is quantized to [0, levels); counts co-occurrences for the given
    (drow, dcol) offset.  Returns an (levels x levels) probability matrix.
    """
    img = np.asarray(image)
    q = np.clip((img - img.min()) / (img.max() - img.min() + 1e-12) * (levels - 1),
                0, levels - 1).astype(int)
    G = np.zeros((levels, levels))
    dr, dc = offset
    nr, nc = q.shape
    for r in range(nr - max(dr, 0)):
        for c in range(nc - max(dc, 0)):
            i, j = q[r, c], q[r + dr, c + dc]
            G[i, j] += 1
            G[j, i] += 1
    s = G.sum()
    return G / s if s > 0 else G


def glcm_contrast(G):
    """GLCM contrast  sum_ij G(i,j)*(i-j)^2  (Eq. 2)."""
    n = G.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return float(np.sum(G * (i - j) ** 2))


def glcm_energy(G):
    """GLCM energy (angular second moment)  sum_ij G(i,j)^2  (Eq. 3)."""
    return float(np.sum(G ** 2))


def experimental_variogram(trace, max_lag):
    """Vertical experimental variogram  gamma(h) = mean((GL_i - GL_{i+h})^2)/2  (Eq. 5)."""
    x = np.asarray(trace, float)
    g = []
    for h in range(1, max_lag + 1):
        diff = x[:-h] - x[h:]
        g.append(0.5 * np.mean(diff ** 2))
    return np.array(g)


# ---------------------------------------------- clustering --------------

def kmeans(X, k, iters=100, seed=0):
    """Minimal k-means; returns (labels, centers)."""
    X = np.asarray(X, float)
    idx = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
    for _ in range(1, k):
        d = np.min([np.linalg.norm(X - X[j], axis=1) for j in idx], axis=0)
        idx.append(int(np.argmax(d)))
    centers = X[idx].copy()
    labels = np.zeros(len(X), int)
    for _ in range(iters):
        d = np.stack([np.linalg.norm(X - c, axis=1) for c in centers], axis=1)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            if np.any(labels == c):
                centers[c] = X[labels == c].mean(0)
    return labels, centers


def silhouette(X, labels):
    """Mean silhouette coefficient  s = (b - a)/max(a, b)  (Eq. 6)."""
    X = np.asarray(X, float)
    labels = np.asarray(labels, int)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    s = np.zeros(len(X))
    for idx in range(len(X)):
        own = labels == labels[idx]
        own[idx] = False
        a = D[idx, own].mean() if own.any() else 0.0
        b = min(D[idx, labels == c].mean() for c in uniq if c != labels[idx])
        s[idx] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def permeability_cost(features, core_perm, n_classes, seed=0):
    """RMS error between core perm and its per-class mean estimate  (Eq. 7)."""
    labels, _ = kmeans(features, n_classes, seed=seed)
    est = np.zeros_like(core_perm, float)
    for c in np.unique(labels):
        est[labels == c] = core_perm[labels == c].mean()
    return float(np.sqrt(np.mean((core_perm - est) ** 2)))


def optimal_classes(features, core_perm, candidates, rel_tol=0.15):
    """First class count whose next increment improves the cost by < rel_tol."""
    costs = [permeability_cost(features, core_perm, k) for k in candidates]
    for i in range(len(candidates) - 1):
        improve = (costs[i] - costs[i + 1]) / costs[i] if costs[i] > 0 else 0.0
        if improve < rel_tol:
            return candidates[i], costs
    return candidates[-1], costs


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Multiphysics Rock Classification")
    print("=" * 60)

    # GLCM: a smooth (low-texture) patch has low contrast and high energy along
    # the horizontal offset; a noisy patch has high contrast and low energy
    rng = np.random.default_rng(7)
    smooth = np.tile(np.linspace(0, 7, 8), (8, 1))   # vertical ramp -> rows equal
    smooth = smooth.T                                 # horizontal neighbors equal
    noisy = rng.integers(0, 8, (8, 8)).astype(float)
    G_s = glcm(smooth, levels=8, offset=(0, 1))
    G_n = glcm(noisy, levels=8, offset=(0, 1))
    print(f"  contrast smooth/noisy  = {glcm_contrast(G_s):.2f} / {glcm_contrast(G_n):.2f}")
    print(f"  energy   smooth/noisy  = {glcm_energy(G_s):.3f} / {glcm_energy(G_n):.3f}")
    assert glcm_contrast(G_s) < glcm_contrast(G_n)
    assert glcm_energy(G_s) > glcm_energy(G_n)

    # Variogram rises with lag for a smoothly varying trace
    trace = np.cumsum(rng.standard_normal(60))
    vg = experimental_variogram(trace, max_lag=10)
    assert vg[-1] > vg[0]

    # Three well-separated rock classes in a 2-feature space, each a perm level
    # (small within-class spread so the cost floor is near zero)
    centers = np.array([[0, 0], [6, 0], [0, 6]], float)
    perm_levels = np.array([1.0, 10.0, 100.0])
    feats, perm = [], []
    for c, k in zip(centers, perm_levels):
        pts = c + 0.4 * rng.standard_normal((40, 2))
        feats.append(pts)
        perm.append(np.full(40, k) * np.exp(0.03 * rng.standard_normal(40)))
    feats = np.vstack(feats); perm = np.concatenate(perm)

    labels, _ = kmeans(feats, 3)
    sil = silhouette(feats, labels)
    print(f"  silhouette (k=3)       = {sil:.3f}")
    assert sil > 0.7                                   # tight, well-separated

    # The permeability cost converges when the class count matches the 3 classes
    k_opt, costs = optimal_classes(feats, perm, [2, 3, 4, 5, 6])
    print(f"  cost vs k              = {np.array2string(np.array(costs), precision=2)}")
    print(f"  optimal classes        = {k_opt}")
    assert k_opt == 3
    print("  PASS")
    return {"silhouette": sil, "optimal_classes": k_opt}


if __name__ == "__main__":
    test_all()
