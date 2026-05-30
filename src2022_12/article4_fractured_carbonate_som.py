"""
Article 4: Naturally Fractured Carbonate Reservoir Characterization
Ali Akbar, Nemes, Bihari, Soltesz, Barany, Toth, Borka, Ferincz (2022)
DOI: 10.30632/PJV63N6-2022a4

The paper trains a Spherical Self-Organizing Map (sSOM) to classify
three fracture facies - macrofracture (Class 1), microfracture (Class 2),
host rock (Class 3) - on legacy wells that lack a full triple-combo
suite.  Three input variants are tested:

    Model 1: NGK, GK, POR, RD, VSh, SPBL, VDolo
    Model 2: DTc, POR, RD, VSh, CALI-BS
    Model 3: POR, RD, VSh, CALI-BS

This module implements:

  - Harrison (1995) Russian-log analogue for legacy GK+NGK->Vsh, porosity.
  - A simplified rectangular SOM (Kohonen 1990) that captures the
    competitive-learning + neighbourhood-update dynamics of the
    spherical variant used in the paper.  Spherical topology mainly
    removes border effects; on small lattices the rectangular SOM gives
    similar facies-classification accuracy.
  - Torabi et al. (2019) damage-zone-width vs. fault-displacement
    relationship for laterally distributing macrofracture probability
    around a fault plane.
"""

import numpy as np


# ---------------------------------------------- Harrison (1995) ------------

def harrison_porosity_from_ngk(ngk):
    """Empirical porosity from Russian neutron-gamma log:
        phi = a - b * NGK  (clipped to [0, 0.35])."""
    return np.clip(0.40 - 0.0030 * ngk, 0.0, 0.35)


def vsh_from_gr_linear(gk, gk_clean=10.0, gk_shale=120.0):
    return np.clip((gk - gk_clean) / (gk_shale - gk_clean), 0.0, 1.0)


# ---------------------------------------------- Torabi damage zone ---------

def damage_zone_width(displacement_m):
    """Torabi et al. (2019) power-law: w_dz = 0.12 * d^0.8 m.

    Splits the zone around a fault into four classes by distance:
        x < 0.5 w_dz       -> fault core
        x < 1.5 w_dz       -> high-damage zone
        x < 3.0 w_dz       -> low-damage zone
        otherwise          -> host rock
    """
    return 0.12 * displacement_m ** 0.8


def fault_zone_class(distance_m, displacement_m):
    w = damage_zone_width(displacement_m)
    if distance_m < 0.5 * w:
        return "fault_core"
    if distance_m < 1.5 * w:
        return "high_dz"
    if distance_m < 3.0 * w:
        return "low_dz"
    return "host"


# ---------------------------------------------- SOM ----------------------

class SOM:
    """Bare-bones rectangular SOM with Gaussian neighbourhood."""
    def __init__(self, grid_shape=(6, 6), lr0=0.5, sigma0=2.5, n_iter=2000,
                 seed=0):
        self.grid_shape = grid_shape
        self.lr0 = lr0
        self.sigma0 = sigma0
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)
        self.W = None

    def _bmu(self, x):
        diff = self.W - x
        d2 = (diff ** 2).sum(-1)
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        return i, j

    def fit(self, X):
        # Standardise
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-9
        Xs = (X - self.mean) / self.std
        n, d = Xs.shape
        h, w = self.grid_shape
        self.W = self.rng.standard_normal((h, w, d)) * 0.3
        ii, jj = np.indices((h, w))
        for it in range(self.n_iter):
            x = Xs[self.rng.integers(0, n)]
            i, j = self._bmu(x)
            lam = self.n_iter / np.log(self.sigma0)
            sig = max(0.6, self.sigma0 * np.exp(-it / lam))
            lr = self.lr0 * np.exp(-it / self.n_iter)
            dist2 = (ii - i) ** 2 + (jj - j) ** 2
            h_neigh = np.exp(-dist2 / (2 * sig ** 2))
            self.W += lr * h_neigh[..., None] * (x - self.W)
        return self

    def assign(self, X):
        Xs = (X - self.mean) / self.std
        return np.array([self._bmu(x) for x in Xs])  # (n, 2)


def som_to_class_labels(som, X, classes, k=3):
    """Majority-vote each SOM unit to a class label, then assign new samples."""
    assignments = som.assign(X)
    h, w = som.grid_shape
    label_for_unit = -np.ones((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            mask = (assignments[:, 0] == i) & (assignments[:, 1] == j)
            if mask.any():
                vals, counts = np.unique(classes[mask], return_counts=True)
                label_for_unit[i, j] = int(vals[np.argmax(counts)])
    # Fill any empty units by nearest-occupied label
    for i in range(h):
        for j in range(w):
            if label_for_unit[i, j] >= 0:
                continue
            for r in range(1, max(h, w)):
                neigh = []
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        ii_, jj_ = i + di, j + dj
                        if 0 <= ii_ < h and 0 <= jj_ < w \
                                and label_for_unit[ii_, jj_] >= 0:
                            neigh.append(label_for_unit[ii_, jj_])
                if neigh:
                    label_for_unit[i, j] = int(np.bincount(neigh).argmax())
                    break
    def predict(X_new):
        a = som.assign(X_new)
        return label_for_unit[a[:, 0], a[:, 1]]
    return label_for_unit, predict


# ---------------------------------------------- synthetic dataset --------

def make_carbonate_logs(n_per_class=120, seed=0):
    """Build (POR, RD, VSh, DTc, CALI-BS) features for three fracture classes."""
    rng = np.random.default_rng(seed)
    feats, classes = [], []
    for ci, (por_mu, rd_mu, vsh_mu, dt_mu) in enumerate([
        (0.06, 80.0, 0.10, 60.0),    # Class 0 macrofracture (high RD, low Vsh)
        (0.04, 25.0, 0.20, 65.0),    # Class 1 microfracture
        (0.03,  8.0, 0.40, 75.0),    # Class 2 host rock (shaly)
    ]):
        por = rng.normal(por_mu, 0.015, n_per_class).clip(0.005, 0.30)
        rd = np.exp(rng.normal(np.log(rd_mu), 0.30, n_per_class))
        vsh = rng.normal(vsh_mu, 0.05, n_per_class).clip(0.0, 1.0)
        dt = rng.normal(dt_mu, 3.0, n_per_class)
        cali = rng.normal(0.0 if ci == 2 else 0.6, 0.15, n_per_class)
        feats.append(np.c_[por, rd, vsh, dt, cali])
        classes.extend([ci] * n_per_class)
    X = np.vstack(feats)
    return X, np.array(classes)


# ---------------------------------------------- tests --------------------

def test_all():
    print("=" * 60)
    print("Article 4: Fractured-Carbonate SOM Facies Classifier")
    print("=" * 60)

    # Quick numerical sanity for Harrison + damage zone helpers
    phi = harrison_porosity_from_ngk(np.array([20.0, 80.0, 120.0]))
    print(f"  Harrison phi for NGK=20/80/120 = {phi}")
    print(f"  Damage-zone width for d=50 m   = "
          f"{damage_zone_width(50.0):.2f} m")
    print(f"  Zone class at x=1.0 m, d=50 m  = "
          f"{fault_zone_class(1.0, 50.0)}")

    X, y = make_carbonate_logs()
    n_train = int(0.7 * len(y))
    idx = np.random.default_rng(0).permutation(len(y))
    Xt, yt = X[idx[:n_train]], y[idx[:n_train]]
    Xe, ye = X[idx[n_train:]], y[idx[n_train:]]

    som = SOM(grid_shape=(8, 8), n_iter=4000, seed=1).fit(Xt)
    _, predict = som_to_class_labels(som, Xt, yt)
    y_hat = predict(Xe)
    acc = float((y_hat == ye).mean())
    print(f"  Train n = {n_train}, test n = {len(ye)}")
    print(f"  SOM held-out classification accuracy = {acc:.3f}")

    # Baseline: nearest centroid
    centroids = np.array([X[y == c].mean(0) for c in range(3)])
    Xs = (Xe - Xe.mean(0)) / (Xe.std(0) + 1e-9)
    Cs = (centroids - Xe.mean(0)) / (Xe.std(0) + 1e-9)
    nn_pred = ((Xs[:, None, :] - Cs[None, :, :]) ** 2).sum(-1).argmin(1)
    acc_nn = float((nn_pred == ye).mean())
    print(f"  Nearest-centroid baseline accuracy   = {acc_nn:.3f}")

    assert acc > 0.7, "SOM must clear 70% on this separable synthetic data"
    print("  PASS")
    return {"som_acc": acc, "centroid_acc": acc_nn}


if __name__ == "__main__":
    test_all()
