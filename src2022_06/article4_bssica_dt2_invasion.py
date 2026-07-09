"""
Article 4: NMR-Supported Near-Wellbore Data Analysis to Improve the
Petrophysical Evaluation of a Sandstone Reservoir Drilled With
Barite-Enriched Water-Based Mud
Romero Rojas, Tagarieva, Panchal, AlTurki, Qubian (2022)
DOI: 10.30632/PJV63N3-2022a4

Sidewall NMR analysis of barite-WBM near-wellbore damage in the Greater
Burgan Field, Kuwait.  Implements:

  - Parallel-relaxation rates                                (Eqs. 1, 2)
        1 / T1 = 1 / T1B + 1 / T1S
        1 / T2 = 1 / T2B + 1 / T2D + 1 / T2S
  - Porosity undercall  d_phi = phi_open - phi_NMR           (Eq. 3)
  - Timur-Coates permeability  K = C * phi^m * (FFV/BFV)^n   (Eq. 4)
  - Permeability ratio index   KRI = K_NMR / K_open          (Eq. 5)
  - Blind Source Separation by Independent Component Analysis
    using the linear mixing model  x = A * s                 (Eqs. 6-11)
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Eqs. 1-2: relaxation rates --

def T1_total(T1_bulk, T1_surface):
    return petrolib.nmr.combine_relaxation_times(T1_bulk, T1_surface)


def T2_total(T2_bulk, T2_diffusion, T2_surface):
    return petrolib.nmr.combine_relaxation_times(T2_bulk, T2_diffusion, T2_surface)


# ---------------------------------------------- Eq. 3: porosity undercall --

def porosity_undercall(phi_open, phi_NMR):
    return float(phi_open - phi_NMR)


# ---------------------------------------------- Eq. 4: Timur-Coates ------

def timur_coates(phi, FFV, BFV, C=10.0, m=4.0, n=2.0):
    """K = C * phi^m * (FFV / BFV)^n   (Eq. 4)."""
    return float(petrolib.nmr.timur_coates(
        phi, FFV, max(BFV, 1e-9), C=C, m=m, n=n, form="prefactor"))


# ---------------------------------------------- Eq. 5: KRI -------------

def permeability_ratio_index(phi_open, phi_NMR, ffv_open, ffv_NMR,
                              bfv_open, bfv_NMR, C=10.0, m=4.0, n=2.0):
    """KRI = K_NMR / K_open  (Eq. 5).

    KRI = 1 -> no damage; KRI < 1 -> damage from mud invasion.
    """
    K_open = timur_coates(phi_open, ffv_open, bfv_open, C, m, n)
    K_NMR = timur_coates(phi_NMR, ffv_NMR, bfv_NMR, C, m, n)
    return float(K_NMR / max(K_open, 1e-12))


# ---------------------------------------------- BSS-ICA (Eqs. 6-11) ------

def whiten(X):
    """Centre and PCA-whiten X (samples in rows)."""
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xw = U * np.sqrt(Xc.shape[0])
    return Xw, Vt.T / (s + 1e-9), Xc.mean(0)


def ica_fastica(X, n_components=None, n_iter=200, tol=1e-6, seed=0):
    """FastICA on the rows of X using the cubic non-linearity g(x) = x^3.

    Returns the independent components S (samples x components).
    """
    rng = np.random.default_rng(seed)
    Xw, dewhiten, mean = whiten(X)
    n = Xw.shape[1]
    k = n_components or n
    W = rng.standard_normal((k, n))
    # Symmetric orthogonalisation
    W = np.linalg.qr(W)[0][:k]
    for _ in range(n_iter):
        W_new = np.zeros_like(W)
        for i in range(k):
            w = W[i]
            wx = Xw @ w
            g = wx ** 3
            gp = 3.0 * wx ** 2
            w_new = (Xw * g[:, None]).mean(0) - gp.mean() * w
            W_new[i] = w_new
        # Symmetric orthogonalisation:  W <- (W W^T)^(-1/2) W
        U, s, Vt = np.linalg.svd(W_new @ W_new.T)
        W_new = (U @ np.diag(1.0 / np.sqrt(s + 1e-9)) @ U.T) @ W_new
        if np.linalg.norm(W_new - W) < tol:
            W = W_new
            break
        W = W_new
    return Xw @ W.T


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: BSS-ICA + D-T2 for Near-Wellbore NMR Analysis")
    print("=" * 60)

    # Relaxation rates: surface + diffusion shorten total
    T1 = T1_total(2000.0, 50.0)
    T2 = T2_total(1500.0, 60.0, 20.0)
    print(f"  T1 (bulk=2000 ms, surf=50 ms) = {T1:6.2f} ms")
    print(f"  T2 (bulk=1500 ms, diff=60 ms, surf=20 ms) = {T2:6.2f} ms")
    assert T2 < 25.0 and T1 < 50.0

    # Porosity undercall and KRI (Eqs. 3-5)
    phi_open = 0.22
    phi_NMR = 0.18
    d_phi = porosity_undercall(phi_open, phi_NMR)
    KRI = permeability_ratio_index(phi_open=0.22, phi_NMR=0.18,
                                   ffv_open=0.18, ffv_NMR=0.10,
                                   bfv_open=0.04, bfv_NMR=0.08)
    print(f"  Porosity undercall d_phi (Eq. 3)         = {d_phi:.3f} v/v")
    print(f"  Permeability ratio index KRI (Eq. 5)     = {KRI:.3f}")
    assert KRI < 0.3, "Significant damage expected from these phi/FFV values"

    # BSS-ICA on a 3-source synthetic mixture
    rng = np.random.default_rng(0)
    t = np.linspace(0, 8, 1500)
    s1 = np.sin(2 * np.pi * 1.0 * t)
    s2 = np.sign(np.sin(2 * np.pi * 3.0 * t))   # square wave
    s3 = (rng.random(len(t)) - 0.5)
    S = np.stack([s1, s2, s3], axis=1)
    A = np.array([[1.0, 0.5, 0.2], [-0.5, 1.0, 0.4], [0.3, -0.6, 1.0]])
    X = S @ A.T

    S_hat = ica_fastica(X, n_components=3, seed=0)
    # Match each recovered IC to a true source by abs correlation
    matched = []
    for k in range(3):
        c = [abs(np.corrcoef(S_hat[:, k], S[:, j])[0, 1]) for j in range(3)]
        matched.append(max(c))
    print(f"  ICA best-source corrs (3 ICs)            = "
          f"{[round(c, 3) for c in matched]}")
    assert min(matched) > 0.85, "ICA must recover all three sources"
    print("  PASS")
    return {"d_phi": d_phi, "KRI": KRI, "ica_corr_min": float(min(matched))}


if __name__ == "__main__":
    test_all()
