"""
Article 10: Estimating the Permeability of Rocks by Principal Component
Regressions of NMR and MICP Data
Rios, Azeredo, Moss, Pritchard, Domingues (2022)
DOI: 10.30632/PJV63N3-2022a10

Body text was not present in the available PDF extract, so this module
is a *methodology proxy* guided by the Guest Editor's summary:
Principal Component Analysis (PCA) / Principal Component Regression
(PCR) on joint NMR T1 / T2 distributions and MICP capillary-pressure
curves to predict permeability in carbonate rocks; multivariate
estimators outperform classical Timur-Coates and SDR.

Implements:

  - Combined NMR + MICP feature vector (binned T2 distribution +
    binned MICP pressure-saturation curve).
  - PCA decomposition (SVD-based) retaining k principal components.
  - PCR: linear regression of log10(k) onto the PC scores.
  - Classical SDR and Timur-Coates baselines computed from the same
    NMR distribution, for head-to-head comparison.
"""

import numpy as np


# ---------------------------------------------- baselines ----------------

def timur_coates(phi, FFV, BFV, C=10.0, m=4.0, n=2.0):
    return C * phi ** m * (FFV / np.maximum(BFV, 1e-9)) ** n


def sdr(phi, T2_lm_ms, a=4.6, m=4.0, n=2.0):
    return a * phi ** m * T2_lm_ms ** n


def log_mean_T2(A, T2_axis):
    A = np.maximum(A, 0.0)
    return np.exp((A * np.log(T2_axis)).sum(-1) / (A.sum(-1) + 1e-12))


# ---------------------------------------------- PCA / PCR ---------------

def pca(X, n_components):
    """Return (scores, components, mean, explained)."""
    mean = X.mean(0)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :n_components] * s[:n_components],
            Vt[:n_components],
            mean,
            (s ** 2)[:n_components] / (s ** 2).sum())


def pca_transform(X, components, mean):
    return (X - mean) @ components.T


def pcr_fit(X, y, n_components):
    """log-domain PCR.  Returns the prediction function and PCA artefacts."""
    scores, comps, mean, _ = pca(X, n_components)
    A = np.c_[np.ones(len(y)), scores]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    def predict(X_new):
        sc = pca_transform(X_new, comps, mean)
        return np.c_[np.ones(len(sc)), sc] @ coef
    return predict, dict(coef=coef, components=comps, mean=mean)


# ---------------------------------------------- synthetic dataset ------

def make_nmr_micp_dataset(n=300, n_T2=24, n_Pc=12, seed=0):
    """Generate per-plug feature vectors and ground-truth permeability."""
    rng = np.random.default_rng(seed)
    T2 = np.logspace(-1, 3, n_T2)
    Pc = np.logspace(-2, 2, n_Pc)
    phi = rng.uniform(0.05, 0.30, n)
    # FFV / BFV ratio drawn so wider distributions give higher k
    ratio = rng.uniform(0.3, 4.0, n)
    BFV = phi / (1.0 + ratio)
    FFV = phi - BFV
    # True permeability uses BOTH the NMR-derived FFV/BFV ratio and the
    # MICP entry-pressure information.  We multiply Timur-Coates by an
    # MICP-driven correction factor so that NMR-only baselines miss the
    # Pc contribution and PCR (which sees both) does better.
    Pc_mean = rng.uniform(0.5, 20.0, n)            # MPa
    k_true = (timur_coates(phi, FFV, BFV)
              * (Pc_mean / Pc_mean.mean()) ** (-1.5)
              * np.exp(0.10 * rng.standard_normal(n)))
    # T2 distribution: BFV peaked at 3 ms, FFV peaked at T2_lm derived from Pc
    nmr = np.zeros((n, n_T2))
    micp = np.zeros((n, n_Pc))
    for i in range(n):
        nmr[i] += BFV[i] * np.exp(-((np.log10(T2) - np.log10(3.0)) / 0.3) ** 2)
        nmr[i] += FFV[i] * np.exp(-((np.log10(T2) - np.log10(80.0 / Pc_mean[i])) / 0.3) ** 2)
        # MICP cumulative Hg saturation as a function of Pc (Brooks-Corey-style)
        lam = 2.0
        pe = Pc_mean[i] / 2.0
        micp[i] = np.clip(1.0 - (pe / Pc) ** lam, 0.0, 1.0)
    X = np.c_[nmr, micp, phi[:, None]]
    return X, phi, FFV, BFV, T2, k_true


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 10: PCR Permeability from NMR + MICP (proxy)")
    print("=" * 60)

    X, phi, FFV, BFV, T2, k_true = make_nmr_micp_dataset()
    y = np.log10(np.maximum(k_true, 1e-9))

    idx = np.random.default_rng(0).permutation(len(y))
    cut = int(0.75 * len(y))
    Xt, yt = X[idx[:cut]], y[idx[:cut]]
    Xe, ye = X[idx[cut:]], y[idx[cut:]]
    phi_e = phi[idx[cut:]]
    nmr_e = X[idx[cut:], :len(T2)]
    FFV_e = FFV[idx[cut:]]
    BFV_e = BFV[idx[cut:]]

    # PCR
    predict, model = pcr_fit(Xt, yt, n_components=6)
    y_pcr = predict(Xe)
    rmse_pcr = float(np.sqrt(((y_pcr - ye) ** 2).mean()))

    # Baselines
    T2_lm = log_mean_T2(nmr_e, T2)
    y_sdr = np.log10(sdr(phi_e, T2_lm))
    y_tc = np.log10(timur_coates(phi_e, FFV_e, BFV_e))
    rmse_sdr = float(np.sqrt(((y_sdr - ye) ** 2).mean()))
    rmse_tc = float(np.sqrt(((y_tc - ye) ** 2).mean()))

    print(f"  PCR (6 PCs)        RMSE log10 k = {rmse_pcr:.3f}")
    print(f"  Timur-Coates       RMSE log10 k = {rmse_tc:.3f}")
    print(f"  SDR                RMSE log10 k = {rmse_sdr:.3f}")

    assert rmse_pcr < rmse_tc + 0.05, "PCR must at least match Timur-Coates"
    assert rmse_pcr < rmse_sdr + 0.05, "PCR must at least match SDR"
    print("  PASS")
    return {"rmse_pcr": rmse_pcr, "rmse_tc": rmse_tc, "rmse_sdr": rmse_sdr}


if __name__ == "__main__":
    test_all()
