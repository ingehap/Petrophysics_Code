"""
article3_mineralogical_inversion.py
====================================
Implements the linear-inversion mineralogical-modelling workflow described
in:

    Jácomo, M.H., Hartmann, G.A., Rebelo, T.B., Mattos, N.H., Batezelli, A.,
    Leite, E.P. (2023).  "Mineralogical Modeling and Petrophysical Properties
    of the Barra Velha Formation, Santos Basin, Brazil",
    Petrophysics, Vol. 64, No. 4, pp. 518-543.
    DOI: 10.30632/PJV64N4-2023a3

Key equations (paper numbering):

    Eq. 1  Volumetric photoelectric factor    U = PEF * rho_b
    Eq. 2  Larionov gamma-ray clay volume     Vcl = 0.083 * (2^(3.7*IGR) - 1)
    Eq. 3  Larionov older-rocks variant       Vcl = 0.33  * (2^(2.0*IGR) - 1)
    Eq. 4  NMR clay volume                    V_NR = (NMRtt - NMReff)/NMRtt
    Eqs. 6/7  Hybrid clay volume combining GR-clay and NMR-clay
    Eq. 8  Linear mineralogical-inversion system   ML_j = sum_i alpha_ij * V_i
                                                    s.t.  sum_i V_i = 1
                                                          V_i >= 0
    Eq. 9  Mismatch error  eps = sqrt( sum_j w_j*(ML_j - TL_j)^2 )

The inversion is solved with a non-negative least-squares problem under the
unit-sum constraint enforced as a heavy soft constraint -- equivalent to
the Techlog multicomponent model used by the authors but kept dependency-free.

Run as a script for the synthetic test suite:

    python article3_mineralogical_inversion.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Equation 1 - volumetric photoelectric factor
# ---------------------------------------------------------------------------
def upef(pef, rho_b):
    """U = PEF * rho_b  (barns/cm^3)."""
    return np.asarray(pef, dtype=float) * np.asarray(rho_b, dtype=float)


# ---------------------------------------------------------------------------
# Equation 2/3 - Larionov clay volume from GR
# ---------------------------------------------------------------------------
def gamma_index(gr, gr_min, gr_max):
    return np.clip((np.asarray(gr, dtype=float) - gr_min) / (gr_max - gr_min),
                   0.0, 1.0)


def vclay_larionov_younger(igr):
    """Tertiary / younger rocks (Eq. 2 in Jácomo et al.)"""
    return 0.083 * (2.0 ** (3.7 * np.asarray(igr, dtype=float)) - 1.0)


def vclay_larionov_older(igr):
    """Older / pre-Tertiary (Eq. 3)."""
    return 0.33 * (2.0 ** (2.0 * np.asarray(igr, dtype=float)) - 1.0)


# ---------------------------------------------------------------------------
# Equation 4 - NMR clay volume
# ---------------------------------------------------------------------------
def vclay_nmr(nmr_tt, nmr_eff):
    """V_NR = (NMRtt - NMReff)/NMRtt  --  micro-porosity proxy."""
    nmr_tt = np.asarray(nmr_tt, dtype=float)
    nmr_eff = np.asarray(nmr_eff, dtype=float)
    out = np.zeros_like(nmr_tt)
    mask = nmr_tt > 1.0e-6
    out[mask] = (nmr_tt[mask] - nmr_eff[mask]) / nmr_tt[mask]
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Eqs. 6/7 - hybrid clay volume
# ---------------------------------------------------------------------------
def vclay_hybrid(igr, nmr_tt, nmr_eff):
    """Use Larionov-GR where it exceeds NMR-clay (radioactive shaly zones)
    and the NMR estimate elsewhere (Mg-clay / non-radioactive zones)."""
    vcl_gr = vclay_larionov_younger(igr)
    vcl_nmr = vclay_nmr(nmr_tt, nmr_eff)
    return np.maximum(vcl_gr, vcl_nmr)


# ---------------------------------------------------------------------------
# Equations 8 & 9 - mineralogical inversion
# ---------------------------------------------------------------------------
def forward_logs(volumes, mineral_props):
    """Forward log model.  ML_j = sum_i alpha_ij * V_i  (Eq. 8).

    volumes        : (..., n_min)  mineral volume fractions, sum == 1
    mineral_props  : (n_log, n_min) matrix of per-mineral log responses

    Returns synthetic log responses of shape (..., n_log).
    """
    return np.asarray(volumes) @ np.asarray(mineral_props).T


def inversion_error(measured, modelled, weights=None):
    """Eq. 9 weighted RMS error.  weights default to unity."""
    measured = np.asarray(measured, dtype=float)
    modelled = np.asarray(modelled, dtype=float)
    if weights is None:
        weights = np.ones(measured.shape[-1])
    weights = np.asarray(weights, dtype=float)
    diff2 = weights * (measured - modelled) ** 2
    return np.sqrt(np.sum(diff2, axis=-1))


def invert_mineralogy(measured_logs, mineral_props,
                      weights=None, sum_to_one_weight=1000.0):
    """Solve volumes V given measured log curves and mineral end-members.

    Parameters
    ----------
    measured_logs : (n_depth, n_log) array of measured log values
    mineral_props : (n_log, n_min)   matrix A so that log = A @ V
    weights       : (n_log,) per-log weights (defaults to ones)
    sum_to_one_weight : how strongly to enforce sum(V) = 1

    Returns volumes of shape (n_depth, n_min), all in [0, 1] and summing
    to ~1 for every depth.  Uses non-negative least squares internally.
    """
    A = np.asarray(mineral_props, dtype=float)
    Y = np.atleast_2d(np.asarray(measured_logs, dtype=float))
    n_log, n_min = A.shape
    if weights is None:
        w = np.ones(n_log)
    else:
        w = np.asarray(weights, dtype=float)
    Wsqrt = np.sqrt(w)[:, None]
    # Augment with sum-to-one row (heavy weight)
    A_aug = np.vstack([Wsqrt * A, np.full((1, n_min), np.sqrt(sum_to_one_weight))])
    out = np.zeros((Y.shape[0], n_min))
    try:
        from scipy.optimize import nnls
        for i, y in enumerate(Y):
            y_aug = np.concatenate([(np.sqrt(w) * y),
                                    [np.sqrt(sum_to_one_weight) * 1.0]])
            v, _ = nnls(A_aug, y_aug)
            s = v.sum()
            if s > 0:
                v = v / s  # exact normalisation
            out[i] = v
    except ImportError:
        # Fallback: unconstrained least-squares with renormalisation/clipping
        for i, y in enumerate(Y):
            y_aug = np.concatenate([(np.sqrt(w) * y),
                                    [np.sqrt(sum_to_one_weight) * 1.0]])
            v, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
            v = np.clip(v, 0.0, None)
            s = v.sum()
            if s > 0:
                v = v / s
            out[i] = v
    return out if Y.shape[0] > 1 else out[0]


# Default mineral end-member catalogue (Table 2/3 in the paper) -------------
DEFAULT_MINERALS = {
    # name        rho_b   PEF   NPHI   Vp    Vshear  notes
    "calcite":   dict(rho_b=2.71, pef=5.08, nphi=0.00, vp=6530.0),
    "dolomite":  dict(rho_b=2.87, pef=3.14, nphi=0.02, vp=7050.0),
    "quartz":    dict(rho_b=2.65, pef=1.81, nphi=-0.04, vp=6050.0),
    "clay":      dict(rho_b=2.45, pef=3.42, nphi=0.30, vp=3400.0),
    "stevensite":dict(rho_b=2.35, pef=2.14, nphi=0.40, vp=3100.0),
}


def build_response_matrix(minerals=None, logs=("rho_b", "nphi", "upef", "vp")):
    """Convert the mineral catalogue to the (n_log, n_min) matrix needed by
    `forward_logs` / `invert_mineralogy`.  upef is computed from PEF*rho_b.
    """
    if minerals is None:
        minerals = DEFAULT_MINERALS
    names = list(minerals.keys())
    n_min, n_log = len(names), len(logs)
    A = np.zeros((n_log, n_min))
    for j, name in enumerate(names):
        m = minerals[name]
        for i, lg in enumerate(logs):
            if lg == "upef":
                A[i, j] = m["pef"] * m["rho_b"]
            else:
                A[i, j] = m[lg]
    return A, names


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
def test_all(verbose=True):
    rng = np.random.default_rng(123)

    # --- 1. UPEF helper ----------------------------------------------------
    u = upef([5.08, 1.81], [2.71, 2.65])
    assert abs(u[0] - 13.7668) < 1e-9 and abs(u[1] - 4.7965) < 1e-9
    if verbose:
        print(f"[1] U-PEF helper OK             (calcite U={u[0]:.2f} b/cm3)")

    # --- 2. Larionov clay --------------------------------------------------
    igr = np.linspace(0, 1, 11)
    vyo = vclay_larionov_younger(igr)
    vol = vclay_larionov_older(igr)
    assert vyo[0] == 0.0 and abs(vyo[-1] - 0.996) < 1e-3
    assert vol[0] == 0.0 and abs(vol[-1] - 0.99)  < 1e-3
    if verbose:
        print(f"[2] Larionov clay OK            "
              f"(younger(IGR=0.5)={vyo[5]:.3f}, older={vol[5]:.3f})")

    # --- 3. NMR clay -------------------------------------------------------
    vnmr = vclay_nmr([0.10, 0.20], [0.07, 0.10])
    assert np.allclose(vnmr, [0.30, 0.50])
    if verbose:
        print(f"[3] NMR clay OK                 (V_NR = {vnmr})")

    # --- 4. Hybrid clay ----------------------------------------------------
    vhyb = vclay_hybrid(igr=0.6, nmr_tt=0.20, nmr_eff=0.05)
    assert 0.0 <= vhyb <= 1.0
    if verbose:
        print(f"[4] Hybrid clay OK              (Vcl_hyb = {vhyb:.3f})")

    # --- 5. Forward log model ---------------------------------------------
    A, names = build_response_matrix()
    n_min = len(names)
    # Pure calcite layer
    v_pure = np.zeros(n_min); v_pure[names.index("calcite")] = 1.0
    logs_pure = forward_logs(v_pure, A)
    # density of pure calcite must equal 2.71
    assert abs(logs_pure[0] - 2.71) < 1e-9
    if verbose:
        print(f"[5] Forward log OK              (pure-calcite RHOB={logs_pure[0]:.2f})")

    # --- 6. Inversion round-trip ------------------------------------------
    n_depth = 30
    v_true = rng.dirichlet(alpha=np.ones(n_min) * 1.0, size=n_depth)
    logs_synth = forward_logs(v_true, A)
    # Add small noise (1% of typical magnitude)
    noise = rng.normal(scale=0.02, size=logs_synth.shape) * np.std(logs_synth, 0)
    logs_obs = logs_synth + noise
    v_inv = invert_mineralogy(logs_obs, A,
                              weights=[10.0, 5.0, 2.0, 1.0])
    assert v_inv.shape == (n_depth, n_min)
    sums = v_inv.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-3)
    mae = np.mean(np.abs(v_inv - v_true))
    assert mae < 0.10, f"Inversion mean abs error {mae:.3f} too high"
    if verbose:
        print(f"[6] Mineral inversion OK        (mean |dV| = {mae:.3f}, "
              f"sum(V) ~ 1 within 1e-3)")

    # --- 7. Error metric ---------------------------------------------------
    eps = inversion_error(logs_obs, forward_logs(v_inv, A),
                          weights=[10.0, 5.0, 2.0, 1.0])
    assert eps.shape == (n_depth,)
    if verbose:
        print(f"[7] Error metric OK             (mean eps = {eps.mean():.3f})")

    if verbose:
        print("\nAll article-3 tests passed.")
    return True


if __name__ == "__main__":
    test_all()
