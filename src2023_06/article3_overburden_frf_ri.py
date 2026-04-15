"""
article3_overburden_frf_ri.py
==============================
Implementation of ideas from:

    Nourani, M., Pruno, S., Ghasemi, M., Fazlija, M.M., Gonzalez, B.,
    Rodvelt, H.-E.
    "Analytical Models for Predicting the Formation Resistivity Factor
    and Resistivity Index at Overburden Conditions"
    Petrophysics, Vol. 64, No. 3 (June 2023), pp. 353-366
    DOI: 10.30632/PJV64N3-2023a3

The paper introduces:

    Rock Resistivity Modulus      RRM = (1/Ro)  * dRo/dP
    True Resistivity Modulus      TRM = (1/Rt)  * dRt/dP
    Water Resistivity Modulus     WRM = (1/Rw)  * dRw/dP  (assumed ~0)

Integrating Eq. 5 over a pressure interval (assuming RRM constant) gives

    Ro(P2) = Ro(P1) * exp( - RRM * (P2 - P1) )         (Eq. 8)

and dividing by Rw yields the overburden FRF (Eq. 15)

    FRF(P2) = FRF(P1) * exp( - RRM * dP )

Two predictive models are derived:

    *  Multi-FRF :  RRM is fitted from FRF measured at several pressures
                    via the slope of  ln(FRF2/FRF1)  vs  dP   (Eq. 18)
    *  Single-FRF:  RRM is computed from compressibilities and m
                    (Eq. 16:  RRM ~ -m * (Cp - Cb) )

The same scheme is then extended to the Resistivity Index.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Multi-FRF model:  fit RRM from FRF measurements at multiple pressures
# ---------------------------------------------------------------------------
def estimate_rrm_from_frf(pressures: np.ndarray,
                          frf_measured: np.ndarray) -> tuple[float, float]:
    """
    Fit RRM from a set of FRF measurements at increasing overburden
    pressures.

    Eq. 18 of the paper :   ln(FRF2/FRF1) = -RRM * (P2 - P1)
    A linear regression of ln(FRFi/FRF1) vs (Pi - P1) through the
    origin gives the slope -RRM.

    Returns
    -------
    rrm   : Rock Resistivity Modulus (negative number; RRM is defined
            so that Ro grows with pressure -> RRM < 0)
    frf1  : reference FRF at the lowest pressure
    """
    pressures = np.asarray(pressures, dtype=float)
    frf_measured = np.asarray(frf_measured, dtype=float)
    order = np.argsort(pressures)
    pressures = pressures[order]
    frf_measured = frf_measured[order]

    # Two-parameter linear regression of ln(FRF) vs P:
    #     ln(FRF) = ln(FRF_ref) - RRM * P
    # then the reference FRF1 is recomputed by extrapolating to the
    # lowest pressure, which avoids letting a single noisy measurement
    # bias the slope estimate (cf. discussion of Fig. 3 of the paper).
    log_frf = np.log(frf_measured)
    slope, intercept = np.polyfit(pressures, log_frf, 1)
    rrm = float(-slope)
    frf1 = float(np.exp(intercept + slope * pressures[0]))
    return rrm, frf1


# ---------------------------------------------------------------------------
# Forward predictions
# ---------------------------------------------------------------------------
def predict_frf_overburden(frf1: float, rrm: float,
                           dP: float | np.ndarray) -> float | np.ndarray:
    """Eq. 15:  FRF(P2) = FRF1 * exp(-RRM * dP)."""
    return frf1 * np.exp(-rrm * np.asarray(dP, dtype=float))


def predict_ri_overburden(ri1: float, trm: float,
                          dP: float | np.ndarray) -> float | np.ndarray:
    """Eq. 20:  RI(P2) = RI1 * exp(-TRM * dP)."""
    return ri1 * np.exp(-trm * np.asarray(dP, dtype=float))


# ---------------------------------------------------------------------------
# Single-FRF model (compressibility based):   RRM ~ -m * (Cp - Cb)
# (Eq. 16 of the paper, neglecting WRM)
# ---------------------------------------------------------------------------
def single_frf_rrm(m_cementation: float, cp: float, cb: float = 0.0) -> float:
    """
    Compute RRM from cementation factor and pore / bulk
    compressibilities (Eq. 16, simplified by setting WRM = 0).

    Cp and Cb in 1/Pa, RRM in 1/Pa.
    """
    return -m_cementation * (cp - cb)


# ---------------------------------------------------------------------------
# Saturation-exponent variation with overburden (Eq. 28 / 29 of the paper)
# ---------------------------------------------------------------------------
def saturation_exponent_overburden(n0: float, trm: float, rrm: float,
                                   sw: float = 1.0) -> float:
    """
    The paper shows that the variation of n with pressure is small
    (typically < 8 %) and is given by

        n(P2) = n(P1) * (TRM - RRM) / RRM   (very small correction)

    Here we return the simple first-order estimate.  When TRM ~ RRM
    the result reduces to n0 (water-bearing rock invariant).
    """
    if abs(rrm) < 1e-30:
        return n0
    return n0 * (1.0 + (trm - rrm) / abs(rrm) * (1.0 - sw))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    """Synthetic-data test for module 3 (overburden FRF / RI)."""
    print("[article3] generating synthetic FRF(P) data with RRM=-2e-9 1/Pa")

    # Synthetic core sample - 6 pressure steps as in Fig. 3 of the paper
    rrm_true = -2.0e-9        # 1/Pa  (negative because Ro grows with P)
    frf1 = 25.0
    pressures = np.array([1, 50, 100, 200, 300, 400]) * 1e5  # Pa  (1..400 bar)
    dP = pressures - pressures[0]
    rng = np.random.default_rng(42)
    # 0.3% relative noise: consistent with the high-quality SCAL measurements
    # in the paper (R^2 > 0.997 over 6 pressure steps).
    frf_meas = predict_frf_overburden(frf1, rrm_true, dP) \
        * (1 + 0.003 * rng.standard_normal(dP.size))

    rrm_hat, frf1_hat = estimate_rrm_from_frf(pressures, frf_meas)
    print(f"           true RRM   = {rrm_true:.3e}")
    print(f"           fitted RRM = {rrm_hat:.3e}")
    assert abs(rrm_hat - rrm_true) / abs(rrm_true) < 0.10, \
        "RRM fit error > 10 %"

    print("[article3] testing forward FRF prediction ...")
    frf_pred = predict_frf_overburden(frf1_hat, rrm_hat, dP)
    rel_err = np.max(np.abs(frf_pred - frf_meas) / frf_meas)
    assert rel_err < 0.05, f"max relative FRF error too high: {rel_err:.3f}"

    print("[article3] testing Single-FRF compressibility formula ...")
    m_cem = 1.95
    cp = 1.5e-9   # 1/Pa
    cb = 0.2e-9   # 1/Pa
    rrm_single = single_frf_rrm(m_cem, cp, cb)
    print(f"           Single-FRF RRM = {rrm_single:.3e}  (m={m_cem}, "
          f"Cp={cp:.1e}, Cb={cb:.1e})")
    assert rrm_single < 0

    print("[article3] testing RI overburden prediction ...")
    trm_true = -2.5e-9
    ri1 = 5.0
    ri_pred = predict_ri_overburden(ri1, trm_true, dP)
    assert ri_pred[0] == ri1
    assert ri_pred[-1] > ri1, "RI must grow with overburden pressure"

    print("[article3] OK")


if __name__ == "__main__":
    test_all()
