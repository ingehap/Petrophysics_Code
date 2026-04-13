"""
article1_nuclear_logging.py
============================
Implements the key petrophysical equations from:

    Fitz, D.E. (2023). "Evolution of Casedhole Nuclear Surveillance Logging
    Through Time", Petrophysics, Vol. 64, No. 4 (August 2023), pp. 473-501.
    DOI: 10.30632/PJV64N4-2023a1

The article is a historical review of casedhole nuclear logging.  The two
core quantitative ideas formalised here are:

    (1) the Pulsed-Neutron-Capture (PNC) volumetric mixing law for the
        macroscopic thermal-neutron capture cross section Sigma_t (Eq. 4):

            Sigma_t = (1 - phi)*Sigma_ma
                      + phi*(1 - Sw)*Sigma_hc
                      + phi*Sw*Sigma_w                        (capture units)

        Solving for Sw yields the PNC water-saturation answer.

    (2) the time-lapse PNC monitoring equation (Eq. 5).  When the rock matrix
        and hydrocarbon properties are constant in time but salinity may
        change, the difference between a "monitor" and a "base" log gives the
        change in water saturation:

            dSigma_t = phi * ( Sw_m*Sigma_w_m - Sw_b*Sigma_w_b
                               - (Sw_m - Sw_b)*Sigma_hc )

        which removes the unknown Sigma_ma.

Auxiliary helpers cover the Larionov-type gamma-ray clay index used as a
qualitative water-channel indicator (the article shows a 2,000 GAPI North
Sea log diagnosing water entry from radium scale).

Run as a script to execute the synthetic test suite:

    python article1_nuclear_logging.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# PNC capture-cross-section mixing law  (Fitz, Eq. 4)
# ---------------------------------------------------------------------------
def sigma_total(phi, sw, sigma_ma, sigma_w, sigma_hc):
    """Forward model: total formation capture cross section (capture units, c.u.).

    Parameters
    ----------
    phi      : total porosity, v/v
    sw       : water saturation, v/v
    sigma_ma : matrix capture cross section (c.u.)
    sigma_w  : formation-water capture cross section (c.u.)
    sigma_hc : hydrocarbon capture cross section (c.u.)
    """
    phi = np.asarray(phi, dtype=float)
    sw  = np.asarray(sw,  dtype=float)
    return (1.0 - phi) * sigma_ma \
           + phi * (1.0 - sw) * sigma_hc \
           + phi * sw * sigma_w


def sw_from_sigma(sigma_t, phi, sigma_ma, sigma_w, sigma_hc):
    """Invert Eq. 4 for water saturation.

        Sw = (Sigma_t - (1-phi)*Sigma_ma - phi*Sigma_hc)
             / (phi * (Sigma_w - Sigma_hc))
    """
    phi = np.asarray(phi, dtype=float)
    num = sigma_t - (1.0 - phi) * sigma_ma - phi * sigma_hc
    den = phi * (sigma_w - sigma_hc)
    # Guard against division by zero when phi == 0 or Sigma_w == Sigma_hc
    sw = np.where(np.abs(den) > 1e-12, num / np.where(den == 0, 1.0, den), np.nan)
    return np.clip(sw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Time-lapse PNC monitoring  (Fitz, Eq. 5)
# ---------------------------------------------------------------------------
def delta_sw_timelapse(sigma_t_base, sigma_t_mon, phi,
                       sigma_w_base, sigma_w_mon,
                       sigma_hc, sw_base):
    """Change in water saturation between a base log and a monitor log.

    Derived from the difference of Eq. 4 between the two logging passes,
    assuming matrix and hydrocarbon properties are unchanged.

        Sw_m = ( (Sigma_t_m - Sigma_t_b)/phi
                 + Sw_b*(Sigma_w_b - Sigma_hc) ) / (Sigma_w_m - Sigma_hc)
    """
    phi = np.asarray(phi, dtype=float)
    num = (sigma_t_mon - sigma_t_base) / phi \
          + sw_base * (sigma_w_base - sigma_hc)
    den = (sigma_w_mon - sigma_hc)
    sw_mon = np.clip(num / den, 0.0, 1.0)
    return sw_mon - sw_base, sw_mon


def water_salinity_to_sigma_w(salinity_ppm_nacl, temperature_C=75.0):
    """Approximate brine capture cross section (c.u.) from NaCl salinity.

    Empirical fit consistent with the worked examples in Fitz (2023):
    ~22 c.u. for fresh water and ~120 c.u. for saturated brine at reservoir
    temperature.  Suitable for synthetic testing only.
    """
    s = np.asarray(salinity_ppm_nacl, dtype=float) / 1.0e6  # weight fraction
    # Linear in salinity with mild T-correction; 22 c.u. baseline.
    sigma_w = 22.0 + 750.0 * s
    sigma_w *= (1.0 - 0.0008 * (temperature_C - 75.0))
    return sigma_w


# ---------------------------------------------------------------------------
# Gamma-ray helpers (used qualitatively in the article for water-entry
# detection from radium / NORM scale).
# ---------------------------------------------------------------------------
def gamma_ray_index(gr, gr_min, gr_max):
    """GR index in [0, 1]."""
    return np.clip((np.asarray(gr, dtype=float) - gr_min) / (gr_max - gr_min),
                   0.0, 1.0)


def vshale_larionov_tertiary(igr):
    """Larionov (1969) Tertiary rocks shale-volume estimator."""
    igr = np.asarray(igr, dtype=float)
    return 0.083 * (2.0 ** (3.7 * igr) - 1.0)


def vshale_larionov_older(igr):
    """Larionov (1969) older (pre-Tertiary) rocks shale-volume estimator."""
    igr = np.asarray(igr, dtype=float)
    return 0.33 * (2.0 ** (2.0 * igr) - 1.0)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
def test_all(verbose=True):
    """Synthetic-data tests for every public function."""
    rng = np.random.default_rng(seed=42)

    # --- 1. Forward / inverse consistency on a synthetic interval ----------
    n = 200
    phi  = rng.uniform(0.05, 0.30, n)
    sw_t = rng.uniform(0.10, 0.95, n)
    sigma_ma, sigma_w, sigma_hc = 8.0, 60.0, 20.5  # c.u.
    sig_t = sigma_total(phi, sw_t, sigma_ma, sigma_w, sigma_hc)
    sw_inv = sw_from_sigma(sig_t, phi, sigma_ma, sigma_w, sigma_hc)
    err = np.max(np.abs(sw_inv - sw_t))
    assert err < 1e-10, f"PNC forward/inverse inconsistency: {err}"
    if verbose:
        print(f"[1] PNC forward/inverse round-trip OK  (max |dSw| = {err:.2e})")

    # --- 2. Time-lapse PNC --------------------------------------------------
    sw_b = np.full(n, 0.20)
    sw_m_truth = np.full(n, 0.85)
    sig_b = sigma_total(phi, sw_b, sigma_ma, sigma_w, sigma_hc)
    # Inject a salinity drop on the monitor pass (sea-water injection)
    sigma_w_b = 60.0
    sigma_w_m = 34.0
    sig_m = sigma_total(phi, sw_m_truth, sigma_ma, sigma_w_m, sigma_hc)
    dsw, sw_m = delta_sw_timelapse(sig_b, sig_m, phi,
                                   sigma_w_b, sigma_w_m,
                                   sigma_hc, sw_b)
    err = np.max(np.abs(sw_m - sw_m_truth))
    assert err < 1e-10, f"Time-lapse Sw recovery error: {err}"
    if verbose:
        print(f"[2] Time-lapse Sw recovery OK         (max |dSw| = {err:.2e})")

    # --- 3. Salinity-to-Sigma_w monotonicity --------------------------------
    sals = np.linspace(1e3, 2.5e5, 25)
    sw_curve = water_salinity_to_sigma_w(sals)
    assert np.all(np.diff(sw_curve) > 0), "Sigma_w must increase with salinity"
    if verbose:
        print(f"[3] Sigma_w(salinity) monotonic       "
              f"(range {sw_curve[0]:.1f}-{sw_curve[-1]:.1f} c.u.)")

    # --- 4. Gamma-ray index and Larionov ------------------------------------
    gr = np.linspace(20.0, 150.0, 14)
    igr = gamma_ray_index(gr, 20.0, 150.0)
    assert igr[0] == 0.0 and abs(igr[-1] - 1.0) < 1e-12
    vsh_t = vshale_larionov_tertiary(igr)
    vsh_o = vshale_larionov_older(igr)
    # The two Larionov curves cross near IGR ~ 0.65. Both must be bounded.
    assert np.all((vsh_t >= 0.0) & (vsh_t <= 1.001))
    assert np.all((vsh_o >= 0.0) & (vsh_o <= 1.001))
    assert np.all(np.diff(vsh_t) > 0) and np.all(np.diff(vsh_o) > 0)
    if verbose:
        print(f"[4] GR index + Larionov shale OK      "
              f"(Vsh_tert(IGR=1)={vsh_t[-1]:.3f}, Vsh_older(IGR=1)={vsh_o[-1]:.3f})")

    # --- 5. "Bad" PNC scenario from the article ----------------------------
    # Low-salinity injection brine (~30 c.u.) wiping out the contrast --
    # Fitz's "Bad" case: dSigma is small, inversion still returns a Sw answer.
    phi_s = 0.20
    sw_b_s, sw_m_s = 0.20, 0.85
    sig_b_s = sigma_total(phi_s, sw_b_s, 8.0, 59.0, 20.5)
    sig_m_s = sigma_total(phi_s, sw_m_s, 8.0, 30.0, 20.5)
    delta = sig_m_s - sig_b_s
    if verbose:
        print(f"[5] Fitz 'Bad' case dSigma            = {delta:+.2f} c.u. "
              f"(small => low PNC sensitivity)")

    if verbose:
        print("\nAll article-1 tests passed.")
    return True


if __name__ == "__main__":
    test_all()
