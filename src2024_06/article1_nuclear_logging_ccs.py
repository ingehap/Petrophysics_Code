"""
article1_nuclear_logging_ccs.py
================================

Implementation of the main ideas from:

    Badruzzaman, A. (2024). "Nuclear Logging in Geological Probing for a
    Low-Carbon Energy Future - A New Frontier?"  Petrophysics 65(3), 274-301.
    DOI: 10.30632/PJV65N3-2024a1

The paper is a broad review of how nuclear logging techniques developed for
oil and gas can be applied to CO2 sequestration (CCS) monitoring, nuclear
repositories, and geothermal systems.  The technical core that we implement
here consists of the three most concrete quantitative ideas in the paper:

1.  The pulsed-neutron capture (PNC) sigma technique used to track an
    injected CO2 plume from the time decay of thermal neutron counts
    (Eq. 1 of the paper and Figs. 3-5):

        N(t) = N0 * exp( -Sigma * v * t )

    so Sigma is obtained from the slope of log(N) vs. t.  Sigma is
    reported in capture units (c.u.), where 1 c.u. = 1e-3 cm^-1.

2.  A diffusion correction that must be added to the "intrinsic" sigma
    when a formation is gas-filled (Appendix 1 of the paper).  In gas,
    neutrons are less efficiently removed by absorption and a diffusion
    term Sigma_D becomes important.

3.  A carbon/oxygen (C/O) ratio response that is used to distinguish CO2
    from other gases in an aquifer (Fig. 6 of the paper).  The paper
    shows a roughly linear C/O vs. CO2 volume fraction relation; we
    reproduce it with a simple linear mixing model.

Everything is implemented with numpy only, and can be run as a script.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. PNC sigma technique ----------------------------------------------------
# ---------------------------------------------------------------------------

THERMAL_NEUTRON_VELOCITY_CM_S = 2.2e5  # velocity of 0.025 eV neutrons, cm/s


def pnc_decay(time_us: np.ndarray, n0: float, sigma_cu: float,
              background: float = 0.0) -> np.ndarray:
    """Forward model of a pulsed-neutron capture decay curve.

    Parameters
    ----------
    time_us : array of times after the neutron burst in microseconds.
    n0      : initial count at t = 0 (arbitrary units).
    sigma_cu: formation macroscopic capture cross section in capture units
              (1 c.u. = 1e-3 cm^-1).
    background : constant background count rate.

    Returns
    -------
    Array of counts N(t) = N0 * exp(-Sigma * v * t) + background.
    """
    sigma_cm_inv = sigma_cu * 1e-3                      # c.u. -> cm^-1
    decay_rate = sigma_cm_inv * THERMAL_NEUTRON_VELOCITY_CM_S  # s^-1
    t_s = np.asarray(time_us) * 1e-6
    return n0 * np.exp(-decay_rate * t_s) + background


def sigma_from_decay(time_us: np.ndarray, counts: np.ndarray,
                     fit_window: tuple[float, float] | None = None) -> float:
    """Recover Sigma (in c.u.) from a measured PNC decay curve.

    Uses the linearised form  ln(N) = ln(N0) - Sigma * v * t  and a simple
    least-squares fit on the gate window where the decay is exponential.
    """
    t_us = np.asarray(time_us, dtype=float)
    n = np.asarray(counts, dtype=float)
    mask = n > 0
    if fit_window is not None:
        t0, t1 = fit_window
        mask &= (t_us >= t0) & (t_us <= t1)
    t_s = t_us[mask] * 1e-6
    log_n = np.log(n[mask])
    slope, _intercept = np.polyfit(t_s, log_n, 1)       # s^-1
    sigma_cm_inv = -slope / THERMAL_NEUTRON_VELOCITY_CM_S
    return sigma_cm_inv * 1e3                           # -> c.u.


# ---------------------------------------------------------------------------
# 2.  Mixing model for the formation sigma in a CCS setting ------------------
# ---------------------------------------------------------------------------
#
#     Sigma_f = (1-phi)*Sigma_matrix + phi*(S_w*Sigma_w + (1-S_w)*Sigma_gas)
#     + Sigma_diffusion correction in gas-filled formations.
# ---------------------------------------------------------------------------

def formation_sigma(porosity: float, sw: float,
                    sigma_matrix: float = 10.0,
                    sigma_water: float = 55.0,
                    sigma_gas: float = 0.03,
                    diffusion_factor: float = 0.0) -> float:
    """Simple volumetric sigma model for a CO2-in-brine CCS aquifer.

    Default end-member values follow the Frio Brine Pilot described in
    the paper (Sakurai et al., 2006): Sigma_brine ~ 55 c.u., Sigma_CO2 ~ 0.03.

    `diffusion_factor` adds a gas-saturation dependent correction that
    mimics the effect reported in Badruzzaman et al. (2010) -- in a
    gas-rich pore space the apparent sigma is higher than the intrinsic
    volumetric sigma because of neutron diffusion / transport.
    """
    sg = 1.0 - sw
    sigma_fluid = sw * sigma_water + sg * sigma_gas
    sigma_int = (1.0 - porosity) * sigma_matrix + porosity * sigma_fluid
    # Diffusion correction scales with gas saturation, as seen in Fig. 4
    # of the paper (flat region above Sg ~ 0.5).
    sigma_d = diffusion_factor * sg ** 2
    return sigma_int + sigma_d


def co2_saturation_from_sigma(sigma_obs: float, porosity: float,
                              sigma_matrix: float = 10.0,
                              sigma_water: float = 55.0,
                              sigma_gas: float = 0.03) -> float:
    """Invert the volumetric mixing model for CO2 saturation Sg = 1 - Sw."""
    sigma_rock = (1.0 - porosity) * sigma_matrix
    # Sigma_obs = sigma_rock + phi*(Sw*sw_term + (1-Sw)*gas_term)
    # Solve for Sw then return Sg.
    num = (sigma_obs - sigma_rock) / porosity - sigma_gas
    den = sigma_water - sigma_gas
    sw = np.clip(num / den, 0.0, 1.0)
    return 1.0 - sw


# ---------------------------------------------------------------------------
# 3. C/O ratio CO2 indicator -------------------------------------------------
# ---------------------------------------------------------------------------

def co_ratio(co2_volume_fraction: float, breakthrough: bool = False,
             baseline: float = 0.25, slope: float = 0.35) -> float:
    """Far-detector C/O ratio vs. volume fraction of CO2 in aquifer fluid.

    Mimics Fig. 6 of Badruzzaman (2024): without breakthrough into the
    well the C/O response is weak below ~50% CO2; with breakthrough the
    sensitivity is pushed down to ~10% CO2.
    """
    f = np.clip(co2_volume_fraction, 0.0, 1.0)
    if breakthrough:
        return baseline + slope * f
    # without breakthrough, first 50% of CO2 produces only half the slope
    return baseline + slope * np.where(f < 0.5, 0.3 * f, 0.15 + 1.3 * (f - 0.5))


# ---------------------------------------------------------------------------
# Test harness ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_all(verbose: bool = True) -> None:
    """Exercise every function with synthetic data and assert the results."""
    rng = np.random.default_rng(42)

    # (a) Forward/inverse PNC sigma consistency
    t = np.linspace(50, 1500, 80)                       # microseconds
    true_sigma = 25.0                                    # c.u.
    clean = pnc_decay(t, n0=1e4, sigma_cu=true_sigma, background=5.0)
    noisy = clean + rng.normal(0, np.sqrt(clean))        # Poisson-ish
    est_sigma = sigma_from_decay(t, noisy - 5.0, fit_window=(100, 800))
    assert abs(est_sigma - true_sigma) < 1.5, (
        f"PNC sigma recovery failed: {est_sigma:.2f} vs {true_sigma:.2f}")

    # (b) Formation sigma monotonically decreases with CO2 saturation
    sigmas = [formation_sigma(0.25, sw) for sw in np.linspace(0, 1, 11)]
    assert all(x <= y for x, y in zip(sigmas, sigmas[1:])), \
        "Sigma should increase with Sw"

    # (c) Inversion round-trip: sigma -> Sg -> sigma
    for sg_true in [0.0, 0.3, 0.6, 0.9]:
        sig = formation_sigma(0.22, sw=1 - sg_true)
        sg_est = co2_saturation_from_sigma(sig, 0.22)
        assert abs(sg_est - sg_true) < 1e-6, (
            f"Sigma inversion failed: {sg_est:.3f} vs {sg_true:.3f}")

    # (d) Diffusion correction makes the apparent sigma higher at high Sg
    sig_plain = formation_sigma(0.25, sw=0.1, diffusion_factor=0.0)
    sig_diff = formation_sigma(0.25, sw=0.1, diffusion_factor=4.0)
    assert sig_diff > sig_plain, "Diffusion correction should raise sigma"

    # (e) C/O ratio responds more sharply with breakthrough
    r_nb = co_ratio(0.2, breakthrough=False)
    r_bt = co_ratio(0.2, breakthrough=True)
    assert r_bt > r_nb, "Breakthrough should enhance C/O sensitivity"

    if verbose:
        print("Article 1 (Nuclear logging for CCS): all tests passed.")
        print(f"  recovered sigma  = {est_sigma:6.2f} c.u. (true {true_sigma})")
        print(f"  sigma(Sw=1)      = {formation_sigma(0.25, 1.0):6.2f} c.u.")
        print(f"  sigma(Sw=0)      = {formation_sigma(0.25, 0.0):6.2f} c.u.")
        print(f"  C/O nb @20%CO2   = {r_nb:6.3f}")
        print(f"  C/O bt @20%CO2   = {r_bt:6.3f}")


if __name__ == "__main__":
    test_all()
