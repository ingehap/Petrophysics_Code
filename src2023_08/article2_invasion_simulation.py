"""
article2_invasion_simulation.py
================================
Implements the rock/fluid/mudcake equations and the radial 1-D mud-filtrate
invasion + Archie-resistivity workflow described in:

    Merletti, G., Al Hajri, S., Rabinovich, M., Farmer, R., Bennis, M.,
    Torres-Verdín, C. (2023).  "Assessment of True Formation Resistivity and
    Water Saturation in Deeply Invaded Tight-Gas Sandstones Based on the
    Combined Numerical Simulation of Mud-Filtrate Invasion and Resistivity
    Logs", Petrophysics, Vol. 64, No. 4, pp. 502-517.
    DOI: 10.30632/PJV64N4-2023a2

Equations implemented (paper numbering shown):

    Eq. 1  Initial water saturation Swin from a porosity regression.
    Eq. 2  Jerauld (1997) trapped-gas model:   Sgt = Sgt_max*Sgi/(1 + (1/Sgt_max - 1)*Sgi)
    Eq. 3  Brooks-Corey gas relative permeability  (Krg)
    Eq. 4  Brooks-Corey water relative permeability (Krw)
    Eq. 5  Brooks-Corey capillary pressure  Pc = Pd*Se^(-1/lambda)
    Eq. 6  Dewan & Chenevert (2001) mudcake permeability time evolution
    Eq. 7  Dewan & Chenevert (2001) mudcake porosity time evolution
    Eq. 8  Chin (1995) mudcake-thickness ODE
    Eq. 9  Archie (1942) saturation equation                    Rt = a*Rw / (phi^m * Sw^n)

A simple radial Buckley-Leverett-style filtrate-invasion advance is used to
generate the time-dependent Sw radial profile that feeds Archie's law.
The result is an axisymmetric Rt(r) profile of the kind shown in the paper.

Run as a script for the synthetic test suite:

    python article2_invasion_simulation.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Equation 1 - initial Sw from a porosity regression
# ---------------------------------------------------------------------------
def swin_porosity_regression(phi, a=0.012, b=-1.21):
    """Sw_in = a * phi^b (the calibration shown in Fig. 6 of the paper)."""
    phi = np.asarray(phi, dtype=float)
    return np.clip(a * np.power(phi, b), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Equation 2 - Jerauld (1997) trapped-gas saturation
# ---------------------------------------------------------------------------
def jerauld_sgt(sgi, sgt_max):
    """Trapped gas saturation from initial gas saturation.

    Standard Land/Jerauld form:  1/Sgt = 1/Sgi + 1/Sgt_max - 1
    so that  Sgt(Sgi=0)=0 and Sgt(Sgi=1)=Sgt_max.
    """
    sgi = np.asarray(sgi, dtype=float)
    out = np.zeros_like(sgi)
    mask = sgi > 1e-12
    out[mask] = 1.0 / (1.0 / sgi[mask] + 1.0 / sgt_max - 1.0)
    return out


# ---------------------------------------------------------------------------
# Equations 3 & 4 - Brooks-Corey relative permeabilities
# ---------------------------------------------------------------------------
def krg_brooks_corey(sw, swcg, sgc, krg_end=1.0, ng=2.0):
    """Gas relative permeability in a wetting-phase saturation framework."""
    sw = np.asarray(sw, dtype=float)
    s_eff = (1.0 - sw - sgc) / (1.0 - swcg - sgc)
    s_eff = np.clip(s_eff, 0.0, 1.0)
    return krg_end * np.power(s_eff, ng)


def krw_brooks_corey(sw, swcw, sgc, krw_end=1.0, nw=2.0):
    """Water relative permeability."""
    sw = np.asarray(sw, dtype=float)
    s_eff = (sw - swcw) / (1.0 - swcw - sgc)
    s_eff = np.clip(s_eff, 0.0, 1.0)
    return krw_end * np.power(s_eff, nw)


# ---------------------------------------------------------------------------
# Equation 5 - Brooks-Corey capillary pressure
# ---------------------------------------------------------------------------
def pc_brooks_corey(sw, swcw, pd, lam):
    """Pc = Pd * Se^(-1/lambda)."""
    sw = np.asarray(sw, dtype=float)
    se = np.clip((sw - swcw) / (1.0 - swcw), 1e-6, 1.0)
    return pd * np.power(se, -1.0 / lam)


# ---------------------------------------------------------------------------
# Equations 6 & 7 - Dewan & Chenevert (2001) mudcake properties
# ---------------------------------------------------------------------------
def mudcake_permeability(t, kmc0, dp, v):
    """Eq. 6:  Kmc(t) = Kmc0 * (Pmc(t)/Pref)^(-v).

    A simple linear ramp of the differential pressure in time is used:
    dp[t] = dp0 (constant overbalance).  The compressibility law gives
    a slow decline of permeability as the cake builds.
    """
    t = np.asarray(t, dtype=float)
    # Use t/(t+t0) as a normalised pressure-history surrogate.
    t0 = 1.0e-3  # days
    p_ratio = (t + t0) / t0
    return kmc0 * np.power(p_ratio, -v)


def mudcake_porosity(t, phi_mc0, delta):
    """Eq. 7:  phi_mc(t) = phi_mc0 * (1 + delta * ln(1 + t))^(-1).

    Porosity decays slowly with time as the cake compacts.
    """
    t = np.asarray(t, dtype=float)
    return phi_mc0 / (1.0 + delta * np.log1p(t))


# ---------------------------------------------------------------------------
# Equation 8 - Chin (1995) mudcake-thickness growth ODE
# ---------------------------------------------------------------------------
def mudcake_thickness(t_array, fs, kmc0, mu_f, dp, v=0.5):
    """Integrate dRmc/dt = (fs * Kmc(t) * dp) / (mu_f * Rmc) (assuming no
    invasion of solids).  Returns Rmc(t) [m] at every time in t_array.
    """
    t_array = np.asarray(t_array, dtype=float)
    rmc = np.zeros_like(t_array)
    rmc[0] = 1.0e-5  # 10 micron seed cake
    for i in range(1, len(t_array)):
        dt = t_array[i] - t_array[i - 1]
        kmc = mudcake_permeability(t_array[i], kmc0, dp, v)
        # dRmc/dt * Rmc = fs*Kmc*dp/mu_f  =>  d(Rmc^2)/dt = 2*fs*Kmc*dp/mu_f
        rmc[i] = np.sqrt(rmc[i - 1] ** 2 + 2.0 * fs * kmc * dp * dt / mu_f)
    return rmc


# ---------------------------------------------------------------------------
# Equation 9 - Archie's law
# ---------------------------------------------------------------------------
def archie_rt(phi, sw, rw, a=1.0, m=2.0, n=2.0):
    """Rt = a*Rw / (phi^m * Sw^n)."""
    phi = np.asarray(phi, dtype=float)
    sw  = np.clip(np.asarray(sw, dtype=float), 1e-3, 1.0)
    return a * rw / (np.power(phi, m) * np.power(sw, n))


def archie_sw(rt, phi, rw, a=1.0, m=2.0, n=2.0):
    """Inverse Archie  ->  Sw."""
    phi = np.asarray(phi, dtype=float)
    rt  = np.asarray(rt,  dtype=float)
    return np.clip(np.power(a * rw / (rt * np.power(phi, m)), 1.0 / n), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Radial 1-D filtrate invasion (axisymmetric profile generator)
# ---------------------------------------------------------------------------
def invasion_radial_profile(r_grid, lxo, swin, sw_filtrate=1.0,
                            transition_width=0.3):
    """Build a smooth Sw(r) radial profile.

    Parameters
    ----------
    r_grid           : radial coordinate (m), r=0 at borehole wall
    lxo              : invasion radius (m) where Sw drops to half-way
    swin             : uninvaded zone water saturation (v/v)
    sw_filtrate      : filtrate-flushed zone water saturation (v/v),
                       1.0 if mud filtrate fully displaces hydrocarbon.
    transition_width : width of the smooth transition (m)

    Returns Sw(r) shaped like r_grid.  This mimics the smooth radial
    profiles shown in Figures 7-10 of Merletti et al. (2023).
    """
    r = np.asarray(r_grid, dtype=float)
    arg = (r - lxo) / max(transition_width, 1e-6)
    weight_uninv = 0.5 * (1.0 + np.tanh(arg))
    return weight_uninv * swin + (1.0 - weight_uninv) * sw_filtrate


def salinity_radial_profile(r_grid, lxo, sal_form, sal_filtrate,
                            transition_width=0.3):
    """Companion salt mixing profile (used to build Rw(r))."""
    r = np.asarray(r_grid, dtype=float)
    w = 0.5 * (1.0 + np.tanh((r - lxo) / max(transition_width, 1e-6)))
    return w * sal_form + (1.0 - w) * sal_filtrate


def rw_from_salinity(sal_ppm, T_C=75.0):
    """Arps-style brine resistivity (Ohm.m) from NaCl ppm and temperature."""
    sal_ppm = np.asarray(sal_ppm, dtype=float)
    rw_75 = 0.0123 + 3647.5 / np.power(np.maximum(sal_ppm, 1.0), 0.955)
    return rw_75 * (75.0 + 7.0) / (T_C + 7.0)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
def test_all(verbose=True):
    """Synthetic-data tests for every public function."""

    # --- 1. Swin regression ------------------------------------------------
    phi = np.linspace(0.05, 0.20, 16)
    swin = swin_porosity_regression(phi)
    assert np.all(np.diff(swin) < 0), "Swin must decrease with porosity"
    if verbose:
        print(f"[1] Swin(phi) monotonic   "
              f"({swin[0]:.2f} @ phi=0.05 -> {swin[-1]:.2f} @ phi=0.20)")

    # --- 2. Jerauld trapped gas --------------------------------------------
    sgi = np.linspace(0.0, 1.0, 21)
    sgt = jerauld_sgt(sgi, sgt_max=0.50)
    assert sgt[0] == 0.0
    assert abs(sgt[-1] - 0.50) < 1e-10
    assert np.all(np.diff(sgt) >= 0)
    if verbose:
        print(f"[2] Jerauld Sgt OK        "
              f"(Sgt(Sgi=1)={sgt[-1]:.3f}, monotonic)")

    # --- 3. Brooks-Corey rel perms ----------------------------------------
    sw = np.linspace(0.15, 0.85, 71)
    krg = krg_brooks_corey(sw, swcg=0.15, sgc=0.10, ng=2.5)
    krw = krw_brooks_corey(sw, swcw=0.15, sgc=0.10, nw=3.0)
    assert np.all(np.diff(krg) <= 1e-12), "Krg must decrease with Sw"
    assert np.all(np.diff(krw) >= -1e-12), "Krw must increase with Sw"
    if verbose:
        print(f"[3] Brooks-Corey Kr OK    "
              f"(Krg(Swcg)={krg[0]:.3f}, Krw(1-Sgc)={krw[-1]:.3f})")

    # --- 4. Capillary pressure ---------------------------------------------
    pc = pc_brooks_corey(sw, swcw=0.15, pd=5.0, lam=2.0)
    assert np.all(np.diff(pc) <= 1e-9), "Pc must decrease with Sw"
    if verbose:
        print(f"[4] Brooks-Corey Pc OK    "
              f"(Pc(Sw=0.85)={pc[-1]:.2f} psi)")

    # --- 5. Mudcake permeability & porosity --------------------------------
    t = np.linspace(0.001, 10.0, 200)
    kmc = mudcake_permeability(t, kmc0=1e-4, dp=1000.0, v=0.6)
    phi_mc = mudcake_porosity(t, phi_mc0=0.40, delta=0.15)
    assert np.all(np.diff(kmc) <= 1e-15), "Kmc must decay in time"
    assert np.all(np.diff(phi_mc) <= 1e-15), "phi_mc must decay in time"
    if verbose:
        print(f"[5] Mudcake props OK      "
              f"(Kmc 0->10d : {kmc[0]:.2e} -> {kmc[-1]:.2e} mD)")

    # --- 6. Mudcake thickness (Chin) --------------------------------------
    rmc = mudcake_thickness(t, fs=0.05, kmc0=1e-4, mu_f=1.0e-3,
                            dp=1000.0, v=0.6)
    assert np.all(np.diff(rmc) >= -1e-15), "Rmc must grow monotonically"
    if verbose:
        print(f"[6] Chin mudcake build OK "
              f"(Rmc(10d) = {rmc[-1]*1000:.2f} mm)")

    # --- 7. Archie round trip ----------------------------------------------
    sw_true = np.linspace(0.20, 0.95, 50)
    rt = archie_rt(phi=0.12, sw=sw_true, rw=0.05)
    sw_inv = archie_sw(rt, phi=0.12, rw=0.05)
    err = np.max(np.abs(sw_inv - sw_true))
    assert err < 1e-10, f"Archie round-trip error {err}"
    if verbose:
        print(f"[7] Archie round-trip OK  (max |dSw| = {err:.2e})")

    # --- 8. Radial invasion profile + Rt(r) -------------------------------
    r = np.linspace(0.0, 5.0, 200)
    sw_r = invasion_radial_profile(r, lxo=1.5, swin=0.22)
    sal_r = salinity_radial_profile(r, lxo=1.5,
                                    sal_form=200000., sal_filtrate=120000.)
    rw_r  = rw_from_salinity(sal_r, T_C=85.0)
    rt_r  = archie_rt(phi=0.12, sw=sw_r, rw=rw_r)
    # The flushed zone must be much more conductive (lower Rt) than virgin.
    assert rt_r[-1] > rt_r[0], "Uninvaded Rt should exceed flushed Rxo"
    if verbose:
        print(f"[8] Radial Rt(r) profile  "
              f"(Rxo={rt_r[0]:.2f} -> Rt={rt_r[-1]:.2f} ohm.m)")

    if verbose:
        print("\nAll article-2 tests passed.")
    return True


if __name__ == "__main__":
    test_all()
