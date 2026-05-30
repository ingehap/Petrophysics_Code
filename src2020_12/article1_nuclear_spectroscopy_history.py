"""
Article 1: A History of Nuclear Spectroscopy in Well Logging
Pemper (2020)
DOI: 10.30632/PJV61N6-2020a1

The opening review of this nuclear-spectroscopy special issue traces gamma-ray,
neutron-porosity, sigma (thermal-neutron capture), carbon/oxygen, and
spectral-gamma logging from the 1950s onward.  It is a historical review with no
numbered equations, so this module implements the canonical quantitative
relations the review discusses, so the rest of the issue's modules can build on
a shared nuclear-physics core.

Implements:

  - Macroscopic capture cross section  Sigma = sum_i N_i sigma_i  (capture units)
  - Capture units from the thermal-neutron decay (die-away) time  Sigma = 4550/tau
  - Number density from bulk density and atomic mass  N = rho*NA*w/A
  - Carbon/oxygen ratio as an oil-vs-water indicator
  - Spectral gamma-ray total from K, U, Th (API-style weighting)

Note: this issue's PDF text layer preserves equation numbers and variable
definitions but drops the typeset formula glyphs, so these are standard-form
nuclear-logging relations (the review itself publishes no numbered equations).
Capture cross section in capture units (1 c.u. = 1e-3 cm^-1); tau in us.
"""

import numpy as np

NA = 6.022e23            # Avogadro
V_THERMAL_CONV = 4550.0  # Sigma[c.u.] = 4550 / tau[us]  (2200 m/s thermal neutron)


# ---------------------------------------------- sigma / capture ---------

def number_density(rho_g_cc, weight_frac, atomic_mass):
    """Atomic number density  N = rho * NA * w / A  (atoms/cm^3)."""
    return rho_g_cc * NA * np.asarray(weight_frac, float) / np.asarray(atomic_mass, float)


def macroscopic_sigma(number_densities, micro_sigma_barns):
    """Macroscopic capture cross section in capture units.

        Sigma = sum_i N_i * sigma_i
    N in atoms/cm^3, micro sigma in barns (1e-24 cm^2) -> Sigma in cm^-1, then
    x1000 to capture units (1 c.u. = 1e-3 cm^-1).
    """
    N = np.asarray(number_densities, float)
    s = np.asarray(micro_sigma_barns, float) * 1e-24
    return float(np.sum(N * s) * 1e3)


def sigma_from_decay(tau_us):
    """Capture units from the thermal-neutron decay time  Sigma = 4550/tau."""
    return V_THERMAL_CONV / np.asarray(tau_us, float)


def decay_from_sigma(sigma_cu):
    """Inverse: thermal-neutron decay time (us) from sigma (c.u.)."""
    return V_THERMAL_CONV / np.asarray(sigma_cu, float)


# ---------------------------------------------- C/O and spectral GR -----

def carbon_oxygen_ratio(c_yield, o_yield):
    """Carbon/oxygen ratio (oil rises, water falls)  COR = Y_C / Y_O."""
    return np.asarray(c_yield, float) / np.asarray(o_yield, float)


def spectral_gr(k_pct, u_ppm, th_ppm):
    """Total (computed) gamma ray from K, U, Th  (API-style weighting).

        GR = 16.32*K(%) + 8.09*U(ppm) + 3.93*Th(ppm)
    """
    return 16.32 * k_pct + 8.09 * u_ppm + 3.93 * th_ppm


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: A History of Nuclear Spectroscopy")
    print("=" * 60)

    # Sigma of pure chlorine-free water from H and O number densities.  Water
    # sigma is dominated by hydrogen capture; the canonical value is ~22 c.u.
    rho_w = 1.0
    n_h = number_density(rho_w, 2 * 1.008 / 18.015, 1.008)
    n_o = number_density(rho_w, 16.0 / 18.015, 16.0)
    sig_w = macroscopic_sigma([n_h, n_o], [0.3326, 0.00019])   # H, O capture barns
    print(f"  sigma fresh water      = {sig_w:.1f} c.u.  (~22 expected)")
    assert 18.0 < sig_w < 26.0

    # Sigma <-> decay time round-trips through the 4550 conversion
    tau = decay_from_sigma(22.0)
    print(f"  tau at 22 c.u.         = {tau:.1f} us")
    assert abs(sigma_from_decay(tau) - 22.0) < 1e-9
    # higher sigma (more capture) -> faster decay (shorter tau)
    assert decay_from_sigma(40.0) < decay_from_sigma(20.0)

    # C/O ratio rises for oil-bearing rock
    cor_oil = carbon_oxygen_ratio(0.45, 0.55)
    cor_wat = carbon_oxygen_ratio(0.20, 0.80)
    print(f"  C/O oil / water        = {cor_oil:.3f} / {cor_wat:.3f}")
    assert cor_oil > cor_wat

    # Spectral GR: a hot shale (K + Th) reads higher than a clean sand
    gr_shale = spectral_gr(3.5, 4.0, 14.0)
    gr_sand = spectral_gr(0.3, 0.5, 2.0)
    print(f"  GR shale / sand        = {gr_shale:.0f} / {gr_sand:.0f} API")
    assert gr_shale > gr_sand
    print("  PASS")
    return {"sigma_water": sig_w, "tau_22cu": float(tau),
            "cor_oil": cor_oil, "gr_shale": gr_shale}


if __name__ == "__main__":
    test_all()
