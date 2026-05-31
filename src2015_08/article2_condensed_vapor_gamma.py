"""
Article 2: In-Situ Evaluation of Vapor Properties Using Condensed Vapor Gamma
O'Sullivan (2015)
Reference: Petrophysics Vol. 56, No. 4 (August 2015), pp. 334-345
DOI: none assigned (this issue predates SPWLA DOI assignment)

In heavy-oil steamfloods, gamma-ray logs through hot vapor-filled sands can
exceed 1,000 (even 10,000) GAPI - far above the liquid-filled response.  The
mechanism (condensed vapor gamma, CVG) is radon-222: radon is ~10x more soluble
in hydrocarbon vapor than in water, so a vapor cloud carries a high radon
partial pressure; when vapor condenses around the chilled wellbore the droplets
concentrate the energetic radon (by factors of >=200), whose decay produces the
anomalous gamma signal.

Implements:

  - Clausius-Clapeyron vapor pressure vs. temperature
  - Radon decay constant, radioactive decay and activity (222Rn, 3.82-day half-life)
  - Radon vapor/liquid partition (Ostwald solubility ratio)
  - Condensation concentration factor and resulting gamma amplitude

Note: this is an observational/mechanistic paper with no display equations; the
relations below are the standard physics it relies on (Clausius-Clapeyron,
radioactive decay, Ostwald solubility).  Pressure in Pa, temperature in K, time
in days, activity in Bq.
"""

import numpy as np

R_GAS = 8.314                 # J/(mol*K)
RN222_HALF_LIFE_DAYS = 3.8235 # days


# ---------------------------------------------- vapor pressure --------------

def clausius_clapeyron(p_ref, t_ref, temperature, latent_heat, molar_mass):
    """Vapor pressure vs. temperature (Clausius-Clapeyron)

        P = P_ref * exp[ -(L*M/R)*(1/T - 1/T_ref) ],

    with L the specific latent heat (J/kg) and M the molar mass (kg/mol); a
    higher-vapor-pressure fluid (butane) populates the vapor phase more than
    water at a given temperature.
    """
    dh_molar = latent_heat * molar_mass
    return p_ref * np.exp(-(dh_molar / R_GAS) * (1.0 / temperature - 1.0 / t_ref))


# ---------------------------------------------- radon decay --------------

def decay_constant(half_life=RN222_HALF_LIFE_DAYS):
    """Radioactive decay constant  lambda = ln(2)/half_life."""
    return np.log(2.0) / half_life


def radioactive_decay(n0, time, half_life=RN222_HALF_LIFE_DAYS):
    """Remaining radon atoms  N = N0*exp(-lambda*t)."""
    return n0 * np.exp(-decay_constant(half_life) * np.asarray(time, float))


def activity(n, half_life=RN222_HALF_LIFE_DAYS):
    """Radon activity (decays per unit time)  A = lambda*N,

    proportional to the gamma signal from the radon progeny.
    """
    return decay_constant(half_life) * n


# ---------------------------------------------- partition / concentration --------------

def radon_partition(c_vapor, ostwald_ratio):
    """Radon concentration in the condensed liquid from the vapor concentration

        C_liquid = ostwald_ratio * C_vapor,

    with the Ostwald solubility ratio ~10x larger for hydrocarbon than water.
    """
    return ostwald_ratio * c_vapor


def condensed_gamma_amplitude(c_vapor, ostwald_ratio, concentration_factor, gamma_per_unit):
    """Condensed-vapor-gamma amplitude (GAPI, relative)

        GR ~ gamma_per_unit * (ostwald_ratio * concentration_factor) * C_vapor,

    the radon swept from the vapor cloud, concentrated by condensation around the
    chilled wellbore, times a gamma-yield calibration.
    """
    return gamma_per_unit * radon_partition(c_vapor, ostwald_ratio) * concentration_factor


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Condensed Vapor Gamma")
    print("=" * 60)

    # Vapor pressure rises with temperature; butane > water at the same T
    p_water = clausius_clapeyron(101325.0, 373.15, 393.15, 2.26e6, 0.018)
    p_butane = clausius_clapeyron(101325.0, 272.65, 393.15, 3.86e5, 0.058)
    print(f"  P water / butane @393K = {p_water/1e3:.0f} / {p_butane/1e3:.0f} kPa")
    assert clausius_clapeyron(101325.0, 373.15, 393.15, 2.26e6, 0.018) > 101325.0
    assert p_butane > p_water

    # Radon decays with its 3.82-day half-life
    n0 = 1.0e6
    assert np.isclose(radioactive_decay(n0, RN222_HALF_LIFE_DAYS), 0.5 * n0)
    a = activity(n0)
    print(f"  radon activity         = {a:.3e} (per day)")
    assert a > 0 and radioactive_decay(n0, 10.0) < radioactive_decay(n0, 5.0)

    # Radon is far more soluble in hydrocarbon vapor -> larger liquid concentration
    assert radon_partition(1.0, 10.0) > radon_partition(1.0, 1.0)

    # Condensation concentrates radon, raising the gamma amplitude well above liquid
    gr_hc = condensed_gamma_amplitude(1.0, ostwald_ratio=10.0, concentration_factor=200.0,
                                      gamma_per_unit=1.0)
    gr_water = condensed_gamma_amplitude(1.0, ostwald_ratio=1.0, concentration_factor=200.0,
                                         gamma_per_unit=1.0)
    print(f"  CVG amplitude HC/water = {gr_hc:.0f} / {gr_water:.0f}")
    assert gr_hc > gr_water and gr_hc >= 1000.0
    print("  PASS")
    return {"P_butane_kPa": float(p_butane / 1e3), "activity": float(a), "CVG_hc": float(gr_hc)}


if __name__ == "__main__":
    test_all()
