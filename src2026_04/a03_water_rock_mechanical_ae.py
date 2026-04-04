"""
Effect of Water-Rock Interactions on Mechanical Properties and Acoustic
Emission Characteristics of Sandstones

Reference:
    Zhao, Y. (2026). Effect of Water-Rock Interactions on Mechanical
    Properties and Acoustic Emission Characteristics of Sandstones.
    Petrophysics, 67(2), 280–293. DOI: 10.30632/PJV67N2-2026a3

Implements:
  - Fluid-saturation porosity calculation (Eq. 1)
  - Negative-exponential weakening of compressive strength and elastic
    modulus with soaking duration (Table 3 / Figs. 6–7)
  - Elastic-wave attenuation coefficient (Eq. 2)
  - AE energy accelerated-release theory (Eqs. 3–4)
  - Normal-distribution fitting to AE energy during loading
  - Transition from brittle to ductile failure indicators
"""

import numpy as np
import scipy.stats as stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Porosity from fluid saturation method (Eq. 1)
# ---------------------------------------------------------------------------

def porosity_fluid_saturation(m_saturated: float,
                               m_dry: float,
                               rho_water: float,
                               V_bulk: float) -> float:
    """
    Rock porosity from mass difference before/after water saturation (Eq. 1).

    Phi = (m_saturated - m_dry) / (rho_water * V_bulk)

    Parameters
    ----------
    m_saturated : Mass of vacuum-saturated specimen, g
    m_dry       : Dry mass of specimen, g
    rho_water   : Water density, g/cm³  (typically 1.0)
    V_bulk      : Total specimen volume, cm³

    Returns
    -------
    phi : Porosity as a fraction (0–1)
    """
    return (m_saturated - m_dry) / (rho_water * V_bulk)


# ---------------------------------------------------------------------------
# 2. Mechanical property degradation (negative exponential, Eqs. fitted in
#    Table 3 of the paper)
# ---------------------------------------------------------------------------

def mechanical_property_decay(t: float,
                               a: float, b: float, c: float) -> float:
    """
    Negative-exponential weakening model for strength or elastic modulus
    as a function of soaking duration (Figs. 6–7).

        P(t) = a * exp(-b * t) + c

    Parameters
    ----------
    t : Soaking duration, months
    a : Initial above-asymptote value (MPa or GPa)
    b : Decay rate coefficient (1/month)
    c : Long-term (asymptotic) value, MPa or GPa

    Returns
    -------
    P : Mechanical property at soaking duration t
    """
    return a * np.exp(-b * t) + c


# Paper-reported best-fit parameters (Table 3)
STRENGTH_PARAMS = dict(a=35.0, b=0.55, c=24.99)    # compressive strength, MPa
MODULUS_PARAMS  = dict(a=4.20, b=0.80, c=8.00)     # elastic modulus, GPa


def compressive_strength(t_months: float) -> float:
    """Compressive strength at soaking duration t (MPa)."""
    return mechanical_property_decay(t_months, **STRENGTH_PARAMS)


def elastic_modulus(t_months: float) -> float:
    """Elastic modulus at soaking duration t (GPa)."""
    return mechanical_property_decay(t_months, **MODULUS_PARAMS)


# ---------------------------------------------------------------------------
# 3. Elastic-wave attenuation (Eq. 2)
# ---------------------------------------------------------------------------

def attenuation_coefficient(A0: float, Ax: float, x: float) -> float:
    """
    Elastic-wave amplitude attenuation coefficient (Eq. 2).

        A(x) = A0 * exp(-α * x)  →  α = -ln(A(x) / A0) / x

    Parameters
    ----------
    A0 : Initial amplitude, dB
    Ax : Amplitude after propagating distance x, dB
    x  : Propagation distance, cm

    Returns
    -------
    alpha : Attenuation coefficient, dB/cm (or Np/cm if using natural units)
    """
    if Ax <= 0 or A0 <= 0 or x <= 0:
        return float("nan")
    return -np.log(Ax / A0) / x


def predict_amplitude(A0: float, alpha: float, x: float) -> float:
    """Predict wave amplitude at distance x given attenuation coefficient."""
    return A0 * np.exp(-alpha * x)


# ---------------------------------------------------------------------------
# 4. AE energy accelerated-release theory (Eqs. 3–4)
# ---------------------------------------------------------------------------

def accumulated_ae_energy_model(t: np.ndarray,
                                 tc: float,
                                 A: float,
                                 m: float) -> np.ndarray:
    """
    Accumulated AE energy based on the accelerated energy-release model
    (Eqs. 3–4 of the paper, after Zhang et al. 2004).

        E(tk) = A * (tc - tk)^(-m)    for tk < tc  (accelerated release)

    Captures the power-law increase in cumulative AE energy as the specimen
    approaches macroscopic fracture at time tc.

    Parameters
    ----------
    t  : Time array, s (must have all elements < tc)
    tc : Critical time (macroscopic fracture at peak strength), s
    A  : Scaling amplitude
    m  : Exponent (m > 0; higher m → sharper acceleration near failure)

    Returns
    -------
    E  : Accumulated AE energy array, aJ
    """
    dt = tc - t
    dt = np.where(dt > 0, dt, np.finfo(float).eps)
    return A * dt**(-m)


def ae_energy_release_rate(E: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Numerical derivative of accumulated AE energy → instantaneous
    energy release rate (dE/dt).
    """
    return np.gradient(E, t)


# ---------------------------------------------------------------------------
# 5. AE energy normal-distribution fitting
# ---------------------------------------------------------------------------

def fit_ae_energy_distribution(ae_energy_values: np.ndarray
                                ) -> Tuple[float, float]:
    """
    Fit a normal distribution to AE energy values recorded during loading.
    The paper shows that AE energy follows a normal distribution (μ, σ)
    where water-saturated specimens have significantly lower μ.

    Parameters
    ----------
    ae_energy_values : 1-D array of AE energy measurements, aJ

    Returns
    -------
    (mu, sigma) : Mean and standard deviation of fitted normal distribution
    """
    mu, sigma = stats.norm.fit(ae_energy_values)
    return mu, sigma


def ae_energy_expectation_with_soaking(t_months: float,
                                        mu_dry: float = 1000.0,
                                        k: float = 0.35) -> float:
    """
    Estimate the AE energy expectation (μ) for a specimen soaked for
    t_months, relative to the dry specimen value.

    Based on the paper's observation that water soaking significantly
    reduces AE energy expectation via wave attenuation and micro-crack
    lubrication.

        mu(t) = mu_dry * exp(-k * t)

    Parameters
    ----------
    t_months : Soaking duration, months
    mu_dry   : AE energy expectation for dry specimen, aJ
    k        : Attenuation rate constant, 1/month

    Returns
    -------
    mu_t : Expected AE energy at soaking duration t, aJ
    """
    return mu_dry * np.exp(-k * t_months)


# ---------------------------------------------------------------------------
# 6. Failure-mode indicator: brittle → ductile transition
# ---------------------------------------------------------------------------

def failure_mode_indicator(t_months: float,
                            sigma_c: float,
                            E_MPa: float) -> str:
    """
    Heuristic classifier for rock failure mode based on water exposure.

    The paper shows that dry specimens exhibit abrupt, accelerated AE
    energy release (brittle failure), while water-saturated specimens
    show gradual release (progressive plastic failure).

    Parameters
    ----------
    t_months : Soaking duration, months
    sigma_c  : Compressive strength, MPa
    E_MPa    : Elastic modulus, MPa (input in MPa for brittleness ratio)

    Returns
    -------
    mode : 'Brittle', 'Transitional', or 'Ductile'
    """
    # Brittleness index: ratio of E to strength (higher → more brittle)
    if sigma_c <= 0:
        return "Unknown"
    BI = E_MPa / sigma_c
    if BI > 250 and t_months < 1:
        return "Brittle"
    elif BI < 150 or t_months > 4:
        return "Ductile"
    else:
        return "Transitional"


# ---------------------------------------------------------------------------
# 7. Full weakening profile
# ---------------------------------------------------------------------------

def soaking_weakening_profile(t_max_months: float = 6.0,
                               n_points: int = 50) -> Dict:
    """
    Generate the complete mechanical weakening profile over soaking time.

    Returns
    -------
    dict with 't', 'strength_MPa', 'modulus_GPa', 'ae_energy_mu'
    """
    t = np.linspace(0, t_max_months, n_points)
    return {
        "t_months":      t,
        "strength_MPa":  compressive_strength(t),
        "modulus_GPa":   elastic_modulus(t),
        "ae_energy_mu":  ae_energy_expectation_with_soaking(t),
    }


# ---------------------------------------------------------------------------
# 8. Example workflow
# ---------------------------------------------------------------------------

def example_workflow():
    print("=" * 60)
    print("Water-Rock Interaction – Mechanical & AE Properties")
    print("Ref: Zhao, Petrophysics 67(2) 2026")
    print("=" * 60)

    # Porosity example
    phi = porosity_fluid_saturation(m_saturated=285.5, m_dry=276.3,
                                     rho_water=1.0, V_bulk=98.2)
    print(f"\nPorosity (fluid saturation method): {phi*100:.2f} %")

    # Mechanical degradation
    print("\nMechanical Property Degradation:")
    print(f"{'Soak (mo)':<12} {'Strength (MPa)':<18} {'E-modulus (GPa)':<18} {'Failure Mode'}")
    print("-" * 65)
    for t in [0, 1, 2, 3, 4, 6]:
        sc = compressive_strength(t)
        em = elastic_modulus(t)
        mode = failure_mode_indicator(t, sc, em * 1000)
        print(f"{t:<12} {sc:<18.1f} {em:<18.2f} {mode}")

    # Attenuation
    alpha = attenuation_coefficient(A0=100.0, Ax=62.5, x=5.0)
    print(f"\nAttenuation coefficient (dry): {alpha:.4f} /cm")
    alpha_sat = attenuation_coefficient(A0=100.0, Ax=48.0, x=5.0)
    print(f"Attenuation coefficient (6-month soak): {alpha_sat:.4f} /cm")

    # AE energy model
    tc = 300.0  # failure at 300 s
    t_arr = np.linspace(0, 295, 200)
    E_arr = accumulated_ae_energy_model(t_arr, tc=tc, A=5.0, m=0.3)
    peak_rate = ae_energy_release_rate(E_arr, t_arr).max()
    print(f"\nAE energy peak release rate (dry): {peak_rate:.2f} aJ/s")

    profile = soaking_weakening_profile()
    final_t = profile["t_months"][-1]
    print(f"\nAt {final_t:.0f} months: "
          f"strength = {profile['strength_MPa'][-1]:.1f} MPa, "
          f"E = {profile['modulus_GPa'][-1]:.2f} GPa")

    return profile


if __name__ == "__main__":
    example_workflow()
