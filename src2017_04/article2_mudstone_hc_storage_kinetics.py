"""
Article 2: Characterizing Hydrocarbon Storage in Organic-Rich Mudstones by
           Integrating Core Measurements, Kinetic Modeling, and Pore-Scale
           Observations: Application to South Texas Organic-Rich Mudstones
Capsan, Sanchez-Ramirez (2017)
Reference: Petrophysics Vol. 58, No. 2 (April 2017), pp. 97-115
DOI: none assigned (this issue predates SPWLA DOI assignment)

Hydrocarbon pore volume in organic-rich mudstones is reconciled from three
independent estimates - Dean-Stark water, Dean-Stark oil (with a formation volume
factor), and pyrolysis S1 - and a source-rock kinetic model caps the
organic-hosted nanoporosity from the converted kerogen mass.

Implements:

  - Hydrocarbon pore volume  HPV = phi_t*(1 - Sw_t)
  - Dean-Stark formation volume factor  FVF = (S_oil + S_gas)/S_oil
  - HPV from oil and from pyrolysis S1
  - Arrhenius first-order kerogen conversion and max organic nanoporosity

Note: this issue's PDF has a text layer; Eqs. 1 and 5 survived as ASCII, while
the others lost their glyphs and are faithful standard-form reconstructions.
Porosities fractional, densities g/cm^3, S1 in mg HC/g rock.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

R_GAS = 8.314                # J/(mol K)


# ---------------------------------------------- pore volume --------------

def hydrocarbon_pore_volume(phi_t, sw_t):
    """Hydrocarbon pore volume  HPV = phi_t*(1 - Sw_t)  (Eq. 1)."""
    return petrolib.porosity_lithology.hydrocarbon_pore_volume(phi_t, sw_t)


def formation_volume_factor(s_oil, s_gas):
    """Dean-Stark formation volume factor  FVF = (S_oil + S_gas)/S_oil  (Eq. 6)."""
    return (s_oil + s_gas) / s_oil


def hpv_from_oil(phi_t, s_oil, fvf):
    """HPV from Dean-Stark oil  HPV = phi_t*S_oil*FVF  (Eq. 5)."""
    return phi_t * s_oil * fvf


def hpv_from_s1(s1, rho_b, rho_oil_surface):
    """HPV from pyrolysis S1  HPV = (S1*rho_b)/(1000*rho_oil_surface)  (Eqs. 7-8).

    S1 (mg HC/g rock) -> oil mass per rock volume (via rho_b) -> oil volume per
    rock volume (via surface oil density); the 1000 converts mg to g.
    """
    return s1 * rho_b / (1000.0 * rho_oil_surface)


# ---------------------------------------------- kinetics --------------

def arrhenius_rate(frequency_factor, activation_energy, temperature_k):
    """Arrhenius rate constant  k = A*exp(-E/(R*T))  (Eq. 10)."""
    return frequency_factor * np.exp(-activation_energy / (R_GAS * temperature_k))


def kerogen_remaining(initial_conc, rate, time):
    """First-order kerogen remaining  c = c_i*exp(-k*t)  (Eq. 9)."""
    return initial_conc * np.exp(-rate * np.asarray(time, float))


def max_organic_nanoporosity(toc_initial, toc, carbon_fraction, rho_b, rho_kerogen):
    """Max organic nanoporosity from converted kerogen (Eq. 12)

        phi_MAX_ONP = ((TOCi - TOC)/c)*(rho_b/rho_k).
    """
    return (toc_initial - toc) / carbon_fraction * (rho_b / rho_kerogen)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Mudstone Hydrocarbon Storage & Kinetics")
    print("=" * 60)

    # HPV and the two volumetric estimates
    hpv = hydrocarbon_pore_volume(0.08, 0.45)
    fvf = formation_volume_factor(s_oil=0.30, s_gas=0.06)
    print(f"  HPV / FVF              = {hpv:.4f} / {fvf:.3f}")
    assert np.isclose(hpv, 0.08 * 0.55) and fvf > 1.0
    assert hpv_from_oil(0.08, 0.30, fvf) > 0 and hpv_from_s1(2.0, 2.4, 0.85) > 0

    # Arrhenius rate rises with temperature; kerogen decays with time
    assert arrhenius_rate(1e13, 2.1e5, 450.0) > arrhenius_rate(1e13, 2.1e5, 400.0)
    c = kerogen_remaining(1.0, 1e-3, np.array([0.0, 1000.0, 5000.0]))
    assert c[0] == 1.0 and np.all(np.diff(c) < 0)

    # Organic nanoporosity grows with the converted (lost) kerogen
    onp = max_organic_nanoporosity(toc_initial=0.08, toc=0.05, carbon_fraction=0.8,
                                   rho_b=2.4, rho_kerogen=1.3)
    print(f"  max organic nanoporosity = {onp:.4f}")
    assert onp > 0
    print("  PASS")
    return {"HPV": float(hpv), "phi_ONP": float(onp)}


if __name__ == "__main__":
    test_all()
