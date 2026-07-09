"""
Article 5: Advances in Quantification of Miscible Contamination in Hydrocarbon
           and Water Samples From Downhole to Surface Laboratories
Zuo, Gisolf, Pfeiffer, Achourov, Chen, Mullins, Edmundson, Partouche (2017)
Reference: Petrophysics Vol. 58, No. 4 (August 2017), pp. 397-410
DOI: none assigned (this issue predates SPWLA DOI assignment)

Quantifies oil/water-base-mud (OBM/WBM) filtrate contamination in fluid samples.
A reservoir fluid's heavy-end composition is exponential in carbon number, a
two-endpoint mass balance separates native fluid from filtrate, and downhole
optical-density / property logs clean up as a power law of pumping time or pumped
volume, whose extrapolated endpoint gives the native value and hence the
contamination level (converted from volume to weight fraction by density).

Implements:

  - Exponential heavy-end composition  z_i = exp(A*C_i + B)
  - Two-endpoint mass balance  z_i = m*y_i + (1 - m)*x_i  and native recovery
  - Power-law optical-density cleanup  OD(t) = C - D*t^(-5/12)
  - Power-law property vs pumped volume and volume->weight conversion

Note: this issue's PDF has a text layer; several equations survived in ASCII
(reproduced here), while others lost their glyphs and are faithful standard-form
reconstructions.  Compositions as mole fractions; OD dimensionless.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- composition --------------

def exponential_composition(carbon_number, a, b):
    """Heavy-end composition  z_i = exp(A*C_i + B)  (Eq. 1), for C_i >= 8."""
    return np.exp(a * np.asarray(carbon_number, float) + b)


def mass_balance(m, y_filtrate, x_native):
    """Contaminated composition  z_i = m*y_i + (1 - m)*x_i  (Eq. 2)."""
    return petrolib.geochem_fluids.contamination.mix_linear(x_native, y_filtrate, m)


def native_composition(z_contaminated, y_filtrate, m):
    """Recover native composition  x_i = (z_i - m*y_i)/(1 - m)  (Eq. 3)."""
    return (np.asarray(z_contaminated, float) - m * np.asarray(y_filtrate, float)) / (1.0 - m)


# ---------------------------------------------- cleanup --------------

def od_cleanup(t, c_native, d, exponent=-5.0 / 12.0):
    """Optical-density cleanup  OD(t) = C - D*t^(-5/12)  (Eq. 4).

    OD -> C (native) as t -> infinity; the filtrate OD is taken as 0.
    """
    return c_native - d * np.asarray(t, float) ** exponent


def power_law_property(volume, endpoint, beta, gamma):
    """Property vs pumped volume  P(V) = endpoint + beta*V^(-gamma)  (Eq. 7).

    The native endpoint is the V -> infinity limit; gamma ~ 5/12 (single probe)
    to 2/3 (focused / low-contamination).
    """
    return endpoint + beta * np.asarray(volume, float) ** (-gamma)


def vol_to_weight_percent(vol_fraction, rho_filtrate, rho_mixture):
    """Convert a filtrate volume fraction to weight fraction  = vol*rho_filt/rho_mix (Eq. 9)."""
    return vol_fraction * rho_filtrate / rho_mixture


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Contamination Quantification")
    print("=" * 60)

    # Heavy-end composition decays with carbon number (A < 0)
    z = exponential_composition(np.array([8, 12, 20, 30]), a=-0.15, b=-1.0)
    assert np.all(np.diff(z) < 0)

    # Mass balance then native recovery round-trips
    x_native = exponential_composition(np.array([8, 12, 20]), -0.15, -1.0)
    y_filt = exponential_composition(np.array([8, 12, 20]), -0.30, 0.5)
    z_cont = mass_balance(0.15, y_filt, x_native)
    assert np.allclose(native_composition(z_cont, y_filt, 0.15), x_native)

    # OD cleanup approaches the native value as time grows
    od_early = od_cleanup(10.0, c_native=0.886, d=0.5)
    od_late = od_cleanup(1e9, c_native=0.886, d=0.5)
    print(f"  OD early / late        = {od_early:.3f} / {od_late:.4f}")
    assert od_late > od_early and abs(od_late - 0.886) < 1e-3

    # Power-law property tends to its endpoint with more pumped volume
    rho0 = power_law_property(1e9, 0.7205, beta=0.05, gamma=2.2)
    assert abs(rho0 - 0.7205) < 1e-3

    # Volume-to-weight conversion (denser filtrate -> higher wt%)
    wt = vol_to_weight_percent(0.019, rho_filtrate=0.7847, rho_mixture=0.7205)
    print(f"  1.9 vol% OBM -> wt%    = {wt * 100:.2f} %")
    assert wt > 0.019                                  # filtrate denser than mixture here
    print("  PASS")
    return {"OD_native": float(od_late), "wt_fraction": float(wt)}


if __name__ == "__main__":
    test_all()
