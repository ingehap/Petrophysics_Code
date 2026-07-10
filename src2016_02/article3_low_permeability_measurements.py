"""
Article 3: Low-Permeability Measurements: Insights
Profice, Hamon, Nicot (2016)
Reference: Petrophysics Vol. 57, No. 1 (February 2016), pp. 30-40
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best Papers of the 2015 SCA Symposium.  Different low-permeability techniques
(step-decay, pulse-decay, steady-state gas and liquid) are compared on
ultralow-permeability pyrophyllite.  When the Darcy-Klinkenberg model is valid
the methods agree; this module implements the underlying relations: the
Klinkenberg gas-slippage correction, compressible (gas) and incompressible
(liquid) Darcy permeability, the gas mean free path and Knudsen number that mark
the slip/transition-flow regimes, and a deviation indicator comparing two
estimates.

Implements:

  - Klinkenberg apparent permeability  k = kl*(1 + b/Pm)  (Eq. 1)
  - Steady-state compressible (gas) Darcy permeability (Eq. 2)
  - Darcy liquid permeability (Eq. 3)
  - Gas mean free path (Eq. 4) and Knudsen number (Eq. 5)
  - Klinkenberg fit (kl, b) from k vs. 1/Pm
  - Deviation indicator between two estimates (Eq. 6)

Note: this issue's PDF has a text layer; the Klinkenberg/Darcy and mean-free-path
relations (Eqs. 1-6) are transcribed from the body, while the typeset glyphs were
dropped and reconstructed in standard form.  SI units unless noted: permeability
in m^2, pressure in Pa, viscosity in Pa*s, length/area in m.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

R_GAS = 8.314                 # J/(mol*K)


# ---------------------------------------------- Klinkenberg / Darcy --------------

def klinkenberg_apparent_k(kl, b, pm):
    """Klinkenberg (1941) apparent gas permeability (Eq. 1)

        k = kl*(1 + b/Pm),

    with kl the intrinsic (liquid-equivalent) permeability, b the gas-slippage
    coefficient and Pm the mean pore pressure.
    """
    return petrolib.flow_transport.klinkenberg_apparent(kl, b=b, p_mean=pm)


def darcy_gas_permeability(q, viscosity, length, area, p0, p1, p_ref):
    """Steady-state compressible (gas) Darcy permeability (Eq. 2)

        k = 2*q*mu*L*P_ref / (A*(P0^2 - P1^2)),

    the isothermal integration of Darcy's law for a compressible gas (q measured
    at reference pressure P_ref, e.g. the outlet/atmospheric pressure).
    """
    return petrolib.flow_transport.darcy_gas_permeability(
        q, mu=viscosity, length=length, area=area, p_up=p0, p_down=p1, p_ref=p_ref)


def darcy_liquid_permeability(q, viscosity, length, area, dp):
    """Incompressible (liquid) Darcy permeability  k = q*mu*L/(A*dP)  (Eq. 3)."""
    return petrolib.flow_transport.darcy_permeability(
        q, mu=viscosity, length=length, area=area, dp=dp)


# ---------------------------------------------- gas flow regime --------------

def mean_free_path(viscosity, pressure, temperature, molar_mass):
    """Gas mean free path (Eq. 4)

        lambda = (mu/P)*sqrt(pi*R*T/(2*M)),

    from kinetic theory; mu viscosity, P pressure, M molar mass (kg/mol).
    """
    return petrolib.flow_transport.mean_free_path(
        pressure=pressure, temperature=temperature, mu=viscosity, molar_mass=molar_mass)


def knudsen_number(mean_free_path_value, pore_radius):
    """Knudsen number  Kn = lambda/Rp.

    Kn < 0.01 continuum (Darcy), 0.01-0.1 slip flow, 0.1-10 transition flow.
    """
    return petrolib.flow_transport.knudsen_number(mean_free_path_value, pore_radius)


# ---------------------------------------------- fitting / comparison --------------

def klinkenberg_fit(pm, k_apparent):
    """Fit kl and b from the Klinkenberg plot of apparent k vs. 1/Pm

        k = kl + kl*b*(1/Pm)  ->  intercept = kl, slope = kl*b.

    Returns (kl, b).
    """
    return petrolib.flow_transport.fit_klinkenberg(pm, k_apparent)


def deviation_indicator(x1, x2):
    """Symmetric deviation between two estimates (Eq. 6)

        D = 2*|x1 - x2|/(x1 + x2),

    the fractional discrepancy used to compare permeability/porosity from
    different techniques.
    """
    return float(petrolib.data_qc_io.clean.relative_discrepancy(x1, x2))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Low-Permeability Measurements")
    print("=" * 60)

    # Klinkenberg: apparent k exceeds kl and rises as mean pressure falls
    kl, b = 1e-19, 5e5
    assert klinkenberg_apparent_k(kl, b, 1e6) > kl
    assert klinkenberg_apparent_k(kl, b, 5e5) > klinkenberg_apparent_k(kl, b, 5e6)

    # Klinkenberg fit recovers kl and b from a synthetic k vs. 1/Pm line
    pm = np.array([2e5, 5e5, 1e6, 2e6, 5e6])
    k_app = klinkenberg_apparent_k(kl, b, pm)
    kl_fit, b_fit = klinkenberg_fit(pm, k_app)
    print(f"  fitted kl / b          = {kl_fit:.2e} / {b_fit:.2e}")
    assert np.isclose(kl_fit, kl, rtol=1e-6) and np.isclose(b_fit, b, rtol=1e-6)

    # Gas and liquid Darcy permeabilities are positive
    kg = darcy_gas_permeability(1e-7, 1.8e-5, 0.05, 1.1e-3, 8.0e6, 2.0e5, 1.0e5)
    kliq = darcy_liquid_permeability(1e-10, 1.5e-3, 0.05, 1.1e-3, 6.0e6)
    print(f"  Darcy k gas / liquid   = {kg:.2e} / {kliq:.2e} m^2")
    assert kg > 0 and kliq > 0

    # Mean free path and Knudsen number place pyrophyllite in slip/transition flow
    lam = mean_free_path(1.8e-5, 1.0e6, 297.0, 0.028)   # N2 at ~10 bar
    kn = knudsen_number(lam, 25e-9)
    print(f"  mean free path / Kn    = {lam*1e9:.1f} nm / {kn:.3f}")
    assert lam > 0 and kn > 0.01            # beyond the continuum regime

    # Deviation indicator is zero for equal estimates, positive otherwise
    assert deviation_indicator(1.0, 1.0) == 0.0
    assert np.isclose(deviation_indicator(1.0, 1.2), 2 * 0.2 / 2.2)
    print("  PASS")
    return {"kl": float(kl_fit), "b": float(b_fit), "Kn": float(kn)}


if __name__ == "__main__":
    test_all()
