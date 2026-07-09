"""
Article 10: Review of Micro/Nanofluidic Insights on Fluid Transport Controls in
            Tight Rocks
Mehmani, Kelly, Torres-Verdin (2019)
DOI: 10.30632/PJV60N6-2019a10

A review of how nanoscale confinement controls fluid transport in tight rocks.
In nanopores the gas mean free path becomes comparable to the pore size, so flow
departs from continuum Darcy behavior: the Knudsen number sets the flow regime,
gas slip (Klinkenberg) and Knudsen diffusion enhance the apparent permeability,
and the capillary number governs displacement.

Implements:

  - Mean free path and Knudsen number  Kn = lambda/d
  - Flow-regime classification by Knudsen number
  - Klinkenberg slip-corrected apparent permeability  k_app = k_inf*(1 + b/P)
  - Knudsen-diffusion / slip apparent-permeability enhancement factor
  - Capillary number  N_c = mu*v/sigma

Note: this issue's source-PDF text extract ended before this article (a review,
present only as a table-of-contents entry), so this module is a faithful
methodology proxy implementing the standard micro/nanofluidic transport
relations the review surveys.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

KB = 1.380649e-23        # J/K


# ---------------------------------------------- Knudsen -----------------

def mean_free_path(T, P, d_molecule=3.8e-10):
    """Gas mean free path  lambda = kB*T/(sqrt(2)*pi*d^2*P)  (m).  P in Pa."""
    return petrolib.flow_transport.mean_free_path(pressure=P, temperature=T, d_collision=d_molecule)


def knudsen_number(mfp, pore_diameter):
    """Knudsen number  Kn = lambda / d."""
    return petrolib.flow_transport.knudsen_number(mfp, pore_diameter)


def flow_regime(kn):
    """Classify flow by Knudsen number.

      Kn < 0.001        -> continuum (Darcy)
      0.001 <= Kn < 0.1 -> slip flow (Klinkenberg)
      0.1 <= Kn < 10    -> transition
      Kn >= 10          -> free-molecular (Knudsen)
    """
    return petrolib.flow_transport.flow_regime(kn)


# ---------------------------------------------- apparent permeability ---

def klinkenberg(k_inf, b, p_mean):
    """Klinkenberg slip-corrected apparent permeability  k_app = k_inf*(1+b/P)."""
    return petrolib.flow_transport.klinkenberg_apparent(k_inf, b=b, p_mean=p_mean)


def apparent_permeability_factor(kn, alpha=1.0):
    """Beskok-Karniadakis-style apparent-permeability enhancement vs Kn.

        f(Kn) = (1 + alpha*Kn)*(1 + 4*Kn/(1+Kn))
    Approaches 1 in the continuum limit and grows in confined nanopores.
    """
    kn = np.asarray(kn, float)
    return (1.0 + alpha * kn) * (1.0 + 4.0 * kn / (1.0 + kn))


# ---------------------------------------------- capillary ---------------

def capillary_number(mu, v, sigma):
    """Capillary number  N_c = mu*v/sigma."""
    return petrolib.relperm_wettability.capillary_number(mu=mu, v=v, sigma=sigma)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 10: Micro/Nanofluidic Transport in Tight Rocks")
    print("=" * 60)

    # Mean free path rises as pressure falls; Knudsen number rises in small pores
    mfp_hi = mean_free_path(350.0, 1e5)        # low pressure
    mfp_lo = mean_free_path(350.0, 1e7)        # high pressure
    print(f"  mean free path 1bar/100bar = {mfp_hi*1e9:.1f} / {mfp_lo*1e9:.3f} nm")
    assert mfp_hi > mfp_lo

    # Flow regimes across pore sizes at low reservoir pressure
    kn_nano = knudsen_number(mean_free_path(350.0, 1e6), 5e-9)    # 5 nm pore
    kn_micro = knudsen_number(mean_free_path(350.0, 1e6), 5e-6)   # 5 um pore
    print(f"  Kn 5nm / 5um           = {kn_nano:.2f} / {kn_micro:.2e}")
    assert flow_regime(kn_nano) in ("transition", "free-molecular")
    assert flow_regime(kn_micro) in ("continuum", "slip")
    assert flow_regime(0.0005) == "continuum" and flow_regime(20.0) == "free-molecular"

    # Klinkenberg slip raises apparent permeability at low pressure
    assert klinkenberg(1.0, 10.0, 50.0) > klinkenberg(1.0, 10.0, 500.0) > 1.0

    # Apparent-permeability factor -> 1 in continuum, grows with Kn
    assert abs(apparent_permeability_factor(1e-5) - 1.0) < 1e-3
    assert apparent_permeability_factor(1.0) > apparent_permeability_factor(0.01)

    # Capillary number scales with velocity
    assert capillary_number(1e-3, 1e-5, 0.03) < capillary_number(1e-3, 1e-3, 0.03)
    print("  PASS")
    return {"Kn_5nm": float(kn_nano), "regime_5nm": flow_regime(kn_nano),
            "kfac_Kn1": float(apparent_permeability_factor(1.0))}


if __name__ == "__main__":
    test_all()
