"""
Article 6: A Method of Determining Formation Density Based on Fast-Neutron Gamma
           Coupled Field Theory
Zhang, Zhang, Liu, Wu, Wu, Jia, Ti, Li (2017)
Reference: Petrophysics Vol. 58, No. 4 (August 2017), pp. 411-425
DOI: none assigned (this issue predates SPWLA DOI assignment)

A sourceless (pulsed-neutron) density method: fast neutrons induce inelastic and
capture gamma rays whose count depends on the formation electron density through
Compton scattering, so the inelastic-gamma count attenuates with density.  This
*methodology proxy* implements the standard neutron-gamma density relations:
the gamma count's exponential attenuation with density, the inversion of count
to density, the two-detector (near/far ratio) compensated density, and a
spine-and-ribs standoff correction.

Implements:

  - Inelastic-gamma count  N = N0*exp(-mu*rho*d)
  - Density inverted from the count  rho = -ln(N/N0)/(mu*d)
  - Two-detector compensated density  rho = a + b*ln(N_near/N_far)
  - Spine-and-ribs standoff (DRHO) correction

Note: this article's body was beyond this issue's machine extraction (the source
text ended at p408), so - consistent with the methodology proxies elsewhere in
this repository - the relations below are the standard neutron-gamma density-
logging forms the title describes, not formulas transcribed from the paper.
Density in g/cm^3, lengths in cm.
"""

import numpy as np


# ---------------------------------------------- gamma count --------------

def gamma_count(rho, n0, mu, spacing):
    """Inelastic/capture gamma count  N = N0*exp(-mu*rho*d).

    mu = mass attenuation coefficient (cm^2/g), spacing d (cm); denser formation
    scatters more gamma rays away, lowering the count.
    """
    return n0 * np.exp(-mu * np.asarray(rho, float) * spacing)


def density_from_count(n, n0, mu, spacing):
    """Formation density inverted from the gamma count  rho = -ln(N/N0)/(mu*d)."""
    return -np.log(np.asarray(n, float) / n0) / (mu * spacing)


# ---------------------------------------------- compensated --------------

def two_detector_density(near_count, far_count, a, b):
    """Two-detector compensated density  rho = a + b*ln(N_near/N_far).

    The near/far ratio cancels source-strength drift; a, b are calibration
    constants.
    """
    return a + b * np.log(np.asarray(near_count, float) / np.asarray(far_count, float))


def spine_ribs_correction(rho_apparent, drho):
    """Spine-and-ribs standoff correction  rho = rho_apparent + DRHO."""
    return rho_apparent + drho


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Fast-Neutron Gamma Density (proxy)")
    print("=" * 60)

    n0, mu, d = 1e6, 0.06, 30.0
    # Denser formation -> fewer detected gamma rays
    assert gamma_count(2.6, n0, mu, d) < gamma_count(2.0, n0, mu, d)

    # Density inverted from the count round-trips
    rho_true = 2.45
    n = gamma_count(rho_true, n0, mu, d)
    rho = density_from_count(n, n0, mu, d)
    print(f"  recovered density      = {rho:.3f} g/cc (true 2.450)")
    assert np.isclose(rho, rho_true)

    # Two-detector density increases as the near/far ratio rises
    lo = two_detector_density(1.0e5, 9.0e4, a=2.0, b=0.5)
    hi = two_detector_density(1.2e5, 9.0e4, a=2.0, b=0.5)
    assert hi > lo

    # Spine-and-ribs standoff correction adds DRHO
    assert np.isclose(spine_ribs_correction(2.40, 0.05), 2.45)
    print("  PASS")
    return {"density": float(rho)}


if __name__ == "__main__":
    test_all()
