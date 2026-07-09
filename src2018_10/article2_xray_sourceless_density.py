"""
Article 2: A Novel X-Ray Tool for True Sourceless Density Logging
Simon, Tkabladze, Beekman, Atobatele, De Looz, Grover, Hamichi, Jundt,
McFarland, Mlcak, Reijonen, Revol, Stewart, Yeboah, Zhang (2018)
DOI: 10.30632/PJV59N5-2018a1

A sourceless density tool uses an electronic X-ray generator (instead of a
chemical gamma source) to measure formation bulk density from Compton-scattered
photon attenuation, plus the photoelectric factor.  A two-detector spine-and-ribs
correction removes the mudcake/standoff effect, giving a corrected density (RHOX)
and a density correction (DRHO) comparable to the conventional tool.

Implements:

  - Compton attenuation density response  count ~ exp(-mu*rho*spacing)
  - Density from the near/far count ratio
  - Spine-and-ribs mudcake/standoff correction (DRHO)
  - Photoelectric factor from the low-energy window

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the density-logging physics the tool is built on.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- density response --------

def detector_count(rho, spacing, mu_mass=0.06, n0=1e6):
    """Compton detector count  N = N0*exp(-mu_mass*rho*spacing)."""
    return petrolib.nuclear.gamma_count(rho, n0=n0, mu_mass=mu_mass, spacing_cm=spacing)


def density_from_count(count, spacing, mu_mass=0.06, n0=1e6):
    """Invert the Compton response for bulk density."""
    return petrolib.nuclear.density_from_count(count, n0=n0, mu_mass=mu_mass, spacing_cm=spacing)


def spine_ribs_correction(rho_long, rho_short):
    """Spine-and-ribs corrected density and DRHO from long/short spacings.

    The corrected density follows the "rib": rho = rho_long + alpha*(rho_long -
    rho_short); DRHO = rho - rho_long is the correction (positive for mudcake).
    """
    return petrolib.nuclear.spine_ribs(rho_long, rho_short, rib_coeffs=(0.0, 1.0, 0.0))


def photoelectric_factor(soft_count, hard_count, k=1.0):
    """Photoelectric factor proxy from the soft/hard energy-window ratio."""
    return k * np.log(np.asarray(soft_count, float) / np.asarray(hard_count, float))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: X-Ray Sourceless Density Logging")
    print("=" * 60)

    # Density response round-trip
    rho = 2.45
    N = detector_count(rho, spacing=0.4)
    print(f"  count at 2.45 g/cc     = {N:.0f}")
    assert abs(density_from_count(N, 0.4) - rho) < 1e-9
    # denser formation -> fewer counts
    assert detector_count(2.7, 0.4) < detector_count(2.3, 0.4)

    # Spine-and-ribs: with mudcake, the short-spacing reads lighter; the
    # correction pushes the answer back up and DRHO is positive
    rho_long, rho_short = 2.45, 2.30          # mudcake biases short spacing low
    rho_corr, drho = spine_ribs_correction(rho_long, rho_short)
    print(f"  corrected rho / DRHO   = {rho_corr:.3f} / {drho:+.3f}")
    assert rho_corr > rho_long and drho > 0
    # no mudcake -> no correction
    rc0, d0 = spine_ribs_correction(2.45, 2.45)
    assert abs(d0) < 1e-9 and abs(rc0 - 2.45) < 1e-9

    # Photoelectric factor rises with the soft/hard ratio (high-Z minerals)
    assert photoelectric_factor(5e5, 1e5) > photoelectric_factor(2e5, 1e5)
    print("  PASS")
    return {"count_2.45": float(N), "DRHO": float(drho)}


if __name__ == "__main__":
    test_all()
