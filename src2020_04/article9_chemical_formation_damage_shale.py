"""
Article 9: Chemically Induced Formation Damage in Shale
Wick, Taneja, Gupta, Sondergeld, Rai (2020)
DOI: 10.30632/PJV61N2-2020a9

Fluid-shale chemical interactions (clay swelling, fines migration, precipitation)
reduce near-wellbore permeability.  Damage is quantified by the retained-
permeability ratio; clay swelling shrinks pore throats / fracture apertures, and
the cubic law links aperture loss to fracture permeability loss.

Implements:

  - Retained-permeability (damage) ratio  k_after/k_before
  - Clay-swelling permeability reduction  k/k0 = (1 - eps_swell)^n
  - Kozeny-Carman porosity-permeability sensitivity  k/k0 = (phi/phi0)^3
  - Fracture cubic law  k_f ~ aperture^2  (k ~ w^3 / (12 L) per unit area)

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard formation-damage relations the paper's title
describes.
"""

import numpy as np


# ---------------------------------------------- damage ratio ------------

def damage_ratio(k_before, k_after):
    """Retained-permeability ratio  k_after/k_before (1 = no damage, 0 = total)."""
    return np.asarray(k_after, float) / k_before


def damage_percent(k_before, k_after):
    """Permeability impairment (%)  (1 - k_after/k_before)*100."""
    return (1.0 - damage_ratio(k_before, k_after)) * 100.0


# ---------------------------------------------- swelling / Kozeny -------

def swelling_permeability(k0, eps_swell, n=3.0):
    """Clay-swelling permeability reduction  k = k0*(1 - eps_swell)^n.

    eps_swell is the fractional pore/aperture closure from clay hydration.
    """
    return k0 * (1.0 - np.asarray(eps_swell, float)) ** n


def kozeny_carman(k0, phi, phi0):
    """Kozeny-Carman porosity-permeability sensitivity  k/k0 = (phi/phi0)^3."""
    return k0 * (np.asarray(phi, float) / phi0) ** 3


def fracture_permeability(aperture_m, length_m=1.0):
    """Fracture-plane permeability via the cubic law  k = w^2/12 (per unit area).

    For parallel plates the conductivity is w^3/(12 L); the intrinsic
    permeability of the fracture is w^2/12.
    """
    return np.asarray(aperture_m, float) ** 2 / 12.0


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Chemically Induced Formation Damage in Shale")
    print("=" * 60)

    # Damage ratio and percent
    assert abs(damage_ratio(100.0, 40.0) - 0.4) < 1e-12
    assert abs(damage_percent(100.0, 40.0) - 60.0) < 1e-9
    print(f"  retained / impairment  = {damage_ratio(100.0, 40.0):.2f} / "
          f"{damage_percent(100.0, 40.0):.0f}%")

    # Clay swelling reduces permeability; more swelling -> more damage
    k_mild = swelling_permeability(1.0, 0.1)
    k_severe = swelling_permeability(1.0, 0.3)
    print(f"  k after 10% / 30% swell = {k_mild:.3f} / {k_severe:.3f}")
    assert k_severe < k_mild < 1.0

    # Kozeny-Carman: a 10% porosity loss drops permeability ~27%
    k = kozeny_carman(1.0, 0.9 * 0.15, 0.15)
    print(f"  k/k0 after 10% phi loss = {k:.3f}")
    assert abs(k - 0.9 ** 3) < 1e-9

    # Cubic law: halving the aperture quarters the fracture permeability
    kf1 = fracture_permeability(20e-6)
    kf2 = fracture_permeability(10e-6)
    assert abs(kf1 / kf2 - 4.0) < 1e-9
    print(f"  fracture k 20um/10um   = {kf1:.2e} / {kf2:.2e} m^2")
    print("  PASS")
    return {"retained": 0.4, "k_severe_swell": float(k_severe)}


if __name__ == "__main__":
    test_all()
