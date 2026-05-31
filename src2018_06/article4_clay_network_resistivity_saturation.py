"""
Article 4: A New Resistivity-Based Model for Improved Hydrocarbon Saturation
           Assessment in Clay-Rich Formations Using Quantitative Geometry of the
           Clay Network
Garcia, Jagadisan, Rostami, Heidari (2018)
DOI: 10.30632/PJV59N3-2018a3

Instead of assuming laminated/dispersed/structural clay, this model upscales the
conductivity of each rock component from pore-scale image geometry: the
percolating clay network conducts in proportion to its directional connectivity,
constriction, and (inverse) tortuosity, while dispersed clay, hydrocarbon, and
grains are added through a Maxwell-Garnett inclusion mixing.  The total rock
conductivity is the sum of the percolating-network and bulk contributions, from
which water saturation follows.

Implements:

  - Directional electrical tortuosity  tau = Le/L
  - Percolating clay-network conductivity  sigma_PC = sigma_C*psi*Cc
  - Maxwell-Garnett inclusion mixing of dispersed components
  - Total rock conductivity  sigma_R = sigma_other + sigma_PC
  - Archie water saturation  Sw = (sigma_R/(sigma_w*phi^m))^(1/n)

Note: this issue's PDF has a text layer but its typeset display-equation glyphs
were dropped in extraction, so the numbered relations (Eqs. 1-11) are faithful
standard-form reconstructions from the variable definitions and the described
six-step image workflow.  Conductivities in S/m.
"""

import numpy as np


# ---------------------------------------------- geometry --------------

def directional_tortuosity(le, length):
    """Directional electrical tortuosity  tau = Le/L  (Eq. 3), >= 1.

    Le = charge pathway length, L = sample length in that direction.
    """
    return np.asarray(le, float) / length


def percolating_clay_conductivity(sigma_c, connectivity, constriction):
    """Percolating clay-network conductivity  sigma_PC = sigma_C*psi*Cc  (Eq. 1).

    sigma_C = local clay conductivity, psi = directional connectivity (0..1),
    Cc = constriction factor (0..1).
    """
    return sigma_c * connectivity * constriction


def maxwell_garnett(sigma_host, sigma_incl, vol_frac, depol=1.0 / 3.0):
    """Maxwell-Garnett effective conductivity of inclusions in a host (Eqs. 4-9)

        sigma = sigma_h*[1 + f*(s_i-s_h)/(s_h + L*(1-f)*(s_i-s_h))].

    depol = depolarization factor L (1/3 for spheres); f = inclusion volume
    fraction.  Used to fold dispersed clay / hydrocarbon / grains into the host.
    """
    ds = sigma_incl - sigma_host
    return sigma_host * (1.0 + vol_frac * ds / (sigma_host + depol * (1.0 - vol_frac) * ds))


def total_conductivity(sigma_other, sigma_pc):
    """Total rock conductivity  sigma_R = sigma_other + sigma_PC  (Eqs. 10-11)."""
    return sigma_other + sigma_pc


def archie_saturation(sigma_r, sigma_w, phi, m=2.0, n=2.0):
    """Water saturation from the modeled conductivity (Archie)

        sigma_R = sigma_w*phi^m*Sw^n  ->  Sw = (sigma_R/(sigma_w*phi^m))^(1/n),

    clipped to [0, 1].
    """
    sw = (np.asarray(sigma_r, float) / (sigma_w * phi ** m)) ** (1.0 / n)
    return np.clip(sw, 0.0, 1.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Clay-Network Resistivity Saturation")
    print("=" * 60)

    # Tortuosity is >= 1; a more winding path conducts less
    assert directional_tortuosity(1.4, 1.0) > 1.0

    # Percolating-network conductivity scales with connectivity
    pc_lo = percolating_clay_conductivity(2.0, 0.3, 0.8)
    pc_hi = percolating_clay_conductivity(2.0, 0.7, 0.8)
    print(f"  sigma_PC low/high psi  = {pc_lo:.3f} / {pc_hi:.3f} S/m")
    assert pc_hi > pc_lo

    # Maxwell-Garnett stays between host and inclusion, and is exact at f=0/1
    mg = maxwell_garnett(1.0, 5.0, 0.3)
    assert 1.0 < mg < 5.0
    assert np.isclose(maxwell_garnett(1.0, 5.0, 0.0), 1.0)

    # Adding a conductive percolating clay network raises sigma_R and lowers Sw
    sig_other = maxwell_garnett(0.05, 1.0, 0.2)        # brine inclusions in grains
    sw_no_clay = archie_saturation(total_conductivity(sig_other, 0.0), 5.0, 0.15)
    sw_clay = archie_saturation(total_conductivity(sig_other, pc_hi), 5.0, 0.15)
    print(f"  Sw without/with clay   = {sw_no_clay:.3f} / {sw_clay:.3f}")
    assert sw_clay > sw_no_clay                         # clay conductivity biases Sw up
    print("  PASS")
    return {"sigma_PC": float(pc_hi), "Sw_clay": float(sw_clay)}


if __name__ == "__main__":
    test_all()
