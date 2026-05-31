"""
Article 4: Petrophysical Characterization of the Pore Space in Permian Wolfcamp
           Rocks
Rafatian, Capsan (2015)
Reference: Petrophysics Vol. 56, No. 1 (February 2015), pp. 45-57
DOI: none assigned (this issue predates SPWLA DOI assignment)

Mercury-injection capillary-pressure (MICP) data characterize the pore space of
Wolfcamp mudrocks.  The Washburn equation converts injection pressure to a
pore-throat radius; several characteristic-radius models (Swanson, Winland r35,
Katz-Thompson, Kozeny-Carman) then relate permeability to porosity and a
characteristic radius.  Except for Swanson, the models share the form
k ~ phi^a * R^2.

Implements:

  - Washburn pore-throat radius from injection pressure (Eq. 7)
  - Winland r35 pore-throat radius and the inverse permeability estimate
  - Swanson permeability from the (Sb/Pc)max apex
  - Characteristic-radius permeability  k = c*phi^a*R^2 (Table 1 form)

Note: this issue's PDF has a text layer; the Washburn relation (Eq. 7) and the
characteristic-radius permeability models are transcribed from the body, while
the typeset glyphs were dropped and reconstructed in standard form (Nelson,
1994; Pittman, 1992; Swanson, 1981).  Pc in psi, IFT in dyne/cm, radius in
microns, permeability in mD, porosity as a fraction.
"""

import numpy as np


# ---------------------------------------------- Washburn --------------

def washburn_pore_radius(pc, ift=480.0, theta=140.0):
    """Pore-throat radius from mercury injection pressure (Washburn; Eq. 7)

        r = 2*IFT*cos(theta)/Pc,

    with the air-mercury IFT (~480 dyne/cm) and contact angle (~140 deg).
    Returns r in microns for Pc in psi (with the 0.145 dyne/cm <-> psi factor).
    """
    # 2*IFT[dyne/cm]*|cos|/Pc[psi]; 1 psi = 68947.6 dyne/cm^2, 1 cm = 1e4 um
    return 2.0 * ift * abs(np.cos(np.radians(theta))) / (np.asarray(pc, float) * 68947.6) * 1e4


# ---------------------------------------------- Winland --------------

def winland_r35(k, phi):
    """Winland r35 pore-throat radius (Pittman, 1992)

        log10(r35) = 0.732 + 0.588*log10(k) - 0.864*log10(phi_pct),

    with k in mD and porosity in percent; r35 (um) is the throat radius at 35%
    mercury saturation.
    """
    return 10.0 ** (0.732 + 0.588 * np.log10(k) - 0.864 * np.log10(np.asarray(phi, float) * 100.0))


def winland_permeability(r35, phi):
    """Permeability from the Winland r35 and porosity (inverting the r35 relation)

        log10(k) = (log10(r35) - 0.732 + 0.864*log10(phi_pct))/0.588.
    """
    phi_pct = np.asarray(phi, float) * 100.0
    return 10.0 ** ((np.log10(r35) - 0.732 + 0.864 * np.log10(phi_pct)) / 0.588)


# ---------------------------------------------- Swanson / characteristic radius --------------

def swanson_permeability(sb_pc_max, c=399.0, d=1.691):
    """Swanson (1981) permeability from the capillary-pressure apex

        k = c*[(Sb/Pc)max]^d,

    where (Sb/Pc)max is the maximum ratio of bulk mercury saturation to
    injection pressure (the apex of the curve).
    """
    return c * np.asarray(sb_pc_max, float) ** d


def characteristic_radius_permeability(phi, radius, c=1.0, a=1.0):
    """Characteristic-radius permeability model (Table 1 common form)

        k = c*phi^a*R^2,

    shared by the Winland/Katz-Thompson/Kozeny-Carman families (R the
    characteristic pore radius); the constants and exponent a are model-specific.
    """
    return c * np.asarray(phi, float) ** a * radius ** 2


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Wolfcamp Pore-Space Characterization")
    print("=" * 60)

    # Washburn: higher injection pressure resolves smaller pore throats
    r_hi_p = washburn_pore_radius(10000.0)
    r_lo_p = washburn_pore_radius(1000.0)
    print(f"  Washburn r @10k/1k psi = {r_hi_p:.4f} / {r_lo_p:.4f} um")
    assert 0 < r_hi_p < r_lo_p

    # Winland r35 and its inverse round-trip
    k, phi = 0.01, 0.08
    r35 = winland_r35(k, phi)
    print(f"  Winland r35            = {r35:.4f} um")
    assert r35 > 0 and np.isclose(winland_permeability(r35, phi), k, rtol=1e-6)

    # Swanson permeability increases with the (Sb/Pc) apex
    assert swanson_permeability(0.5) > swanson_permeability(0.2) > 0

    # Characteristic-radius model: k rises with porosity and radius^2
    k1 = characteristic_radius_permeability(0.10, 0.05, c=100.0, a=1.0)
    k2 = characteristic_radius_permeability(0.20, 0.05, c=100.0, a=1.0)
    assert k2 > k1 and np.isclose(k2 / k1, 2.0)
    assert characteristic_radius_permeability(0.1, 0.10, 100.0) == 4.0 * characteristic_radius_permeability(0.1, 0.05, 100.0)
    print("  PASS")
    return {"r35": float(r35), "k_winland": float(winland_permeability(r35, phi))}


if __name__ == "__main__":
    test_all()
