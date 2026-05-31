"""
Article 6: A Case Study on Integrated Petrophysical Characterization of a
           Carbonate Reservoir Pore System in the Offshore Red River Basin of
           Vietnam
Giao, Chung (2017)
Reference: Petrophysics Vol. 58, No. 3 (June 2017), pp. 289-301
DOI: none assigned (this issue predates SPWLA DOI assignment)

An integrated well-log + thin-section workflow characterizes a carbonate pore
system (interparticle, separate-vug, and touching-vug porosity) and rock types
(limestone vs dolostone).  This *methodology proxy* implements the standard
carbonate-petrophysics relations the workflow uses: density porosity, PEF rock
typing, the vuggy/interparticle porosity partition, the Lucia rock-fabric
permeability, and fracture porosity from a resistivity model.

Implements:

  - Density porosity  phi = (rho_ma - rho_b)/(rho_ma - rho_fl)
  - Rock typing from the photoelectric factor (limestone vs dolostone)
  - Vuggy / interparticle porosity partition
  - Lucia rock-fabric permeability and a dual-porosity fracture porosity

Note: this article's body was beyond this issue's machine extraction (the text
ended at the "METHODOLOGY OF STUDY" heading), so - consistent with the
methodology proxies elsewhere in this repository - the relations below are the
standard carbonate-characterization formulas the abstract describes, not formulas
transcribed from the paper.  Porosities fractional, density in g/cm^3.
"""

import numpy as np

PEF_LIMESTONE = 5.08
PEF_DOLOMITE = 3.14


# ---------------------------------------------- porosity / rock type --------------

def density_porosity(rho_b, rho_ma=2.71, rho_fl=1.0):
    """Density porosity  phi = (rho_ma - rho_b)/(rho_ma - rho_fl)  (limestone matrix default)."""
    return (rho_ma - np.asarray(rho_b, float)) / (rho_ma - rho_fl)


def rock_type(pef):
    """Carbonate rock type from PEF: limestone (PEF ~ 5.08) vs dolostone (PEF ~ 3.14)."""
    return "limestone" if pef >= 0.5 * (PEF_LIMESTONE + PEF_DOLOMITE) else "dolostone"


def vuggy_porosity(total_porosity, interparticle_porosity):
    """Vuggy (separate + touching) porosity = total - interparticle."""
    return total_porosity - interparticle_porosity


# ---------------------------------------------- permeability / fractures --------------

def lucia_permeability(phi, rock_fabric_number):
    """Lucia rock-fabric permeability (global transform, log-linear in porosity)

        log10(k) = (A - B*log10(RFN)) + (C - D*log10(RFN))*log10(phi).

    RFN = Lucia rock-fabric number (~0.5 coarse/grain-dominated, high perm ... 4
    fine/mud-dominated, low perm); a smaller RFN gives a higher permeability.
    k in mD.
    """
    a, b, c, d = 9.7982, 3.6700, 8.0735, 3.0791       # Lucia (1995/2007) coefficients
    rfn = np.asarray(rock_fabric_number, float)
    log_k = (a - b * np.log10(rfn)) + (c - d * np.log10(rfn)) * np.log10(phi)
    return 10.0 ** log_k


def fracture_porosity(rt, rt_matrix, m_fracture=1.0):
    """Dual-porosity fracture porosity from a resistivity contrast

        phi_f = (Rt_matrix/Rt)^(1/m_fracture) scaled to a small fracture fraction.

    A lower measured Rt than the matrix indicates conductive fractures.
    """
    return np.clip((rt_matrix / np.asarray(rt, float)) ** (1.0 / m_fracture) - 1.0, 0.0, None) * 0.01


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Carbonate Pore System (proxy)")
    print("=" * 60)

    # Density porosity and PEF rock typing
    assert density_porosity(2.71) == 0.0 and density_porosity(2.51) > 0
    print(f"  PEF 5.0 / 3.2 -> {rock_type(5.0)} / {rock_type(3.2)}")
    assert rock_type(5.0) == "limestone" and rock_type(3.2) == "dolostone"

    # Vuggy = total - interparticle
    assert np.isclose(vuggy_porosity(0.12, 0.08), 0.04)

    # Lucia: permeability rises with porosity and with a coarser fabric (lower RFN)
    k_coarse = lucia_permeability(0.15, rock_fabric_number=1.0)
    k_fine = lucia_permeability(0.15, rock_fabric_number=3.0)
    print(f"  Lucia k coarse/fine RFN = {k_coarse:.3f} / {k_fine:.3f} mD")
    assert lucia_permeability(0.20, 1.0) > lucia_permeability(0.10, 1.0)
    assert k_coarse > k_fine

    # Fracture porosity: zero when matched, positive when Rt < matrix
    assert fracture_porosity(20.0, 20.0) == 0.0 and fracture_porosity(10.0, 20.0) > 0
    print("  PASS")
    return {"k_coarse": float(k_coarse)}


if __name__ == "__main__":
    test_all()
