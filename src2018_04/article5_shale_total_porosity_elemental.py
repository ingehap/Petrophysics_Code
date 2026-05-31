"""
Article 5: Calculating the Total Porosity of Shale Reservoirs by Combining
           Conventional Logging and Elemental Logging to Eliminate the Effects
           of Gas Saturation
Zhu, Zhang, Guo, Jiao, Chen, Zhou, Zhang, Zhang (2018)
DOI: 10.30632/PJV59N2-2018a4

A five-component shale model (clay, organic matter, nonclay minerals, water,
hydrocarbons) computes total porosity while algebraically eliminating the
unknown hydrocarbon saturation.  The density and neutron logs each give a
response equation summing the component contributions; written in terms of the
porosity (phi) and the hydrocarbon-filled porosity (phi*Sh) they form two linear
equations, so solving the pair removes the saturation dependence.  Matrix
density/neutron come from elemental (ECS) logging and the organic-matter volume
from TOC.

Implements:

  - Organic-matter volume from TOC  V_OM = rho*TOC*k/(rho_OM*100)
  - Density and neutron response equations (5-component)
  - Total porosity by solving the 2x2 system (eliminates Sh)
  - Hydrocarbon saturation recovered as a by-product

Note: this issue's PDF has a text layer but the response/porosity equations
(Eqs. 1-5) lost their typeset glyphs in extraction, so they are faithful
standard-form reconstructions from the surviving variable definitions.
Densities in g/cm^3, neutron/porosity fractional, TOC in wt%.
"""

import numpy as np


# ---------------------------------------------- organic matter --------------

def organic_matter_volume(rho, toc_pct, k=1.25, rho_om=1.1):
    """Organic-matter volume fraction  V_OM = rho*TOC*k/(rho_OM*100)  (Eq. 1).

    rho = bulk density, TOC in wt%, k = organic-carbon conversion factor
    (1.25-1.57, kerogen-type dependent), rho_OM = organic-matter density.
    """
    return rho * np.asarray(toc_pct, float) * k / (rho_om * 100.0)


# ---------------------------------------------- total porosity --------------

def total_porosity(rho_log, n_log, rho_ma, n_ma, v_om,
                   rho_om=1.1, n_om=0.6, rho_w=1.0, n_w=1.0, rho_h=0.2, n_h=0.3):
    """Total porosity and hydrocarbon saturation, eliminating Sh (Eqs. 2-5).

    Unknowns are phi and the hydrocarbon-filled porosity y = phi*Sh.  The density
    and neutron responses,

        rho = rho_ma*(1-phi-V_OM) + rho_OM*V_OM + (phi-y)*rho_w + y*rho_h
        N   = N_ma*(1-phi-V_OM)   + N_OM*V_OM   + (phi-y)*N_w   + y*N_h,

    are linear in (phi, y); solving the 2x2 system gives phi without needing Sh,
    then Sh = y/phi.
    """
    b = np.array([rho_log - rho_ma * (1 - v_om) - rho_om * v_om,
                  n_log - n_ma * (1 - v_om) - n_om * v_om])
    a = np.array([[rho_w - rho_ma, rho_h - rho_w],
                  [n_w - n_ma, n_h - n_w]])
    phi, y = np.linalg.solve(a, b)
    return phi, (y / phi if phi else 0.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Shale Total Porosity (elemental)")
    print("=" * 60)

    # Organic-matter volume grows with TOC
    v_om = organic_matter_volume(rho=2.45, toc_pct=4.0)
    print(f"  V_OM for 4% TOC        = {v_om:.4f}")
    assert v_om > 0 and organic_matter_volume(2.45, 8.0) > v_om

    # Forward-model a gas shale, then recover phi and Sh from the two logs
    rho_ma, n_ma, vom = 2.65, 0.0, 0.05
    phi_true, sh_true = 0.08, 0.60
    y = phi_true * sh_true
    rho_log = rho_ma * (1 - phi_true - vom) + 1.1 * vom + (phi_true - y) * 1.0 + y * 0.2
    n_log = n_ma * (1 - phi_true - vom) + 0.6 * vom + (phi_true - y) * 1.0 + y * 0.3
    phi, sh = total_porosity(rho_log, n_log, rho_ma, n_ma, vom)
    print(f"  recovered phi / Sh     = {phi:.3f} / {sh:.3f}  (true 0.080 / 0.600)")
    assert np.isclose(phi, phi_true) and np.isclose(sh, sh_true)
    print("  PASS")
    return {"phi": float(phi), "Sh": float(sh), "V_OM": float(v_om)}


if __name__ == "__main__":
    test_all()
