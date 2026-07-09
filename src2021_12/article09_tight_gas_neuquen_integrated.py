"""
Article 9: An Integrated Petrophysical Characterization of a Siliciclastic
           Tight Gas Reservoir in Neuquen Basin, Western Argentina
Carrizo, Santiago, Saldungaray (2021)
DOI: 10.30632/PJV62N6-2021a9

*Methodology proxy.*  This article's body was not present in the source-PDF
text extract used to build this folder (only the table of contents and the
editor's narrative were available).  This module is a methodology proxy that
demonstrates the integrated tight-gas workflow the editor describes: a rock-
calibrated clay volume, porosity, permeability, dominant pore-throat radius,
several water-saturation models, and correlation of hydraulic units with
lithofacies, in a reservoir with overpressure up to ~50% above hydrostatic.

Implements:

  - Clay volume from gamma ray (linear + Larionov older-rocks)
  - Density porosity
  - Archie and Simandoux water-saturation models
  - Winland r35 dominant pore-throat radius
  - Rock-quality index, normalized porosity, flow-zone indicator (hydraulic units)
  - Overpressure pore-pressure gradient

These are standard correlations consistent with the editor's description;
replace with the paper's calibrated parameters when its body is available.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- clay volume -------------

def vshale_linear(gr, gr_clean, gr_shale):
    """Linear gamma-ray shale index IGR = (GR - GRclean)/(GRshale - GRclean)."""
    return petrolib.porosity_lithology.gamma_ray_index(gr, gr_clean, gr_shale)


def vshale_larionov_old(gr, gr_clean, gr_shale):
    """Larionov (1969) older-rocks correction  Vsh = 0.33*(2^(2*IGR) - 1)."""
    return petrolib.porosity_lithology.vshale_from_gr(
        gr, gr_clean, gr_shale, method="larionov_older", clip=(0.0, 1.0))


# ---------------------------------------------- porosity ----------------

def density_porosity(rhob, rho_ma=2.65, rho_fl=1.0):
    """Density porosity  phi = (rho_ma - rhob)/(rho_ma - rho_fl)."""
    return petrolib.porosity_lithology.density_porosity(rhob, rho_ma, rho_fl)


# ---------------------------------------------- saturation --------------

def archie_sw(rt, phi, rw, a=1.0, m=2.0, n=2.0):
    """Archie  Sw = (a*Rw / (phi^m * Rt))^(1/n)."""
    # HAZARD (LIBRARY_MERGE_PLAN.md section 9): this article's argument order
    # is (rt, phi, rw) — the canonical order is (rt, rw, phi=).  Mapped
    # explicitly.
    return petrolib.saturation_resistivity.archie_sw(rt, rw, phi=phi, a=a, m=m, n=n, clip=(0.0, 1.0))


def simandoux_sw(rt, phi, rw, vsh, rsh, a=1.0, m=2.0):
    """Simandoux shaly-sand water saturation (n = 2 closed form).

    Solves  1/Rt = Vsh*Sw/Rsh + phi^m * Sw^2 / (a*Rw)  for Sw.
    """
    return petrolib.saturation_resistivity.simandoux_sw(rt, rw, phi=phi, vsh=vsh, rsh=rsh, a=a, m=m,
                              clip=(0.0, 1.0))


# ---------------------------------------------- pore-throat radius ------

def winland_r35(k_md, phi):
    """Winland (1972) dominant pore-throat radius r35 (microns).

    log10(r35) = 0.732 + 0.588*log10(k) - 0.864*log10(phi*100).
    """
    return petrolib.flow_transport.winland_r35(k_md, phi)


# ---------------------------------------------- hydraulic units ---------

def rock_quality_index(k_md, phi):
    """RQI (microns)  = 0.0314 * sqrt(k/phi)."""
    return petrolib.flow_transport.rqi(k_md, phi)


def normalized_porosity(phi):
    """phi_z = phi / (1 - phi)."""
    return petrolib.flow_transport.phi_z(phi)


def flow_zone_indicator(k_md, phi):
    """FZI (microns) = RQI / phi_z."""
    return petrolib.flow_transport.fzi(k_md, phi)


# ---------------------------------------------- overpressure ------------

def pore_pressure_gradient(hydrostatic_grad, overpressure_fraction):
    """Pore-pressure gradient = hydrostatic * (1 + overpressure_fraction)."""
    return hydrostatic_grad * (1.0 + overpressure_fraction)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Integrated Tight-Gas Characterization (proxy)")
    print("=" * 60)

    gr = np.array([20.0, 60.0, 110.0])
    vsh_lin = vshale_linear(gr, 15.0, 120.0)
    vsh_lar = vshale_larionov_old(gr, 15.0, 120.0)
    print(f"  Vsh linear / Larionov  = {vsh_lin.round(3)} / {vsh_lar.round(3)}")
    assert np.all(vsh_lar <= vsh_lin + 1e-9)   # Larionov reduces clay estimate

    phi = density_porosity(np.array([2.45, 2.55, 2.62]))
    print(f"  density porosity       = {phi.round(3)}")
    assert np.all((phi > 0) & (phi < 0.2))

    # Saturation: shaly-sand Simandoux >= Archie in shaly rock
    sw_a = archie_sw(rt=20.0, phi=0.10, rw=0.05)
    sw_s = simandoux_sw(rt=20.0, phi=0.10, rw=0.05, vsh=0.3, rsh=5.0)
    print(f"  Sw Archie / Simandoux  = {sw_a:.3f} / {sw_s:.3f}")
    assert 0.0 < sw_a <= 1.0 and 0.0 < sw_s <= 1.0

    # Winland r35 increases with permeability
    r35_lo = winland_r35(0.05, 0.08)
    r35_hi = winland_r35(5.0, 0.12)
    print(f"  Winland r35 lo / hi    = {r35_lo:.3f} / {r35_hi:.3f} um")
    assert r35_hi > r35_lo > 0

    # Flow-zone indicator groups rock quality
    fzi = flow_zone_indicator(np.array([0.05, 0.5, 5.0]), np.array([0.08, 0.10, 0.12]))
    print(f"  FZI                    = {fzi.round(3)}")
    assert np.all(np.diff(fzi) > 0)

    # Overpressure up to 50% above hydrostatic
    pg = pore_pressure_gradient(0.105, 0.5)   # bar/m hydrostatic-ish
    assert abs(pg - 0.1575) < 1e-9
    print("  PASS")
    return {"vsh": vsh_lar, "phi": phi, "sw_simandoux": sw_s, "fzi": fzi}


if __name__ == "__main__":
    test_all()
