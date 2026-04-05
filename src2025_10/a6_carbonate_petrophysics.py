#!/usr/bin/env python3
"""
Article 6: Enhanced Learning Experience for New Petrophysicists Using
           Open-Source Carbonate Data and Python Programming
Author: Imran M. Fadhil
Ref: Petrophysics, Vol. 66, No. 5 (October 2025), pp. 807-838.
     DOI: 10.30632/PJV66N5-2025a6

Implements a standard petrophysical evaluation workflow for carbonates:
  - Shale volume (linear & Larionov)
  - Effective porosity from density-neutron
  - Water saturation (Archie, Indonesian, Simandoux)
  - Permeability (Timur-Coates, Winland R35)
  - Net-pay flagging
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shale volume
# ---------------------------------------------------------------------------

def vshale_linear(gr, gr_clean, gr_shale):
    """Linear shale volume from gamma ray."""
    igr = np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0, 1)
    return igr


def vshale_larionov_tertiary(gr, gr_clean, gr_shale):
    """Larionov (1969) non-linear Vsh for Tertiary rocks."""
    igr = np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0, 1)
    return 0.083 * (2.0 ** (3.7 * igr) - 1.0)


def vshale_larionov_older(gr, gr_clean, gr_shale):
    """Larionov (1969) non-linear Vsh for older (pre-Tertiary) rocks."""
    igr = np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0, 1)
    return 0.33 * (2.0 ** (2.0 * igr) - 1.0)


# ---------------------------------------------------------------------------
# Porosity
# ---------------------------------------------------------------------------

def density_porosity(rho_b, rho_ma, rho_fl):
    """Density porosity.  phi_d = (rho_ma - rho_b) / (rho_ma - rho_fl)."""
    return np.clip((rho_ma - rho_b) / (rho_ma - rho_fl), 0, 0.6)


def neutron_density_porosity(nphi, dphi):
    """Combined neutron-density porosity (RMS average)."""
    return np.sqrt((nphi ** 2 + dphi ** 2) / 2.0)


def effective_porosity(phi_total, vsh, phi_sh=0.10):
    """Effective porosity = phi_total - Vsh * phi_shale."""
    return np.clip(phi_total - vsh * phi_sh, 0, 0.6)


# ---------------------------------------------------------------------------
# Water saturation models
# ---------------------------------------------------------------------------

def sw_archie(rt, rw, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation.
    Sw^n = a * Rw / (phi^m * Rt)
    """
    phi = np.maximum(phi, 0.001)
    rt = np.maximum(rt, 0.001)
    sw_n = a * rw / (phi ** m * rt)
    return np.clip(sw_n ** (1.0 / n), 0, 1)


def sw_simandoux(rt, rw, phi, vsh, rsh, a=1.0, m=2.0, n=2.0):
    """Simandoux water saturation (simplified)."""
    phi = np.maximum(phi, 0.001)
    rt = np.maximum(rt, 0.001)
    c = (1.0 - vsh) * a * rw / (phi ** m)
    d = c * vsh / (2.0 * rsh)
    e = c / rt
    sw = (-d + np.sqrt(np.maximum(d ** 2 + e, 0))) ** (2.0 / n)
    return np.clip(sw, 0, 1)


def sw_indonesian(rt, rw, phi, vsh, rsh, a=1.0, m=2.0, n=2.0):
    """Indonesian (Poupon-Leveaux) water saturation."""
    phi = np.maximum(phi, 0.001)
    rt = np.maximum(rt, 0.001)
    vsh_term = vsh ** (1.0 - 0.5 * vsh) / np.sqrt(np.maximum(rsh, 0.01))
    phi_term = np.sqrt(phi ** m / (a * rw))
    denom = (vsh_term + phi_term) ** 2
    sw = (1.0 / (rt * denom + 1e-30)) ** (1.0 / n)
    return np.clip(sw, 0, 1)


# ---------------------------------------------------------------------------
# Permeability
# ---------------------------------------------------------------------------

def perm_timur(phi, swir):
    """Timur (1968) permeability (md).
    k = 0.136 * phi^4.4 / Swir^2   (phi in fraction)
    """
    phi = np.maximum(phi, 0.001)
    swir = np.maximum(swir, 0.01)
    return 0.136 * (phi ** 4.4) / (swir ** 2) * 1e4  # md


def perm_winland_r35(phi, k_md):
    """Winland R35 pore-throat radius (microns).
    log(R35) = 0.732 + 0.588*log(k) - 0.864*log(phi*100)
    """
    k = np.maximum(k_md, 0.001)
    p = np.maximum(phi * 100.0, 0.1)
    log_r35 = 0.732 + 0.588 * np.log10(k) - 0.864 * np.log10(p)
    return 10.0 ** log_r35


# ---------------------------------------------------------------------------
# Net pay flagging
# ---------------------------------------------------------------------------

def net_pay_flag(phi, sw, vsh, phi_cut=0.05, sw_cut=0.6, vsh_cut=0.5):
    """Flag net pay based on cutoffs."""
    return ((phi >= phi_cut) & (sw <= sw_cut) & (vsh <= vsh_cut)).astype(int)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Article 6: Carbonate Petrophysics Workflow Demo ===\n")
    np.random.seed(42)
    n = 150
    depth = np.linspace(3000, 3150, n)

    gr = 20 + 60 * np.random.rand(n)
    rho_b = 2.71 - 0.4 * np.random.rand(n)
    nphi = 0.05 + 0.25 * np.random.rand(n)
    rt = 5.0 + 100.0 * np.random.rand(n)

    vsh = vshale_larionov_older(gr, 15, 100)
    dphi = density_porosity(rho_b, 2.71, 1.00)
    phi_nd = neutron_density_porosity(nphi, dphi)
    phie = effective_porosity(phi_nd, vsh)

    rw = 0.03
    sw = sw_archie(rt, rw, phie, a=1.0, m=2.0, n=2.0)
    sw_ind = sw_indonesian(rt, rw, phie, vsh, rsh=5.0)

    k = perm_timur(phie, np.maximum(sw, 0.1))
    flag = net_pay_flag(phie, sw, vsh)

    print(f"Vsh range         : {vsh.min():.3f} – {vsh.max():.3f}")
    print(f"PHIE range        : {phie.min():.3f} – {phie.max():.3f}")
    print(f"Sw (Archie) range : {sw.min():.3f} – {sw.max():.3f}")
    print(f"Sw (Indon.) range : {sw_ind.min():.3f} – {sw_ind.max():.3f}")
    print(f"Perm range        : {k.min():.2f} – {k.max():.2f} md")
    print(f"Net pay intervals : {flag.sum()} / {n}")
    print()


if __name__ == "__main__":
    demo()
