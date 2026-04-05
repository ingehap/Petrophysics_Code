#!/usr/bin/env python3
"""
Article 10: That Pesky and Unpredictable Neutron Log Response in Shales
Author: John Rasmus
Ref: Petrophysics, Vol. 66, No. 5 (October 2025), pp. 887-893.
     DOI: 10.30632/PJV66N5-2025a10

Implements:
  - Migration length (Lm), slowing-down length (Ls), diffusion length (Ld)
    calculations for mineral mixtures (SNUPAR-like)
  - Neutron porosity transforms for clean lithologies (SS, LS, DOL)
  - Effective Lm* accounting for tool/borehole
  - Shale neutron response modelling (linear vs nonlinear mixing)
  - Apparent porosity for clay-mineral mixtures
  - Slowing power index (SPI) based mineral mixing
"""

import numpy as np


# ---------------------------------------------------------------------------
# Nuclear parameters for pure minerals / fluids
# (SNUPAR-like lookup; values are representative, not exact vendor data)
# ---------------------------------------------------------------------------

NUCLEAR_PARAMS = {
    # mineral:       (Lm cm, Ls cm, Ld cm, density g/cm3, SPI relative to LS)
    'quartz':        (16.0, 12.0, 3.5, 2.65, 1.04),
    'calcite':       (15.5, 11.5, 3.3, 2.71, 1.00),  # reference = limestone
    'dolomite':      (14.8, 11.0, 3.1, 2.87, 1.10),
    'illite_wet':    (10.5,  7.5, 2.5, 2.65, 1.50),   # includes bound water
    'kaolinite_wet': (11.0,  8.0, 2.6, 2.60, 1.40),
    'smectite_wet':  ( 9.5,  7.0, 2.3, 2.30, 1.65),
    'chlorite_wet':  (10.0,  7.2, 2.4, 2.80, 1.55),
    'water':         ( 6.0,  4.2, 1.5, 1.00, 3.00),
    'salt':          (20.0, 15.0, 5.0, 2.17, 0.20),
    'anhydrite':     (17.5, 13.0, 4.0, 2.96, 0.55),
}


# ---------------------------------------------------------------------------
# Migration length for a mixture
# ---------------------------------------------------------------------------

def migration_length(mineral_volumes, porosity, fluid='water'):
    """Compute bulk migration length Lm for a mineral + fluid mixture.

    mineral_volumes: dict mineral_name -> volume fraction (of solid).
    porosity: total porosity (fraction).
    Fluids fill the porosity.

    Lm_mix = 1 / sum(Vi / Lm_i)  (harmonic average, approximate).
    """
    inv_lm = 0.0
    for mineral, vol_frac in mineral_volumes.items():
        if mineral in NUCLEAR_PARAMS:
            lm = NUCLEAR_PARAMS[mineral][0]
            inv_lm += vol_frac * (1.0 - porosity) / lm
    if fluid in NUCLEAR_PARAMS:
        lm_fl = NUCLEAR_PARAMS[fluid][0]
        inv_lm += porosity / lm_fl
    return 1.0 / max(inv_lm, 1e-10)


def slowing_down_length(mineral_volumes, porosity, fluid='water'):
    """Bulk slowing down length Ls (harmonic average)."""
    inv_ls = 0.0
    for mineral, vol_frac in mineral_volumes.items():
        if mineral in NUCLEAR_PARAMS:
            ls = NUCLEAR_PARAMS[mineral][1]
            inv_ls += vol_frac * (1.0 - porosity) / ls
    if fluid in NUCLEAR_PARAMS:
        ls_fl = NUCLEAR_PARAMS[fluid][1]
        inv_ls += porosity / ls_fl
    return 1.0 / max(inv_ls, 1e-10)


# ---------------------------------------------------------------------------
# Neutron porosity transforms (clean lithology)
# ---------------------------------------------------------------------------

def neutron_porosity_limestone(Lm_star):
    """Apparent limestone porosity from effective Lm* (approximate transform).
    Calibrated so that Lm*=6.0 -> phi=1.0, Lm*=15.5 -> phi=0.0.
    """
    phi = (15.5 - Lm_star) / (15.5 - 6.0)
    return np.clip(phi, -0.05, 1.0)


def neutron_porosity_sandstone(Lm_star):
    """Apparent sandstone porosity from effective Lm*."""
    phi = (16.0 - Lm_star) / (16.0 - 6.0)
    return np.clip(phi, -0.05, 1.0)


def neutron_porosity_dolomite(Lm_star):
    """Apparent dolomite porosity from effective Lm*."""
    phi = (14.8 - Lm_star) / (14.8 - 6.0)
    return np.clip(phi, -0.05, 1.0)


# ---------------------------------------------------------------------------
# Effective Lm* (tool + borehole effect)
# ---------------------------------------------------------------------------

def effective_lm_star(Lm_bulk, a=0.85, b=1.2):
    """Empirical transform from bulk Lm to tool-effective Lm*.
    Lm* = a * Lm + b   (linear approximation of MCNP calibration).
    """
    return a * np.asarray(Lm_bulk, dtype=float) + b


# ---------------------------------------------------------------------------
# Shale neutron response (linear mixing law and its error)
# ---------------------------------------------------------------------------

def apparent_shale_porosity_ss(vol_shale, shale_endpoint_pu=20.4):
    """Apparent SS porosity in a shale using linear mixing law.
    TNPH_measured = V_shale * endpoint + V_quartz * 0 + V_porosity * 1.0

    Returns apparent porosity (in fraction, divide by 100 if needed).
    """
    return vol_shale * shale_endpoint_pu / 100.0


def nonlinear_shale_response(vol_shale, porosity, shale_minerals=None):
    """Model nonlinear TNPH response for shale + water mixtures.

    Uses Lm-based calculation rather than linear mixing.
    """
    if shale_minerals is None:
        shale_minerals = {'illite_wet': 0.5, 'quartz': 0.5}

    scaled = {}
    for m, v in shale_minerals.items():
        scaled[m] = v * vol_shale
    # add quartz to fill remaining solid
    solid_sum = sum(scaled.values())
    if solid_sum < (1.0 - porosity):
        scaled['quartz'] = scaled.get('quartz', 0) + (1.0 - porosity - solid_sum)

    Lm_bulk = migration_length(scaled, porosity)
    Lm_star = effective_lm_star(Lm_bulk)
    return neutron_porosity_sandstone(Lm_star)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Article 10: Neutron Log Response in Shales Demo ===\n")

    # Clean sandstone at various porosities
    phis = [0.0, 0.10, 0.20, 0.30]
    print("Clean sandstone Lm and apparent TNPH_SS:")
    for phi in phis:
        Lm = migration_length({'quartz': 1.0}, phi)
        Lm_s = effective_lm_star(Lm)
        tnph = neutron_porosity_sandstone(Lm_s)
        print(f"  phi={phi:.2f}  Lm={Lm:.2f} cm  Lm*={Lm_s:.2f} cm  TNPH_SS={tnph:.3f}")

    # Shale (50-50 illite-quartz) at varying porosity
    print("\n50-50 illite-quartz shale: nonlinear vs linear TNPH_SS:")
    shale_min = {'illite_wet': 0.5, 'quartz': 0.5}
    for phi in [0.0, 0.05, 0.10, 0.15, 0.20]:
        vsh = 1.0 - phi  # shale fills all solid
        tnph_nl = nonlinear_shale_response(vsh, phi, shale_min)
        tnph_lin = phi + apparent_shale_porosity_ss(vsh)
        print(f"  phi={phi:.2f}  TNPH(nonlinear)={tnph_nl:.3f}  "
              f"TNPH(linear)={tnph_lin:.3f}  error={abs(tnph_nl-tnph_lin):.4f}")
    print()


if __name__ == "__main__":
    demo()
