"""
article8_thz_porosity.py
=========================
Implementation of ideas from:

    Eichmann, S.L., Bouchard, J., Ow, H., Petkie, D., Poitzsch, M.E.
    "THz Imaging to Map the Lateral Microporosity Distribution in
     Carbonate Rocks"
    Petrophysics, Vol. 64, No. 3 (June 2023), pp. 438-447
    DOI: 10.30632/PJV64N3-2023a8

The workflow rests on three weighings (dry, fully saturated, partially
saturated after centrifugation) and three terahertz time-domain
spectroscopy intensity maps (THz_dry, THz_sat, THz_cent).

Mass-balance porosities (paragraph 'Porosity and Microporosity by Mass')

    phi_total  = (m_sat - m_dry) / (rho_w * V_bulk)
    phi_micro  = (m_cent - m_dry_f) / (rho_w * V_bulk)
    phi_macro  = phi_total - phi_micro

THz attenuation maps (Beer-Lambert through the slab thickness h):

    A(i,j) = -ln( I(i,j) / I_dry(i,j) )

The contribution of water in a pixel of thickness h to the
log-attenuation is proportional to phi(i,j)*h.  Therefore a
calibration constant k can be extracted from the bulk porosity:

    phi_map = (A_map / mean(A_map)) * phi_bulk

This gives a 2D porosity map; subtracting the centrifuged-state map
from the saturated-state map yields a macroporosity map; the
centrifuged-state map by itself yields the microporosity map.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Bulk (mass-balance) porosities
# ---------------------------------------------------------------------------
def porosity_total(m_dry: float, m_sat: float, V_bulk: float,
                   rho_w: float = 1000.0) -> float:
    """Total porosity = (m_sat - m_dry) / (rho_w * V_bulk)."""
    return (m_sat - m_dry) / (rho_w * V_bulk)


def porosity_micro(m_dry_f: float, m_cent: float, V_bulk: float,
                   rho_w: float = 1000.0) -> float:
    """Micro-porosity from the centrifuged-state mass-balance."""
    return (m_cent - m_dry_f) / (rho_w * V_bulk)


def porosity_macro(phi_total: float, phi_micro: float) -> float:
    return phi_total - phi_micro


# ---------------------------------------------------------------------------
# THz attenuation map (Beer-Lambert)
# ---------------------------------------------------------------------------
def thz_attenuation(I: np.ndarray, I_dry: np.ndarray) -> np.ndarray:
    """A = -ln(I / I_dry).  Both arrays have the same shape."""
    safe = np.where(I_dry == 0, 1e-30, I_dry)
    ratio = np.clip(np.asarray(I) / safe, 1e-30, None)
    return -np.log(ratio)


# ---------------------------------------------------------------------------
# Map calibration: scale the attenuation map so that its mean equals
# the independently measured bulk porosity.
# ---------------------------------------------------------------------------
def porosity_map(att_map: np.ndarray, phi_bulk: float) -> np.ndarray:
    """
    Scale the attenuation map to match the bulk porosity in average.
    Pixels with negative attenuation (noise) are clipped to zero.
    """
    att = np.maximum(att_map, 0.0)
    m = att.mean()
    if m <= 0:
        return np.zeros_like(att)
    return att * (phi_bulk / m)


def porosity_maps_workflow(I_sat: np.ndarray, I_cent: np.ndarray,
                           I_dry: np.ndarray,
                           m_dry: float, m_sat: float, m_cent: float,
                           m_dry_f: float, V_bulk: float,
                           rho_w: float = 1000.0) -> dict:
    """
    Implements the full workflow: from three THz maps + four masses
    return the three porosity maps and their bulk values.
    """
    phi_t = porosity_total(m_dry, m_sat, V_bulk, rho_w)
    phi_mu = porosity_micro(m_dry_f, m_cent, V_bulk, rho_w)
    phi_M = porosity_macro(phi_t, phi_mu)

    A_sat = thz_attenuation(I_sat, I_dry)
    A_cent = thz_attenuation(I_cent, I_dry)

    phi_total_map = porosity_map(A_sat, phi_t)
    phi_micro_map = porosity_map(A_cent, phi_mu)
    phi_macro_map = np.maximum(phi_total_map - phi_micro_map, 0.0)

    return {
        "phi_total": phi_t,
        "phi_micro": phi_mu,
        "phi_macro": phi_M,
        "phi_total_map": phi_total_map,
        "phi_micro_map": phi_micro_map,
        "phi_macro_map": phi_macro_map,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    """Synthetic-data test for module 8 (THz imaging of porosity)."""
    print("[article8] testing mass-balance porosity calculations ...")
    rho_w = 1000.0  # kg/m^3
    V_bulk = 1e-5   # 10 cm^3 in SI
    m_dry, m_sat = 0.0220, 0.0240        # kg  -> 2.0 g of water
    m_cent, m_dry_f = 0.0225, 0.0220     # kg  -> 0.5 g micro water
    phi_t = porosity_total(m_dry, m_sat, V_bulk, rho_w)
    phi_mu = porosity_micro(m_dry_f, m_cent, V_bulk, rho_w)
    phi_M = porosity_macro(phi_t, phi_mu)
    print(f"           phi_total = {phi_t:.3f}, phi_micro = {phi_mu:.3f}, "
          f"phi_macro = {phi_M:.3f}")
    assert abs(phi_t - 0.20) < 1e-6
    assert abs(phi_mu - 0.05) < 1e-6
    assert abs(phi_M - 0.15) < 1e-6

    print("[article8] testing THz attenuation -> porosity map workflow ...")
    rng = np.random.default_rng(11)
    H, W = 40, 80
    # Synthetic dry intensity field with smooth background variations
    I_dry = 1.0 + 0.05 * rng.standard_normal((H, W))

    # Underlying ground-truth porosity:
    # right half of the slab is more porous than the left half.
    true_phi_total = np.full((H, W), 0.15)
    true_phi_total[:, W // 2:] = 0.25
    true_phi_micro = np.full((H, W), 0.04)
    true_phi_micro[:, W // 2:] = 0.06

    # Beer-Lambert with a known absorption-per-porosity-unit
    k = 1.0
    I_sat = I_dry * np.exp(-k * true_phi_total)
    I_cent = I_dry * np.exp(-k * true_phi_micro)

    # Bulk masses consistent with the ground-truth maps
    bulk_phi_t = float(true_phi_total.mean())
    bulk_phi_mu = float(true_phi_micro.mean())
    m_dry = 0.020
    m_sat = m_dry + rho_w * V_bulk * bulk_phi_t
    m_cent = m_dry + rho_w * V_bulk * bulk_phi_mu

    out = porosity_maps_workflow(I_sat, I_cent, I_dry,
                                 m_dry=m_dry, m_sat=m_sat, m_cent=m_cent,
                                 m_dry_f=m_dry, V_bulk=V_bulk, rho_w=rho_w)

    print(f"           bulk phi_total = {out['phi_total']:.3f}  "
          f"(expected {bulk_phi_t:.3f})")
    print(f"           bulk phi_micro = {out['phi_micro']:.3f}  "
          f"(expected {bulk_phi_mu:.3f})")
    assert abs(out["phi_total"] - bulk_phi_t) < 1e-3
    assert abs(out["phi_micro"] - bulk_phi_mu) < 1e-3

    # The right-half mean porosity should be > the left-half mean porosity
    right_mean = out["phi_total_map"][:, W // 2:].mean()
    left_mean = out["phi_total_map"][:, :W // 2].mean()
    print(f"           porosity map: left mean={left_mean:.3f}  "
          f"right mean={right_mean:.3f}")
    assert right_mean > left_mean
    print("[article8] OK")


if __name__ == "__main__":
    test_all()
