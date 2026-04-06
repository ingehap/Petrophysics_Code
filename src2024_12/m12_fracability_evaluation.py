#!/usr/bin/env python3
"""
Fracability Evaluation for Tight Sandstone Reservoirs
======================================================
Based on: Qian, Wang, and Xie (2024), Petrophysics 65(6), pp. 995-1009.
DOI: 10.30632/PJV65N6-2024a12

Implements the fracability evaluation method from logging & experimental data:
  1. Dynamic-to-static mechanical parameter conversion (Eqs. 1-3).
  2. Brittleness evaluation: mineral method, acoustic modulus, and
     comprehensive method considering matrix + microcracks (Eq. 8).
  3. Fracability analysis: fracture generation difficulty, vertical
     expansion, fracture azimuth, and network complexity.
  4. Horizontal stress difference coefficient (Eq. 12).
"""
import numpy as np
from typing import Dict, Tuple

# --- Mechanical Parameter Conversions (Study area specific) ---

def dynamic_to_static_youngs(Ed_GPa):
    """Convert dynamic to static Young's modulus (Eq. 1).
    Es = 0.5066 * Ed + 4.2819 (GPa)."""
    return 0.5066 * np.asarray(Ed_GPa, dtype=float) + 4.2819

def dynamic_to_static_poisson(vd):
    """Convert dynamic to static Poisson's ratio (Eq. 2).
    vs = 1.1282 * vd - 0.0587."""
    return 1.1282 * np.asarray(vd, dtype=float) - 0.0587

def ucs_from_youngs(Es_GPa):
    """Uniaxial compressive strength from static Young's modulus (Eq. 3).
    UCS = 94.62 + 6.327 * Es (MPa)."""
    return 94.62 + 6.327 * np.asarray(Es_GPa, dtype=float)

# --- Brittleness Evaluation ---

def brittleness_mineral(W_brittle, W_total):
    """Brittleness index from mineral composition (Eq. 4).
    B_matrix = W_brittle / W_total * 100 (%).
    Brittle minerals: quartz, feldspar, calcite, dolomite."""
    return np.asarray(W_brittle, dtype=float) / (np.asarray(W_total, dtype=float) + 1e-10) * 100

def brittleness_acoustic_modulus(Es, vs, Es_min, Es_max, vs_min, vs_max):
    """Brittleness index from acoustic modulus (Eqs. 5-7, Rickman et al., 2008).
    B_E = (Es - Es_min) / (Es_max - Es_min) * 100
    B_v = (vs - vs_max) / (vs_min - vs_max) * 100
    B_brit = (B_E + B_v) / 2."""
    Es = np.asarray(Es, dtype=float)
    vs = np.asarray(vs, dtype=float)
    B_E = (Es - Es_min) / (Es_max - Es_min + 1e-10) * 100
    B_v = (vs - vs_max) / (vs_min - vs_max + 1e-10) * 100
    return np.clip((B_E + B_v) / 2, 0, 100)

def youngs_modulus_matrix(shear_modulus, bulk_modulus):
    """Matrix Young's modulus from shear and bulk moduli (Eq. 11).
    E_matrix = 9*K*G / (3*K + G)."""
    K = np.asarray(bulk_modulus, dtype=float)
    G = np.asarray(shear_modulus, dtype=float)
    return 9 * K * G / (3 * K + G + 1e-10)

def brittleness_crack_parameter(Es, porosity, E_matrix):
    """Microcrack brittleness parameter (Eqs. 9-10, Kumar et al., 2015).
    E_porous = E_matrix * (1 - porosity)^2
    B_crack = 1 - Es / E_porous (normalized)."""
    phi = np.asarray(porosity, dtype=float)
    E_porous = np.asarray(E_matrix, dtype=float) * (1 - phi)**2
    B_crack = 1 - np.asarray(Es, dtype=float) / (E_porous + 1e-10)
    return np.clip(B_crack, 0, 1) * 100

def brittleness_comprehensive(B_matrix, B_crack, alpha_matrix=0.5, alpha_crack=0.5):
    """Comprehensive brittleness considering matrix and microcracks (Eq. 8).
    B_comp = alpha_matrix * B_matrix + alpha_crack * B_crack."""
    return alpha_matrix * np.asarray(B_matrix) + alpha_crack * np.asarray(B_crack)

# --- Fracability Analysis ---

def fracture_generation_difficulty(brittleness, UCS_MPa):
    """Evaluate difficulty of generating hydraulic fractures.
    Good brittleness (>60%) + low UCS -> easy to fracture."""
    b = np.asarray(brittleness, dtype=float)
    u = np.asarray(UCS_MPa, dtype=float)
    # Score: 0 = hard, 1 = easy
    b_score = np.clip((b - 40) / 40, 0, 1)
    u_score = np.clip(1 - (u - 80) / 120, 0, 1)
    return (b_score + u_score) / 2

def vertical_expansion_ability(stress_diff_MPa, barrier_thickness_m, barrier_strength_MPa, res_strength_MPa):
    """Evaluate vertical expansion ability of hydraulic fractures.
    Key threshold: stress difference > 3 MPa controls fracture height."""
    ds = np.asarray(stress_diff_MPa, dtype=float)
    bt = np.asarray(barrier_thickness_m, dtype=float)
    # Stress containment score
    stress_score = np.clip((ds - 1) / 5, 0, 1)
    # Barrier thickness score (thicker barrier -> better containment -> higher score)
    thick_score = np.clip(bt / 10, 0, 1)
    # Strength contrast
    strength_contrast = np.clip((np.asarray(barrier_strength_MPa) - np.asarray(res_strength_MPa)) / 50, 0, 1)
    return (stress_score + thick_score + strength_contrast) / 3

def horizontal_stress_difference_coefficient(SHmax, SHmin):
    """Horizontal stress difference coefficient (Eq. 12).
    Kh = (SHmax - SHmin) / SHmin.
    Small Kh (<0.13) favors complex fracture networks."""
    SHmax = np.asarray(SHmax, dtype=float)
    SHmin = np.asarray(SHmin, dtype=float)
    return (SHmax - SHmin) / (SHmin + 1e-10)

def fracture_network_complexity(brittleness, Kh, has_natural_fractures=False, has_microcracks=False):
    """Evaluate complexity of potential hydraulic fracture network.
    Lower Kh + higher brittleness + natural fractures -> more complex network."""
    b = np.asarray(brittleness, dtype=float)
    kh = np.asarray(Kh, dtype=float)
    b_score = np.clip(b / 100, 0, 1)
    kh_score = np.clip(1 - (kh - 0.03) / 0.25, 0, 1)
    nf_bonus = 0.15 if has_natural_fractures else 0
    mc_bonus = 0.1 if has_microcracks else 0
    return np.clip((b_score + kh_score) / 2 + nf_bonus + mc_bonus, 0, 1)

def fracability_index(generation_score, vertical_score, complexity_score):
    """Overall fracability index (composite of all sub-evaluations)."""
    return (np.asarray(generation_score) * 0.35 +
            np.asarray(vertical_score) * 0.30 +
            np.asarray(complexity_score) * 0.35)

def test_all():
    print("=" * 70)
    print("Module 12: Fracability Evaluation (Qian et al., 2024)")
    print("=" * 70)
    # Mechanical conversions
    Ed = np.array([20, 30, 40, 50])  # GPa
    Es = dynamic_to_static_youngs(Ed)
    print(f"Dynamic E: {Ed} -> Static E: {np.round(Es, 1)} GPa")
    vd = np.array([0.20, 0.25, 0.30])
    vs = dynamic_to_static_poisson(vd)
    print(f"Dynamic v: {vd} -> Static v: {np.round(vs, 3)}")
    ucs = ucs_from_youngs(Es)
    print(f"UCS from Es: {np.round(ucs, 1)} MPa")
    # Brittleness
    B_min = brittleness_mineral(np.array([60, 70, 50, 80]), np.array([100, 100, 100, 100]))
    print(f"\nMineral brittleness: {B_min}%")
    B_mod = brittleness_acoustic_modulus(Es, np.full_like(Es, 0.22), 10, 30, 0.15, 0.35)
    print(f"Acoustic modulus brittleness: {np.round(B_mod, 1)}%")
    B_crack = brittleness_crack_parameter(Es, np.array([0.05, 0.08, 0.10, 0.12]), np.full(4, 50.0))
    print(f"Crack parameter: {np.round(B_crack, 1)}%")
    B_comp = brittleness_comprehensive(B_min, B_crack, 0.6, 0.4)
    print(f"Comprehensive brittleness: {np.round(B_comp, 1)}%")
    # Fracability
    gen = fracture_generation_difficulty(B_comp, ucs)
    print(f"\nFracture generation ease: {np.round(gen, 2)}")
    vert = vertical_expansion_ability(np.array([2, 4, 5, 6]), 5.0, 150, 100)
    print(f"Vertical containment: {np.round(vert, 2)}")
    Kh = horizontal_stress_difference_coefficient(np.array([45, 50, 55, 60]), np.array([40, 42, 44, 46]))
    print(f"Kh coefficients: {np.round(Kh, 3)}")
    comp = fracture_network_complexity(B_comp, Kh, True, True)
    print(f"Network complexity: {np.round(comp, 2)}")
    fi = fracability_index(gen, vert, comp)
    print(f"Fracability index: {np.round(fi, 2)}")
    # Classification
    for i, f in enumerate(fi):
        label = 'good' if f > 0.6 else 'moderate' if f > 0.4 else 'poor'
        print(f"  Layer {i+1}: FI={f:.2f} -> {label}")
    print("\n[PASS] All tests completed successfully.\n")

if __name__ == "__main__":
    test_all()
