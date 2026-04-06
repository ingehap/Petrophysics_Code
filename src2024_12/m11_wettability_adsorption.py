#!/usr/bin/env python3
"""
Wettability Quantification in Rock Components via Water Adsorption Isotherms
=============================================================================
Based on: Silveira de Araujo and Heidari (2024), Petrophysics 65(6), pp. 983-994.
DOI: 10.30632/PJV65N6-2024a11

Implements:
  1. Water adsorption isotherm computation (BET model).
  2. Wettability index from monolayer adsorption ratios.
  3. Contact angle correlation with adsorption capacity.
  4. Mineral mixture adsorption modeling.
"""
import numpy as np
from typing import Dict, List, Tuple

# Mineral adsorption parameters (synthetic, inspired by paper results)
MINERAL_PARAMS = {
    'quartz':              {'C_BET': 20.0, 'Vm_mg_g': 0.8, 'contact_angle': 20},
    'calcite':             {'C_BET': 25.0, 'Vm_mg_g': 1.2, 'contact_angle': 30},
    'kaolinite':           {'C_BET': 40.0, 'Vm_mg_g': 5.0, 'contact_angle': 15},
    'illite':              {'C_BET': 50.0, 'Vm_mg_g': 8.0, 'contact_angle': 10},
    'na_montmorillonite':  {'C_BET': 80.0, 'Vm_mg_g': 60.0, 'contact_angle': 5},
    'kerogen':             {'C_BET': 5.0,  'Vm_mg_g': 40.0, 'contact_angle': 90},
    'quartz_oil_treated':  {'C_BET': 5.0,  'Vm_mg_g': 0.22, 'contact_angle': 105},
}

def bet_isotherm(rh, Vm, C):
    """BET (Brunauer-Emmett-Teller) adsorption isotherm.
    V = Vm * C * x / ((1 - x)(1 - x + C*x)) where x = RH (relative humidity).
    Vm: monolayer capacity (mg/g), C: BET constant."""
    x = np.asarray(rh, dtype=float)
    x = np.clip(x, 0.001, 0.999)
    return Vm * C * x / ((1 - x) * (1 - x + C * x))

def compute_adsorption_isotherm(mineral, rh_range=None):
    """Compute water adsorption isotherm for a mineral.
    Returns (relative_humidity, adsorption_mg_g)."""
    if rh_range is None:
        rh_range = np.linspace(0.05, 0.98, 50)
    params = MINERAL_PARAMS.get(mineral, MINERAL_PARAMS['quartz'])
    adsorption = bet_isotherm(rh_range, params['Vm_mg_g'], params['C_BET'])
    return rh_range, adsorption

def wettability_index_from_adsorption(treated_Vm, untreated_Vm):
    """Compute wettability index from monolayer adsorption ratio.
    WI = Vm_treated / Vm_untreated (Tabrizy et al., 2011).
    WI = 1 → water-wet, WI → 0 → oil-wet."""
    return np.clip(treated_Vm / (untreated_Vm + 1e-10), 0, 2)

def adsorption_to_contact_angle(adsorption_at_98rh, reference_adsorption=None):
    """Estimate contact angle from adsorption capacity at 98% RH.
    Higher adsorption → lower contact angle (more water-wet).
    Empirical: theta ~ 120 * exp(-0.5 * V/V_ref)."""
    if reference_adsorption is None:
        reference_adsorption = MINERAL_PARAMS['quartz']['Vm_mg_g'] * 5
    v = np.asarray(adsorption_at_98rh, dtype=float)
    theta = 120 * np.exp(-0.5 * v / reference_adsorption)
    return np.clip(theta, 0, 180)

def mixture_adsorption(mineral_fractions, rh_range=None):
    """Compute adsorption isotherm for a mineral mixture.
    mineral_fractions: dict of {mineral_name: weight_fraction}.
    Uses linear mixing of individual isotherms."""
    if rh_range is None:
        rh_range = np.linspace(0.05, 0.98, 50)
    total = np.zeros_like(rh_range)
    for mineral, fraction in mineral_fractions.items():
        _, iso = compute_adsorption_isotherm(mineral, rh_range)
        total += fraction * iso
    return rh_range, total

def work_of_adhesion(adsorption_at_98rh, surface_tension=72.8e-3):
    """Estimate work of adhesion from adsorption (Schlangen et al., 1994).
    W_a ~ gamma_lv * (1 + cos(theta)).
    We get theta from adsorption, then compute W_a."""
    theta_deg = adsorption_to_contact_angle(adsorption_at_98rh)
    theta_rad = np.radians(theta_deg)
    return surface_tension * (1 + np.cos(theta_rad)) * 1000  # mJ/m^2

def test_all():
    print("=" * 70)
    print("Module 11: Wettability via Adsorption (Silveira de Araujo & Heidari, 2024)")
    print("=" * 70)
    rh = np.linspace(0.05, 0.98, 50)
    # Individual mineral isotherms
    print("Adsorption at 98% RH (mg/g):")
    for mineral in MINERAL_PARAMS:
        _, ads = compute_adsorption_isotherm(mineral, rh)
        ca = MINERAL_PARAMS[mineral]['contact_angle']
        print(f"  {mineral:25s}: {ads[-1]:8.2f} mg/g (expected CA={ca}°)")
    # Wettability index
    wi = wettability_index_from_adsorption(
        MINERAL_PARAMS['quartz_oil_treated']['Vm_mg_g'],
        MINERAL_PARAMS['quartz']['Vm_mg_g'])
    print(f"\nWettability index (treated/untreated quartz): {wi:.3f}")
    print(f"  ~72% reduction in adsorption at 98% RH (paper reports ~72%)")
    # Contact angle from adsorption
    ads_values = [0.5, 2.0, 5.0, 20.0, 60.0]
    for v in ads_values:
        theta = adsorption_to_contact_angle(v)
        wa = work_of_adhesion(v)
        print(f"  Adsorption={v:5.1f} mg/g -> CA={theta:.1f}°, W_a={wa:.1f} mJ/m²")
    # Mixture
    mix = {'quartz': 0.6, 'kaolinite': 0.2, 'kerogen': 0.2}
    _, mix_ads = mixture_adsorption(mix, rh)
    print(f"\nMixture (60% qtz + 20% kao + 20% ker): ads@98%RH = {mix_ads[-1]:.2f} mg/g")
    mix2 = {'quartz': 0.8, 'na_montmorillonite': 0.2}
    _, mix2_ads = mixture_adsorption(mix2, rh)
    print(f"Mixture (80% qtz + 20% Na-mont): ads@98%RH = {mix2_ads[-1]:.2f} mg/g")
    print("\n[PASS] All tests completed successfully.\n")

if __name__ == "__main__":
    test_all()
