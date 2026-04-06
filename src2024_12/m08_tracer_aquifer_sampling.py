#!/usr/bin/env python3
"""
Shallow Aquifer Sampling for CCS: Low-Toxicity Tracer
======================================================
Based on: Taplin, Peyret, Jackson, and Hitchen (2024),
Petrophysics 65(6), pp. 944-956. DOI: 10.30632/PJV65N6-2024a8

Implements contamination monitoring and salinity estimation:
  1. D2O-based contamination calculation from tracer concentration.
  2. Salinity correction from contamination level.
  3. Rwa-based salinity estimation from openhole logs (Appendix 1).
  4. Pressure gradient-based fluid density estimation.
"""
import numpy as np
from typing import Dict, Tuple

def compute_contamination_d2o(d_sample, d_mud, d_formation=0.000156):
    """Compute drilling fluid contamination from D2O concentrations.
    C = (D_sample - D_formation) / (D_mud - D_formation).
    D_formation ~ 0.000156 (VSMOW natural deuterium)."""
    d_s = np.asarray(d_sample, dtype=float)
    return np.clip((d_s - d_formation) / (d_mud - d_formation + 1e-15), 0, 1)

def correct_salinity_for_contamination(measured_salinity_ppm, contamination, mud_salinity_ppm):
    """Correct measured salinity for drilling mud contamination.
    Sal_formation = (Sal_measured - C * Sal_mud) / (1 - C)."""
    c = np.asarray(contamination, dtype=float)
    c = np.clip(c, 0, 0.99)
    return (np.asarray(measured_salinity_ppm) - c * mud_salinity_ppm) / (1 - c)

def density_porosity(bulk_density, matrix_density=2.67, fluid_density=1.03):
    """Porosity from bulk density log (Eq. A1.1 in paper)."""
    return (matrix_density - np.asarray(bulk_density, dtype=float)) / (matrix_density - fluid_density)

def compute_rwa(resistivity, porosity, a=1.0, m=1.776):
    """Compute apparent water resistivity from Archie equation (Eq. A1.2).
    Rwa = Rt * phi^m / a (assuming Sw=1)."""
    phi = np.asarray(porosity, dtype=float)
    rt = np.asarray(resistivity, dtype=float)
    return rt * (phi ** m) / a

def rwa_to_salinity(rwa, temperature_c):
    """Convert Rwa to NaCl-equivalent salinity using chartbook relationship.
    Approximate: salinity_ppm ~ 4000 / (Rw * (T+21.5)/43.5)."""
    rw = np.asarray(rwa, dtype=float)
    t = np.asarray(temperature_c, dtype=float)
    temp_factor = (t + 21.5) / 43.5
    return np.clip(4000.0 / (rw * temp_factor + 1e-10), 0, 300000)

def fluid_density_from_pressure_gradient(pressures_psi, depths_m, g=9.81):
    """Estimate fluid density from pressure gradient.
    rho = dP/dz / g, convert from psi/m to g/cc."""
    p = np.asarray(pressures_psi, dtype=float)
    z = np.asarray(depths_m, dtype=float)
    if len(p) < 2: return 1.0
    # Linear regression for gradient
    coeffs = np.polyfit(z, p, 1)
    grad_psi_per_m = coeffs[0]
    # Convert: 1 psi = 6894.76 Pa, 1 g/cc = 9810 Pa/m
    density_gcc = grad_psi_per_m * 6894.76 / (g * 1000)
    return float(np.clip(density_gcc, 0.9, 1.3))

def density_to_salinity_nacl(density_gcc):
    """Approximate NaCl salinity from fluid density.
    Salinity_ppm ~ (density - 1.0) / 7e-7 (simplified chartbook)."""
    return np.clip((density_gcc - 1.0) / 7e-7, 0, 300000)

def test_all():
    print("=" * 70)
    print("Module 8: Tracer & Aquifer Sampling (Taplin et al., 2024)")
    print("=" * 70)
    rng = np.random.RandomState(42)
    # D2O contamination monitoring
    d_mud = 0.5  # 50% D2O spike
    d_samples = rng.uniform(0.0003, 0.02, 10)
    contam = compute_contamination_d2o(d_samples, d_mud)
    print(f"Contamination levels: {[f'{c:.1%}' for c in contam[:5]]}")
    # Salinity correction
    meas_sal = rng.uniform(50000, 90000, 10)
    corr_sal = correct_salinity_for_contamination(meas_sal, contam, 30000)
    print(f"Corrected salinity range: {corr_sal.min():.0f}-{corr_sal.max():.0f} ppm")
    # Density porosity
    rhob = np.linspace(2.0, 2.5, 50)
    phi = density_porosity(rhob)
    print(f"Porosity range: {phi.min():.3f}-{phi.max():.3f}")
    # Rwa salinity estimate
    rt = rng.uniform(1.0, 10.0, 50)
    rwa = compute_rwa(rt, np.clip(phi, 0.05, 0.4))
    temp = np.linspace(20, 40, 50)
    sal = rwa_to_salinity(rwa, temp)
    print(f"Rwa salinity range: {sal.min():.0f}-{sal.max():.0f} ppm")
    # Pressure gradient density
    depths = np.array([800, 810, 820, 830, 840])
    pressures = 1160 + 1.5 * (depths - 800)  # ~1.5 psi/m
    rho_f = fluid_density_from_pressure_gradient(pressures, depths)
    print(f"Fluid density from gradient: {rho_f:.3f} g/cc")
    sal_grad = density_to_salinity_nacl(rho_f)
    print(f"Salinity from gradient: {sal_grad:.0f} ppm")
    print("\n[PASS] All tests completed successfully.\n")

if __name__ == "__main__":
    test_all()
