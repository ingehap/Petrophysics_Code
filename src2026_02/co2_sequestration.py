"""
Article 8: Effect of CO2 Sequestration on Carbonate Formation Integrity -
Aspects of Static Interactions.

Authors: Al-Hamad, Sindt, Ma, and Abdallah (2026)
DOI: 10.30632/PJV67N1-2026a8

Implements models for assessing the effect of CO2-saturated brine on
carbonate rock properties: porosity, permeability, NMR T2, ultrasonic
velocity, and resistivity changes under static and dynamic aging.

Key ideas implemented:
    - CO2 solubility in brine (pressure/temperature dependent)
    - pH estimation for CO2-saturated brine (carbonic acid)
    - Rock property change models (static vs. dynamic aging)
    - NMR T2 distribution shift due to surface dissolution
    - Dynamic elastic modulus from ultrasonic measurements

References
----------
Al-Hamad et al. (2026), Petrophysics, 67(1), 106-122.
Rosenbauer and Koksalan (2002); Steel et al. (2016).
Singer et al. (2023, 2024); Wang and Ehlig-Economides (2023).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CarbonateRockSample:
    """Properties of a carbonate rock sample before and after CO2 aging.

    Attributes
    ----------
    porosity : float
        Porosity (fraction).
    permeability : float
        Permeability (mD).
    resistivity : float
        Electrical resistivity (ohm-m).
    vp : float
        Compressional wave velocity (m/s).
    vs : float
        Shear wave velocity (m/s).
    bulk_density : float
        Bulk density (g/cc).
    calcite_fraction : float
        Fraction of calcite in mineralogy.
    dolomite_fraction : float
        Fraction of dolomite in mineralogy.
    """
    porosity: float = 0.15
    permeability: float = 10.0
    resistivity: float = 50.0
    vp: float = 4500.0
    vs: float = 2500.0
    bulk_density: float = 2.50
    calcite_fraction: float = 0.5
    dolomite_fraction: float = 0.5


def co2_solubility_in_brine(temperature: float,
                            pressure: float,
                            salinity: float = 0.0) -> float:
    """Estimate CO2 solubility in brine (mol/kg).

    Simplified model based on experimental data. CO2 solubility increases
    with pressure and decreases with temperature and salinity.

    Parameters
    ----------
    temperature : float
        Temperature (°C).
    pressure : float
        Pressure (psi).
    salinity : float
        Total dissolved solids (mass fraction, e.g. 0.1 for 10%).

    Returns
    -------
    float
        CO2 solubility in mol/kg of water.
    """
    # Simplified Duan model
    P_MPa = pressure * 0.00689476
    T_K = temperature + 273.15

    # Base solubility in pure water
    x_co2 = 0.034 * P_MPa / (1 + 0.015 * P_MPa) * np.exp(-0.001 * (T_K - 323.15)**2)

    # Salinity correction (salting-out effect)
    x_co2 *= (1 - 1.5 * salinity)

    return max(0.0, x_co2)


def ph_co2_saturated_brine(co2_mol_kg: float,
                           temperature: float) -> float:
    """Estimate pH of CO2-saturated brine.

    CO2 + H2O ⇌ H2CO3 ⇌ H+ + HCO3-

    Parameters
    ----------
    co2_mol_kg : float
        Dissolved CO2 concentration (mol/kg).
    temperature : float
        Temperature (°C).

    Returns
    -------
    float
        Estimated pH.
    """
    if co2_mol_kg <= 0:
        return 7.0

    # Ka1 for carbonic acid, temperature-dependent
    T_K = temperature + 273.15
    pKa1 = 6.35 + 0.01 * (T_K - 298.15)
    Ka1 = 10**(-pKa1)

    # Approximate pH from weak acid equilibrium
    # [H+] ≈ sqrt(Ka1 * C_CO2)
    h_plus = np.sqrt(Ka1 * co2_mol_kg)
    ph = -np.log10(h_plus + 1e-30)
    return float(np.clip(ph, 2.5, 7.0))


def static_aging_property_change(sample: CarbonateRockSample,
                                 months: float,
                                 co2_mol_kg: float,
                                 temperature: float) -> dict:
    """Model property changes during static aging in CO2-saturated brine.

    Static aging: rock immersed in a fixed volume of CO2-saturated brine.
    Changes are minimal (up to 6 months), mainly in NMR T2 distribution
    and slight pore surface morphology changes.

    Parameters
    ----------
    sample : CarbonateRockSample
        Initial rock properties.
    months : float
        Aging duration (months).
    co2_mol_kg : float
        CO2 concentration in brine (mol/kg).
    temperature : float
        Aging temperature (°C).

    Returns
    -------
    dict
        Property changes: delta_phi, delta_k, delta_resistivity,
        delta_vp, delta_vs, t2_shift_factor.
    """
    ph = ph_co2_saturated_brine(co2_mol_kg, temperature)
    acid_strength = max(0, 7.0 - ph)  # Higher = more acidic

    # Static aging: limited dissolution, mainly surface effects
    # Calcite dissolves faster than dolomite
    dissolution_rate = (sample.calcite_fraction * 0.5 +
                        sample.dolomite_fraction * 0.1) * acid_strength

    time_factor = 1 - np.exp(-0.3 * months)  # Saturating effect

    # Porosity change: minimal in static (<1% relative)
    delta_phi = sample.porosity * dissolution_rate * 0.005 * time_factor

    # Permeability change: slightly increases with dissolution
    delta_k = sample.permeability * dissolution_rate * 0.01 * time_factor

    # Resistivity: minimal change
    delta_resistivity = -sample.resistivity * dissolution_rate * 0.003 * time_factor

    # Ultrasonic: dolomite maintains stiffness; calcite shows changes
    delta_vp = -sample.vp * sample.calcite_fraction * dissolution_rate * 0.002 * time_factor
    delta_vs = -sample.vs * sample.calcite_fraction * dissolution_rate * 0.001 * time_factor

    # NMR T2 shift: clear change in pore-size distribution
    t2_shift = 1.0 + dissolution_rate * 0.1 * time_factor

    return {
        "delta_phi": delta_phi,
        "delta_k": delta_k,
        "delta_resistivity": delta_resistivity,
        "delta_vp": delta_vp,
        "delta_vs": delta_vs,
        "t2_shift_factor": t2_shift,
    }


def dynamic_aging_property_change(sample: CarbonateRockSample,
                                  months: float,
                                  co2_mol_kg: float,
                                  temperature: float,
                                  flow_rate_cc_min: float = 0.1) -> dict:
    """Model property changes during dynamic aging (continuous flooding).

    Dynamic aging produces more pronounced changes than static aging because
    fresh CO2-saturated brine continuously contacts the rock, preventing
    equilibrium and sustaining dissolution.

    Parameters
    ----------
    sample : CarbonateRockSample
        Initial rock properties.
    months : float
        Aging duration (months).
    co2_mol_kg : float
        CO2 concentration in brine (mol/kg).
    temperature : float
        Aging temperature (°C).
    flow_rate_cc_min : float
        Flow rate through the core (cc/min).

    Returns
    -------
    dict
        Same structure as static_aging_property_change, but larger magnitudes.
    """
    ph = ph_co2_saturated_brine(co2_mol_kg, temperature)
    acid_strength = max(0, 7.0 - ph)

    dissolution_rate = (sample.calcite_fraction * 1.5 +
                        sample.dolomite_fraction * 0.3) * acid_strength

    # Flow enhancement factor
    flow_factor = 1 + 2.0 * np.log1p(flow_rate_cc_min)
    time_factor = 1 - np.exp(-0.5 * months)

    effective_rate = dissolution_rate * flow_factor * time_factor

    # More pronounced changes than static
    delta_phi = sample.porosity * effective_rate * 0.02
    delta_k = sample.permeability * effective_rate * 0.05
    delta_resistivity = -sample.resistivity * effective_rate * 0.01
    delta_vp = -sample.vp * sample.calcite_fraction * effective_rate * 0.01
    delta_vs = -sample.vs * sample.calcite_fraction * effective_rate * 0.005
    t2_shift = 1.0 + effective_rate * 0.2

    return {
        "delta_phi": delta_phi,
        "delta_k": delta_k,
        "delta_resistivity": delta_resistivity,
        "delta_vp": delta_vp,
        "delta_vs": delta_vs,
        "t2_shift_factor": t2_shift,
    }


def dynamic_shear_modulus(vs: float, density: float) -> float:
    """Compute dynamic shear modulus from shear velocity and density.

    G = ρ * Vs²

    Parameters
    ----------
    vs : float
        Shear wave velocity (m/s).
    density : float
        Bulk density (kg/m³ or g/cc * 1000).

    Returns
    -------
    float
        Dynamic shear modulus (GPa).
    """
    if isinstance(density, float) and density < 10:
        density *= 1000  # Convert g/cc to kg/m³
    return density * vs**2 / 1e9


def assess_formation_integrity(sample: CarbonateRockSample,
                               aging_results: dict) -> dict:
    """Assess formation integrity after CO2 exposure.

    Evaluates whether property changes are within acceptable limits for
    CO2 sequestration projects.

    Parameters
    ----------
    sample : CarbonateRockSample
        Original rock properties.
    aging_results : dict
        Output from static_aging or dynamic_aging_property_change.

    Returns
    -------
    dict
        'porosity_change_pct', 'permeability_change_pct',
        'stiffness_maintained': bool, 'integrity_assessment': str.
    """
    phi_change = aging_results["delta_phi"] / (sample.porosity + 1e-30) * 100
    k_change = aging_results["delta_k"] / (sample.permeability + 1e-30) * 100

    G_orig = dynamic_shear_modulus(sample.vs, sample.bulk_density)
    vs_new = sample.vs + aging_results["delta_vs"]
    G_new = dynamic_shear_modulus(vs_new, sample.bulk_density)
    stiffness_change = (G_new - G_orig) / (G_orig + 1e-30) * 100

    stiffness_ok = abs(stiffness_change) < 5.0  # <5% change

    if abs(phi_change) < 2 and abs(k_change) < 5 and stiffness_ok:
        assessment = "favorable"
    elif abs(phi_change) < 5 and abs(k_change) < 15:
        assessment = "acceptable"
    else:
        assessment = "requires_monitoring"

    return {
        "porosity_change_pct": phi_change,
        "permeability_change_pct": k_change,
        "stiffness_change_pct": stiffness_change,
        "stiffness_maintained": stiffness_ok,
        "integrity_assessment": assessment,
    }
