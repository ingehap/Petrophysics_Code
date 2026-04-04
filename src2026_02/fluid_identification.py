"""
Article 5: Beyond Gas Bubbles in Norwegian Oil Fields: An Integrated
Technique to Understand Reservoir Fluid Distribution.

Authors: Bravo, Roblero Nunez, Donnadieu, Ungar, Yerkinkyzy, Yang, and
Fristad (2026)
DOI: 10.30632/PJV67N1-2026a5

Implements an integrated workflow for fluid identification in depleted oil
fields by combining advanced mud gas analysis, petrophysical logs, and
formation sampling data.

Key ideas implemented:
    - Mud gas ratio analysis for fluid-phase discrimination
    - Neutron-density fluid typing (crossover interpretation)
    - Gas-oil ratio estimation from mud gas composition
    - Fluid contact identification from log-gas integration
    - Wetness and balance ratios for hydrocarbon characterization

References
----------
Bravo et al. (2026), Petrophysics, 67(1), 68-80.
Cely et al. (2023); Bravo et al. (2024a, 2024b).
Yang et al. (2021); Yerkinkyzy et al. (2024).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MudGasData:
    """Advanced mud gas measurements at a given depth.

    Attributes
    ----------
    depth : float
        Measured depth (m).
    c1 : float
        Methane concentration (ppm or %).
    c2 : float
        Ethane concentration.
    c3 : float
        Propane concentration.
    ic4 : float
        Iso-butane concentration.
    nc4 : float
        Normal-butane concentration.
    ic5 : float
        Iso-pentane concentration.
    nc5 : float
        Normal-pentane concentration.
    co2 : float
        Carbon dioxide concentration.
    h2s : float
        Hydrogen sulfide concentration.
    total_gas : float
        Total gas reading.
    """
    depth: float
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    ic4: float = 0.0
    nc4: float = 0.0
    ic5: float = 0.0
    nc5: float = 0.0
    co2: float = 0.0
    h2s: float = 0.0
    total_gas: float = 0.0


def wetness_ratio(gas: MudGasData) -> float:
    """Compute wetness ratio Wh for hydrocarbon characterization.

    Wh = (C2 + C3 + C4 + C5) / (C1 + C2 + C3 + C4 + C5)

    Wetness ratio indicates the relative proportion of heavy gases.
    - Dry gas: Wh < 0.05
    - Wet gas: 0.05 < Wh < 0.20
    - Oil: Wh > 0.20

    Parameters
    ----------
    gas : MudGasData
        Mud gas composition.

    Returns
    -------
    float
        Wetness ratio (0 to 1).
    """
    c4 = gas.ic4 + gas.nc4
    c5 = gas.ic5 + gas.nc5
    heavy = gas.c2 + gas.c3 + c4 + c5
    total = gas.c1 + heavy
    if total < 1e-30:
        return 0.0
    return heavy / total


def balance_ratio(gas: MudGasData) -> float:
    """Compute balance (character) ratio for fluid typing.

    Bh = (C1 + C2) / (C3 + C4 + C5)

    - High values indicate gas-prone zones
    - Low values indicate oil-prone zones

    Parameters
    ----------
    gas : MudGasData
        Mud gas composition.

    Returns
    -------
    float
        Balance ratio.
    """
    c4 = gas.ic4 + gas.nc4
    c5 = gas.ic5 + gas.nc5
    light = gas.c1 + gas.c2
    heavy = gas.c3 + c4 + c5
    if heavy < 1e-30:
        return np.inf
    return light / heavy


def gas_oil_ratio_from_mud_gas(gas: MudGasData) -> float:
    """Estimate gas-oil ratio (GOR) from mud gas composition.

    Based on the empirical correlation between mud gas ratios and
    reservoir fluid GOR (Yang et al., 2021).

    GOR ∝ C1 / (C3 + C4 + C5)

    Parameters
    ----------
    gas : MudGasData
        Mud gas composition.

    Returns
    -------
    float
        Estimated GOR (scf/bbl), approximate.
    """
    c4 = gas.ic4 + gas.nc4
    c5 = gas.ic5 + gas.nc5
    heavy = gas.c3 + c4 + c5
    if heavy < 1e-30:
        return np.inf
    # Empirical scaling (approximate)
    ratio = gas.c1 / heavy
    gor = 50.0 * ratio  # Approximate empirical coefficient
    return gor


def classify_fluid_from_mud_gas(gas: MudGasData) -> str:
    """Classify reservoir fluid type from mud gas composition.

    Uses wetness ratio and balance ratio to discriminate:
    - Dry gas, wet gas/condensate, volatile oil, black oil, heavy oil

    Parameters
    ----------
    gas : MudGasData
        Mud gas composition.

    Returns
    -------
    str
        Fluid classification.
    """
    wh = wetness_ratio(gas)
    bh = balance_ratio(gas)

    if wh < 0.02:
        return "dry_gas"
    elif wh < 0.05:
        return "wet_gas"
    elif wh < 0.15 and bh > 50:
        return "gas_condensate"
    elif wh < 0.30 and bh > 10:
        return "volatile_oil"
    elif wh < 0.50:
        return "black_oil"
    else:
        return "heavy_oil"


def neutron_density_fluid_indicator(nphi: float,
                                    rhob: float,
                                    matrix_density: float = 2.65,
                                    fluid_density: float = 1.0) -> dict:
    """Interpret neutron-density log response for fluid typing.

    Neutron-density crossover (NPHI < DPHI) indicates gas effect.
    Large separation (NPHI > DPHI) indicates oil/water.

    Parameters
    ----------
    nphi : float
        Neutron porosity (fraction).
    rhob : float
        Bulk density (g/cc).
    matrix_density : float
        Rock matrix density (g/cc), default sandstone 2.65.
    fluid_density : float
        Assumed fluid density (g/cc).

    Returns
    -------
    dict
        'dphi': density porosity, 'separation': NPHI-DPHI,
        'gas_flag': True if gas effect detected.
    """
    dphi = (matrix_density - rhob) / (matrix_density - fluid_density)
    separation = nphi - dphi

    return {
        "dphi": dphi,
        "separation": separation,
        "gas_flag": separation < -0.02,
    }


def identify_fluid_contacts(depths: np.ndarray,
                            gas_data: list,
                            resistivity: np.ndarray,
                            nphi: np.ndarray,
                            rhob: np.ndarray,
                            sigma: Optional[np.ndarray] = None) -> dict:
    """Identify fluid contacts from integrated log and mud gas data.

    Combines mud gas fluid classification, neutron-density crossover, and
    resistivity changes to locate gas-oil and oil-water contacts.

    Parameters
    ----------
    depths : np.ndarray
        Measured depths (m).
    gas_data : list[MudGasData]
        Mud gas data at each depth.
    resistivity : np.ndarray
        Deep resistivity log (ohm-m).
    nphi : np.ndarray
        Neutron porosity log (fraction).
    rhob : np.ndarray
        Bulk density log (g/cc).
    sigma : np.ndarray, optional
        Formation sigma log (capture units) if available.

    Returns
    -------
    dict
        'fluid_zones': list of (depth_top, depth_bottom, fluid_type),
        'goc_depth': estimated gas-oil contact depth or None,
        'owc_depth': estimated oil-water contact depth or None.
    """
    n = len(depths)
    fluid_types = []

    for i in range(n):
        gas_type = classify_fluid_from_mud_gas(gas_data[i])
        nd = neutron_density_fluid_indicator(nphi[i], rhob[i])

        # Integrated classification
        if nd["gas_flag"] and gas_type in ("dry_gas", "wet_gas", "gas_condensate"):
            fluid_types.append("gas")
        elif gas_type in ("volatile_oil", "black_oil", "heavy_oil"):
            fluid_types.append("oil")
        elif resistivity[i] < 2.0:  # Low resistivity -> water
            fluid_types.append("water")
        else:
            fluid_types.append("uncertain")

    # Identify contacts as transition points
    goc_depth = None
    owc_depth = None
    for i in range(1, n):
        if fluid_types[i-1] == "gas" and fluid_types[i] == "oil":
            goc_depth = (depths[i-1] + depths[i]) / 2.0
        elif fluid_types[i-1] == "oil" and fluid_types[i] == "water":
            owc_depth = (depths[i-1] + depths[i]) / 2.0

    # Build zone list
    zones = []
    zone_start = 0
    for i in range(1, n):
        if fluid_types[i] != fluid_types[zone_start]:
            zones.append((depths[zone_start], depths[i-1], fluid_types[zone_start]))
            zone_start = i
    zones.append((depths[zone_start], depths[-1], fluid_types[zone_start]))

    return {
        "fluid_zones": zones,
        "goc_depth": goc_depth,
        "owc_depth": owc_depth,
    }
