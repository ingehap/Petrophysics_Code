"""
Laboratory Measurement of Total Porosity and Fluid Saturations
for Unconventional Tight Rocks
================================================================
Based on: Cheng et al., "Laboratory Measurement of Total Porosity and
Fluid Saturations for Unconventional Tight Rocks: Methodologies,
Challenges, and Comparison",
Petrophysics, Vol. 66, No. 2, April 2025, pp. 250–266.

Implements:
  - Crushed Rock Analysis (CRA/GRI) porosity and saturation
  - Retort porosity and saturation with temperature cutoffs
  - NMR T2 distribution porosity
  - Comparison framework and volumetric modelling

Reference: https://doi.org/10.30632/PJV66N2-2025a5 (SPWLA)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class RockSample:
    """Properties of an unconventional tight rock sample."""
    dry_weight_g: float
    bulk_volume_cm3: float
    grain_density_g_cm3: float = 2.65
    toc_wt_percent: float = 3.0       # Total organic carbon
    clay_wt_percent: float = 20.0     # Clay mineral content
    smectite_fraction: float = 0.05   # Fraction of clay that is smectite
    brine_salinity_ppm: float = 250000.0  # Formation brine salinity


@dataclass
class CRAResult:
    """Results from Crushed Rock Analysis (GRI method)."""
    grain_volume_cm3: float
    total_porosity: float
    oil_volume_cm3: float
    water_volume_cm3: float
    gas_volume_cm3: float
    oil_saturation: float
    water_saturation: float
    gas_saturation: float


@dataclass
class RetortResult:
    """Results from Retort analysis with temperature cutoffs."""
    total_porosity: float
    free_water_volume_cm3: float        # Released at T < 121°C
    bound_water_volume_cm3: float       # Released at 121°C < T < 316°C
    oil_volume_cm3: float               # Released at T < 300°C
    free_water_saturation: float
    bound_water_saturation: float
    oil_saturation: float
    gas_saturation: float


@dataclass
class NMRResult:
    """Results from NMR T2 distribution analysis."""
    total_porosity: float
    t2_distribution: np.ndarray
    t2_times_ms: np.ndarray
    bound_fluid_porosity: float     # T2 < cutoff
    free_fluid_porosity: float      # T2 > cutoff
    t2_cutoff_ms: float


def crushed_rock_analysis(sample: RockSample,
                          as_received_weight_g: float,
                          toluene_extracted_weight_g: float,
                          chloroform_methanol_extracted_weight_g: float,
                          helium_grain_volume_cm3: float) -> CRAResult:
    """
    Perform Crushed Rock Analysis (CRA/GRI method).

    The GRI method crushes samples to increase surface-area-to-volume ratio,
    enabling helium access to all connected porosity. Solvents extract
    fluids: toluene for oil, chloroform-methanol for remaining oil and salt.

    Limitations noted by Cheng et al. (2025): CRA may overestimate total
    porosity and oil volume for organic-rich samples because bitumen
    (immobile) is removed by organic solvent extraction.

    Parameters
    ----------
    sample : RockSample
    as_received_weight_g : float
    toluene_extracted_weight_g : float  (weight loss after toluene)
    chloroform_methanol_extracted_weight_g : float
    helium_grain_volume_cm3 : float

    Returns
    -------
    CRAResult
    """
    # Oil volume from toluene extraction (density ~ 0.85 g/cm³)
    oil_density = 0.85
    oil_mass = as_received_weight_g - toluene_extracted_weight_g
    oil_volume = max(oil_mass / oil_density, 0.0)

    # Water volume (corrected for salt) from remaining solvent extraction
    salt_correction = sample.brine_salinity_ppm / 1e6
    water_mass = toluene_extracted_weight_g - chloroform_methanol_extracted_weight_g
    # Correct for dissolved salt
    water_mass_corrected = water_mass * (1.0 - salt_correction)
    water_volume = max(water_mass_corrected / 1.0, 0.0)  # water density ~ 1.0

    # Total porosity
    pore_volume = sample.bulk_volume_cm3 - helium_grain_volume_cm3
    total_porosity = pore_volume / sample.bulk_volume_cm3

    # Gas volume
    gas_volume = max(pore_volume - oil_volume - water_volume, 0.0)

    # Saturations
    if pore_volume > 0:
        So = oil_volume / pore_volume
        Sw = water_volume / pore_volume
        Sg = gas_volume / pore_volume
    else:
        So = Sw = Sg = 0.0

    return CRAResult(
        grain_volume_cm3=helium_grain_volume_cm3,
        total_porosity=total_porosity,
        oil_volume_cm3=oil_volume,
        water_volume_cm3=water_volume,
        gas_volume_cm3=gas_volume,
        oil_saturation=So,
        water_saturation=Sw,
        gas_saturation=Sg,
    )


def retort_analysis(sample: RockSample,
                    water_121C_cm3: float,
                    water_316C_cm3: float,
                    oil_300C_cm3: float,
                    post_retort_grain_vol_cm3: float,
                    closed_system: bool = True) -> RetortResult:
    """
    Perform Retort analysis with temperature cutoffs.

    Temperature cutoffs from Handwerger et al. (2011):
      - 121°C: separates free pore water from bound water
      - 316°C: separates bound water from structural water (clay OH)
      - 300°C: maximum temperature to avoid kerogen cracking

    The closed retort setup collects 8-10% more water than open retort
    (Nikitin et al., 2019; Perry et al., 2021).

    Parameters
    ----------
    sample : RockSample
    water_121C_cm3 : float  Water collected at T < 121°C
    water_316C_cm3 : float  Additional water at 121°C < T < 316°C
    oil_300C_cm3 : float    Oil collected at T < 300°C
    post_retort_grain_vol_cm3 : float
    closed_system : bool

    Returns
    -------
    RetortResult
    """
    # Correct for smectite interlayer water released as "free water"
    smectite_water_correction = 0.0
    if sample.smectite_fraction > 0.1:
        # High smectite: some interlayer water comes out below 121°C
        smectite_water_correction = water_121C_cm3 * 0.1 * sample.smectite_fraction

    free_water = max(water_121C_cm3 - smectite_water_correction, 0.0)
    bound_water = water_316C_cm3 + smectite_water_correction

    # Open retort correction factor
    if not closed_system:
        free_water *= 1.09   # ~9% loss correction
        bound_water *= 1.09
        oil_300C_cm3 *= 1.05

    # Pore volume
    total_fluid = free_water + bound_water + oil_300C_cm3
    pore_volume = sample.bulk_volume_cm3 - post_retort_grain_vol_cm3
    total_porosity = pore_volume / sample.bulk_volume_cm3

    # Gas volume
    gas_volume = max(pore_volume - total_fluid, 0.0)

    # Saturations
    if pore_volume > 0:
        Sfw = free_water / pore_volume
        Sbw = bound_water / pore_volume
        So = oil_300C_cm3 / pore_volume
        Sg = gas_volume / pore_volume
    else:
        Sfw = Sbw = So = Sg = 0.0

    return RetortResult(
        total_porosity=total_porosity,
        free_water_volume_cm3=free_water,
        bound_water_volume_cm3=bound_water,
        oil_volume_cm3=oil_300C_cm3,
        free_water_saturation=Sfw,
        bound_water_saturation=Sbw,
        oil_saturation=So,
        gas_saturation=Sg,
    )


def nmr_porosity(t2_times_ms: np.ndarray,
                 t2_amplitudes: np.ndarray,
                 hydrogen_index: float = 1.0,
                 t2_cutoff_ms: float = 3.0) -> NMRResult:
    """
    Compute porosity from NMR T2 distribution.

    The T2 distribution reflects pore size distribution. A T2 cutoff
    separates bound fluid from free fluid. For unconventional tight
    rocks, typical T2 cutoffs are 3 ms (shale) or 33 ms (sandstone).

    Parameters
    ----------
    t2_times_ms : np.ndarray
    t2_amplitudes : np.ndarray
    hydrogen_index : float  (1.0 for fresh water)
    t2_cutoff_ms : float

    Returns
    -------
    NMRResult
    """
    # Normalize amplitudes
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    total_amplitude = _trapz(t2_amplitudes, np.log10(t2_times_ms))
    total_porosity = total_amplitude / hydrogen_index

    # Split by cutoff
    bound_mask = t2_times_ms <= t2_cutoff_ms
    free_mask = ~bound_mask

    bound_amp = _trapz(
        t2_amplitudes[bound_mask],
        np.log10(t2_times_ms[bound_mask])
    ) if np.sum(bound_mask) > 1 else 0.0

    free_amp = _trapz(
        t2_amplitudes[free_mask],
        np.log10(t2_times_ms[free_mask])
    ) if np.sum(free_mask) > 1 else 0.0

    bound_porosity = bound_amp / hydrogen_index
    free_porosity = free_amp / hydrogen_index

    return NMRResult(
        total_porosity=total_porosity,
        t2_distribution=t2_amplitudes,
        t2_times_ms=t2_times_ms,
        bound_fluid_porosity=bound_porosity,
        free_fluid_porosity=free_porosity,
        t2_cutoff_ms=t2_cutoff_ms,
    )


def compare_methods(cra: CRAResult,
                    retort: RetortResult,
                    nmr: NMRResult) -> Dict:
    """
    Compare total porosity and saturations across the three methods.

    Returns
    -------
    Dict with comparison metrics.
    """
    porosities = {
        "CRA": cra.total_porosity,
        "Retort": retort.total_porosity,
        "NMR": nmr.total_porosity,
    }
    water_sats = {
        "CRA_Sw": cra.water_saturation,
        "Retort_Sw_free": retort.free_water_saturation,
        "Retort_Sw_bound": retort.bound_water_saturation,
        "Retort_Sw_total": retort.free_water_saturation + retort.bound_water_saturation,
    }
    oil_sats = {
        "CRA_So": cra.oil_saturation,
        "Retort_So": retort.oil_saturation,
    }

    por_values = list(porosities.values())
    return {
        "porosities": porosities,
        "water_saturations": water_sats,
        "oil_saturations": oil_sats,
        "porosity_range": max(por_values) - min(por_values),
        "porosity_mean": np.mean(por_values),
    }


def generate_synthetic_t2(porosity: float = 0.08,
                          bound_fraction: float = 0.6,
                          n_points: int = 100,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic NMR T2 distribution for testing."""
    rng = np.random.RandomState(seed)
    t2 = np.logspace(-1, 4, n_points)  # 0.1 to 10,000 ms

    # Bimodal distribution: bound + free fluid peaks
    bound_peak = porosity * bound_fraction * \
        np.exp(-0.5 * ((np.log10(t2) - np.log10(1.0)) / 0.4) ** 2)
    free_peak = porosity * (1 - bound_fraction) * \
        np.exp(-0.5 * ((np.log10(t2) - np.log10(50.0)) / 0.5) ** 2)

    amplitudes = bound_peak + free_peak + 0.001 * rng.rand(n_points) * porosity
    return t2, amplitudes


def test_all():
    """Test all functions with synthetic data."""
    print("=" * 70)
    print("Testing: unconventional_porosity (Cheng et al., 2025)")
    print("=" * 70)

    sample = RockSample(
        dry_weight_g=100.0,
        bulk_volume_cm3=40.0,
        grain_density_g_cm3=2.65,
        toc_wt_percent=4.0,
        clay_wt_percent=25.0,
        smectite_fraction=0.05,
        brine_salinity_ppm=250000.0,
    )

    # CRA test
    cra = crushed_rock_analysis(
        sample,
        as_received_weight_g=100.0,
        toluene_extracted_weight_g=98.5,
        chloroform_methanol_extracted_weight_g=97.8,
        helium_grain_volume_cm3=36.5,
    )
    print(f"  CRA porosity: {cra.total_porosity:.3f} ({cra.total_porosity:.1%})")
    print(f"  CRA So={cra.oil_saturation:.3f}, Sw={cra.water_saturation:.3f}, "
          f"Sg={cra.gas_saturation:.3f}")
    assert 0 < cra.total_porosity < 0.3

    # Retort test
    retort = retort_analysis(
        sample,
        water_121C_cm3=0.8,    # Free water
        water_316C_cm3=0.5,    # Bound water
        oil_300C_cm3=0.3,
        post_retort_grain_vol_cm3=36.8,
        closed_system=True,
    )
    print(f"  Retort porosity: {retort.total_porosity:.3f} ({retort.total_porosity:.1%})")
    print(f"  Retort Sfw={retort.free_water_saturation:.3f}, "
          f"Sbw={retort.bound_water_saturation:.3f}, "
          f"So={retort.oil_saturation:.3f}")
    assert 0 < retort.total_porosity < 0.3

    # NMR test
    t2_times, t2_amps = generate_synthetic_t2(porosity=0.08)
    nmr = nmr_porosity(t2_times, t2_amps, t2_cutoff_ms=3.0)
    print(f"  NMR total porosity: {nmr.total_porosity:.3f}")
    print(f"  NMR bound porosity: {nmr.bound_fluid_porosity:.3f}, "
          f"free porosity: {nmr.free_fluid_porosity:.3f}")
    assert nmr.total_porosity > 0

    # Comparison
    comp = compare_methods(cra, retort, nmr)
    print(f"  Porosity comparison: {comp['porosities']}")
    print(f"  Porosity range: {comp['porosity_range']:.3f}")

    print("  All tests PASSED.\n")


if __name__ == "__main__":
    test_all()
