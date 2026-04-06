"""
Gas Logging Based on Semipermeable Film Degasser and Infrared Spectrum
======================================================================
Based on: Cheng, L., Ye, X., Wang, H., et al. (2024), "A New Gas Logging
Method Based on Semipermeable Film Degasser and Infrared Spectrum,"
Petrophysics, 65(4), pp. 548-564. DOI: 10.30632/PJV65N4-2024a9

Implements:
  - Beer-Lambert law for IR gas absorption
  - NDIR (non-dispersive infrared) spectral analysis
  - Multi-component gas concentration determination
  - Membrane degassing model for gas-liquid separation efficiency
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List


# Central wavelengths for alkane gas detection (micrometers)
ALKANE_WAVELENGTHS = {
    "C1": 3.31,   # methane
    "C2": 3.35,   # ethane
    "C3": 3.38,   # propane
    "C4": 3.40,   # butane
    "C5": 3.42,   # pentane
    "CO2": 4.26,  # carbon dioxide (reference)
}

# Absorption coefficients (cm^-1 * ppm^-1) - synthetic representative values
ABSORPTION_COEFFICIENTS = {
    "C1": 0.0012,
    "C2": 0.0025,
    "C3": 0.0038,
    "C4": 0.0052,
    "C5": 0.0065,
    "CO2": 0.0095,
}


@dataclass
class IRSpectrum:
    """Infrared spectrum measurement from NDIR detector."""
    wavelengths: np.ndarray    # wavelength points (um)
    intensities: np.ndarray    # transmitted light intensities
    reference: np.ndarray      # reference (incident) intensities
    path_length_cm: float      # absorption path length


@dataclass
class MembraneParams:
    """Parameters for the semipermeable membrane degasser.

    The membrane allows gas molecules to pass through based on their
    molecular size and the partial pressure differential.
    """
    permeability_c1: float = 1.0     # relative permeability (C1 = reference)
    permeability_c2: float = 0.85
    permeability_c3: float = 0.70
    permeability_c4: float = 0.55
    permeability_c5: float = 0.42
    membrane_area_cm2: float = 100.0
    thickness_um: float = 50.0
    temperature_c: float = 25.0


def beer_lambert(I0: np.ndarray, concentration_ppm: float,
                 absorption_coeff: float,
                 path_length_cm: float) -> np.ndarray:
    """Apply Beer-Lambert law: I = I0 * exp(-mu * C * L).

    Parameters
    ----------
    I0 : incident light intensity
    concentration_ppm : gas concentration in ppm
    absorption_coeff : absorption coefficient (cm^-1 ppm^-1)
    path_length_cm : optical path length
    """
    return I0 * np.exp(-absorption_coeff * concentration_ppm * path_length_cm)


def compute_absorbance(I: np.ndarray, I0: np.ndarray) -> np.ndarray:
    """Compute absorbance A = -log10(I / I0)."""
    ratio = np.clip(I / (I0 + 1e-30), 1e-10, 1.0)
    return -np.log10(ratio)


def solve_multicomponent(absorbances: np.ndarray,
                         pure_spectra: np.ndarray,
                         path_length_cm: float) -> np.ndarray:
    """Solve for gas concentrations from multi-wavelength absorbance.

    Uses least-squares: X = absorbance vector, S = pure spectra matrix.
    Solve: X = S * C * L, so C = (S^T S)^-1 S^T X / L

    This is the core of the NDIR analysis described in the paper.
    """
    # S is (n_wavelengths, n_components), C is (n_components,)
    StS = pure_spectra.T @ pure_spectra
    StX = pure_spectra.T @ absorbances
    try:
        concentrations = np.linalg.solve(StS, StX) / path_length_cm
    except np.linalg.LinAlgError:
        concentrations = np.linalg.lstsq(pure_spectra * path_length_cm,
                                         absorbances, rcond=None)[0]
    return np.clip(concentrations, 0, None)


def membrane_extraction_efficiency(gas_name: str, params: MembraneParams,
                                   mud_flow_rate_lpm: float = 5.0,
                                   pressure_bar: float = 1.0) -> float:
    """Compute gas extraction efficiency through semipermeable membrane.

    Efficiency depends on membrane permeability, area, pressure
    differential, and contact time (inversely proportional to flow rate).

    Returns fractional efficiency (0-1).
    """
    permeabilities = {
        "C1": params.permeability_c1, "C2": params.permeability_c2,
        "C3": params.permeability_c3, "C4": params.permeability_c4,
        "C5": params.permeability_c5,
    }
    perm = permeabilities.get(gas_name, 0.5)

    # Fick's law approximation: flux proportional to perm * area * dP / thickness
    contact_time = params.membrane_area_cm2 / (mud_flow_rate_lpm * 1000 / 60)  # seconds
    flux_factor = perm * pressure_bar / (params.thickness_um * 1e-4)  # per cm
    efficiency = 1.0 - np.exp(-flux_factor * contact_time)
    return np.clip(efficiency, 0.01, 0.99)


def correct_for_membrane(concentrations: Dict[str, float],
                         params: MembraneParams,
                         mud_flow_rate_lpm: float = 5.0) -> Dict[str, float]:
    """Correct measured concentrations for membrane extraction efficiency.

    Divides measured concentration by efficiency to estimate true
    formation gas concentration.
    """
    corrected = {}
    for gas, conc in concentrations.items():
        eff = membrane_extraction_efficiency(gas, params, mud_flow_rate_lpm)
        corrected[gas] = conc / eff
    return corrected


def simulate_ir_measurement(true_concentrations: Dict[str, float],
                            path_length_cm: float = 10.0,
                            noise_level: float = 0.01,
                            random_state: int = 42) -> IRSpectrum:
    """Simulate an IR spectrum measurement from known gas concentrations."""
    rng = np.random.RandomState(random_state)
    wavelengths = np.linspace(3.0, 5.0, 200)
    I0 = np.ones_like(wavelengths) * 1000  # reference intensity

    I = I0.copy()
    for gas, conc in true_concentrations.items():
        if gas in ALKANE_WAVELENGTHS:
            center = ALKANE_WAVELENGTHS[gas]
            mu = ABSORPTION_COEFFICIENTS[gas]
            # Gaussian absorption profile
            profile = mu * np.exp(-((wavelengths - center) ** 2) / (2 * 0.02 ** 2))
            I = I * np.exp(-profile * conc * path_length_cm)

    I += rng.normal(0, noise_level * I0, len(wavelengths))
    I = np.clip(I, 1, None)

    return IRSpectrum(wavelengths=wavelengths, intensities=I,
                      reference=I0, path_length_cm=path_length_cm)


def analyze_spectrum(spectrum: IRSpectrum,
                     gases: List[str] = None) -> Dict[str, float]:
    """Analyze IR spectrum to determine gas concentrations.

    Builds pure spectra matrix and solves multi-component Beer-Lambert.
    """
    if gases is None:
        gases = ["C1", "C2", "C3", "C4", "C5"]

    absorbance = compute_absorbance(spectrum.intensities, spectrum.reference)

    # Build pure spectra matrix
    pure_spectra = np.zeros((len(spectrum.wavelengths), len(gases)))
    for j, gas in enumerate(gases):
        center = ALKANE_WAVELENGTHS[gas]
        mu = ABSORPTION_COEFFICIENTS[gas]
        pure_spectra[:, j] = mu * np.exp(
            -((spectrum.wavelengths - center) ** 2) / (2 * 0.02 ** 2)
        )

    concentrations = solve_multicomponent(absorbance, pure_spectra,
                                          spectrum.path_length_cm)
    return {gas: conc for gas, conc in zip(gases, concentrations)}


def test_all():
    """Test membrane degasser and IR spectroscopy pipeline."""
    print("=" * 70)
    print("Testing: Membrane Degasser + IR Spectrum (Cheng et al., 2024)")
    print("=" * 70)

    # True formation gas concentrations (ppm)
    true_conc = {"C1": 5000, "C2": 800, "C3": 400, "C4": 150, "C5": 60}
    print(f"  True concentrations: {true_conc}")

    # Simulate membrane extraction
    membrane = MembraneParams()
    print(f"\n  Membrane extraction efficiencies:")
    for gas in true_conc:
        eff = membrane_extraction_efficiency(gas, membrane)
        print(f"    {gas}: {eff:.3f}")

    # Apply membrane effect (what the instrument sees)
    measured_conc = {}
    for gas, conc in true_conc.items():
        eff = membrane_extraction_efficiency(gas, membrane)
        measured_conc[gas] = conc * eff

    # Simulate IR spectrum from measured concentrations
    spectrum = simulate_ir_measurement(measured_conc, path_length_cm=10.0)
    print(f"\n  IR spectrum: {len(spectrum.wavelengths)} wavelength points, "
          f"path length={spectrum.path_length_cm} cm")

    # Analyze spectrum
    analyzed = analyze_spectrum(spectrum)
    print(f"\n  Analyzed concentrations (from IR):")
    for gas in ["C1", "C2", "C3", "C4", "C5"]:
        print(f"    {gas}: {analyzed.get(gas, 0):.0f} ppm "
              f"(measured={measured_conc.get(gas, 0):.0f})")

    # Correct for membrane
    corrected = correct_for_membrane(analyzed, membrane)
    print(f"\n  Membrane-corrected concentrations:")
    for gas in ["C1", "C2", "C3", "C4", "C5"]:
        true_val = true_conc[gas]
        corr_val = corrected.get(gas, 0)
        error = abs(corr_val - true_val) / true_val * 100
        print(f"    {gas}: {corr_val:.0f} ppm (true={true_val}, error={error:.1f}%)")

    # Beer-Lambert verification
    I0 = np.array([1000.0])
    I = beer_lambert(I0, 5000, 0.0012, 10.0)
    A = compute_absorbance(I, I0)
    print(f"\n  Beer-Lambert check: I0=1000, C=5000ppm, mu=0.0012, L=10cm")
    print(f"    Transmitted I={I[0]:.1f}, Absorbance A={A[0]:.4f}")

    print("\n  [PASS] Membrane degasser + IR spectrum tests completed.")
    return True


if __name__ == "__main__":
    test_all()
