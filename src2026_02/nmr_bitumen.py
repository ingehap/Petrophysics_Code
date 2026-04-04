"""
Article 7: Petrophysical Characterization of Secondary Organic Matter and
Hydrocarbons in the Early Jurassic Formation Using Laboratory NMR Techniques.

Authors: Al Mershed, Al Hammadi, Baqer, Acharya, Xie, Hawley, Woodroof,
Dolan, and Appel (2026)
DOI: 10.30632/PJV67N1-2026a7

Implements NMR-based methods for quantifying solid bitumen (pyrobitumen)
in carbonate reservoirs, including deficit porosity, Gaussian decomposition
of T2 distributions, and Arrhenius-based temperature correction.

Key ideas implemented:
    - NMR deficit porosity method for bitumen detection
    - Three-Gaussian decomposition of T2 distributions
    - Arrhenius model for temperature-dependent T2 relaxation
    - Bitumen-adjusted permeability model
    - Soluble-to-insoluble bitumen ratio estimation

References
----------
Al Mershed et al. (2026), Petrophysics, 67(1), 92-105.
Khatibi et al. (2019); Tucker (2013); Liu et al. (2024).
"""

import numpy as np
from typing import Optional


def deficit_porosity(phi_density: float,
                     phi_nmr: float) -> float:
    """Compute NMR deficit porosity for bitumen detection.

    Deficit porosity = phi_density - phi_NMR

    Solid organic matter (bitumen/pyrobitumen) occupies pore space visible
    to density logs but may be invisible to NMR due to very short T2
    (< minimum TE). The deficit indicates bitumen volume.

    Note: This method may underestimate bitumen volume if some bitumen
    signal is detected by the NMR tool at reservoir temperature.

    Parameters
    ----------
    phi_density : float
        Density-derived porosity (fraction).
    phi_nmr : float
        NMR total porosity (fraction).

    Returns
    -------
    float
        Deficit porosity (fraction), proxy for bitumen volume.
    """
    return max(0.0, phi_density - phi_nmr)


def gaussian_t2_decomposition(t2_dist: np.ndarray,
                              t2_basis: np.ndarray,
                              n_gaussians: int = 3,
                              max_iter: int = 200) -> dict:
    """Decompose a T2 distribution into a sum of log-Gaussian functions.

    f(T2) = Σ A_k * exp(-(log(T2) - μ_k)² / (2σ_k²))

    The three components typically correspond to:
        - Short T2: solid-like bitumen (temperature-independent)
        - Medium T2: mobilizable bitumen (temperature-dependent, Arrhenius)
        - Long T2: free fluids (temperature-dependent, Arrhenius)

    Parameters
    ----------
    t2_dist : np.ndarray
        T2 distribution (amplitudes at each T2 bin).
    t2_basis : np.ndarray
        T2 values (ms) for each bin.
    n_gaussians : int
        Number of Gaussian components (default 3).
    max_iter : int
        Maximum optimization iterations.

    Returns
    -------
    dict
        'amplitudes': peak amplitudes,
        'means': log-T2 means (log10 ms),
        'stds': log-T2 standard deviations,
        'components': individual Gaussian distributions.
    """
    log_t2 = np.log10(t2_basis)

    # Initialize with equally spaced Gaussians
    log_range = log_t2[-1] - log_t2[0]
    means = np.array([log_t2[0] + (k + 0.5) * log_range / n_gaussians
                       for k in range(n_gaussians)])
    stds = np.full(n_gaussians, log_range / (2 * n_gaussians))
    amplitudes = np.full(n_gaussians, np.max(t2_dist) / n_gaussians)

    # Iterative fitting (simplified EM-like algorithm)
    for _ in range(max_iter):
        # E-step: compute responsibilities
        components = np.zeros((n_gaussians, len(t2_basis)))
        for k in range(n_gaussians):
            components[k] = amplitudes[k] * np.exp(
                -(log_t2 - means[k])**2 / (2 * stds[k]**2 + 1e-10)
            )
        total = np.sum(components, axis=0) + 1e-30
        weights = components / total

        # M-step: update parameters
        for k in range(n_gaussians):
            w = weights[k] * t2_dist
            w_sum = np.sum(w)
            if w_sum < 1e-30:
                continue
            amplitudes[k] = np.max(components[k])
            means[k] = np.sum(w * log_t2) / w_sum
            stds[k] = np.sqrt(np.sum(w * (log_t2 - means[k])**2) / w_sum)
            stds[k] = max(stds[k], 0.05)

    # Final components
    final_components = np.zeros((n_gaussians, len(t2_basis)))
    for k in range(n_gaussians):
        final_components[k] = amplitudes[k] * np.exp(
            -(log_t2 - means[k])**2 / (2 * stds[k]**2 + 1e-10)
        )

    return {
        "amplitudes": amplitudes,
        "means": means,
        "stds": stds,
        "components": final_components,
    }


def arrhenius_t2_correction(t2_ref: float,
                            temp_ref: float,
                            temp_target: float,
                            activation_energy: float = 25.0) -> float:
    """Arrhenius temperature correction for T2 relaxation time.

    T2(T) = T2_ref * exp(-Ea/R * (1/T - 1/T_ref))

    The mobilizable bitumen component follows the Arrhenius relationship,
    while the solid-like (pyrobitumen) component is temperature-independent.

    Parameters
    ----------
    t2_ref : float
        T2 at reference temperature (ms).
    temp_ref : float
        Reference temperature (°C).
    temp_target : float
        Target temperature (°C).
    activation_energy : float
        Apparent activation energy (kJ/mol), default 25.

    Returns
    -------
    float
        T2 at the target temperature (ms).
    """
    R = 8.314e-3  # kJ/(mol·K)
    T_ref_K = temp_ref + 273.15
    T_target_K = temp_target + 273.15

    t2_target = t2_ref * np.exp(
        -activation_energy / R * (1.0 / T_target_K - 1.0 / T_ref_K)
    )
    return t2_target


def bitumen_volume_from_nmr(phi_density: float,
                            phi_nmr: float,
                            t2_dist: np.ndarray,
                            t2_basis: np.ndarray,
                            te: float = 0.2,
                            temperature: float = 25.0,
                            reservoir_temp: float = 120.0) -> dict:
    """Estimate bitumen volume from NMR measurements.

    Combines deficit porosity with Gaussian decomposition to quantify
    total bitumen and partition into soluble and insoluble (pyrobitumen).

    Parameters
    ----------
    phi_density : float
        Density porosity (fraction).
    phi_nmr : float
        NMR porosity measured at lab/logging TE (fraction).
    t2_dist : np.ndarray
        T2 distribution.
    t2_basis : np.ndarray
        T2 basis values (ms).
    te : float
        Inter-echo spacing of NMR measurement (ms).
    temperature : float
        Measurement temperature (°C).
    reservoir_temp : float
        Reservoir temperature (°C).

    Returns
    -------
    dict
        'deficit_phi': deficit porosity,
        'bitumen_volume': estimated total bitumen volume fraction,
        'pyrobitumen_fraction': fraction of insoluble pyrobitumen,
        'soluble_bitumen_fraction': fraction of soluble bitumen.
    """
    def_phi = deficit_porosity(phi_density, phi_nmr)

    # Decompose T2 distribution
    decomp = gaussian_t2_decomposition(t2_dist, t2_basis, n_gaussians=3)

    # Shortest T2 component: solid-like (temperature-independent) -> pyrobitumen
    # Medium T2 component: mobilizable bitumen (Arrhenius)
    sorted_idx = np.argsort(decomp["means"])
    short_amp = np.sum(decomp["components"][sorted_idx[0]])
    medium_amp = np.sum(decomp["components"][sorted_idx[1]])
    long_amp = np.sum(decomp["components"][sorted_idx[2]])
    total_signal = short_amp + medium_amp + long_amp + 1e-30

    # At reservoir temperature, more bitumen signal becomes visible
    t2_medium_res = arrhenius_t2_correction(
        10**decomp["means"][sorted_idx[1]], temperature, reservoir_temp
    )

    # Bitumen volume: deficit + short T2 component not captured
    # At LF NMR (2 MHz), signals with T2 < TE/2 are not captured
    min_detectable_t2 = te / 2.0
    undetected_fraction = short_amp / total_signal if total_signal > 0 else 0

    bitumen_total = def_phi + phi_nmr * undetected_fraction
    pyro_fraction = short_amp / (short_amp + medium_amp + 1e-30)

    return {
        "deficit_phi": def_phi,
        "bitumen_volume": float(np.clip(bitumen_total, 0, phi_density)),
        "pyrobitumen_fraction": float(pyro_fraction),
        "soluble_bitumen_fraction": float(1.0 - pyro_fraction),
    }


def bitumen_permeability_model(k_clean: float,
                               phi_total: float,
                               bitumen_volume: float,
                               exponent: float = 3.0) -> float:
    """Permeability model for bitumen-prone intervals.

    Permeability reduction due to pore-occluding bitumen:
        k_effective = k_clean * ((phi_total - V_bit) / phi_total)^n

    Parameters
    ----------
    k_clean : float
        Clean (bitumen-free) permeability (mD).
    phi_total : float
        Total porosity (fraction).
    bitumen_volume : float
        Volume fraction occupied by bitumen.
    exponent : float
        Reduction exponent (default 3.0, Kozeny-like).

    Returns
    -------
    float
        Effective permeability (mD).
    """
    if phi_total <= 0 or bitumen_volume >= phi_total:
        return 0.0
    effective_phi_ratio = (phi_total - bitumen_volume) / phi_total
    return k_clean * effective_phi_ratio**exponent
