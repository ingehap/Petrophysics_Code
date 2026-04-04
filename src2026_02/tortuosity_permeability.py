"""
Article 9: Tortuosity Assessment for Reliable Permeability Quantification
Using Integration of Hydraulic and Electric Current Flow in Complex Carbonates.

Authors: Arrieta, Azizoglu, Sahu, Heidari, and Câncio (2026)
DOI: 10.30632/PJV67N1-2026a9

Implements a workflow integrating NMR, MICP, and pore-scale imaging to
estimate depth-by-depth permeability accounting for hydraulic and
electrical tortuosity.

Key ideas implemented:
    - Garcia et al. (2018) permeability model with tortuosity (Eq. 1)
    - Kozeny-Carman permeability from pore-body size and tortuosity
    - Electrical vs. hydraulic tortuosity comparison
    - Constriction factor from pore-body and pore-throat distributions
    - Permeability improvement using hydraulic tortuosity

References
----------
Arrieta et al. (2026), Petrophysics, 67(1), 123-139.
Garcia et al. (2018); August et al. (2022); Katz and Thompson (1986).
Kozeny (1927); Carman (1937).
"""

import numpy as np
from typing import Optional


def kozeny_carman_permeability(porosity: float,
                               specific_surface: float,
                               kozeny_constant: float = 5.0) -> float:
    """Classic Kozeny-Carman permeability model.

    k = phi³ / (Kk * Sb²)

    Parameters
    ----------
    porosity : float
        Connected porosity (fraction).
    specific_surface : float
        Specific surface area Sb with respect to bulk volume (1/μm).
    kozeny_constant : float
        Kozeny constant Kk = Fs * τ² (default 5.0).

    Returns
    -------
    float
        Permeability in μm² (≈ mD × 1.013e-3).
    """
    if porosity <= 0 or specific_surface <= 0:
        return 0.0
    return porosity**3 / (kozeny_constant * specific_surface**2)


def garcia_permeability_model(pore_radii: np.ndarray,
                              volume_fractions: np.ndarray,
                              connected_porosity: float,
                              constriction_factor: float,
                              tortuosity: float) -> float:
    """Permeability model from Garcia et al. (2018), Eq. 1 of the article.

    k = (phi_c / (Ce * τ²)) * Σ x_i * r_p,i²

    Parameters
    ----------
    pore_radii : np.ndarray
        Pore-body radii r_p,i (μm) for each pore size class.
    volume_fractions : np.ndarray
        Volumetric fractions x_i for each pore size class.
    connected_porosity : float
        Connected porosity phi_c (fraction).
    constriction_factor : float
        Electric or hydraulic constriction factor Ce.
    tortuosity : float
        Electric or hydraulic tortuosity τ.

    Returns
    -------
    float
        Permeability in μm².
    """
    if connected_porosity <= 0 or constriction_factor <= 0 or tortuosity <= 0:
        return 0.0

    mean_r2 = np.sum(volume_fractions * pore_radii**2)
    k = (connected_porosity / (constriction_factor * tortuosity**2)) * mean_r2
    return k


def estimate_constriction_factor(pore_body_radii: np.ndarray,
                                 pore_throat_radii: np.ndarray,
                                 volume_fractions: np.ndarray) -> float:
    """Estimate constriction factor from pore body and throat sizes.

    The constriction factor captures the flow restriction due to narrow
    pore throats connecting larger pore bodies.

    Ce = <r_body²> / <r_throat²>

    Parameters
    ----------
    pore_body_radii : np.ndarray
        Pore body radii from NMR T2 (μm).
    pore_throat_radii : np.ndarray
        Pore throat radii from MICP (μm).
    volume_fractions : np.ndarray
        Volume fractions for each pore class.

    Returns
    -------
    float
        Constriction factor (dimensionless, >= 1).
    """
    mean_body_r2 = np.sum(volume_fractions * pore_body_radii**2)
    mean_throat_r2 = np.sum(volume_fractions * pore_throat_radii**2)
    if mean_throat_r2 <= 0:
        return np.inf
    return mean_body_r2 / mean_throat_r2


def electrical_tortuosity_from_resistivity(formation_factor: float,
                                           porosity: float,
                                           constriction_factor: float = 1.0) -> float:
    """Estimate electrical tortuosity from formation factor.

    F = τ_e² * Ce / phi

    where F = R0/Rw is the formation resistivity factor.

    Parameters
    ----------
    formation_factor : float
        Formation factor F = R0/Rw.
    porosity : float
        Connected porosity (fraction).
    constriction_factor : float
        Constriction factor Ce.

    Returns
    -------
    float
        Electrical tortuosity τ_e.
    """
    if porosity <= 0 or formation_factor <= 0:
        return 1.0
    tau_sq = formation_factor * porosity / constriction_factor
    return np.sqrt(max(tau_sq, 1.0))


def hydraulic_tortuosity_from_simulation(streamline_lengths: np.ndarray,
                                          sample_length: float) -> float:
    """Compute hydraulic tortuosity from pore-scale flow simulation.

    τ_h = <L_streamline> / L_sample

    where L_streamline is the actual path length of fluid flow
    and L_sample is the straight-line distance.

    Parameters
    ----------
    streamline_lengths : np.ndarray
        Path lengths of simulated flow streamlines (μm).
    sample_length : float
        Straight-line sample dimension in flow direction (μm).

    Returns
    -------
    float
        Hydraulic tortuosity τ_h (≥ 1).
    """
    if sample_length <= 0 or len(streamline_lengths) == 0:
        return 1.0
    mean_length = np.mean(streamline_lengths)
    return max(mean_length / sample_length, 1.0)


def permeability_workflow(nmr_t2: np.ndarray,
                          t2_basis: np.ndarray,
                          micp_throat_radii: np.ndarray,
                          micp_volume_fractions: np.ndarray,
                          porosity: float,
                          formation_factor: float,
                          surface_relaxivity: float = 5.0,
                          use_hydraulic_tortuosity: bool = True,
                          hydraulic_tortuosity: Optional[float] = None) -> dict:
    """Complete permeability estimation workflow.

    Integrates NMR T2 (pore-body sizes), MICP (pore-throat sizes),
    and tortuosity for improved permeability assessment.

    Parameters
    ----------
    nmr_t2 : np.ndarray
        NMR T2 distribution amplitudes.
    t2_basis : np.ndarray
        T2 relaxation times (ms).
    micp_throat_radii : np.ndarray
        Pore throat radii from MICP (μm).
    micp_volume_fractions : np.ndarray
        Volume fractions from MICP.
    porosity : float
        Connected porosity (fraction).
    formation_factor : float
        Formation factor F = R0/Rw.
    surface_relaxivity : float
        NMR surface relaxivity ρ (μm/s).
    use_hydraulic_tortuosity : bool
        If True and hydraulic_tortuosity provided, use it. Otherwise use electrical.
    hydraulic_tortuosity : float, optional
        Hydraulic tortuosity from simulation.

    Returns
    -------
    dict
        'permeability_electric': using electrical tortuosity,
        'permeability_hydraulic': using hydraulic tortuosity (if available),
        'electric_tortuosity', 'hydraulic_tortuosity',
        'constriction_factor', 'pore_body_radii'.
    """
    # Convert T2 to pore body radii: r = ρ * T2 / 1000 (T2 in ms -> s)
    total_signal = np.sum(nmr_t2)
    if total_signal > 0:
        vol_fracs_nmr = nmr_t2 / total_signal
    else:
        vol_fracs_nmr = np.ones_like(nmr_t2) / len(nmr_t2)

    pore_body_radii = surface_relaxivity * t2_basis / 1000.0  # μm

    # Constriction factor
    # Match NMR and MICP distributions by rank
    n_classes = min(len(pore_body_radii), len(micp_throat_radii))
    idx_nmr = np.linspace(0, len(pore_body_radii)-1, n_classes, dtype=int)
    idx_micp = np.linspace(0, len(micp_throat_radii)-1, n_classes, dtype=int)

    Ce = estimate_constriction_factor(
        pore_body_radii[idx_nmr],
        micp_throat_radii[idx_micp],
        vol_fracs_nmr[idx_nmr] / (np.sum(vol_fracs_nmr[idx_nmr]) + 1e-30)
    )
    Ce = min(Ce, 100.0)

    # Electrical tortuosity
    tau_e = electrical_tortuosity_from_resistivity(formation_factor, porosity, Ce)

    # Permeability with electrical tortuosity
    k_electric = garcia_permeability_model(
        pore_body_radii, vol_fracs_nmr, porosity, Ce, tau_e
    )

    # Permeability with hydraulic tortuosity
    k_hydraulic = None
    tau_h = hydraulic_tortuosity
    if use_hydraulic_tortuosity and tau_h is not None:
        k_hydraulic = garcia_permeability_model(
            pore_body_radii, vol_fracs_nmr, porosity, Ce, tau_h
        )

    return {
        "permeability_electric": k_electric,
        "permeability_hydraulic": k_hydraulic,
        "electric_tortuosity": tau_e,
        "hydraulic_tortuosity": tau_h,
        "constriction_factor": Ce,
        "pore_body_radii": pore_body_radii,
    }
