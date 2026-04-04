"""
Article 11: A Robust Joint Inversion for Improved Structural Mapping in
UDAR Applications Using Multiple Measurement Sensitivities and Uncertainties.

Authors: Wu, Wu, Yan, Ma, Fan, Lozinsky, and Bittar (2026)
DOI: 10.30632/PJV67N1-2026a11

Implements joint inversion of ultradeep azimuthal resistivity (UDAR) and
deep azimuthal resistivity (DAR) measurements for reservoir mapping.

Key ideas:
    - 1D layered earth forward model for EM tools
    - Joint inversion combining UDAR and DAR with different sensitivities
    - Uncertainty-weighted cost function
    - Layer boundary detection from combined measurements

References
----------
Wu et al. (2026), Petrophysics, 67(1), 161-172.
"""

import numpy as np
from typing import Optional


def layered_earth_response_1d(layer_resistivities: np.ndarray,
                              layer_boundaries: np.ndarray,
                              tool_depth: float,
                              spacing: float,
                              frequency: float = 2e5) -> complex:
    """Simplified 1D layered-earth EM response for a horizontal coil tool.

    Computes the apparent resistivity seen by a two-coil tool in a
    horizontally layered formation using geometric factor integration.

    Parameters
    ----------
    layer_resistivities : np.ndarray
        Resistivities of each layer (ohm-m), from top to bottom.
    layer_boundaries : np.ndarray
        TVD of layer boundaries (m), shape (n_layers - 1,).
    tool_depth : float
        TVD of the tool (m).
    spacing : float
        Transmitter-receiver spacing (m).
    frequency : float
        Operating frequency (Hz).

    Returns
    -------
    complex
        Complex apparent conductivity response.
    """
    mu0 = 4e-7 * np.pi
    omega = 2 * np.pi * frequency

    # Determine which layer the tool is in
    n_layers = len(layer_resistivities)
    tool_layer = 0
    for i, bd in enumerate(layer_boundaries):
        if tool_depth > bd:
            tool_layer = i + 1

    # Geometric factor approach: each layer contributes based on distance
    sigma_apparent = 0.0 + 0j
    for i in range(n_layers):
        sigma_i = 1.0 / layer_resistivities[i]

        if i == 0:
            z_top = -np.inf
        else:
            z_top = layer_boundaries[i - 1]

        if i == n_layers - 1:
            z_bot = np.inf
        else:
            z_bot = layer_boundaries[i]

        # Geometric factor for this layer
        skin_depth = np.sqrt(2 * layer_resistivities[i] / (mu0 * omega))
        dist_top = abs(tool_depth - z_top) if z_top != -np.inf else spacing * 5
        dist_bot = abs(tool_depth - z_bot) if z_bot != np.inf else spacing * 5

        # Weight decreases with distance (exponential decay ~ skin depth)
        weight = np.exp(-min(dist_top, dist_bot) / (skin_depth + 1e-10))
        if i == tool_layer:
            weight = 1.0  # Primary contribution

        sigma_apparent += sigma_i * weight

    return sigma_apparent


def udar_forward_model(model: dict,
                       tool_depth: float,
                       n_spacings: int = 5) -> np.ndarray:
    """UDAR forward model: compute measurements at multiple spacings.

    Parameters
    ----------
    model : dict
        'resistivities': layer resistivities (ohm-m),
        'boundaries': layer boundaries (m TVD).
    tool_depth : float
        Tool depth (m TVD).
    n_spacings : int
        Number of T-R spacings (from short to ultradeep).

    Returns
    -------
    np.ndarray
        Apparent resistivities at each spacing.
    """
    spacings = np.array([0.5, 1.0, 2.0, 5.0, 10.0])[:n_spacings]
    resistivities = model["resistivities"]
    boundaries = model["boundaries"]

    responses = np.zeros(n_spacings)
    for i, sp in enumerate(spacings):
        sigma = layered_earth_response_1d(resistivities, boundaries,
                                          tool_depth, sp)
        responses[i] = 1.0 / (abs(sigma) + 1e-30)

    return responses


def joint_inversion_cost(params: np.ndarray,
                         data_udar: np.ndarray,
                         data_dar: np.ndarray,
                         uncertainties_udar: np.ndarray,
                         uncertainties_dar: np.ndarray,
                         tool_depth: float,
                         boundaries: np.ndarray,
                         regularization: float = 0.1) -> float:
    """Joint UDAR/DAR inversion cost function.

    C = Σ ((d_udar - f_udar(m))² / σ_udar²) +
        Σ ((d_dar - f_dar(m))² / σ_dar²) +
        λ * ||∇m||²

    Parameters
    ----------
    params : np.ndarray
        Log-resistivities of each layer (ln(ohm-m)).
    data_udar : np.ndarray
        UDAR measurement data (apparent resistivities).
    data_dar : np.ndarray
        DAR measurement data (apparent resistivities).
    uncertainties_udar : np.ndarray
        UDAR measurement uncertainties.
    uncertainties_dar : np.ndarray
        DAR measurement uncertainties.
    tool_depth : float
        Tool depth (m TVD).
    boundaries : np.ndarray
        Layer boundaries (m TVD).
    regularization : float
        Smoothness regularization weight.

    Returns
    -------
    float
        Total cost.
    """
    resistivities = np.exp(params)
    model = {"resistivities": resistivities, "boundaries": boundaries}

    pred_udar = udar_forward_model(model, tool_depth, len(data_udar))
    pred_dar = udar_forward_model(model, tool_depth, len(data_dar))

    # Data misfit
    misfit_udar = np.sum(((data_udar - pred_udar) / (uncertainties_udar + 1e-10))**2)
    misfit_dar = np.sum(((data_dar - pred_dar) / (uncertainties_dar + 1e-10))**2)

    # Regularization (smoothness between adjacent layers)
    reg = regularization * np.sum(np.diff(params)**2)

    return misfit_udar + misfit_dar + reg


def run_joint_inversion(data_udar: np.ndarray,
                        data_dar: np.ndarray,
                        uncertainties_udar: np.ndarray,
                        uncertainties_dar: np.ndarray,
                        tool_depth: float,
                        n_layers: int = 5,
                        max_iter: int = 100,
                        regularization: float = 0.1) -> dict:
    """Run joint UDAR/DAR inversion to estimate layer resistivities.

    Parameters
    ----------
    data_udar : np.ndarray
        UDAR measurements.
    data_dar : np.ndarray
        DAR measurements.
    uncertainties_udar, uncertainties_dar : np.ndarray
        Measurement uncertainties.
    tool_depth : float
        Tool depth (m TVD).
    n_layers : int
        Number of layers in the model.
    max_iter : int
        Maximum iterations.
    regularization : float
        Smoothness constraint weight.

    Returns
    -------
    dict
        'resistivities': estimated layer resistivities,
        'boundaries': layer boundaries,
        'cost_history': optimization history.
    """
    # Set up layer boundaries evenly around the tool
    spacing = 5.0  # meters between boundaries
    boundaries = np.array([tool_depth + (i - n_layers // 2) * spacing
                           for i in range(n_layers - 1)])

    # Initial guess: geometric mean of measurements
    initial_res = np.mean(np.concatenate([data_udar, data_dar]))
    params = np.log(np.full(n_layers, initial_res))

    # Simple gradient descent
    step_size = 0.01
    cost_history = []

    for _ in range(max_iter):
        cost = joint_inversion_cost(params, data_udar, data_dar,
                                    uncertainties_udar, uncertainties_dar,
                                    tool_depth, boundaries, regularization)
        cost_history.append(cost)

        # Numerical gradient
        grad = np.zeros_like(params)
        h = 0.01
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += h
            cost_plus = joint_inversion_cost(params_plus, data_udar, data_dar,
                                             uncertainties_udar, uncertainties_dar,
                                             tool_depth, boundaries, regularization)
            grad[i] = (cost_plus - cost) / h

        params -= step_size * grad

        if len(cost_history) > 2 and abs(cost_history[-1] - cost_history[-2]) < 1e-6:
            break

    return {
        "resistivities": np.exp(params),
        "boundaries": boundaries,
        "cost_history": cost_history,
    }


# =========================================================================
# Article 12: Multidimensional UDAR Inversion
# DOI: 10.30632/PJV67N1-2026a12
# Saputra, Torres-Verdín, Ambia, et al. (2026)
# =========================================================================

def occam_regularized_inversion(data: np.ndarray,
                                forward_func,
                                initial_model: np.ndarray,
                                noise_level: float,
                                max_iter: int = 50) -> dict:
    """Occam-type regularized inversion for EM measurements.

    Seeks the smoothest model that fits the data to within the noise.
    Progressively reduces regularization until target misfit is reached.

    Parameters
    ----------
    data : np.ndarray
        Measured data.
    forward_func : callable
        Forward modeling function: model -> predicted data.
    initial_model : np.ndarray
        Initial model parameters.
    noise_level : float
        Expected noise standard deviation.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    dict
        'model': estimated model, 'misfit_history': list,
        'regularization_history': list.
    """
    model = initial_model.copy()
    n = len(model)
    target_misfit = len(data) * noise_level**2

    # Smoothness operator (second difference)
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i] = 1
        L[i, i + 1] = -2
        L[i, i + 2] = 1

    reg_factor = 100.0
    misfit_history = []
    reg_history = []

    for iteration in range(max_iter):
        pred = forward_func(model)
        residual = data - pred
        misfit = np.sum(residual**2)
        misfit_history.append(misfit)
        reg_history.append(reg_factor)

        if misfit < target_misfit:
            break

        # Jacobian (numerical)
        J = np.zeros((len(data), n))
        h = 0.01
        for j in range(n):
            model_p = model.copy()
            model_p[j] += h
            J[:, j] = (forward_func(model_p) - pred) / h

        # Occam update: (J^T J + λ L^T L) dm = J^T r
        JtJ = J.T @ J
        LtL = L.T @ L
        Jtr = J.T @ residual

        dm = np.linalg.solve(JtJ + reg_factor * LtL + 1e-10 * np.eye(n), Jtr)
        model = model + 0.5 * dm

        # Reduce regularization
        if misfit > 2 * target_misfit:
            reg_factor *= 0.8
        elif misfit > target_misfit:
            reg_factor *= 0.95

    return {
        "model": model,
        "misfit_history": misfit_history,
        "regularization_history": reg_history,
    }


# =========================================================================
# Article 13: Sand Injectite Geobody Mapping
# DOI: 10.30632/PJV67N1-2026a13
# Ahmad, Mouatt, Tosi, Marcy, and Clegg (2026)
# =========================================================================

def estimate_geobody_volume(thickness: np.ndarray,
                            widths: np.ndarray,
                            dip_angles: np.ndarray,
                            well_spacing: float) -> dict:
    """Estimate sand injectite geobody dimensions from UDAR and log data.

    Parameters
    ----------
    thickness : np.ndarray
        Net sand thickness at each well/section (m).
    widths : np.ndarray
        Estimated lateral extents from UDAR inversions (m).
    dip_angles : np.ndarray
        Dip angles of sills/dykes (degrees).
    well_spacing : float
        Distance between adjacent wells (m).

    Returns
    -------
    dict
        'area': estimated area (m²),
        'volume': estimated volume (m³),
        'mean_thickness': average thickness (m),
        'geometry_type': 'sill', 'dyke', or 'wing'.
    """
    mean_thickness = np.mean(thickness)
    mean_width = np.mean(widths)
    mean_dip = np.mean(np.abs(dip_angles))

    # Classify geometry based on dip
    if mean_dip < 15:
        geom_type = "sill"
    elif mean_dip > 60:
        geom_type = "dyke"
    else:
        geom_type = "wing"

    # Area estimation (elliptical approximation)
    length = well_spacing  # Along-well dimension
    area = np.pi / 4 * length * mean_width

    # Volume
    volume = area * mean_thickness

    return {
        "area": float(area),
        "volume": float(volume),
        "mean_thickness": float(mean_thickness),
        "mean_width": float(mean_width),
        "geometry_type": geom_type,
    }


# =========================================================================
# Article 14: Active Resistivity Ranging for Near-Parallel Wells
# DOI: 10.30632/PJV67N1-2026a14
# Salim, Chen, Liang, Denichou, et al. (2026)
# =========================================================================

def ranging_distance_from_udar(signal_amplitude: float,
                               formation_resistivity: float,
                               casing_resistivity: float = 1e-6,
                               frequency: float = 2e5,
                               calibration_factor: float = 1.0) -> float:
    """Estimate distance to a nearby cased well from UDAR signal strength.

    The UDAR tool detects cased wells due to the high resistivity contrast
    between the steel casing and formation. Signal strength decays with
    distance.

    Parameters
    ----------
    signal_amplitude : float
        Measured UDAR signal amplitude (normalized).
    formation_resistivity : float
        Background formation resistivity (ohm-m).
    casing_resistivity : float
        Casing resistivity (ohm-m), steel ≈ 1e-6.
    frequency : float
        Tool frequency (Hz).
    calibration_factor : float
        Tool-specific calibration factor.

    Returns
    -------
    float
        Estimated distance to the target well (m).
    """
    mu0 = 4e-7 * np.pi
    skin_depth = np.sqrt(2 * formation_resistivity / (mu0 * 2 * np.pi * frequency))

    # Signal decays approximately as exp(-d / skin_depth) / d
    if signal_amplitude <= 0:
        return np.inf

    # Invert the relationship: amplitude ∝ contrast * exp(-d/δ) / d
    contrast = formation_resistivity / (casing_resistivity + 1e-30)
    ref_signal = calibration_factor * contrast

    # Newton's method to solve: A * exp(-d/δ) / d = signal
    d = skin_depth  # Initial guess
    for _ in range(20):
        f_d = ref_signal * np.exp(-d / skin_depth) / (d + 1e-10) - signal_amplitude
        df_d = ref_signal * np.exp(-d / skin_depth) * (
            -1.0 / (skin_depth * (d + 1e-10)) - 1.0 / (d + 1e-10)**2
        )
        if abs(df_d) < 1e-30:
            break
        d = d - f_d / df_d
        d = max(d, 0.1)

    return float(d)


def ranging_azimuth_from_harmonics(second_harmonic_phase: float,
                                   tool_face_angle: float) -> float:
    """Determine azimuthal direction to nearby well from 2nd harmonic phase.

    Parameters
    ----------
    second_harmonic_phase : float
        Phase of the UDAR second harmonic signal (degrees).
    tool_face_angle : float
        Tool face angle (degrees from high side).

    Returns
    -------
    float
        Azimuthal direction to the target well (degrees from high side).
    """
    azimuth = (second_harmonic_phase / 2.0 + tool_face_angle) % 360.0
    return azimuth


# =========================================================================
# Article 15: UDAR Look-Ahead Mapping for Horizontal Wells
# DOI: 10.30632/PJV67N1-2026a15
# Ma, Clegg, Walmsley, Suarez Arcano, et al. (2026)
# =========================================================================

def look_ahead_inversion_3d(tensor_measurements: np.ndarray,
                            background_model: dict,
                            transmitter_position: float,
                            bit_position: float,
                            n_ahead_cells: int = 10,
                            cell_size: float = 1.0,
                            max_iter: int = 30) -> dict:
    """3D EM inversion for look-ahead fault detection in horizontal wells.

    Uses 9 tensor components from UDAR measurements to resolve resistivity
    changes ahead of the bit. The transmitter is placed ~3 m from the bit
    for maximum look-ahead sensitivity.

    Parameters
    ----------
    tensor_measurements : np.ndarray
        9-component EM tensor data, shape (9,) or (n_stations, 9).
    background_model : dict
        'resistivity': background formation resistivity (ohm-m).
    transmitter_position : float
        MD of transmitter (m).
    bit_position : float
        MD of bit (m).
    n_ahead_cells : int
        Number of cells ahead of the bit to resolve.
    cell_size : float
        Size of each look-ahead cell (m).
    max_iter : int
        Maximum inversion iterations.

    Returns
    -------
    dict
        'resistivity_ahead': estimated resistivities ahead of the bit,
        'distances_ahead': distances from bit (m),
        'fault_detected': bool, 'fault_distance': distance to fault (m).
    """
    background_res = background_model["resistivity"]
    tx_to_bit = abs(bit_position - transmitter_position)

    distances = np.arange(1, n_ahead_cells + 1) * cell_size
    # Initialize model: all background resistivity
    model_ahead = np.full(n_ahead_cells, np.log(background_res))

    # Simplified: detect anomaly from tensor measurement magnitude
    if tensor_measurements.ndim == 1:
        tensor_magnitude = np.linalg.norm(tensor_measurements)
    else:
        tensor_magnitude = np.mean(np.linalg.norm(tensor_measurements, axis=1))

    # Expected magnitude for homogeneous formation
    expected_magnitude = background_res * 10.0  # Simplified

    # Anomaly detection
    anomaly_ratio = tensor_magnitude / (expected_magnitude + 1e-30)

    # If significant anomaly, place resistivity contrast ahead
    fault_detected = False
    fault_distance = np.inf

    if anomaly_ratio < 0.5 or anomaly_ratio > 2.0:
        fault_detected = True
        # Distance estimation from sensitivity decay
        for i in range(n_ahead_cells):
            sensitivity = np.exp(-distances[i] / (tx_to_bit * 2))
            if sensitivity > 0.1:
                model_ahead[i] = np.log(background_res * anomaly_ratio)
            else:
                break
        fault_distance = distances[0]  # Nearest detectable distance

    resistivity_ahead = np.exp(model_ahead)

    return {
        "resistivity_ahead": resistivity_ahead,
        "distances_ahead": distances,
        "fault_detected": fault_detected,
        "fault_distance": float(fault_distance) if fault_detected else None,
    }
