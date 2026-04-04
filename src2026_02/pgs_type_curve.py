"""
Article 10: A Novel Type Curve for Sandstone Rock Typing.

Authors: Musu, Akbar, Permadi, and Widarsono (2026)
DOI: 10.30632/PJV67N1-2026a10

Implements the Pore Geometry Structure (PGS) crossplot and type curve for
rock typing in heterogeneous sandstone reservoirs, based on the Kozeny
equation and J-Function relationships.

Key ideas implemented:
    - Pore geometry (k/φ)^0.5 and pore structure (k/φ³) computation
    - PGS power-law fitting: (k/φ)^0.5 = a * (k/φ³)^b
    - Type curve convergence point identification
    - Leverett J-Function computation and grouping
    - Comparison with Hydraulic Flow Unit (HFU/FZI) method

References
----------
Musu et al. (2026), Petrophysics, 67(1), 140-160.
Kozeny (1927); Carman (1937); Leverett (1941); El-Khatib (1995).
Permadi and Wibowo (2013); Amaefule et al. (1993).
"""

import numpy as np
from typing import Optional


def pore_geometry(k: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Compute pore geometry parameter (k/φ)^0.5.

    Also known as "mean hydraulic radius." Represents the characteristic
    pore dimension.

    Parameters
    ----------
    k : np.ndarray
        Permeability (mD).
    phi : np.ndarray
        Porosity (fraction).

    Returns
    -------
    np.ndarray
        Pore geometry values (√mD).
    """
    return np.sqrt(k / (phi + 1e-30))


def pore_structure(k: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Compute pore structure parameter k/φ³.

    Captures all features of the internal pore network, related to
    the Kozeny constant and specific surface area.

    Parameters
    ----------
    k : np.ndarray
        Permeability (mD).
    phi : np.ndarray
        Porosity (fraction).

    Returns
    -------
    np.ndarray
        Pore structure values (mD).
    """
    return k / (phi**3 + 1e-30)


def fit_pgs_power_law(k: np.ndarray,
                      phi: np.ndarray,
                      labels: Optional[np.ndarray] = None) -> dict:
    """Fit PGS power-law relationships for rock type groups (Eq. 6).

    (k/φ)^0.5 = a * (k/φ³)^b

    In log-log space: log(PG) = log(a) + b * log(PS)

    For an ideal porous medium (bundle of capillary tubes), a = 1 and b = 0.5.
    For real rocks, a < 1 (flow efficiency) and b reflects pore structure
    complexity.

    Parameters
    ----------
    k : np.ndarray
        Permeability (mD).
    phi : np.ndarray
        Porosity (fraction).
    labels : np.ndarray, optional
        Rock type labels for each sample. If None, fit all data as one group.

    Returns
    -------
    dict
        'groups': list of dicts with 'a', 'b', 'r_squared' for each group,
        'convergence_point': (PS_conv, PG_conv) if lines converge.
    """
    pg = pore_geometry(k, phi)
    ps = pore_structure(k, phi)

    valid = (pg > 0) & (ps > 0) & np.isfinite(pg) & np.isfinite(ps)

    if labels is None:
        labels = np.zeros(len(k), dtype=int)

    unique_labels = np.unique(labels)
    groups = []

    for label in unique_labels:
        mask = (labels == label) & valid
        if np.sum(mask) < 3:
            continue

        log_pg = np.log10(pg[mask])
        log_ps = np.log10(ps[mask])

        # Linear regression in log-log space
        n = len(log_pg)
        x_mean = np.mean(log_ps)
        y_mean = np.mean(log_pg)
        b = (np.sum((log_ps - x_mean) * (log_pg - y_mean)) /
             (np.sum((log_ps - x_mean)**2) + 1e-30))
        log_a = y_mean - b * x_mean
        a = 10**log_a

        # R-squared
        y_pred = log_a + b * log_ps
        ss_res = np.sum((log_pg - y_pred)**2)
        ss_tot = np.sum((log_pg - y_mean)**2)
        r_sq = 1 - ss_res / (ss_tot + 1e-30)

        groups.append({
            "label": int(label),
            "a": float(a),
            "b": float(b),
            "log_a": float(log_a),
            "r_squared": float(r_sq),
            "n_samples": int(np.sum(mask)),
        })

    # Find convergence point (where all lines intersect)
    convergence_point = None
    if len(groups) >= 2:
        # Pairwise intersections
        intersections_ps = []
        intersections_pg = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                b_i, a_i = groups[i]["b"], groups[i]["log_a"]
                b_j, a_j = groups[j]["b"], groups[j]["log_a"]
                if abs(b_i - b_j) > 1e-6:
                    log_ps_int = (a_j - a_i) / (b_i - b_j)
                    log_pg_int = a_i + b_i * log_ps_int
                    intersections_ps.append(log_ps_int)
                    intersections_pg.append(log_pg_int)

        if intersections_ps:
            convergence_point = (
                10**np.median(intersections_ps),
                10**np.median(intersections_pg),
            )

    return {
        "groups": groups,
        "convergence_point": convergence_point,
    }


def leverett_j_function(pc: np.ndarray,
                        k: float,
                        phi: float,
                        sigma: float = 0.048,
                        theta: float = 0.0) -> np.ndarray:
    """Compute the Leverett J-Function (Eq. 8).

    J(Sw) = Pc * √(k/φ) / (σ * cos(θ))

    Parameters
    ----------
    pc : np.ndarray
        Capillary pressure (Pa or consistent units).
    k : float
        Permeability (m² or consistent units with Pc).
    phi : float
        Porosity (fraction).
    sigma : float
        Interfacial tension (N/m), default mercury-air 0.048.
    theta : float
        Contact angle (degrees), default 0 (strongly wetting).

    Returns
    -------
    np.ndarray
        J-Function values.
    """
    cos_theta = np.cos(np.radians(theta))
    hydraulic_radius = np.sqrt(k / (phi + 1e-30))
    j = pc * hydraulic_radius / (sigma * cos_theta + 1e-30)
    return j


def flow_zone_indicator(k: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Compute Flow Zone Indicator (FZI) for comparison with PGS.

    FZI = RQI / PHI_z

    where RQI = 0.0314 * √(k/φ) and PHI_z = φ/(1-φ)

    Amaefule et al. (1993) HFU method.

    Parameters
    ----------
    k : np.ndarray
        Permeability (mD).
    phi : np.ndarray
        Porosity (fraction).

    Returns
    -------
    np.ndarray
        Flow Zone Indicator values.
    """
    rqi = 0.0314 * np.sqrt(k / (phi + 1e-30))
    phi_z = phi / (1 - phi + 1e-30)
    return rqi / (phi_z + 1e-30)


def classify_rock_types_pgs(k: np.ndarray,
                            phi: np.ndarray,
                            n_types: int = 5) -> np.ndarray:
    """Classify rock types using the PGS crossplot method.

    Groups data into clusters on the log(PG) vs. log(PS) crossplot
    using the slope (b parameter) as the primary discriminator.

    Parameters
    ----------
    k : np.ndarray
        Permeability (mD).
    phi : np.ndarray
        Porosity (fraction).
    n_types : int
        Number of rock types to classify.

    Returns
    -------
    np.ndarray
        Rock type labels (0 to n_types-1).
    """
    pg = pore_geometry(k, phi)
    ps = pore_structure(k, phi)
    valid = (pg > 0) & (ps > 0) & np.isfinite(pg) & np.isfinite(ps)

    log_pg = np.log10(np.where(valid, pg, 1.0))
    log_ps = np.log10(np.where(valid, ps, 1.0))

    # Cluster using FZI as primary discriminator
    fzi = flow_zone_indicator(k, phi)
    log_fzi = np.log10(np.where(fzi > 0, fzi, 1e-10))

    # Equal-frequency binning on log(FZI)
    valid_fzi = log_fzi[valid]
    if len(valid_fzi) < n_types:
        return np.zeros(len(k), dtype=int)

    percentiles = np.linspace(0, 100, n_types + 1)
    boundaries = np.percentile(valid_fzi, percentiles)

    labels = np.zeros(len(k), dtype=int)
    for i in range(n_types):
        mask = (log_fzi >= boundaries[i]) & (log_fzi < boundaries[i + 1])
        labels[mask] = i
    labels[log_fzi >= boundaries[-1]] = n_types - 1

    return labels


def predict_capillary_pressure(sw: np.ndarray,
                               k: float,
                               phi: float,
                               pgs_group: dict,
                               sigma: float = 0.048,
                               theta: float = 0.0) -> np.ndarray:
    """Predict capillary pressure curve from PGS rock type parameters.

    Uses the relationship between PGS parameters and J-Function:
    From Eq. 7: J(Sw) is related to the b parameter (self-similarity).

    Parameters
    ----------
    sw : np.ndarray
        Water saturation values (fraction).
    k : float
        Permeability (mD, converted internally).
    phi : float
        Porosity (fraction).
    pgs_group : dict
        PGS group parameters with 'a' and 'b'.
    sigma : float
        Interfacial tension (N/m).
    theta : float
        Contact angle (degrees).

    Returns
    -------
    np.ndarray
        Predicted capillary pressure (Pa).
    """
    cos_theta = np.cos(np.radians(theta))
    hydraulic_radius = np.sqrt(k * 9.869e-16 / (phi + 1e-30))  # mD to m²

    b = pgs_group["b"]
    # J-Function model: J = C / (Sw - Swirr)^(1/b)
    swirr = 0.05 * (1 - b)  # Approximate Swirr from b
    swirr = np.clip(swirr, 0.01, 0.4)

    sw_eff = np.clip(sw - swirr, 0.01, 1.0)
    J = pgs_group["a"] / sw_eff**(1.0 / max(b, 0.1))

    pc = J * sigma * cos_theta / (hydraulic_radius + 1e-30)
    return pc
