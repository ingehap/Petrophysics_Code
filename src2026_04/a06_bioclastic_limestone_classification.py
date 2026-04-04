"""
A Novel Lithological Classification Method for Marine Bioclastic
Limestones Through Integrated Analysis of Geological and
Petrophysical Facies

Reference:
    Guo, X., Duan, G., Du, Y., Ma, S., Fu, M., Li, D., Deng, H.,
    and Li, R. (2026). A Novel Lithological Classification Method for
    Marine Bioclastic Limestones Through Integrated Analysis of
    Geological and Petrophysical Facies.
    Petrophysics, 67(2), 336–350. DOI: 10.30632/PJV67N2-2026a6

Implements:
  - Grain-energy classification (high / mixed / low) from thin-section data
  - Seven-type geological-facies classification scheme (A, B, C1, C2-1,
    C2-2, C3-1, C3-2) using grain proportions
  - Petrophysical cluster analysis on (phi, k, R35, Pc_discharge,
    r_median) parameter space
  - Integrated lithofacies classifier combining geological + petrophysical
    clusters
  - Mercury-injection capillary-pressure parameter extraction (R35, Pd, rm)
  - Productivity index ranking from rock type
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# 1. Rock-type definitions (paper's seven-type classification)
# ---------------------------------------------------------------------------

class RockType(Enum):
    A    = "Mudstone"
    B    = "Wackestone"
    C1   = "Limestone-bearing low-energy grains"
    C2_1 = "Limestone-bearing mixed-energy grains I"
    C2_2 = "Limestone-bearing mixed-energy grains II"
    C3_1 = "Limestone-bearing high-energy grains I"
    C3_2 = "Limestone-bearing high-energy grains II"


PRODUCTIVITY_RANK = {
    RockType.A:    1,
    RockType.B:    2,
    RockType.C1:   3,
    RockType.C2_1: 4,
    RockType.C2_2: 4,
    RockType.C3_1: 5,
    RockType.C3_2: 5,
}


# ---------------------------------------------------------------------------
# 2. Grain-energy classification from thin-section proportions
# ---------------------------------------------------------------------------

def classify_grain_energy(f_high_energy: float,
                           f_mixed_energy: float,
                           f_low_energy: float) -> str:
    """
    Classify depositional grain energy from component proportions
    (fractions, must sum ≤ 1; remainder = micritic matrix).

    Parameters
    ----------
    f_high_energy  : fraction of high-energy grains (rudists, coralline algae)
    f_mixed_energy : fraction of mixed-energy grains
    f_low_energy   : fraction of low-energy grains (peloids, bioclasts)

    Returns
    -------
    'high', 'mixed', or 'low'
    """
    total = f_high_energy + f_mixed_energy + f_low_energy
    if total <= 0:
        return "low"
    dominant = max(f_high_energy, f_mixed_energy, f_low_energy)
    if dominant == f_high_energy and f_high_energy > 0.4:
        return "high"
    elif dominant == f_mixed_energy and f_mixed_energy > 0.3:
        return "mixed"
    else:
        return "low"


# ---------------------------------------------------------------------------
# 3. Geological-facies classification (7 types from grain proportions)
# ---------------------------------------------------------------------------

def classify_geological_facies(f_high: float,
                                 f_mixed: float,
                                 f_low: float,
                                 f_matrix: float) -> RockType:
    """
    Assign one of seven geological facies based on grain proportions
    (Fig. / Table logic from the paper).

    Parameters
    ----------
    f_high, f_mixed, f_low : weight fractions of grain categories
    f_matrix               : micritic matrix fraction

    Returns
    -------
    RockType enum member
    """
    # Type A: >80% matrix → mudstone
    if f_matrix > 0.80:
        return RockType.A

    # Type B: >50% low-energy grains, <10% high-energy → wackestone
    if f_low > 0.50 and f_high < 0.10:
        return RockType.B

    # Type C1: low-energy dominant with moderate matrix
    if f_low > 0.30 and f_high < 0.20 and f_mixed < 0.25:
        return RockType.C1

    # Types C2: mixed-energy dominant
    if f_mixed >= 0.25:
        # Sub-type based on high-energy grain presence
        if f_high >= 0.20:
            return RockType.C2_2
        else:
            return RockType.C2_1

    # Types C3: high-energy dominant (> 40%)
    if f_high >= 0.40:
        # C3-2: very high-energy, rudist packstone/grainstone
        if f_high >= 0.60:
            return RockType.C3_2
        return RockType.C3_1

    # Default fallback
    return RockType.C1


# ---------------------------------------------------------------------------
# 4. Mercury-injection capillary-pressure parameters
# ---------------------------------------------------------------------------

def r35_pore_throat_radius(k_md: float, phi_frac: float) -> float:
    """
    R35 median pore-throat radius at 35% mercury saturation (Winland method).

        log(R35) = 0.732 + 0.588*log(k) - 0.864*log(phi*100)

    Parameters
    ----------
    k_md    : Permeability, mD
    phi_frac: Porosity as fraction (0–1)

    Returns
    -------
    R35 : Median pore-throat radius at 35% Hg saturation, μm
    """
    if k_md <= 0 or phi_frac <= 0:
        return float("nan")
    log_R35 = 0.732 + 0.588 * np.log10(k_md) - 0.864 * np.log10(phi_frac * 100)
    return 10**log_R35


def displacement_pressure(k_md: float, phi_frac: float,
                           IFT_mNm: float = 485.0,
                           contact_angle_deg: float = 140.0) -> float:
    """
    Mercury displacement (threshold) pressure, Pd (psi), estimated from
    a Purcell-type capillary entry pressure relationship.

        Pd = (2 * IFT * cos(θ)) / r_entry   (Laplace)
    where r_entry ≈ 2 * R35.

    Parameters
    ----------
    k_md           : Permeability, mD
    phi_frac       : Porosity fraction
    IFT_mNm        : Hg-air interfacial tension, mN/m  (default 485)
    contact_angle_deg: Contact angle degrees  (default 140° for Hg-air)

    Returns
    -------
    Pd : Displacement pressure, psi
    """
    R35_um = r35_pore_throat_radius(k_md, phi_frac)
    if np.isnan(R35_um) or R35_um <= 0:
        return float("nan")
    r_m = R35_um * 1e-6  # convert μm → m
    theta = np.radians(contact_angle_deg)
    Pd_Pa = 2.0 * (IFT_mNm * 1e-3) * abs(np.cos(theta)) / r_m
    return Pd_Pa / 6894.76  # Pa → psi


def median_pore_throat_radius(k_md: float, phi_frac: float) -> float:
    """
    Median pore-throat radius (50% Hg saturation) using Swanson parameter.
        rm = 0.1 * sqrt(k / phi)     μm (empirical Swanson-type)
    """
    if k_md <= 0 or phi_frac <= 0:
        return float("nan")
    return 0.1 * np.sqrt(k_md / phi_frac)


# ---------------------------------------------------------------------------
# 5. Petrophysical parameter vector
# ---------------------------------------------------------------------------

@dataclass
class PetrophysicalSample:
    """Five petrophysical parameters used for cluster analysis."""
    phi:    float   # porosity, fraction
    k_md:   float   # permeability, mD
    R35:    float   # R35 pore-throat radius, μm
    Pd:     float   # displacement pressure, psi
    r_med:  float   # median pore-throat radius, μm

    @classmethod
    def from_core(cls, phi: float, k_md: float) -> "PetrophysicalSample":
        """Compute all derived parameters from core phi and k."""
        return cls(
            phi   = phi,
            k_md  = k_md,
            R35   = r35_pore_throat_radius(k_md, phi),
            Pd    = displacement_pressure(k_md, phi),
            r_med = median_pore_throat_radius(k_md, phi),
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.phi, self.k_md, self.R35, self.Pd, self.r_med])


# ---------------------------------------------------------------------------
# 6. K-means clustering on petrophysical space (7 clusters)
# ---------------------------------------------------------------------------

def kmeans_petrophysical(samples: List[PetrophysicalSample],
                          n_clusters: int = 7,
                          seed: int = 42) -> np.ndarray:
    """
    Simple k-means clustering on the log-transformed 5D petrophysical
    parameter space.

    Parameters
    ----------
    samples    : List of PetrophysicalSample objects
    n_clusters : Number of clusters (7 to match geological facies count)
    seed       : Random seed

    Returns
    -------
    labels : Integer cluster labels, shape (n_samples,)
    """
    X = np.array([s.as_array() for s in samples])
    # Log-transform positive parameters for scale invariance
    X_log = np.where(X > 0, np.log10(X), -6.0)
    # Standardise
    mu    = X_log.mean(axis=0)
    sigma = X_log.std(axis=0) + 1e-12
    Xn    = (X_log - mu) / sigma

    # K-means (simple implementation)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xn), n_clusters, replace=False)
    centroids = Xn[idx].copy()

    for _ in range(200):
        dists  = np.linalg.norm(Xn[:, None, :] - centroids[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        new_c  = np.array([Xn[labels == k].mean(axis=0)
                           if (labels == k).any() else centroids[k]
                           for k in range(n_clusters)])
        if np.allclose(new_c, centroids, atol=1e-6):
            break
        centroids = new_c

    return labels


# ---------------------------------------------------------------------------
# 7. Integrated lithofacies classification
# ---------------------------------------------------------------------------

def integrated_classify(geo_facies: RockType,
                         petro_cluster: int,
                         cluster_to_geo: Dict[int, RockType]) -> RockType:
    """
    Combine geological facies with petrophysical cluster membership.
    If both agree, return that type; otherwise use geological facies
    (dominant in the paper's workflow).

    Parameters
    ----------
    geo_facies       : Geological facies from classify_geological_facies()
    petro_cluster    : Cluster index from k-means
    cluster_to_geo   : Pre-built mapping of cluster index → rock type
                       (established by comparing cluster statistics to
                       known core data)

    Returns
    -------
    RockType
    """
    petro_guess = cluster_to_geo.get(petro_cluster, geo_facies)
    # Geological facies takes precedence; petrophysical refines sub-types
    if geo_facies == petro_guess:
        return geo_facies
    # If geo says C2 or C3 but petro distinguishes sub-type, use petro
    if (geo_facies in (RockType.C2_1, RockType.C2_2) and
            petro_guess in (RockType.C2_1, RockType.C2_2)):
        return petro_guess
    if (geo_facies in (RockType.C3_1, RockType.C3_2) and
            petro_guess in (RockType.C3_1, RockType.C3_2)):
        return petro_guess
    return geo_facies


# ---------------------------------------------------------------------------
# 8. Productivity ranking
# ---------------------------------------------------------------------------

def productivity_rank(rock_type: RockType) -> int:
    """
    Return relative productivity rank (1 = lowest, 5 = highest).
    Based on paper's finding that C3 types have highest productivity.
    """
    return PRODUCTIVITY_RANK.get(rock_type, 0)


# ---------------------------------------------------------------------------
# 9. Example workflow
# ---------------------------------------------------------------------------

def example_workflow():
    print("=" * 60)
    print("Bioclastic Limestone Lithological Classification")
    print("Ref: Guo et al., Petrophysics 67(2) 2026")
    print("=" * 60)

    # Synthetic sample set
    data = [
        # (f_high, f_mixed, f_low, f_matrix, phi, k_md)
        (0.00, 0.05, 0.10, 0.85, 0.04, 0.01),   # Type A
        (0.05, 0.10, 0.55, 0.30, 0.08, 0.10),   # Type B
        (0.10, 0.15, 0.35, 0.40, 0.12, 0.50),   # Type C1
        (0.25, 0.30, 0.20, 0.25, 0.15, 2.00),   # Type C2-1
        (0.30, 0.35, 0.15, 0.20, 0.18, 5.00),   # Type C2-2
        (0.55, 0.20, 0.10, 0.15, 0.22, 20.0),   # Type C3-1
        (0.70, 0.15, 0.05, 0.10, 0.28, 80.0),   # Type C3-2
    ]

    samples_petro  = []
    geo_types      = []

    print(f"\n{'Sample':<8} {'Geo Type':<35} {'phi%':<7} {'k(mD)':<9}"
          f" {'R35(μm)':<10} {'Pd(psi)':<9} {'Rank'}")
    print("-" * 85)

    for i, (fh, fm, fl, fmat, phi, k) in enumerate(data):
        gt = classify_geological_facies(fh, fm, fl, fmat)
        ps = PetrophysicalSample.from_core(phi, k)
        geo_types.append(gt)
        samples_petro.append(ps)
        rank = productivity_rank(gt)
        print(f"{i+1:<8} {gt.value:<35} {phi*100:<7.1f} {k:<9.2f}"
              f" {ps.R35:<10.3f} {ps.Pd:<9.1f} {'★'*rank}")

    # K-means clustering
    labels = kmeans_petrophysical(samples_petro, n_clusters=7)
    print(f"\nK-means cluster labels: {labels}")

    return geo_types, samples_petro


if __name__ == "__main__":
    example_workflow()
