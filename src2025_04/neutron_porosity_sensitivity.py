"""
Neutron Porosity Logging Sensitivity Functions in Casedhole
=============================================================
Based on: Varignier et al., "Laboratory Experimental Validation of
Sensitivity Functions for a Neutron Porosity Logging Tool in Casedhole
Environments",
Petrophysics, Vol. 66, No. 2, April 2025, pp. 294–317.

Implements:
  - Flux Sensitivity Functions (FSF) using weight window method (Eq. 1)
  - Interaction Sensitivity Functions (ISF) using particle tracking (Eq. 3)
  - FSF ↔ ISF relationship via macroscopic cross sections (Eq. 4)
  - Neutron porosity calibration from near/far detector count rates
  - Casedhole environmental corrections
  - Fast-forward modelling (FFM) for porosity prediction

Reference: https://doi.org/10.30632/PJV66N2-2025a8 (SPWLA)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class CylindricalMesh:
    """Regular cylindrical mesh for sensitivity function computation."""
    r_edges_cm: np.ndarray = field(default_factory=lambda:
        np.linspace(0, 50.0, 26))     # Radial bin edges
    z_edges_cm: np.ndarray = field(default_factory=lambda:
        np.linspace(-100, 100, 41))    # Axial bin edges
    theta_edges_rad: np.ndarray = field(default_factory=lambda:
        np.linspace(0, 2 * np.pi, 13))  # Azimuthal bin edges

    @property
    def n_cells(self) -> int:
        return (len(self.r_edges_cm) - 1) * \
               (len(self.z_edges_cm) - 1) * \
               (len(self.theta_edges_rad) - 1)

    def cell_volumes(self) -> np.ndarray:
        """Compute volume of each cylindrical cell (cm³)."""
        nr = len(self.r_edges_cm) - 1
        nz = len(self.z_edges_cm) - 1
        nt = len(self.theta_edges_rad) - 1
        volumes = np.zeros((nr, nz, nt))
        for i in range(nr):
            r1, r2 = self.r_edges_cm[i], self.r_edges_cm[i + 1]
            for j in range(nz):
                dz = abs(self.z_edges_cm[j + 1] - self.z_edges_cm[j])
                for k in range(nt):
                    dtheta = self.theta_edges_rad[k + 1] - self.theta_edges_rad[k]
                    volumes[i, j, k] = 0.5 * (r2 ** 2 - r1 ** 2) * dz * dtheta
        return volumes


@dataclass
class WellArchitecture:
    """Well geometry for casedhole neutron porosity modelling."""
    borehole_radius_cm: float = 10.795  # 8.5 in borehole
    casing_od_cm: float = 9.525        # 7-in casing OD
    casing_id_cm: float = 8.0          # Casing ID
    cement_od_cm: float = 10.795       # Cement outer radius = borehole
    casing_density_g_cm3: float = 7.85  # Steel
    cement_density_g_cm3: float = 1.90  # Class G Portland


@dataclass
class FormationProperties:
    """Formation properties for neutron transport."""
    porosity_pu: float = 20.0
    matrix_density_g_cm3: float = 2.71  # Limestone
    fluid_density_g_cm3: float = 1.0    # Fresh water
    hydrogen_index: float = 1.0
    sigma_scattering_cm_inv: float = 0.1  # Macroscopic scattering cross section


def flux_sensitivity_ww(flux_map: np.ndarray,
                        importance_map: np.ndarray) -> np.ndarray:
    """
    Compute Flux Sensitivity Function using weight window method (Eq. 1).

    FSF_j = FMESH_j * WWG_j

    where FMESH is the neutron flux map and WWG is the importance map.

    Parameters
    ----------
    flux_map : np.ndarray
        3D neutron flux in each mesh cell (from MCNP FMESH tally).
    importance_map : np.ndarray
        3D importance weights (from MCNP weight window generator).

    Returns
    -------
    np.ndarray : Normalized FSF values (sum = 1).
    """
    fsf = flux_map * importance_map
    total = np.sum(fsf)
    if total > 0:
        fsf /= total
    return fsf


def flux_sensitivity_pt(track_lengths: np.ndarray,
                        cell_volumes: np.ndarray) -> np.ndarray:
    """
    Compute Flux Sensitivity Function using particle tracking (Eq. 2).

    FSF_PT_j = sum_k(T_kj) / V_j

    where T_kj is the track length of neutron k in cell j,
    and V_j is the volume of cell j.

    Parameters
    ----------
    track_lengths : np.ndarray
        Total track length per cell summed over detected neutrons (cm).
    cell_volumes : np.ndarray
        Volume of each cell (cm³).

    Returns
    -------
    np.ndarray : Normalized FSF values (sum = 1).
    """
    fsf = np.where(cell_volumes > 0, track_lengths / cell_volumes, 0.0)
    total = np.sum(fsf)
    if total > 0:
        fsf /= total
    return fsf


def interaction_sensitivity_pt(n_elastic: np.ndarray,
                               n_inelastic: np.ndarray) -> np.ndarray:
    """
    Compute Interaction Sensitivity Function using particle tracking (Eq. 3).

    ISF_j = sum_k(N_elastic_kj + N_inelastic_kj)

    Parameters
    ----------
    n_elastic : np.ndarray  Elastic scattering counts per cell.
    n_inelastic : np.ndarray  Inelastic scattering counts per cell.

    Returns
    -------
    np.ndarray : Normalized ISF values (sum = 1).
    """
    isf = n_elastic + n_inelastic
    total = np.sum(isf)
    if total > 0:
        isf /= total
    return isf


def isf_to_fsf(isf: np.ndarray,
               sigma_scattering: np.ndarray) -> np.ndarray:
    """
    Convert ISF to FSF using macroscopic cross sections (Eq. 4).

    FSF_j ≈ ISF_j / sigma_scattering_j

    Parameters
    ----------
    isf : np.ndarray
    sigma_scattering : np.ndarray  Macroscopic scattering cross section (cm⁻¹)

    Returns
    -------
    np.ndarray : Estimated FSF.
    """
    fsf = np.where(sigma_scattering > 0, isf / sigma_scattering, 0.0)
    total = np.sum(fsf)
    if total > 0:
        fsf /= total
    return fsf


def neutron_count_rate(porosity_pu: float,
                       detector: str = "near",
                       cased: bool = False) -> float:
    """
    Simplified model for neutron detector count rate.

    Based on the calibration relationships between porosity and count
    rate for AmBe source neutron-neutron tools.

    Parameters
    ----------
    porosity_pu : float
        Formation porosity in porosity units.
    detector : str
        "near" or "far" detector.
    cased : bool
        Whether measurement is in cased hole.

    Returns
    -------
    float : Count rate (counts/second).
    """
    # Base count rates at 0% and 100% porosity (approximate)
    if detector == "near":
        cr_0 = 2000.0   # High count at low porosity
        cr_100 = 200.0   # Low count at high porosity
        decay_rate = 0.023
    else:  # far
        cr_0 = 800.0
        cr_100 = 50.0
        decay_rate = 0.035

    cr = cr_0 * np.exp(-decay_rate * porosity_pu)

    # Casing attenuation
    if cased:
        cr *= 0.75  # ~25% signal reduction through casing

    return cr


def compensated_porosity(near_cr: float,
                         far_cr: float,
                         calib_a: float = 0.0,
                         calib_b: float = -30.0,
                         calib_c: float = 45.0) -> float:
    """
    Compute compensated neutron porosity from near/far ratio.

    porosity = a + b * ln(ratio) + c * ln(ratio)^2

    Parameters
    ----------
    near_cr, far_cr : float  Near and far detector count rates.
    calib_a, calib_b, calib_c : float  Calibration coefficients.

    Returns
    -------
    float : Compensated porosity (p.u.).
    """
    ratio = near_cr / max(far_cr, 0.01)
    ln_ratio = np.log(ratio)
    porosity = calib_a + calib_b * ln_ratio + calib_c * ln_ratio ** 2
    return np.clip(porosity, 0, 60)


def casedhole_correction(porosity_pu: float,
                         casing_thickness_cm: float,
                         cement_thickness_cm: float,
                         casing_density: float = 7.85,
                         cement_density: float = 1.90) -> float:
    """
    Apply environmental correction for casedhole neutron porosity.

    Corrects for neutron absorption and scattering in casing and cement.

    Parameters
    ----------
    porosity_pu : float
    casing_thickness_cm : float
    cement_thickness_cm : float
    casing_density, cement_density : float  (g/cm³)

    Returns
    -------
    float : Corrected porosity (p.u.).
    """
    # Casing correction (approximately linear with thickness * density)
    casing_correction = -0.5 * casing_thickness_cm * (casing_density / 7.85)

    # Cement correction
    cement_correction = -0.3 * cement_thickness_cm * (cement_density / 1.90 - 1.0)

    return porosity_pu + casing_correction + cement_correction


def fast_forward_model(porosity_pu: float,
                       fsf_formation: float,
                       well: WellArchitecture) -> Tuple[float, float]:
    """
    Fast-forward model (FFM) for neutron porosity prediction.

    Uses sensitivity functions to predict tool response without
    full Monte Carlo simulation.

    Parameters
    ----------
    porosity_pu : float
    fsf_formation : float  Fraction of sensitivity in formation.
    well : WellArchitecture

    Returns
    -------
    Tuple[float, float] : (predicted_near_cr, predicted_far_cr)
    """
    near_cr = neutron_count_rate(porosity_pu, "near", cased=True) * \
              (0.5 + 0.5 * fsf_formation)
    far_cr = neutron_count_rate(porosity_pu, "far", cased=True) * \
             (0.3 + 0.7 * fsf_formation)
    return near_cr, far_cr


def test_all():
    """Test all functions with synthetic data."""
    print("=" * 70)
    print("Testing: neutron_porosity_sensitivity (Varignier et al., 2025)")
    print("=" * 70)

    mesh = CylindricalMesh()
    vols = mesh.cell_volumes()
    print(f"  Mesh cells: {mesh.n_cells}, volume shape: {vols.shape}")

    # Test FSF weight window
    rng = np.random.RandomState(42)
    flux = rng.exponential(1.0, vols.shape)
    importance = rng.exponential(0.5, vols.shape)
    fsf_ww = flux_sensitivity_ww(flux, importance)
    assert abs(np.sum(fsf_ww) - 1.0) < 1e-10, "FSF should be normalized"
    print(f"  FSF (WW) sum: {np.sum(fsf_ww):.6f}")

    # Test FSF particle tracking
    tracks = rng.exponential(2.0, vols.shape)
    fsf_pt = flux_sensitivity_pt(tracks, vols)
    assert abs(np.sum(fsf_pt) - 1.0) < 1e-10
    print(f"  FSF (PT) sum: {np.sum(fsf_pt):.6f}")

    # Test ISF
    n_el = rng.poisson(10, vols.shape).astype(float)
    n_in = rng.poisson(3, vols.shape).astype(float)
    isf = interaction_sensitivity_pt(n_el, n_in)
    assert abs(np.sum(isf) - 1.0) < 1e-10
    print(f"  ISF sum: {np.sum(isf):.6f}")

    # Test ISF to FSF conversion
    sigma = np.full(vols.shape, 0.1)
    fsf_from_isf = isf_to_fsf(isf, sigma)
    print(f"  FSF from ISF sum: {np.sum(fsf_from_isf):.6f}")

    # Test count rates and porosity
    for por in [5, 15, 30]:
        near = neutron_count_rate(por, "near", cased=False)
        far = neutron_count_rate(por, "far", cased=False)
        comp_por = compensated_porosity(near, far)
        print(f"  Porosity={por} p.u.: near={near:.0f}, far={far:.0f}, "
              f"compensated={comp_por:.1f} p.u.")

    # Test casedhole correction
    por_raw = 20.0
    well = WellArchitecture()
    casing_thick = (well.casing_od_cm - well.casing_id_cm) / 2.0
    cement_thick = well.cement_od_cm - well.casing_od_cm
    por_corr = casedhole_correction(por_raw, casing_thick, cement_thick)
    print(f"  Casedhole correction: {por_raw:.1f} → {por_corr:.1f} p.u.")

    # Test FFM
    near_ffm, far_ffm = fast_forward_model(20.0, 0.8, well)
    por_ffm = compensated_porosity(near_ffm, far_ffm)
    print(f"  FFM prediction at 20 p.u.: near={near_ffm:.0f}, far={far_ffm:.0f}, "
          f"porosity={por_ffm:.1f} p.u.")

    print("  All tests PASSED.\n")


if __name__ == "__main__":
    test_all()
