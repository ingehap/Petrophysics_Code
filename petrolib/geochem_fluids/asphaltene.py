"""Asphaltene gravity gradients (Flory-Huggins-Zuo) and Yen-Mullins sizes.

The gravity-only FHZ optical-density (concentration) ratio between two depths, the
Yen-Mullins particle-size classes, and molar-volume <-> diameter converters.

Convention (fixed here to defeat the corpus's three sign/basis conventions): the
per-mole form ``OD(h2)/OD(h1) = exp[ Va*g*(rho_a - rho_o)*(h2 - h1)/(R*T) ]`` with
``h`` positive downward, so a deeper h2 gives a ratio > 1.  ``Va`` in m3/mol,
densities kg/m3, T in K, delta densities and solubility parameters in SI.  The
full iterative FHZ solubility solver stays in its article facade.  Sources:
src2014_04/article4, src2017_04/article5, src2023_02/article1, src2023_10/article_10.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_04/article4 -- Article 4: Fault Block Migrations Inferred from Asphaltene Gradients.
  Chengli Dong, Melton P. Hows, Peter M.W. Cornelisse, Hani Elshahawi (2014). Petrophysics Vol. 55,
  No. 2 (April 2014), pp. 113-123. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2014_04/article4_asphaltene_fault_block_migration -- Article 4: Fault Block Migrations Inferred
  from Asphaltene Gradients. Chengli Dong, Melton P. Hows, Peter M.W. Cornelisse, Hani Elshahawi
  (2014). Petrophysics Vol. 55, No. 2 (April 2014), pp. 113-123. DOI: none assigned (this issue
  predates SPWLA DOI assignment).
src2017_04/article5 -- Article 5: Scanning Electron Micrographs of Tar-Mat Intervals Formed by
  Asphaltene Phase Transition. Pfeiffer, Di Primio, Achourov, Mullins (2017). Petrophysics Vol. 58,
  No. 2 (April 2017), pp. 141-152. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2017_04/article5_tarmat_asphaltene_phase -- Article 5: Scanning Electron Micrographs of Tar-Mat
  Intervals Formed by Asphaltene Phase Transition. Pfeiffer, Di Primio, Achourov, Mullins (2017).
  Petrophysics Vol. 58, No. 2 (April 2017), pp. 141-152. DOI: none assigned (this issue predates
  SPWLA DOI assignment).
src2023_02/article1 -- Article 1: Enigmatic Reservoir Properties Deciphered Using Petroleum System
  Modeling and Reservoir Fluid Geodynamics. Pierpont, Birkeland, Cely, Yang, Chen, Achourov,
  Betancourt, Canas, Forsythe, Pomerantz, Yang, Datir, Mullins (2023). DOI:
  10.30632/PJV64N1-2023a1. Petrophysics Vol. 64 No. 1 (Feb 2023).
src2023_02/article1_rfg_petroleum_system -- Article 1: Enigmatic Reservoir Properties Deciphered
  Using Petroleum System Modeling and Reservoir Fluid Geodynamics. Pierpont, Birkeland, Cely, Yang,
  Chen, Achourov, Betancourt, Canas, Forsythe, Pomerantz, Yang, Datir, Mullins (2023). DOI:
  10.30632/PJV64N1-2023a1. Petrophysics Vol. 64 No. 1 (Feb 2023).
src2023_10/article_10 -- Mohamed, T. S., Torres-Verdin, C., and Mullins, O. C. (2023). "Enhanced
  Reservoir Description via Areal Data Integration and Reservoir Fluid Geodynamics: A Case Study
  From Deepwater Gulf of Mexico." Petrophysics, 64(5), 773-795. DOI: 10.30632/PJV64N5-2023a10.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]

R_GAS = 8.314  # J/(mol*K)
G_ACCEL = 9.81  # m/s2
N_AVOGADRO = 6.022e23  # 1/mol

# Yen-Mullins hierarchy nominal particle diameters (m).
YEN_MULLINS_DIAMETERS_M = {
    "molecule": 1.5e-9,
    "nanoaggregate": 2.0e-9,
    "cluster": 5.0e-9,
}


def molar_volume_from_diameter(d_m: ArrayLike) -> _Float:
    """Molar volume (m3/mol) of a sphere of diameter ``d_m`` ``= (pi/6)*d**3 * NA``.

    Sources: src2014_04/article4_asphaltene_fault_block_migration.
    """
    d = np.asarray(d_m, np.float64)
    return np.asarray(np.pi / 6.0 * d**3 * N_AVOGADRO)


def diameter_from_molar_volume(va_m3mol: ArrayLike) -> _Float:
    """Sphere diameter (m) from molar volume ``= (6*(Va/NA)/pi)**(1/3)``.

    Sources: src2014_04/article4_asphaltene_fault_block_migration.
    """
    va = np.asarray(va_m3mol, np.float64)
    return np.asarray((6.0 * (va / N_AVOGADRO) / np.pi) ** (1.0 / 3.0))


def fhz_ratio(
    dz_m: ArrayLike,
    va_m3mol: ArrayLike,
    delta_rho: ArrayLike,
    temp_k: ArrayLike,
    *,
    entropy: ArrayLike = 0.0,
    dsol2_pa: ArrayLike = 0.0,
) -> _Float:
    """Flory-Huggins-Zuo concentration ratio between two depths (gravity + optional terms).

    ``OD(z+dz)/OD(z) = exp[ Va*g*delta_rho*dz/(R*T) + entropy - Va*dsol2/(R*T) ]``
    with ``dz`` positive downward, ``delta_rho = rho_asphaltene - rho_oil`` (kg/m3),
    ``Va`` in m3/mol, ``dsol2_pa`` the solubility-squared difference (Pa).  The
    entropy and solubility terms default off (gravity-only).

    Sources: src2014_04/article4_asphaltene_fault_block_migration,
    src2017_04/article5_tarmat_asphaltene_phase, src2023_02/article1_rfg_petroleum_system.
    """
    va = np.asarray(va_m3mol, np.float64)
    rt = R_GAS * np.asarray(temp_k, np.float64)
    grav = va * G_ACCEL * np.asarray(delta_rho, np.float64) * np.asarray(dz_m, np.float64) / rt
    sol = -va * np.asarray(dsol2_pa, np.float64) / rt
    return np.asarray(np.exp(grav + np.asarray(entropy, np.float64) + sol))


def fhz_profile(
    depth_m: ArrayLike,
    od_ref: ArrayLike,
    depth_ref: ArrayLike,
    va_m3mol: ArrayLike,
    delta_rho: ArrayLike,
    temp_k: ArrayLike,
) -> _Float:
    """Optical-density profile ``OD(z) = OD_ref * fhz_ratio(z - z_ref, ...)`` (gravity-only)."""
    dz = np.asarray(depth_m, np.float64) - np.asarray(depth_ref, np.float64)
    return np.asarray(np.asarray(od_ref, np.float64) * fhz_ratio(dz, va_m3mol, delta_rho, temp_k))


def fhz_invert_molar_volume(
    od1: ArrayLike,
    z1: ArrayLike,
    od2: ArrayLike,
    z2: ArrayLike,
    delta_rho: ArrayLike,
    temp_k: ArrayLike,
) -> _Float:
    """Recover the asphaltene molar volume (m3/mol) from two OD/depth points.

    Inverts the gravity-only FHZ ratio: ``Va = R*T*ln(OD2/OD1)/(g*delta_rho*(z2 - z1))``.

    Sources: src2014_04/article4_asphaltene_fault_block_migration.
    """
    rt = R_GAS * np.asarray(temp_k, np.float64)
    num = rt * np.log(np.asarray(od2, np.float64) / np.asarray(od1, np.float64))
    den = (
        G_ACCEL
        * np.asarray(delta_rho, np.float64)
        * (np.asarray(z2, np.float64) - np.asarray(z1, np.float64))
    )
    return np.asarray(num / den)


def nearest_yen_mullins(d_m: float, *, rtol: float = 0.25) -> tuple[str, float, bool]:
    """Nearest Yen-Mullins class for a particle diameter (m) -> ``(name, ref_d, agrees)``.

    ``agrees`` is True when the diameter is within ``rtol`` (relative) of the class.
    """
    best_name = ""
    best_ref = 0.0
    best_rel = np.inf
    for name, ref in YEN_MULLINS_DIAMETERS_M.items():
        rel = abs(float(d_m) - ref) / ref
        if rel < best_rel:
            best_rel, best_name, best_ref = rel, name, ref
    return best_name, best_ref, bool(best_rel <= rtol)
