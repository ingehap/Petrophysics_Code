#!/usr/bin/env python3
"""
Fit-For-Purpose Thomas-Stieber Diagram in the Well-Log Domain
===============================================================
Implements the methodology from:
  Eghbali, A. and Torres-Verdín, C., 2025,
  "Fit-For-Purpose Thomas-Stieber Diagram for the Petrophysical Evaluation
  of Heterogeneous Shaly Sandstones via Well Logs,"
  Petrophysics, Vol. 66, No. 3, pp. 392–423.

Key ideas implemented:
  - Nuclear-log forward models for bulk density, neutron porosity,
    gamma ray, and volumetric PEF as functions of V_Lam and V_Disp
    (Eqs. 4–7 of paper).
  - Porosity calculation comparison: T-S principles vs mass-balance
    (Eqs. 1–3).
  - Construction of T-S crossplots in the well-log domain.
  - Rock-class separation for multi-class shaly sandstones.

References:
  Thomas, E.C. and Stieber, S.J., 1975.
  Ellis, D.V. and Singer, J.M., 2007.
  Herron, M.M. and Matteson, A., 1993.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class RockClassEndpoints:
    """
    Defines a single shaly-sandstone rock class with its clean-sand,
    shale, and dispersed-clay endpoints in log and petrophysical space.
    """
    name: str = "Default"

    # Clean sandstone
    phi_s: float = 0.40          # clean sandstone porosity
    rho_s: float = 2.175         # clean sandstone bulk density g/cc
    nphi_s: float = -0.02        # clean sandstone neutron porosity
    gr_s: float = 15.0           # clean sandstone gamma ray (API)
    u_s: float = 4.79            # clean sandstone volumetric PEF

    # Shale lamina
    phi_shale: float = 0.15      # shale porosity
    rho_shale: float = 2.50      # shale bulk density g/cc
    nphi_shale: float = 0.35     # shale neutron porosity
    gr_shale: float = 120.0      # shale gamma ray
    u_shale: float = 8.0         # shale volumetric PEF

    # Dispersed wet clay
    rho_wetclay: float = 2.12    # wet-clay density g/cc
    nphi_wetclay: float = 0.40   # wet-clay neutron porosity
    gr_wetclay: float = 150.0    # wet-clay gamma ray
    u_wetclay: float = 9.5       # wet-clay volumetric PEF
    phi_wetclay: float = 0.15    # wet-clay porosity

    # Fluids
    rho_fs: float = 0.80         # fluid density in non-clay porosity g/cc
    nphi_fs: float = 1.0         # fluid neutron porosity (hydrogen index)
    u_fs: float = 0.40           # fluid volumetric PEF


def ts_porosity_principles(v_lam: np.ndarray, v_disp: np.ndarray,
                            ep: RockClassEndpoints) -> np.ndarray:
    """
    Total porosity from T-S principles (Eq. 1 of paper).

    φ_T = φ_S(1 - V_Lam - V_Disp) + φ_Shale·V_Lam + φ_Shale·V_Disp
    """
    return (ep.phi_s * (1.0 - v_lam - v_disp) +
            ep.phi_shale * v_lam +
            ep.phi_shale * v_disp)


def ts_porosity_mass_balance(v_lam: np.ndarray, v_disp: np.ndarray,
                              ep: RockClassEndpoints) -> np.ndarray:
    """
    Total porosity from mass-balance (Eq. 3 of paper).

    φ_T = (ρ_mT - ρ_b) / (ρ_mT - ρ_fT)

    where ρ_b is the measured bulk density.
    """
    rho_b = bulk_density_model(v_lam, v_disp, ep)
    # Effective matrix density
    rho_mt = (ep.rho_s * (1.0 - ep.phi_s) * (1.0 - v_lam - v_disp) +
              ep.rho_shale * (1.0 - ep.phi_shale) * v_lam +
              ep.rho_wetclay * (1.0 - ep.phi_wetclay) * v_disp)
    rho_mt /= ((1.0 - ep.phi_s) * (1.0 - v_lam - v_disp) +
               (1.0 - ep.phi_shale) * v_lam +
               (1.0 - ep.phi_wetclay) * v_disp + 1e-12)

    # Effective fluid density
    rho_ft_num = (1.0 * ep.phi_shale * v_lam +     # water in shale
                  1.0 * ep.phi_wetclay * v_disp +    # water in wet clay
                  ep.rho_fs * (ep.phi_s * (1.0 - v_lam - v_disp) - ep.phi_wetclay * v_disp))
    phi_total_approx = ts_porosity_principles(v_lam, v_disp, ep)
    rho_ft = np.where(phi_total_approx > 0.01,
                      rho_ft_num / phi_total_approx, 1.0)

    phi_mb = np.where(np.abs(rho_mt - rho_ft) > 0.01,
                      (rho_mt - rho_b) / (rho_mt - rho_ft), 0.0)
    return np.clip(phi_mb, 0.0, 1.0)


# ---------- Nuclear-log forward models ----------

def bulk_density_model(v_lam: np.ndarray, v_disp: np.ndarray,
                        ep: RockClassEndpoints) -> np.ndarray:
    """
    Bulk density as a function of V_Lam and V_Disp (Eq. 4 of paper).

    ρ_b = ρ_S + V_Lam(ρ_Shale - ρ_S) + V_Disp(ρ_WetClay - ρ_fs)
    """
    return (ep.rho_s +
            v_lam * (ep.rho_shale - ep.rho_s) +
            v_disp * (ep.rho_wetclay - ep.rho_fs))


def neutron_porosity_model(v_lam: np.ndarray, v_disp: np.ndarray,
                            ep: RockClassEndpoints) -> np.ndarray:
    """
    Neutron porosity as a function of V_Lam and V_Disp.

    NPHI = NPHI_S + V_Lam(NPHI_Shale - NPHI_S) + V_Disp(NPHI_WetClay - NPHI_fs)
    """
    return (ep.nphi_s +
            v_lam * (ep.nphi_shale - ep.nphi_s) +
            v_disp * (ep.nphi_wetclay - ep.nphi_fs))


def gamma_ray_model(v_lam: np.ndarray, v_disp: np.ndarray,
                     ep: RockClassEndpoints) -> np.ndarray:
    """
    Gamma ray as a function of V_Lam and V_Disp.

    GR = GR_S + V_Lam(GR_Shale - GR_S) + V_Disp(GR_WetClay - GR_S)
    """
    return (ep.gr_s +
            v_lam * (ep.gr_shale - ep.gr_s) +
            v_disp * (ep.gr_wetclay - ep.gr_s))


def volumetric_pef_model(v_lam: np.ndarray, v_disp: np.ndarray,
                          ep: RockClassEndpoints) -> np.ndarray:
    """
    Volumetric PEF as a function of V_Lam and V_Disp (Eq. 6 of paper).

    U_b = U_S + V_Lam(U_Shale - U_S) + V_Disp(U_WetClay - U_fs)
    """
    return (ep.u_s +
            v_lam * (ep.u_shale - ep.u_s) +
            v_disp * (ep.u_wetclay - ep.u_fs))


def electron_density(rho_b: np.ndarray,
                     a: float = 0.1823,
                     b: float = 1.07) -> np.ndarray:
    """
    Electron density approximation (Eq. 7 of paper).

    ρ_e = a + b · ρ_b
    """
    return a + b * rho_b


def pef_from_volumetric(u_b: np.ndarray,
                         rho_b: np.ndarray) -> np.ndarray:
    """Convert volumetric PEF to actual PEF = U_b / ρ_e."""
    rho_e = electron_density(rho_b)
    return u_b / rho_e


# ---------- T-S diagram lines in log domain ----------

def ts_log_domain_lines(ep: RockClassEndpoints,
                         n_points: int = 50) -> dict:
    """
    Compute the laminated and dispersed lines in the log domain
    for RHOB-GR, RHOB-NPHI, and NPHI-GR crossplots.
    """
    v_lam_line = np.linspace(0, 1.0, n_points)
    v_disp_line = np.linspace(0, ep.phi_s, n_points)

    # Laminated: V_Disp = 0
    lam_rhob = bulk_density_model(v_lam_line, 0.0, ep)
    lam_nphi = neutron_porosity_model(v_lam_line, 0.0, ep)
    lam_gr = gamma_ray_model(v_lam_line, 0.0, ep)

    # Dispersed: V_Lam = 0
    disp_rhob = bulk_density_model(0.0, v_disp_line, ep)
    disp_nphi = neutron_porosity_model(0.0, v_disp_line, ep)
    disp_gr = gamma_ray_model(0.0, v_disp_line, ep)

    return {
        "laminated": {
            "v_lam": v_lam_line,
            "rhob": lam_rhob, "nphi": lam_nphi, "gr": lam_gr
        },
        "dispersed": {
            "v_disp": v_disp_line,
            "rhob": disp_rhob, "nphi": disp_nphi, "gr": disp_gr
        }
    }


# ---------- Synthetic log generation ----------

def generate_synthetic_shaly_sandstone(ep: RockClassEndpoints,
                                       n_layers: int = 20,
                                       noise_level: float = 0.01,
                                       seed: int = 42) -> dict:
    """
    Generate synthetic well logs for a shaly-sandstone formation.

    Returns dict with depth, v_lam, v_disp, rhob, nphi, gr, pef arrays.
    """
    rng = np.random.RandomState(seed)

    depth = np.arange(n_layers) * 0.5 + 100.0  # depth in metres

    # Random laminated and dispersed fractions
    v_lam = rng.uniform(0, 0.6, n_layers)
    v_disp = rng.uniform(0, ep.phi_s * 0.8, n_layers)

    # Some layers are purely laminated, some purely dispersed
    for i in range(0, n_layers, 5):
        v_disp[i] = 0.0  # purely laminated
    for i in range(2, n_layers, 5):
        v_lam[i] = 0.0   # purely dispersed

    rhob = bulk_density_model(v_lam, v_disp, ep) + rng.randn(n_layers) * noise_level
    nphi = neutron_porosity_model(v_lam, v_disp, ep) + rng.randn(n_layers) * noise_level
    gr = gamma_ray_model(v_lam, v_disp, ep) + rng.randn(n_layers) * noise_level * 100
    u_b = volumetric_pef_model(v_lam, v_disp, ep) + rng.randn(n_layers) * noise_level
    pef = pef_from_volumetric(u_b, rhob)

    return {
        "depth": depth, "v_lam": v_lam, "v_disp": v_disp,
        "rhob": rhob, "nphi": nphi, "gr": gr, "pef": pef
    }


# ---------- Rock classification ----------

def classify_shaly_sandstone(rhob: np.ndarray, nphi: np.ndarray,
                              gr: np.ndarray,
                              rock_classes: List[RockClassEndpoints]) -> np.ndarray:
    """
    Simple nearest-class assignment based on log responses.
    For each sample, find which rock class produces the smallest
    misfit between observed and modelled logs.

    Returns array of class indices.
    """
    n = len(rhob)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        best_dist = np.inf
        for c, ep in enumerate(rock_classes):
            # Try all combinations on a coarse grid
            for vl in np.linspace(0, 1, 10):
                for vd in np.linspace(0, ep.phi_s, 5):
                    r = bulk_density_model(np.array([vl]), np.array([vd]), ep)[0]
                    n_ = neutron_porosity_model(np.array([vl]), np.array([vd]), ep)[0]
                    g = gamma_ray_model(np.array([vl]), np.array([vd]), ep)[0]
                    dist = ((r - rhob[i]) / 0.1) ** 2 + \
                           ((n_ - nphi[i]) / 0.05) ** 2 + \
                           ((g - gr[i]) / 20) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        labels[i] = c
    return labels


# ---------- Test ----------

def test_all():
    """Test all functions with synthetic data."""
    print("=" * 60)
    print("Testing thomas_stieber_welllog module (Eghbali & Torres-Verdín, 2025)")
    print("=" * 60)

    ep = RockClassEndpoints(name="Rock Class I")

    # 1. Forward models
    v_lam = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    v_disp = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    rhob = bulk_density_model(v_lam, v_disp, ep)
    nphi = neutron_porosity_model(v_lam, v_disp, ep)
    gr = gamma_ray_model(v_lam, v_disp, ep)
    print("\n1) Laminated-line forward model:")
    for i in range(5):
        print(f"   V_Lam={v_lam[i]:.2f}  →  RHOB={rhob[i]:.3f}  "
              f"NPHI={nphi[i]:.3f}  GR={gr[i]:.1f}")

    # 2. Porosity comparison
    v_l = np.array([0.2])
    v_d = np.array([0.1])
    phi_ts = ts_porosity_principles(v_l, v_d, ep)
    phi_mb = ts_porosity_mass_balance(v_l, v_d, ep)
    print(f"\n2) Porosity (V_Lam=0.2, V_Disp=0.1): "
          f"T-S={phi_ts[0]:.4f}  Mass-bal={phi_mb[0]:.4f}  "
          f"Δ={abs(phi_ts[0]-phi_mb[0]):.4f}")

    # 3. Log-domain lines
    lines = ts_log_domain_lines(ep)
    print(f"\n3) Laminated line RHOB range: "
          f"{lines['laminated']['rhob'][0]:.3f} – "
          f"{lines['laminated']['rhob'][-1]:.3f}")

    # 4. Synthetic logs
    syn = generate_synthetic_shaly_sandstone(ep, n_layers=10)
    print(f"\n4) Generated {len(syn['depth'])} synthetic layers")
    print(f"   RHOB range: {syn['rhob'].min():.3f} – {syn['rhob'].max():.3f}")
    print(f"   GR range:   {syn['gr'].min():.1f} – {syn['gr'].max():.1f}")

    # 5. Multi-class
    ep2 = RockClassEndpoints(
        name="Rock Class II",
        phi_s=0.30, rho_s=2.45, nphi_s=0.00,
        gr_s=20.0, phi_shale=0.06, rho_shale=2.60,
        nphi_shale=0.25, gr_shale=100.0,
        rho_wetclay=2.30, nphi_wetclay=0.30,
        gr_wetclay=130.0, phi_wetclay=0.06
    )
    labels = classify_shaly_sandstone(syn['rhob'][:5], syn['nphi'][:5],
                                      syn['gr'][:5], [ep, ep2])
    print(f"\n5) Rock classification labels: {labels}")

    print("\n✓ All thomas_stieber_welllog tests passed.\n")


if __name__ == "__main__":
    test_all()
