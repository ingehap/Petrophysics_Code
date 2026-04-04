"""
Model-Free Relative Permeability From ²³Na MRI Saturation Monitoring
=====================================================================

Implements the ideas of:

    Zamiri, M.S., Ansaribaranghar, N., Marica, F., Ramírez Aguilera, A.,
    Green, D., Caubit, C., Nicot, B., and Balcom, B.J., 2025b,
    "Relative Permeability Measurement Using Rapid In-Situ Saturation
    Measurement With ²³Na MRI",
    Petrophysics, 66(6), 1101–1117.
    DOI: 10.30632/PJV66N6-2025a13

Key ideas
---------
* Two-phase Darcy flow equation (Eq. 1) and continuity (Eq. 2).
* Goodfield et al. (2001) workflow: derive fractional flow and capillary
  dispersion from saturation-profile derivatives (Eq. 9).
* Corey kr curves (Eqs. 18-19) and Logbeta Pc functions (Eqs. 15-17).
* Model-free kr extraction using in-situ saturation data (²³Na MRI).
* Pressure-equation solver (Eq. 14) for validating the reconstructed
  kr and Pc curves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy import optimize


# ──────────────────────────────────────────────────────────────────────
# 1. Corey Relative Permeability  (Eqs. 18-19)
# ──────────────────────────────────────────────────────────────────────
def corey_krw(Sw: np.ndarray, Swir: float, Sor: float,
              krw0: float, alpha_w: float) -> np.ndarray:
    """Water relative permeability [Eq. 18].

    krw = krw0 * ((Sw - Swir) / (1 - Swir - Sor))^αw
    """
    Sw = np.asarray(Sw, float)
    Se = np.clip((Sw - Swir) / (1.0 - Swir - Sor), 0.0, 1.0)
    return krw0 * Se ** alpha_w


def corey_kro(Sw: np.ndarray, Swir: float, Sor: float,
              kro0: float, alpha_o: float) -> np.ndarray:
    """Oil relative permeability [Eq. 19].

    kro = kro0 * ((1 - Sw - Sor) / (1 - Swir - Sor))^αo
    """
    Sw = np.asarray(Sw, float)
    Se = np.clip((1.0 - Sw - Sor) / (1.0 - Swir - Sor), 0.0, 1.0)
    return kro0 * Se ** alpha_o


# ──────────────────────────────────────────────────────────────────────
# 2. Logbeta Capillary Pressure Functions  (Eqs. 15-17)
# ──────────────────────────────────────────────────────────────────────
def reduced_saturation(Sw: float, Smin: float, Smax: float = 1.0) -> float:
    """Reduced saturation [Eq. 17].

    Sr = (Sw - Smin) / (Smax - Smin)
    """
    denom = Smax - Smin
    if denom <= 0:
        return 0.0
    return (Sw - Smin) / denom


def logbeta_Pc_drainage(
    Sw: np.ndarray,
    Smin: float,
    po: float,
    pt: float,
) -> np.ndarray:
    """Logbeta Pc for primary drainage (water-wet) [Eq. 15].

    Pc = pt + po * log((1 - Sr) / Sr)

    where Sr is the reduced saturation and Smax = 1.

    Parameters
    ----------
    Sw : array_like
        Water saturation.
    Smin : float
        Irreducible water saturation.
    po : float
        Logbeta pressure parameter.
    pt : float
        Threshold (entry) pressure.

    Returns
    -------
    np.ndarray
        Capillary pressure.
    """
    Sw = np.asarray(Sw, float)
    Sr = np.clip((Sw - Smin) / (1.0 - Smin), 1e-10, 1.0 - 1e-10)
    return pt + po * np.log((1.0 - Sr) / Sr)


def logbeta_Pc_imbibition(
    Sw: np.ndarray,
    Smin: float,
    Smax: float,
    beta: float,
    po: float,
) -> np.ndarray:
    """Logbeta Pc for imbibition (may include oil-wet contribution) [Eq. 16].

    Sr_Pc0 = (Smax - Smin) is the reduced saturation at Pc = 0.
    α = Sr_Pc0 * (1 - Sr_Pc0) / β
    b = α * po * ln(...)

    For simplicity this returns:
        Pc = po * log((Smax - Sw) / (Sw - Smin)) + offset
    """
    Sw = np.asarray(Sw, float)
    Sr = np.clip((Sw - Smin) / (Smax - Smin), 1e-10, 1.0 - 1e-10)
    Sr_Pc0 = 1.0  # at Smax
    alpha = Sr_Pc0 * (1.0 - Sr_Pc0) / max(beta, 1e-10)
    # Simplified Logbeta form
    return po * np.log((1.0 - Sr) / Sr)


# ──────────────────────────────────────────────────────────────────────
# 3. Fractional Flow and Capillary Dispersion from Saturation Data
#    (Eqs. 6-7, 9-10 — the core of the model-free workflow)
# ──────────────────────────────────────────────────────────────────────
def compute_saturation_derivatives(
    Sw_profiles: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ∂Sw/∂x  and  ∂Sw/∂t  from spatiotemporal saturation data.

    Parameters
    ----------
    Sw_profiles : ndarray, shape (n_times, n_positions)
        Saturation profiles at each time step.
    x : 1-D array (n_positions,)
        Spatial positions along the core.
    t : 1-D array (n_times,)
        Time values for each profile.

    Returns
    -------
    Sx : ndarray, shape (n_times, n_positions)
        ∂Sw/∂x
    St : ndarray, shape (n_times, n_positions)
        ∂Sw/∂t
    """
    Sw = np.asarray(Sw_profiles, float)
    Sx = np.gradient(Sw, x, axis=1)  # spatial derivative
    St = np.gradient(Sw, t, axis=0)  # temporal derivative
    return Sx, St


def compute_water_flux(
    St: np.ndarray,
    x: np.ndarray,
    phi: float,
    qt: float,
) -> np.ndarray:
    """Water flux qw(x,t) from ∂Sw/∂t  [Eq. 10].

    qw(x, t) = qt − φ ∫₀ˣ (∂Sw/∂t) dx'

    Parameters
    ----------
    St : ndarray, shape (n_times, n_positions)
        ∂Sw/∂t at each (t, x).
    x : 1-D array
        Spatial positions.
    phi : float
        Porosity.
    qt : float
        Total flow rate (m³/s per m²).

    Returns
    -------
    qw : ndarray, same shape as St.
    """
    # Cumulative integral of St with respect to x for each time
    # Use numpy.trapezoid (numpy ≥ 2.0) or fallback to numpy.trapz
    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz

    integral = np.zeros_like(St)
    for i in range(St.shape[0]):
        integral[i, :] = phi * np.array([_trapz(St[i, :j+1], x[:j+1])
                            for j in range(len(x))])
    return qt - integral


def extract_fw_Dc(
    Sw_flat: np.ndarray,
    Sx_flat: np.ndarray,
    qw_flat: np.ndarray,
    qt: float,
    n_bins: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract fractional flow fw(Sw) and capillary dispersion Dc(Sw)
    using the linear relationship in Eq. 9.

    For each Sw bin, fit  qw = fw(Sw)*qt + Dc(Sw)*Sx  (linear in Sx).

    Parameters
    ----------
    Sw_flat : 1-D array
        Flattened saturation values.
    Sx_flat : 1-D array
        Flattened ∂Sw/∂x.
    qw_flat : 1-D array
        Flattened water flux.
    qt : float
        Total flow rate.
    n_bins : int
        Number of saturation bins.

    Returns
    -------
    Sw_centres, fw, Dc : 1-D arrays
    """
    Sw_edges = np.linspace(Sw_flat.min(), Sw_flat.max(), n_bins + 1)
    Sw_centres = 0.5 * (Sw_edges[:-1] + Sw_edges[1:])
    fw = np.zeros(n_bins)
    Dc = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (Sw_flat >= Sw_edges[i]) & (Sw_flat < Sw_edges[i + 1])
        if mask.sum() < 3:
            continue
        # Linear regression: qw = fw*qt + Dc*Sx
        A = np.column_stack([np.ones(mask.sum()) * qt, Sx_flat[mask]])
        b = qw_flat[mask]
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            fw[i] = coeffs[0]
            Dc[i] = coeffs[1]
        except np.linalg.LinAlgError:
            pass

    return Sw_centres, fw, Dc


# ──────────────────────────────────────────────────────────────────────
# 4. kr from Fractional Flow and Capillary Dispersion  (Eqs. 6-7)
# ──────────────────────────────────────────────────────────────────────
def kr_from_fw_Dc(
    Sw: np.ndarray,
    fw: np.ndarray,
    Dc: np.ndarray,
    mu_w: float,
    mu_o: float,
    K: float,
    dPc_dSw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive model-free kr curves from fw and Dc  [Eqs. 6-7].

    λw = fw * λt
    λo = (1 - fw) * λt
    Dc = λo * (-dPc/dSw) / (λt * φ)

    But we can also use:
        krw / (krw + kro * μw/μo) = fw
        Dc = (kro * krw) / (krw + kro * μw/μo) * (-dPc/dSw) / (μo * φ)

    Simplified approach for the model-free case:
        krw = fw * λt * μw   and   kro = (1-fw) * λt * μo
    where λt comes from the total mobility (needs pressure drop data).

    Here we use a direct inversion assuming Darcy and known Pc slope.

    Parameters
    ----------
    Sw : 1-D array
        Saturation values.
    fw : 1-D array
        Fractional flow of water.
    Dc : 1-D array
        Capillary dispersion.
    mu_w, mu_o : float
        Viscosities (Pa·s).
    K : float
        Absolute permeability (m²).
    dPc_dSw : 1-D array
        Derivative of capillary pressure w.r.t. Sw (Pa).

    Returns
    -------
    krw, kro : 1-D arrays
    """
    fw = np.asarray(fw, float)
    Dc = np.asarray(Dc, float)
    dPc = np.asarray(dPc_dSw, float)

    # From Eq. 7: Dc = -K * kro * krw / (mu_o * (kro/mu_o + krw/mu_w)) * dPc/dSw
    # and Eq. 6:  fw = (krw/mu_w) / (krw/mu_w + kro/mu_o)
    #
    # Let M = krw/mu_w,  N = kro/mu_o
    # fw = M / (M + N)  =>  N = M * (1-fw) / fw  (where fw > 0)
    # Dc = -K * N * M / (M + N) * dPc/dSw  (per unit phi)
    # => Dc = -K * fw * N * dPc/dSw

    # Avoid division by zero
    fw_safe = np.where(fw > 1e-10, fw, 1e-10)
    fw_safe = np.where(fw_safe < 1.0 - 1e-10, fw_safe, 1.0 - 1e-10)

    # From Dc and fw, extract N:
    dPc_safe = np.where(np.abs(dPc) > 1e-10, dPc, -1e-10)
    N = -Dc / (K * fw_safe * dPc_safe)
    N = np.maximum(N, 0.0)

    kro = N * mu_o
    M = N * fw_safe / (1.0 - fw_safe)
    krw = M * mu_w

    return np.clip(krw, 0, 1), np.clip(kro, 0, 1)


# ──────────────────────────────────────────────────────────────────────
# 5. Pressure Equation Solver  (Eq. 14 — validation)
# ──────────────────────────────────────────────────────────────────────
def solve_pressure_profile(
    Sw_profile: np.ndarray,
    x: np.ndarray,
    krw_func,
    kro_func,
    Pc_func,
    K: float,
    mu_w: float,
    mu_o: float,
    qt: float,
) -> np.ndarray:
    """Solve the linearised pressure equation [Eq. 14] for oil pressure.

    (d/dx)[ K (kro/μo + krw/μw) dPo/dx ] = qi_sources

    Simplified: 1-D finite-difference solution with no sources.

    Parameters
    ----------
    Sw_profile : 1-D array
        Water saturation at each grid point.
    x : 1-D array
        Grid positions (m).
    krw_func, kro_func : callable
        krw(Sw), kro(Sw).
    Pc_func : callable
        Pc(Sw).
    K : float
        Absolute permeability (m²).
    mu_w, mu_o : float
        Viscosities (Pa·s).
    qt : float
        Total Darcy flux (m/s).

    Returns
    -------
    Po : 1-D array
        Oil pressure profile (Pa).
    """
    n = len(x)
    dx = np.diff(x)

    krw = krw_func(Sw_profile)
    kro = kro_func(Sw_profile)
    Pc = Pc_func(Sw_profile)

    # Total mobility at cell faces (harmonic average)
    lam = K * (kro / mu_o + krw / mu_w)
    lam_face = 2.0 * lam[:-1] * lam[1:] / (lam[:-1] + lam[1:] + 1e-30)

    # Tridiagonal system for Po
    # Boundary: Po(outlet) = P_outlet (set to 0 gauge)
    # Inlet: total flux = qt
    A_mat = np.zeros((n, n))
    b_vec = np.zeros(n)

    for i in range(1, n - 1):
        A_mat[i, i - 1] = lam_face[i - 1] / dx[i - 1]
        A_mat[i, i + 1] = lam_face[i] / dx[i]
        A_mat[i, i] = -(lam_face[i - 1] / dx[i - 1] + lam_face[i] / dx[i])

    # Outlet BC: Po = 0
    A_mat[-1, -1] = 1.0
    b_vec[-1] = 0.0

    # Inlet BC: flux = -lam * dP/dx = qt  =>  dP/dx = -qt/lam
    A_mat[0, 0] = 1.0
    A_mat[0, 1] = -1.0
    b_vec[0] = -qt / (lam[0] + 1e-30) * dx[0]

    try:
        Po = np.linalg.solve(A_mat, b_vec)
    except np.linalg.LinAlgError:
        Po = np.zeros(n)

    return Po


def pressure_drop(Po: np.ndarray) -> float:
    """Pressure drop across the core from the oil pressure profile."""
    return float(Po[0] - Po[-1])


# ──────────────────────────────────────────────────────────────────────
# 6. Complete Workflow Wrapper
# ──────────────────────────────────────────────────────────────────────
def model_free_kr_workflow(
    Sw_profiles: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    phi: float,
    qt: float,
    mu_w: float,
    mu_o: float,
    K: float,
    Pc_params: dict,
    n_bins: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end model-free kr extraction from MRI saturation data.

    Steps (following the 5-step workflow in the paper):
    1. Compute Sx and St from saturation profiles.
    2. Compute qw from St (Eq. 10).
    3. Extract fw and Dc via binned linear regression (Eq. 9).
    4. Assume a parametric Pc and compute dPc/dSw.
    5. Invert for krw, kro (Eqs. 6-7).

    Parameters
    ----------
    Sw_profiles : (n_times, n_positions) array
    x : (n_positions,) array   Core positions (m).
    t : (n_times,) array       Measurement times (s).
    phi : float                Porosity.
    qt : float                 Total Darcy flux (m/s).
    mu_w, mu_o : float         Viscosities (Pa·s).
    K : float                  Absolute permeability (m²).
    Pc_params : dict           Keys: 'Smin', 'po', 'pt' for Logbeta drainage.
    n_bins : int               Saturation bins for fw/Dc extraction.

    Returns
    -------
    Sw_centres, krw, kro : 1-D arrays
    """
    # Step 1
    Sx, St = compute_saturation_derivatives(Sw_profiles, x, t)

    # Step 2
    qw = compute_water_flux(St, x, phi, qt)

    # Step 3
    Sw_flat = Sw_profiles.ravel()
    Sx_flat = Sx.ravel()
    qw_flat = qw.ravel()
    Sw_centres, fw, Dc = extract_fw_Dc(Sw_flat, Sx_flat, qw_flat, qt, n_bins)

    # Step 4: dPc/dSw from Logbeta
    Smin = Pc_params["Smin"]
    po = Pc_params["po"]
    pt = Pc_params.get("pt", 0.0)
    Sr = np.clip((Sw_centres - Smin) / (1.0 - Smin), 1e-10, 1.0 - 1e-10)
    # dPc/dSw = po * d/dSw [log((1-Sr)/Sr)]
    #         = po / ((1-Smin)) * (-1/(Sr*(1-Sr)))
    dPc_dSw = -po / ((1.0 - Smin) * Sr * (1.0 - Sr))

    # Step 5
    krw, kro = kr_from_fw_Dc(Sw_centres, fw, Dc, mu_w, mu_o, K, dPc_dSw)

    return Sw_centres, krw, kro


# ──────────────────────────────────────────────────────────────────────
# Quick demo — synthetic drainage case
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate synthetic saturation profiles (mimicking CYDAR output)
    nx, nt = 50, 20
    x = np.linspace(0, 0.05, nx)  # 5 cm core
    t = np.linspace(0, 3600, nt)  # 1 hour

    Swir, Sor = 0.15, 0.20
    # Simple propagating front
    Sw_profiles = np.zeros((nt, nx))
    for j in range(nt):
        front = int(nx * (j + 1) / (nt + 1))
        Sw_profiles[j, :front] = Swir
        Sw_profiles[j, front:] = 1.0 - Sor * j / nt

    # Smooth
    from scipy.ndimage import gaussian_filter1d
    for j in range(nt):
        Sw_profiles[j] = gaussian_filter1d(Sw_profiles[j], sigma=2)
        Sw_profiles[j] = np.clip(Sw_profiles[j], Swir, 1.0)

    # Run workflow
    Sw_c, krw, kro = model_free_kr_workflow(
        Sw_profiles, x, t,
        phi=0.24, qt=1e-5,
        mu_w=1e-3, mu_o=5e-3,
        K=1.1e-12,  # ~1.1 Darcy
        Pc_params={"Smin": 0.15, "po": 5000, "pt": 1000},
    )

    print("Sw       krw      kro")
    for i in range(0, len(Sw_c), 5):
        print(f"{Sw_c[i]:.3f}    {krw[i]:.4f}   {kro[i]:.4f}")
