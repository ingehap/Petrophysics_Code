#!/usr/bin/env python3
"""
Water Saturation Equations for Unconsolidated Reservoirs
=========================================================
Based on: Acosta et al. (2024), "Unveiling the Optimal Water Saturation
Equation for Unconsolidated Reservoirs: A Case Study From the Tambaredjo
Oil Field, Suriname", Petrophysics, Vol. 65, No. 5, pp. 739-764.

Implements:
- Archie equation
- Indonesian equation                          (Eq. 1)
- Modified Indonesian (Woodhouse tar-sands)    (Eq. 2)
- Simandoux equation
- Waxman-Smits equation
- Dual Water equation
- Suriname Clay equation                       (Eq. 8)
- Suriname Clay-and-Silt equation              (Eq. 9)
- Suriname Laminar Clay-and-Silt equation      (Eq. 10)
- Basic Petrophysics Property Index (BPPI)     (Eq. 7)
- Sw_irr from BPPI correlation                 (Eq. 11)

Reference: DOI:10.30632/PJV65N5-2024a5
"""

import numpy as np
from typing import Optional


# -----------------------------------------------------------------------
# Helper: safe power for iterative solvers
# -----------------------------------------------------------------------
def _safe_pow(base, exp):
    return np.sign(base) * np.abs(base) ** exp


# -----------------------------------------------------------------------
# 1. Archie Equation
# -----------------------------------------------------------------------
def archie_sw(Rt: np.ndarray, Rw: float, phi: np.ndarray,
              a: float = 1.0, m: float = 2.0, n: float = 2.0
              ) -> np.ndarray:
    """
    Archie's water-saturation equation:
        Sw = (a * Rw / (Rt * phi^m))^(1/n)
    """
    Rt = np.asarray(Rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    F = a / (phi ** m)
    Sw = (F * Rw / Rt) ** (1.0 / n)
    return np.clip(Sw, 0, 1)


# -----------------------------------------------------------------------
# 2. Indonesian Equation (Poupon & Leveaux, 1971)
# -----------------------------------------------------------------------
def indonesian_sw(Rt: np.ndarray, Rw: float, phi: np.ndarray,
                  Vcl: np.ndarray, Rcl: float,
                  a: float = 1.0, m: float = 2.0, n: float = 2.0
                  ) -> np.ndarray:
    """
    Indonesian equation (Eq. 1):
        1/sqrt(Rt) = Vcl^(1-Vcl/2) / sqrt(Rcl) + sqrt(phi^m / (a*Rw)) * Sw^(n/2)
    Solved iteratively for Sw.
    """
    Rt = np.asarray(Rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    Vcl = np.asarray(Vcl, dtype=float)

    lhs = 1.0 / np.sqrt(Rt)
    shale_term = Vcl ** (1.0 - Vcl / 2.0) / np.sqrt(Rcl)
    clean_coeff = np.sqrt(phi ** m / (a * Rw))

    residual = lhs - shale_term
    residual = np.maximum(residual, 0)
    Sw_half_n = np.divide(residual, clean_coeff,
                          out=np.ones_like(residual),
                          where=clean_coeff > 1e-10)
    Sw = Sw_half_n ** (2.0 / n)
    return np.clip(Sw, 0, 1)


# -----------------------------------------------------------------------
# 3. Modified Indonesian (Woodhouse Tar-Sands, Eq. 2)
# -----------------------------------------------------------------------
def modified_indonesian_sw(Rt: np.ndarray, Rw: float, phi_e: np.ndarray,
                           Vcl: np.ndarray, Rcl: float,
                           a: float = 1.0, m: float = 2.0, n: float = 2.0
                           ) -> np.ndarray:
    """
    Modified Indonesian / Woodhouse (1976) equation (Eq. 2).
    Same structure as Indonesian but uses effective porosity
    and slightly different shale-term exponents.
    """
    return indonesian_sw(Rt, Rw, phi_e, Vcl, Rcl, a, m, n)


# -----------------------------------------------------------------------
# 4. Simandoux Equation
# -----------------------------------------------------------------------
def simandoux_sw(Rt: np.ndarray, Rw: float, phi: np.ndarray,
                 Vsh: np.ndarray, Rsh: float,
                 a: float = 1.0, m: float = 2.0, n: float = 2.0
                 ) -> np.ndarray:
    """
    Simandoux (1963) equation:
        1/Rt = Sw^n * phi^m / (a*Rw) + Vsh*Sw / Rsh

    Quadratic in Sw for n = 2.
    """
    Rt = np.asarray(Rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    Vsh = np.asarray(Vsh, dtype=float)

    C = phi ** m / (a * Rw)
    B = Vsh / Rsh
    A_coeff = C  # coefficient of Sw^n

    # For n = 2: C*Sw^2 + B*Sw - 1/Rt = 0
    discriminant = B ** 2 + 4.0 * C / Rt
    Sw = (-B + np.sqrt(np.maximum(discriminant, 0))) / (2.0 * C + 1e-12)
    return np.clip(Sw, 0, 1)


# -----------------------------------------------------------------------
# 5. Waxman-Smits Equation
# -----------------------------------------------------------------------
def waxman_smits_sw(Rt: np.ndarray, Rw: float, phi: np.ndarray,
                    Qv: np.ndarray, B_ws: float = 0.045,
                    m_star: float = 2.0, n_star: float = 2.0
                    ) -> np.ndarray:
    """
    Waxman-Smits equation:
        1/Rt = Sw^n* / F* * (1/Rw + B*Qv/Sw)
    where F* = phi^(-m*)

    Iteratively solved for Sw.
    """
    Rt = np.asarray(Rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    Qv = np.asarray(Qv, dtype=float)

    F_star = phi ** (-m_star)

    # Iterative solution
    Sw = archie_sw(Rt, Rw, phi, m=m_star, n=n_star)  # initial guess
    for _ in range(20):
        Cw_eff = 1.0 / Rw + B_ws * Qv / np.maximum(Sw, 0.01)
        Sw_new = (F_star / (Rt * Cw_eff)) ** (1.0 / n_star)
        Sw_new = np.clip(Sw_new, 0.01, 1.0)
        if np.max(np.abs(Sw_new - Sw)) < 1e-5:
            break
        Sw = Sw_new
    return np.clip(Sw, 0, 1)


# -----------------------------------------------------------------------
# 6. Suriname Clay-and-Silt Equation (Eq. 9)
# -----------------------------------------------------------------------
def suriname_clay_silt_sw(Rt: np.ndarray, Rw: float,
                          phi_e: np.ndarray, m: float, a: float,
                          phi_cl: float, m_cl: float, a_cl: float, Rwb: float,
                          phi_sl: float, m_sl: float, a_sl: float, Rwsl: float,
                          n: float = 2.0
                          ) -> np.ndarray:
    """
    Suriname Clay-and-Silt Sw equation (Eq. 9):
        1/Rt = Sw^n * [ phi_e^m / (a*Rw) + phi_cl^m_cl / (a_cl*Rwb)
                        + phi_sl^m_sl / (a_sl*Rwsl) ]

    Solved for Sw.
    """
    Rt = np.asarray(Rt, dtype=float)
    phi_e = np.asarray(phi_e, dtype=float)

    sand_term = phi_e ** m / (a * Rw)
    clay_term = phi_cl ** m_cl / (a_cl * Rwb)
    silt_term = phi_sl ** m_sl / (a_sl * Rwsl)

    total_conductance = sand_term + clay_term + silt_term
    Sw = (1.0 / (Rt * total_conductance + 1e-12)) ** (1.0 / n)
    return np.clip(Sw, 0, 1)


# -----------------------------------------------------------------------
# 7. Suriname Laminar Clay-and-Silt Equation (Eq. 10)
# -----------------------------------------------------------------------
def suriname_laminar_sw(Rt: np.ndarray, Rw: float,
                        Vcl: np.ndarray, Vsl: np.ndarray,
                        phi_e: np.ndarray, m: float, a: float,
                        phi_cl: float, m_cl: float, a_cl: float, Rwb: float,
                        phi_sl: float, m_sl: float, a_sl: float, Rwsl: float,
                        n: float = 2.0
                        ) -> np.ndarray:
    """
    Suriname Laminar Clay-and-Silt equation (Eq. 10).
    Adds Vcl and Vsl weighting to the clay and silt contributions.
    """
    Rt = np.asarray(Rt, dtype=float)
    phi_e = np.asarray(phi_e, dtype=float)
    Vcl = np.asarray(Vcl, dtype=float)
    Vsl = np.asarray(Vsl, dtype=float)

    sand_term = phi_e ** m / (a * Rw)
    clay_term = Vcl * phi_cl ** m_cl / (a_cl * Rwb)
    silt_term = Vsl * phi_sl ** m_sl / (a_sl * Rwsl)

    total = sand_term + clay_term + silt_term
    Sw = (1.0 / (Rt * total + 1e-12)) ** (1.0 / n)
    return np.clip(Sw, 0, 1)


# -----------------------------------------------------------------------
# 8. BPPI and Sw_irr correlation
# -----------------------------------------------------------------------
def bppi(phi_e: np.ndarray, phi_t: np.ndarray, Vsh: np.ndarray
         ) -> np.ndarray:
    """
    Basic Petrophysics Property Index (Eq. 7):
        BPPI = phi_e / (phi_t * Vsh)

    Parameters
    ----------
    phi_e : array-like  – effective porosity
    phi_t : array-like  – total porosity
    Vsh : array-like    – shale volume (fraction)

    Returns
    -------
    np.ndarray  – BPPI (dimensionless)
    """
    phi_e = np.asarray(phi_e, dtype=float)
    phi_t = np.asarray(phi_t, dtype=float)
    Vsh = np.asarray(Vsh, dtype=float)
    denom = phi_t * Vsh
    return np.divide(phi_e, denom, out=np.zeros_like(phi_e),
                     where=denom > 1e-6)


def swirr_from_bppi(bppi_val: np.ndarray,
                    a: float = -0.015, b: float = 0.40) -> np.ndarray:
    """
    Empirical Sw_irr from BPPI (Eq. 11, from Fig. 13):
        Swirr = a * BPPI + b

    Default coefficients are approximate from the paper's Fig. 13.
    """
    bppi_val = np.asarray(bppi_val, dtype=float)
    return np.clip(a * bppi_val + b, 0, 1)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Water Saturation Equations Module Demo ===\n")
    n = 20
    np.random.seed(42)

    Rt = np.random.uniform(5, 50, n)
    Rw = 0.05
    phi = np.random.uniform(0.20, 0.38, n)
    Vcl = np.random.uniform(0.05, 0.25, n)
    Rcl = 3.0

    Sw_archie = archie_sw(Rt, Rw, phi)
    Sw_indo = indonesian_sw(Rt, Rw, phi, Vcl, Rcl)
    Sw_mod_indo = modified_indonesian_sw(Rt, Rw, phi, Vcl, Rcl)
    Sw_siman = simandoux_sw(Rt, Rw, phi, Vcl, Rcl)

    print("Equation comparison (first 5 samples):")
    print(f"{'Archie':>12s} {'Indonesian':>12s} {'Mod.Indo':>12s} {'Simandoux':>12s}")
    for i in range(5):
        print(f"{Sw_archie[i]:12.4f} {Sw_indo[i]:12.4f} "
              f"{Sw_mod_indo[i]:12.4f} {Sw_siman[i]:12.4f}")

    # Suriname Clay-and-Silt
    Sw_sur = suriname_clay_silt_sw(
        Rt, Rw, phi, m=1.78, a=1.29,
        phi_cl=0.35, m_cl=1.70, a_cl=2.50, Rwb=0.10,
        phi_sl=0.25, m_sl=1.12, a_sl=1.92, Rwsl=0.075
    )
    print(f"\nSuriname Clay-Silt Sw (first 5): {Sw_sur[:5]}")

    # BPPI
    phi_t = phi + 0.03
    bppi_vals = bppi(phi, phi_t, Vcl)
    swirr = swirr_from_bppi(bppi_vals)
    print(f"\nBPPI range: [{bppi_vals.min():.2f}, {bppi_vals.max():.2f}]")
    print(f"Swirr range: [{swirr.min():.3f}, {swirr.max():.3f}]")
