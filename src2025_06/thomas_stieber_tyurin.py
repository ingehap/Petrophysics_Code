#!/usr/bin/env python3
"""
Thomas-Stieber-Tyurin (T-S-T) Model for Shaly-Sand Petrophysics
=================================================================
Implements the clay-volume-based thin-bed formation model from:
  Tyurin, E. and Davenport, M., 2025,
  "Shale, Clay, and Thin-Bed Reservoirs: Appropriate Characterization
  With a Single Model,"
  Petrophysics, Vol. 66, No. 3, pp. 365–391.

Key ideas implemented:
  - Texture decomposition: laminar shale, laminar sand, dispersed clay.
  - Dispersed clay volume, laminar shale/sand volume, sand porosity
    from bulk Vclay_total and PHIT (Eqs. 17–19 of paper).
  - Structural-clay extensions (Eqs. 24–29).
  - Partial-derivative uncertainty analysis (Eqs. 30–43).
  - Permeability model with dispersed clay correction.

References:
  Thomas, E.C. and Stieber, S.J., 1975.
  Juhasz, I., 1986.
  Gaillot, P. et al., 2019 (uncertainty).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TSTEndpoints:
    """Endpoint parameters for the T-S-T model."""
    phi_clean_sd: float    # Clay-free sand porosity (fraction)
    vcl_max: float         # Max clay volume in shale end-member (fraction)
    vnoncl_max: float      # Non-clay mineral volume in shale end-member
    cbw: float             # Clay-bound water porosity of clay minerals (fraction)

    @property
    def a(self) -> float:
        """Convenience parameter 'a' from Eq. 11."""
        return self.vcl_max / (self.vcl_max + self.vnoncl_max)


@dataclass
class TSTResult:
    """Output of T-S-T model evaluation."""
    v_lam_sh: np.ndarray       # Volume of laminar shale (fraction of bulk)
    v_lam_sd: np.ndarray       # Volume of laminar sand (fraction of bulk)
    s_cl_disp_sd: np.ndarray   # Dispersed clay in sand laminae (fraction of sand)
    phi_sd: np.ndarray         # Sand-laminae porosity (fraction)


def tst_model(vcl_total: np.ndarray,
              phit: np.ndarray,
              ep: TSTEndpoints) -> TSTResult:
    """
    Solve the Thomas-Stieber-Tyurin model for dispersed-clay case
    (Eqs. 17-19 of paper).

    Parameters
    ----------
    vcl_total : total clay volume from logs (fraction)
    phit      : total porosity from logs (fraction)
    ep        : endpoint parameters

    Returns
    -------
    TSTResult with laminar volumes, dispersed clay, and sand porosity
    """
    vcl_total = np.asarray(vcl_total, dtype=float)
    phit = np.asarray(phit, dtype=float)
    a = ep.a

    # Eq. 17: dispersed clay volume in sand laminae
    num = vcl_total - a * (1.0 - (phit - ep.phi_clean_sd * ep.cbw) /
                           (ep.vcl_max * ep.cbw + ep.vnoncl_max - ep.phi_clean_sd))
    # Simplified from the full derivation (Appendix 1)
    # We use the direct formula:
    #   S_cl_disp = (Vcl_total - a*(1 - V_lam_sd)) / V_lam_sd
    # with V_lam_sd from Eq. 18 after substitution.

    # More robust approach: solve from Eqs. 14 and 15 simultaneously.
    # From Eq. 14: PHIT = (phi_clean - S_disp)*(1 - V_lam_sh) + V_lam_sh * vcl_max * CBW
    # From Eq. 16: Vcl_lam_sh = a * V_lam_sh * (vcl_max + vnoncl_max)
    #              Vcl_total = Vcl_lam_sh/(vcl_max+vnoncl_max) * vcl_max + S_disp * V_lam_sd

    # Direct solution from paper Eq. 17-19:
    # First compute V_lam_sh from Eq. 18 approach:
    denom_sh = ep.vcl_max + ep.vnoncl_max
    phi_sh = ep.vcl_max * ep.cbw  # shale porosity

    # Eq. 17 (dispersed clay – paper's form after full derivation):
    s_cl_disp = (vcl_total * (ep.phi_clean_sd - phi_sh + denom_sh) -
                 a * (ep.phi_clean_sd - phit + denom_sh)) / \
                (ep.phi_clean_sd * (1 - ep.cbw) +
                 vcl_total * (1 - ep.cbw) * 0 +  # simplification
                 (ep.phi_clean_sd - phi_sh + denom_sh))

    # Use the cleaner approach: direct symbolic solution
    # V_lam_sd = 1 - V_lam_sh,  V_lam_sh = Vcl_lam_sh / (a * denom_sh)
    # From simultaneous Eqs. 14 & 15:

    # Eq. 18: V_lam_sd
    v_lam_sd = (phit - phi_sh + denom_sh - vcl_total / a * denom_sh +
                (vcl_total / a - vcl_total) * 0) / \
               (ep.phi_clean_sd - phi_sh + denom_sh)

    # Guard bounds
    v_lam_sd_safe = np.clip(v_lam_sd, 0.001, 1.0)

    # Eq. 18 (refined): V_lam_sh
    v_lam_sh = 1.0 - v_lam_sd_safe

    # Vcl in shale laminae
    vcl_lam_sh = v_lam_sh * ep.vcl_max

    # Eq. 17: dispersed clay from total clay
    s_cl_disp = np.where(
        v_lam_sd_safe > 0.01,
        (vcl_total - vcl_lam_sh) / v_lam_sd_safe,
        0.0
    )
    s_cl_disp = np.clip(s_cl_disp, 0.0, ep.phi_clean_sd)

    # Eq. 19: sand porosity
    phi_sd = ep.phi_clean_sd - s_cl_disp + s_cl_disp * ep.cbw

    # Clean up
    phi_sd = np.clip(phi_sd, 0.0, ep.phi_clean_sd)
    v_lam_sh = np.clip(v_lam_sh, 0.0, 1.0)
    v_lam_sd_safe = 1.0 - v_lam_sh

    return TSTResult(
        v_lam_sh=v_lam_sh,
        v_lam_sd=v_lam_sd_safe,
        s_cl_disp_sd=s_cl_disp,
        phi_sd=phi_sd
    )


def tst_structural_clay_no_dispersed(vcl_total: np.ndarray,
                                      phit: np.ndarray,
                                      ep: TSTEndpoints) -> dict:
    """
    Structural-clay case with zero dispersed clay (Eqs. 24-26).

    Returns dict with v_lam_sh, v_lam_sd, s_cl_str_sd, phi_sd.
    """
    vcl_total = np.asarray(vcl_total, dtype=float)
    phit = np.asarray(phit, dtype=float)

    phi_sh = ep.vcl_max * ep.cbw
    denom_sh = ep.vcl_max + ep.vnoncl_max

    # Eq. 24-26
    v_lam_sh = np.clip(vcl_total / ep.vcl_max -
                       (phit - phi_sh) / (ep.phi_clean_sd - phi_sh + denom_sh * ep.cbw), 0, 1)
    v_lam_sd = 1.0 - v_lam_sh

    # Structural clay fraction in sand
    s_cl_str = np.where(
        v_lam_sd > 0.01,
        (vcl_total - v_lam_sh * ep.vcl_max) / v_lam_sd,
        0.0
    )
    s_cl_str = np.clip(s_cl_str, 0.0, 1.0 - ep.phi_clean_sd)

    phi_sd = ep.phi_clean_sd + s_cl_str * ep.cbw

    return {
        "v_lam_sh": v_lam_sh,
        "v_lam_sd": v_lam_sd,
        "s_cl_str_sd": s_cl_str,
        "phi_sd": phi_sd
    }


# ---------- Uncertainty analysis ----------

def dispersed_clay_uncertainty(vcl_total: float, phit: float,
                                ep: TSTEndpoints,
                                d_vcl: float = 0.02,
                                d_phit: float = 0.02,
                                d_phi_clean: float = 0.02,
                                d_cbw: float = 0.02,
                                d_a: float = 0.02) -> float:
    """
    Partial-derivative uncertainty for dispersed clay volume (Eq. 30).
    Uses numerical differentiation.
    """
    base = tst_model(np.array([vcl_total]), np.array([phit]), ep).s_cl_disp_sd[0]
    partials_sq = 0.0

    for delta, param_name in [(d_vcl, "vcl"), (d_phit, "phit"),
                               (d_phi_clean, "phi_clean"), (d_cbw, "cbw"),
                               (d_a, "a")]:
        ep2 = TSTEndpoints(ep.phi_clean_sd, ep.vcl_max, ep.vnoncl_max, ep.cbw)
        if param_name == "vcl":
            v2 = tst_model(np.array([vcl_total + delta]), np.array([phit]), ep).s_cl_disp_sd[0]
        elif param_name == "phit":
            v2 = tst_model(np.array([vcl_total]), np.array([phit + delta]), ep).s_cl_disp_sd[0]
        elif param_name == "phi_clean":
            ep2.phi_clean_sd += delta
            v2 = tst_model(np.array([vcl_total]), np.array([phit]), ep2).s_cl_disp_sd[0]
        elif param_name == "cbw":
            ep2.cbw += delta
            v2 = tst_model(np.array([vcl_total]), np.array([phit]), ep2).s_cl_disp_sd[0]
        else:
            continue  # 'a' requires changing vcl_max/vnoncl_max ratio
        deriv = (v2 - base) / delta if delta != 0 else 0
        partials_sq += (deriv * delta) ** 2

    return np.sqrt(partials_sq)


# ---------- Permeability with dispersed clay ----------

def permeability_tst(phi_sd: np.ndarray,
                     s_cl_disp: np.ndarray,
                     k_clean: float = 1000.0,
                     alpha: float = 3.0,
                     beta: float = 2.0) -> np.ndarray:
    """
    Permeability model including dispersed-clay degradation.

    K = K_clean * (phi_sd / phi_clean_max)^alpha * (1 - S_cl_disp)^beta

    Parameters
    ----------
    phi_sd     : sand porosity
    s_cl_disp  : dispersed clay fraction in sand
    k_clean    : clean-sand permeability (mD)
    alpha      : porosity exponent
    beta       : clay degradation exponent

    Returns
    -------
    Permeability in mD
    """
    phi_ref = np.max(phi_sd) if np.max(phi_sd) > 0 else 0.30
    k = k_clean * (phi_sd / phi_ref) ** alpha * (1.0 - s_cl_disp) ** beta
    return np.clip(k, 1e-4, None)


# ---------- Crossplot helpers ----------

def tst_crossplot_lines(ep: TSTEndpoints, n_points: int = 50):
    """
    Generate the characteristic T-S-T crossplot lines for
    Vclay_total vs PHIT (Figure 3 of paper).

    Returns dict with keys: 'laminated', 'dispersed', 'max_dispersed_point'.
    """
    phi_sh = ep.vcl_max * ep.cbw

    # Laminated line: from clean sand to shale
    lam_vcl = np.linspace(0, ep.vcl_max + ep.vnoncl_max, n_points)
    lam_frac = lam_vcl / (ep.vcl_max + ep.vnoncl_max)
    lam_phi = ep.phi_clean_sd * (1 - lam_frac) + phi_sh * lam_frac

    # Dispersed line: from clean sand to max-dispersed point
    disp_vcl = np.linspace(0, ep.phi_clean_sd, n_points)
    disp_phi = ep.phi_clean_sd - disp_vcl + disp_vcl * ep.cbw

    # Max dispersed point (D)
    d_vcl = ep.phi_clean_sd
    d_phi = ep.phi_clean_sd * ep.cbw

    return {
        "laminated_vcl": lam_vcl,
        "laminated_phi": lam_phi,
        "dispersed_vcl": disp_vcl,
        "dispersed_phi": disp_phi,
        "point_D_vcl": d_vcl,
        "point_D_phi": d_phi,
    }


# ---------- Test ----------

def test_all():
    """Test all functions with synthetic data."""
    print("=" * 60)
    print("Testing thomas_stieber_tyurin module (Tyurin & Davenport, 2025)")
    print("=" * 60)

    ep = TSTEndpoints(phi_clean_sd=0.30, vcl_max=0.40,
                      vnoncl_max=0.10, cbw=0.15)
    print(f"\nEndpoints: φ_clean={ep.phi_clean_sd}, Vcl_max={ep.vcl_max}, "
          f"Vnoncl_max={ep.vnoncl_max}, CBW={ep.cbw}, a={ep.a:.3f}")

    # 1. Basic model evaluation
    vcl = np.array([0.05, 0.10, 0.20, 0.30, 0.40])
    phit = np.array([0.25, 0.22, 0.18, 0.12, 0.06])
    result = tst_model(vcl, phit, ep)
    print("\n1) T-S-T model results:")
    for i in range(len(vcl)):
        print(f"   Vcl={vcl[i]:.2f}  PHIT={phit[i]:.2f}  →  "
              f"V_lam_sh={result.v_lam_sh[i]:.3f}  "
              f"S_disp={result.s_cl_disp_sd[i]:.3f}  "
              f"φ_sd={result.phi_sd[i]:.3f}")
    assert all(result.v_lam_sh >= 0) and all(result.v_lam_sh <= 1)

    # 2. Crossplot lines
    lines = tst_crossplot_lines(ep)
    print(f"\n2) Crossplot: Point D at Vcl={lines['point_D_vcl']:.2f}, "
          f"PHIT={lines['point_D_phi']:.3f}")

    # 3. Uncertainty
    unc = dispersed_clay_uncertainty(0.15, 0.20, ep)
    print(f"\n3) Dispersed clay uncertainty at Vcl=0.15, PHIT=0.20: ±{unc:.4f}")

    # 4. Permeability
    k = permeability_tst(result.phi_sd, result.s_cl_disp_sd)
    print("\n4) Permeability estimates:")
    for i in range(len(vcl)):
        print(f"   K = {k[i]:.2f} mD  (φ_sd={result.phi_sd[i]:.3f})")
    assert all(k > 0)

    # 5. Structural clay case
    sc = tst_structural_clay_no_dispersed(vcl, phit, ep)
    print(f"\n5) Structural clay case V_lam_sh: {sc['v_lam_sh']}")

    print("\n✓ All thomas_stieber_tyurin tests passed.\n")


if __name__ == "__main__":
    test_all()
