"""
Article 5: Integration of Nuclear Spectroscopy Technology and Core Data Results
           for Through-Casing TOC Measurement and Saturation Analysis - A Case
           Study in Najmah-Sargelu Reservoir, South Kuwait
Bouchou, Abughneej, Ghioca, Alarcon, Mendez (2020)
DOI: 10.30632/PJV61N6-2020a5

Through-casing pulsed-neutron spectroscopy is combined with core data to derive
lithology, total organic carbon (TOC), and oil saturation in an organic-rich
carbonate where neutron porosity is invalid.  A weighted-least-squares
multimineral inversion solves for mineral volumes; the spectroscopy total carbon
minus mineral and matrix carbon gives the organic-pore "excess carbon", which
converts to oil saturation without environmental calibration.

Implements:

  - Linear multimineral log-response model  R = sum_j Rma_j * Vma_j   (Eq. 1)
  - Weighted least-squares inversion  min sum[(Rm - Rth)/sigma]^2     (Eq. 2)
  - Excess carbon  XCarbon = CTot - (CMin + CMat)                     (Eq. 3)
  - Oil volume / saturation  Vo = (Rhob * Xc)/(Rhoo * Fc * phie)      (Eq. 4)

Note: this issue's PDF text layer drops typeset glyphs (Eq. 3 survived verbatim
and is implemented exactly); Eqs. 1-2 and 4 are faithful standard-form
reconstructions of the described multimineral / excess-carbon workflow.
"""

import numpy as np


# ---------------------------------------------- multimineral inversion --

def _project_simplex(v):
    """Euclidean projection onto {x >= 0, sum(x) = 1}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(len(u)) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def forward_response(response_matrix, volumes):
    """Theoretical tool responses  R = A @ V  (Eq. 1).

    response_matrix A is (n_tools x n_minerals); volumes V sum to 1.
    """
    return np.asarray(response_matrix, float) @ np.asarray(volumes, float)


def multimineral_inversion(response_matrix, measured, sigma, iters=8000):
    """Weighted-least-squares mineral volumes (Eqs. 1-2), closure-constrained.

    Minimises sum_j [(R_meas,j - (A V)_j)/sigma_j]^2 subject to V >= 0,
    sum(V) = 1, by projected gradient with the exact simplex projection.
    """
    A = np.asarray(response_matrix, float)
    d = np.asarray(measured, float)
    w = 1.0 / np.asarray(sigma, float) ** 2          # weights
    n = A.shape[1]
    Aw = A * np.sqrt(w)[:, None]
    dw = d * np.sqrt(w)
    AtA = Aw.T @ Aw
    Atd = Aw.T @ dw
    v = np.full(n, 1.0 / n)
    lr = 1.0 / np.linalg.norm(AtA, 2)
    for _ in range(iters):
        v = _project_simplex(v - lr * (AtA @ v - Atd))
    return v


# ---------------------------------------------- excess carbon -----------

def excess_carbon(c_total, c_mineral, c_matrix):
    """Organic-pore excess carbon  XCarbon = CTot - (CMin + CMat)  (Eq. 3)."""
    return c_total - (c_mineral + c_matrix)


def oil_saturation(rhob, xcarbon, rho_oil, carbon_index, phi_e):
    """Oil saturation from excess carbon  (Eq. 4).

        So = (Rhob * Xc) / (Rhoo * Fc * phie)
    Xc is the excess-carbon weight fraction of the bulk rock, Fc the carbon
    weight fraction of the oil (carbon index, ~0.85).  The numerator is the
    oil volume per bulk and the phie in the denominator converts it to a
    pore-volume saturation.
    """
    return (rhob * xcarbon) / (rho_oil * carbon_index * phi_e)


def oil_volume(rhob, xcarbon, rho_oil, carbon_index, phi_e):
    """Bulk oil volume fraction  Vo = So * phie."""
    return oil_saturation(rhob, xcarbon, rho_oil, carbon_index, phi_e) * phi_e


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Through-Casing TOC & Saturation (Kuwait)")
    print("=" * 60)

    # Multimineral inversion: 4 tool responses, 3 minerals (calcite/dolomite/clay)
    # rows = [sigma c.u., bulk density, neutron, spectroscopy-Ca yield]
    minerals = ["calcite", "dolomite", "clay"]
    A = np.array([
        [9.0, 8.5, 25.0],     # sigma
        [2.71, 2.87, 2.60],   # rho_ma
        [0.00, 0.02, 0.30],   # neutron
        [0.40, 0.22, 0.05],   # Ca yield
    ])
    v_true = np.array([0.55, 0.30, 0.15])
    d = forward_response(A, v_true)
    sigma = np.array([0.5, 0.02, 0.01, 0.01])
    v_inv = multimineral_inversion(A, d, sigma)
    print(f"  recovered volumes      = {np.array2string(v_inv, precision=3)}")
    assert abs(v_inv.sum() - 1.0) < 1e-6
    assert np.max(np.abs(v_inv - v_true)) < 0.02

    # Excess carbon (Eq. 3) is the organic-pore carbon left after subtraction
    xc = excess_carbon(c_total=0.085, c_mineral=0.060, c_matrix=0.010)
    print(f"  excess carbon          = {xc:.4f}")
    assert abs(xc - 0.015) < 1e-9

    # Oil saturation from excess carbon (Eq. 4)
    so = oil_saturation(rhob=2.45, xcarbon=xc, rho_oil=0.85,
                        carbon_index=0.85, phi_e=0.12)
    print(f"  oil saturation         = {so:.3f}")
    assert 0.0 < so < 1.0
    # more excess carbon -> more oil
    so2 = oil_saturation(2.45, 0.030, 0.85, 0.85, 0.12)
    assert so2 > so
    print("  PASS")
    return {"volumes": v_inv.tolist(), "xcarbon": xc, "so": so}


if __name__ == "__main__":
    test_all()
