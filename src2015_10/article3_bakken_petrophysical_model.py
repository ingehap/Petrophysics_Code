"""
Article 3: Using Advanced Logging Measurements to Develop a Robust Petrophysical
           Model for the Bakken Petroleum System
Simpson, Hohman, Pirie, Horkowitz (2015)
Reference: Petrophysics Vol. 56, No. 5 (October 2015), pp. 457-478
DOI: none assigned (this issue predates SPWLA DOI assignment)

The Bakken petroleum system is a hybrid play (conventional reservoirs plus
shale source/reservoir).  A robust model is built from advanced logs: a
multimineral solution from the linear volumetric log response, NMR permeability
from the carbonate SDR transform, and a bimodal horizontal/vertical resistivity
(Rh-Rv) thin-bed analysis to recover the reservoir (carbonate thin bed)
resistivity in the laminated Three Forks Formation.

Implements:

  - Linear volumetric log response  Mi = sum_j Vj*Rj  (Eqs. 1-2)
  - Multimineral linear inversion (log responses -> volumes)
  - Carbonate SDR NMR permeability  KSDR = A*phi^B*(rho*T2LM)^C  (Eq. 3)
  - NMR pore surface-to-volume from T2  S/V = 1/(rho*T2)  (Eq. 4)
  - Bimodal Rh (parallel) / Rv (series) thin-bed model and solver (Eqs. 5-7)

Note: this issue's PDF has a text layer; the log-response, SDR and Rh-Rv
relations (Eqs. 1-7) are transcribed from the body, while the typeset glyphs
were dropped and reconstructed in standard form.  Resistivity in ohm-m,
permeability in mD, T2 in ms, porosity/fractions dimensionless.
"""

import numpy as np


# ---------------------------------------------- log response --------------

def log_response(volumes, response_params):
    """Linear volumetric log response  Mi = sum_j Vj*Rj  (Eqs. 1-2)."""
    return float(np.sum(np.asarray(volumes, float) * np.asarray(response_params, float)))


def multimineral_inversion(measurements, endpoints):
    """Multimineral solution by constrained linear inversion.

    Solves  endpoints @ volumes = measurements  with the unity constraint
    sum(volumes) = 1 (added as a high-weight equation), least squares.
    `endpoints` is (n_logs, n_minerals) of pure-component tool responses.
    """
    a = np.asarray(endpoints, float)
    b = np.asarray(measurements, float)
    w = 1e3
    a_aug = np.vstack([a, w * np.ones(a.shape[1])])
    b_aug = np.concatenate([b, [w]])
    vols, *_ = np.linalg.lstsq(a_aug, b_aug, rcond=None)
    return vols


# ---------------------------------------------- NMR --------------

def ksdr_permeability(phi, rho, t2lm, a=4.0, b=4.0, c=2.0):
    """Carbonate SDR NMR permeability (Eq. 3)

        KSDR = A*phi^B*(rho*T2LM)^C,

    with rho the surface relaxivity (um/s) and T2LM the log-mean T2.
    """
    return a * phi ** b * (rho * t2lm) ** c


def nmr_surface_to_volume(t2, rho):
    """Pore surface-to-volume ratio from T2  1/T2 = rho*(S/V)  ->  S/V = 1/(rho*T2)
    (Eq. 4)."""
    return 1.0 / (rho * np.asarray(t2, float))


# ---------------------------------------------- thin-bed Rh-Rv --------------

def thinbed_rh(rsand, rshale_h, f_shale):
    """Horizontal (parallel) resistivity  1/Rh = Fsand/Rsand + Fshale/Rshale_h
    (Eqs. 5, 7)."""
    return 1.0 / ((1.0 - f_shale) / rsand + f_shale / rshale_h)


def thinbed_rv(rsand, rshale_v, f_shale):
    """Vertical (series) resistivity  Rv = Fsand*Rsand + Fshale*Rshale_v  (Eq. 6)."""
    return (1.0 - f_shale) * rsand + f_shale * rshale_v


def solve_thinbed(rh, rv, rshale_h, rshale_v):
    """Bimodal Rh-Rv solution for the shale fraction and reservoir resistivity.

    Given measured Rh, Rv and the (anisotropic) shale resistivities, solves
    Eqs. 5-7 for (F_shale, Rsand) by bisection on F_shale in [0, 1).
    """
    def rh_model(fsh):
        rsand = (rv - fsh * rshale_v) / (1.0 - fsh)        # from the series (Rv) eq
        return thinbed_rh(rsand, rshale_h, fsh)

    lo, hi = 0.0, 1.0 - 1e-6
    # Rh increases as the conductive shale fraction decreases; bracket and bisect
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rh_model(mid) > rh:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    f_shale = 0.5 * (lo + hi)
    rsand = (rv - f_shale * rshale_v) / (1.0 - f_shale)
    return f_shale, rsand


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Bakken Petrophysical Model")
    print("=" * 60)

    # Multimineral inversion recovers known volumes (sand / calcite / fluid)
    endpoints = np.array([[2.65, 2.71, 1.0],      # density
                          [-0.02, 0.0, 1.0]])     # neutron
    true_vols = np.array([0.6, 0.25, 0.15])
    meas = endpoints @ true_vols
    vols = multimineral_inversion(meas, endpoints)
    print(f"  inverted volumes       = {np.round(vols, 3)}")
    assert np.allclose(vols, true_vols, atol=1e-3) and np.isclose(vols.sum(), 1.0)
    assert np.isclose(log_response(true_vols, endpoints[0]), meas[0])

    # SDR permeability rises with porosity and T2 log-mean
    k = ksdr_permeability(0.08, rho=20.0, t2lm=30.0)
    print(f"  KSDR permeability      = {k:.4f} mD")
    assert k > 0 and ksdr_permeability(0.12, 20.0, 30.0) > k

    # NMR S/V: smaller pores (shorter T2) give larger S/V
    assert nmr_surface_to_volume(10.0, 20e-3) > nmr_surface_to_volume(100.0, 20e-3)

    # Bimodal Rh-Rv: Rv (series) >= Rh (parallel); solver recovers Fshale, Rsand
    f_sh, rsand = 0.4, 50.0
    rsh_h, rsh_v = 2.0, 3.0
    rh = thinbed_rh(rsand, rsh_h, f_sh)
    rv = thinbed_rv(rsand, rsh_v, f_sh)
    print(f"  Rh / Rv                = {rh:.3f} / {rv:.3f}")
    assert rv > rh
    f_fit, rsand_fit = solve_thinbed(rh, rv, rsh_h, rsh_v)
    print(f"  solved Fshale / Rsand  = {f_fit:.3f} / {rsand_fit:.2f}")
    assert np.isclose(f_fit, f_sh, atol=1e-3) and np.isclose(rsand_fit, rsand, atol=1e-1)
    print("  PASS")
    return {"k_sdr": float(k), "Fshale": float(f_fit), "Rsand": float(rsand_fit)}


if __name__ == "__main__":
    test_all()
