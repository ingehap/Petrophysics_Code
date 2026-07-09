"""
McGlynn et al. (2023), Petrophysics 64(6): 900-918.
New pulsed-neutron spectroscopy instrument with LaBr3 detectors providing
simultaneous C/O ratio, capture sigma, and ratio-based gas measurements
for two- and three-phase saturation analysis.

Implements a forward model for inelastic C/O and capture-sigma response,
plus a least-squares solver for (S_oil, S_gas, S_water).
"""
import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib
from scipy.optimize import nnls


def co_ratio(s_oil, s_gas, s_w, phi,
             c_oil=0.85, c_gas=0.75, c_mat=0.10, o_w=1.0, o_mat=0.55):
    """Inelastic C/O ratio model."""
    return petrolib.nuclear.co_forward_3phase(
        phi, s_oil, s_gas, s_w, c_oil=c_oil, c_gas=c_gas, c_mat=c_mat, o_w=o_w, o_mat=o_mat
    )


def sigma_capture(s_oil, s_gas, s_w, phi,
                  sig_oil=22, sig_gas=8, sig_w=80, sig_mat=10):
    """Capture cross-section in c.u."""
    return petrolib.nuclear.sigma_forward_3phase(
        phi, s_oil, s_gas, s_w, sigma_oil=sig_oil, sigma_gas=sig_gas, sigma_w=sig_w, sigma_ma=sig_mat
    )


def solve_saturations(co_obs, sigma_obs, gas_ratio_obs, phi):
    """Solve constrained 3-phase saturations from 3 measurements."""
    from scipy.optimize import least_squares

    def resid(s):
        so, sg, sw = s
        r = [co_ratio(so, sg, sw, phi) - co_obs,
             sigma_capture(so, sg, sw, phi) - sigma_obs,
             (sg / (so + sg + 1e-9)) - gas_ratio_obs,
             (so + sg + sw) - 1.0]
        return r
    res = least_squares(resid, [0.3, 0.3, 0.4],
                        bounds=([0, 0, 0], [1, 1, 1]))
    return dict(s_oil=res.x[0], s_gas=res.x[1], s_water=res.x[2])


def test_all():
    phi = 0.20
    truth = dict(s_oil=0.35, s_gas=0.25, s_water=0.40)
    co = co_ratio(truth["s_oil"], truth["s_gas"], truth["s_water"], phi)
    sig = sigma_capture(truth["s_oil"], truth["s_gas"], truth["s_water"], phi)
    gr = truth["s_gas"] / (truth["s_oil"] + truth["s_gas"])
    rec = solve_saturations(co, sig, gr, phi)
    print("McGlynn et al. pulsed-neutron 3-phase solver:")
    print(f"  truth    : {truth}")
    print(f"  recovered: { {k: round(v,3) for k,v in rec.items()} }")
    for k in truth:
        assert abs(rec[k] - truth[k]) < 0.05
    print("  PASS")


if __name__ == "__main__":
    test_all()
