"""
McGlynn et al. (2023), Petrophysics 64(6): 900-918.
New pulsed-neutron spectroscopy instrument with LaBr3 detectors providing
simultaneous C/O ratio, capture sigma, and ratio-based gas measurements
for two- and three-phase saturation analysis.

Implements a forward model for inelastic C/O and capture-sigma response,
plus a least-squares solver for (S_oil, S_gas, S_water).
"""
import numpy as np
from scipy.optimize import nnls


def co_ratio(s_oil, s_gas, s_w, phi,
             c_oil=0.85, c_gas=0.75, c_mat=0.10, o_w=1.0, o_mat=0.55):
    """Inelastic C/O ratio model."""
    C = phi * (s_oil * c_oil + s_gas * c_gas) + (1 - phi) * c_mat
    O = phi * s_w * o_w + (1 - phi) * o_mat
    return C / (O + 1e-9)


def sigma_capture(s_oil, s_gas, s_w, phi,
                  sig_oil=22, sig_gas=8, sig_w=80, sig_mat=10):
    """Capture cross-section in c.u."""
    return phi * (s_oil * sig_oil + s_gas * sig_gas + s_w * sig_w) \
           + (1 - phi) * sig_mat


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
