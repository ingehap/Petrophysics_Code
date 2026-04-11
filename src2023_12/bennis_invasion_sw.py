"""
Bennis et al. (2023), Petrophysics 64(6): 931-953.
Estimation of radial/vertical water-saturation distribution around a borehole in
deeply-invaded tight-gas sandstone, by joint inversion of well logs, core, and
formation-tester data.

This module implements a simplified 2D (r,z) saturation model and a
least-squares inversion that fits an Archie-style apparent-resistivity forward
operator to synthetic logs.
"""
import numpy as np
from scipy.optimize import least_squares


def radial_sw_profile(r, r_invaded, sw_inv, sw_virgin, transition=0.1):
    """Smooth radial Sw profile: invaded zone -> virgin zone (tanh transition)."""
    return sw_inv + 0.5 * (sw_virgin - sw_inv) * (1 + np.tanh((r - r_invaded) / transition))


def archie_rt(sw, phi, rw, a=1.0, m=2.0, n=2.0):
    """Archie's equation for true resistivity."""
    return a * rw / (phi ** m * sw ** n)


def apparent_resistivity(r_grid, sw_profile, phi, rw, depths_of_investigation):
    """Volume-weighted apparent resistivity at several DOIs (toy radial average)."""
    rt = archie_rt(sw_profile, phi, rw)
    apps = []
    for doi in depths_of_investigation:
        w = np.exp(-((r_grid - doi) ** 2) / (2 * (doi / 3) ** 2))
        w /= w.sum()
        apps.append(1.0 / np.sum(w / rt))  # parallel-conductivity average
    return np.array(apps)


def invert_sw(observed_app_res, r_grid, phi, rw, dois, r_inv_guess=0.5):
    """Recover (r_invaded, sw_invaded, sw_virgin) from multi-DOI resistivities."""
    def resid(p):
        r_i, sw_i, sw_v = p
        prof = radial_sw_profile(r_grid, r_i, sw_i, sw_v)
        return apparent_resistivity(r_grid, prof, phi, rw, dois) - observed_app_res
    res = least_squares(resid, x0=[r_inv_guess, 0.8, 0.3],
                        bounds=([0.05, 0.05, 0.05], [3.0, 1.0, 1.0]))
    return dict(r_invaded=res.x[0], sw_invaded=res.x[1], sw_virgin=res.x[2], cost=res.cost)


def test_all():
    rng = np.random.default_rng(0)
    r_grid = np.linspace(0.05, 3.0, 200)
    phi, rw = 0.12, 0.05
    true = dict(r_invaded=0.6, sw_inv=0.85, sw_virgin=0.25)
    prof = radial_sw_profile(r_grid, **true)
    true["sw_invaded"] = true.pop("sw_inv")
    dois = np.array([0.2, 0.5, 1.0, 2.0])
    obs = apparent_resistivity(r_grid, prof, phi, rw, dois)
    obs *= rng.normal(1.0, 0.02, obs.shape)
    rec = invert_sw(obs, r_grid, phi, rw, dois)
    print("Bennis et al. inversion:")
    print(f"  true     : {true}")
    print(f"  recovered: {rec}")
    assert abs(rec["sw_virgin"] - true["sw_virgin"]) < 0.1
    print("  PASS")


if __name__ == "__main__":
    test_all()
