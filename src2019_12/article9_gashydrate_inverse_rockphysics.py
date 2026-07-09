"""
Article 9: Joint Interpretation of Elastic and Electrical Data for Petrophysical
           Properties of Gas-Hydrate-Bearing Sediments Using Inverse Rock Physics
           Modeling Method
Pan, Li, Zhang, Chen, Cai, Geng (2019)
DOI: 10.30632/PJV60N6-2019a9

Gas-hydrate saturation is estimated by jointly inverting elastic (P-wave
velocity) and electrical (resistivity) data: a rock-physics model relates hydrate
saturation to velocity (stiffening) and an Archie-type model relates it to
resistivity (hydrate is an electrical insulator).  Combining both measurements
reduces the non-uniqueness of either one alone.

Implements:

  - Velocity vs hydrate saturation (load-bearing stiffening model)
  - Archie resistivity vs hydrate saturation (Rt = a*Rw/(phi^m*(1-Sh)^n))
  - Joint inverse rock-physics modeling (least-squares over Sh)

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard joint elastic-electrical inversion the paper's
title describes.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- forward models ----------

def velocity_model(sh, vp_water=1.8, vp_max=3.2, exponent=1.0):
    """P-wave velocity vs hydrate saturation (monotonic stiffening, km/s).

        Vp(Sh) = Vp_water + (Vp_max - Vp_water)*Sh^exponent
    """
    return vp_water + (vp_max - vp_water) * np.asarray(sh, float) ** exponent


def archie_resistivity(sh, Rw=0.3, phi=0.4, a=1.0, m=2.0, n=2.0):
    """Archie resistivity vs hydrate saturation  Rt = a*Rw/(phi^m*(1-Sh)^n).

    Water saturation = 1 - Sh (hydrate is an insulator displacing brine).
    """
    sw = 1.0 - np.asarray(sh, float)
    return petrolib.saturation_resistivity.archie_rt(sw, Rw, phi=phi, a=a, m=m, n=n)


# ---------------------------------------------- joint inversion ---------

def invert_hydrate_saturation(vp_obs, rt_obs, vp_kw=None, archie_kw=None,
                              grid=None):
    """Joint inverse rock-physics modeling for hydrate saturation.

    Minimizes the normalized misfit of both the velocity and the resistivity
    forward models over a grid of Sh; returns the best Sh.
    """
    vp_kw = vp_kw or {}
    archie_kw = archie_kw or {}
    if grid is None:
        grid = np.linspace(0.0, 0.8, 401)
    vp_pred = velocity_model(grid, **vp_kw)
    rt_pred = archie_resistivity(grid, **archie_kw)
    misfit = ((vp_pred - vp_obs) / vp_obs) ** 2 + ((rt_pred - rt_obs) / rt_obs) ** 2
    return float(grid[np.argmin(misfit)])


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Gas-Hydrate Joint Elastic-Electrical Inversion")
    print("=" * 60)

    # Forward models are monotonic: more hydrate -> faster, more resistive
    assert velocity_model(0.4) > velocity_model(0.1)
    assert archie_resistivity(0.6) > archie_resistivity(0.2)

    # Joint inversion recovers a planted hydrate saturation from both data
    sh_true = 0.45
    vp_obs = velocity_model(sh_true)
    rt_obs = archie_resistivity(sh_true)
    sh_hat = invert_hydrate_saturation(vp_obs, rt_obs)
    print(f"  recovered Sh           = {sh_hat:.3f}  (true {sh_true})")
    assert abs(sh_hat - sh_true) < 0.01

    # Joint inversion is more robust than the noisier single measurement
    # (resistivity-only) because it also uses the precise velocity datum
    rng = np.random.default_rng(9)
    grid = np.linspace(0.0, 0.8, 401)
    rt_pred = archie_resistivity(grid)
    joint_err, rt_only_err = [], []
    for _ in range(50):
        vp_n = vp_obs * (1 + rng.normal(0, 0.03))
        rt_n = rt_obs * (1 + rng.normal(0, 0.10))
        joint_err.append(abs(invert_hydrate_saturation(vp_n, rt_n) - sh_true))
        rt_only_err.append(abs(grid[np.argmin(((rt_pred - rt_n) / rt_n) ** 2)] - sh_true))
    print(f"  mean |dSh| joint / rt-only = {np.mean(joint_err):.4f} / {np.mean(rt_only_err):.4f}")
    assert np.mean(joint_err) < np.mean(rt_only_err)
    print("  PASS")
    return {"sh_hat": sh_hat, "joint_err": float(np.mean(joint_err))}


if __name__ == "__main__":
    test_all()
