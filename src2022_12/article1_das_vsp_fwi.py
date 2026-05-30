"""
Article 1: Full-Waveform Inversion of Fiber-Optic VSP Data From Deviated Wells
Podgornova, Bettinelli, Liang, Le Calvez, Leaney, Perez, Soliman (2022)
DOI: 10.30632/PJV63N6-2022a1

The paper formulates time-domain elastic FWI directly on the DAS strain
observable.  A faithful 2-D / 3-D elastic FWI is far outside the scope of
a self-contained demo module, but the *measurement model* and the
inversion problem can be reproduced exactly in 1-D using a reflectivity-
based forward operator.  This module implements:

  - DAS strain observable: d_DAS = S * eps * tau            (Eqs. 4-5)
    along the fiber tangent vector tau (here axis-aligned).
  - Reflectivity-based seismic forward operator: tr(t) = w(t) * R(t),
    where R(t) is the time-domain reflectivity from a layered impedance
    series.  Strain is the time derivative of the seismogram averaged
    over a gauge-length boxcar (the discretisation S of Eq. 5).
  - Least-squares misfit J = 0.5 * || d_pred - d_obs ||^2     (Eq. 6)
    minimised via Gauss-Newton on per-interface impedance contrasts.
  - Closed-form moment tensors M_vert, M_hor,x, M_45,xz for vertical,
    horizontal and 45-deg deviated wells from tau = (tau_x, tau_y, tau_z)
                                                            (Eqs. 10-11)
"""

import numpy as np


# ---------------------------------------------- moment tensors (Eqs 10-11) -

def das_moment_tensor(tau):
    """Symmetric outer product tau tau^T (Eq. 10-11 of the paper).

    Returns the 3 x 3 moment tensor for a DAS fiber whose local tangent is
    `tau` (3-vector, need not be unit).  The DAS adjoint source for FWI
    becomes a moment-tensor source rather than a body force - this is the
    formula the paper derives for arbitrary well deviation.
    """
    t = np.asarray(tau, float)
    t = t / np.linalg.norm(t)
    return np.outer(t, t)


def well_deviation_examples():
    """Return moment tensors for the three reference cases of Sect. 3."""
    return dict(
        vertical=das_moment_tensor([0, 0, 1]),
        horizontal_x=das_moment_tensor([1, 0, 0]),
        deviated_45_xz=das_moment_tensor([1, 0, 1]),
    )


# ---------------------------------------------- DAS forward model --------

def impedance_to_reflectivity(Z):
    """Acoustic reflectivity series R_i = (Z_{i+1} - Z_i) / (Z_{i+1} + Z_i)."""
    return (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])


def ricker(t, f0, t0):
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def das_forward(Z, t_axis, wavelet, two_way_times, gauge_pts=3):
    """Synthesise the DAS strain trace at a single receiver.

      tr(t) = sum_i R_i * w(t - t_i)
      eps(t) = d tr / dt   (continuous), averaged over a gauge boxcar.

    `two_way_times` are the arrival times of the reflections from each
    interface; the reflectivities are derived from the impedance series Z.
    """
    R = impedance_to_reflectivity(Z)
    tr = np.zeros_like(t_axis)
    dt = t_axis[1] - t_axis[0]
    for r, t_i in zip(R, two_way_times):
        idx = int(round((t_i - t_axis[0]) / dt))
        if 0 <= idx < len(t_axis):
            tr += r * np.roll(wavelet, idx)
    # Differentiate then boxcar-average (the discretisation of S * d/dt)
    eps = np.gradient(tr, dt)
    kernel = np.ones(gauge_pts) / gauge_pts
    return np.convolve(eps, kernel, mode="same")


# ---------------------------------------------- inversion (Eq 6 + GN) ---

def invert_impedance_contrasts(Z_init, t_axis, wavelet, two_way_times,
                               data_obs, n_iter=12, gauge_pts=3, reg=1e-6):
    """Gauss-Newton over the impedance contrasts.

    Parameterisation: x = log Z (one number per layer), so the contrast at
    each interface is differentiable.  At each iter we build the linearised
    Jacobian J = d eps / d x by finite differences on each layer and
    update x_new = x - (J^T J + reg I)^-1 J^T (d_pred - d_obs).
    """
    x = np.log(np.array(Z_init, float))
    misfits = []
    for it in range(n_iter):
        Z = np.exp(x)
        d_pred = das_forward(Z, t_axis, wavelet, two_way_times, gauge_pts)
        r = d_pred - data_obs
        misfits.append(float(0.5 * np.sum(r ** 2)))
        # Jacobian via central differences (small problem -> tractable)
        n_p = len(x)
        J = np.zeros((len(r), n_p))
        delta = 1e-3
        for j in range(n_p):
            x_p = x.copy(); x_p[j] += delta
            x_m = x.copy(); x_m[j] -= delta
            d_p = das_forward(np.exp(x_p), t_axis, wavelet,
                              two_way_times, gauge_pts)
            d_m = das_forward(np.exp(x_m), t_axis, wavelet,
                              two_way_times, gauge_pts)
            J[:, j] = (d_p - d_m) / (2 * delta)
        H = J.T @ J + reg * np.eye(n_p)
        g = J.T @ r
        x = x - np.linalg.solve(H, g)
    return np.exp(x), misfits


# ---------------------------------------------- tests --------------------

def test_all():
    print("=" * 60)
    print("Article 1: DAS-VSP FWI (1-D reflectivity-domain demo)")
    print("=" * 60)

    # Moment-tensor check (Eqs 10-11)
    cases = well_deviation_examples()
    for name, M in cases.items():
        print(f"  M_{name:14s} trace = {np.trace(M):.2f}   "
              f"|M|_F = {np.linalg.norm(M):.2f}")
    # Trace(tau tau^T) = ||tau||^2 = 1 for unit tau
    assert all(np.isclose(np.trace(M), 1.0) for M in cases.values()), \
        "Trace of unit moment tensor must equal 1"

    # 1-D layered impedance model
    Z_true = np.array([5.0e6, 7.5e6, 6.2e6, 9.0e6, 8.4e6])   # kg / (m^2 s)
    two_way_times = np.array([0.10, 0.18, 0.27, 0.35])       # s
    n_t = 800
    dt = 1e-3
    t = np.arange(n_t) * dt
    w = ricker(t, f0=30.0, t0=0.05)
    data_obs = das_forward(Z_true, t, w, two_way_times)

    # Initial guess - all layers same impedance (no contrast)
    Z0 = np.full_like(Z_true, 7.0e6)
    Z_est, mis = invert_impedance_contrasts(Z0, t, w, two_way_times,
                                            data_obs, n_iter=12)
    rel_err = np.abs(Z_est - Z_true) / Z_true
    print(f"  Initial misfit  J = {mis[0]:.3e}")
    print(f"  Final  misfit   J = {mis[-1]:.3e}")
    print(f"  Per-layer |Z_est - Z_true| / Z_true = "
          f"{', '.join(f'{e:.3f}' for e in rel_err)}")

    assert mis[-1] < 1e-3 * mis[0], "Gauss-Newton must crush the misfit"
    assert rel_err.max() < 0.02, "Per-layer impedance recovered within 2 %"
    print("  PASS")
    return {"misfit_0": mis[0], "misfit_final": mis[-1],
            "max_rel_err": float(rel_err.max())}


if __name__ == "__main__":
    test_all()
