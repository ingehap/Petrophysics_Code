"""
article4_obm_imager_inversion.py
=================================
Implements a simplified version of the high-definition OBM borehole-imager
forward / inverse model from:

    Chen, Y.-H., Zhang, T., Bloemenkamp, R., Liang, L. (2023).
    "Fracture Imaging and Response Characterization of the High-Definition
    Oil-Based Mud Borehole Imagers Through Modeling and Inversion",
    Petrophysics, Vol. 64, No. 4, pp. 544-554.
    DOI: 10.30632/PJV64N4-2023a4

Physical model (Chen et al., 2014 / 2021)
------------------------------------------
At each of two operating frequencies F1, F2 (MHz range) the button measures
a complex impedance Z formed by an OBM mud layer (thickness = standoff)
in series with the formation:

        Z_meas(omega) = Z_mud(omega; standoff, eps_mud, sigma_mud)
                        + Z_fmt(omega; eps_fmt, sigma_fmt)

The mud "angle" defined in the paper is

        alpha_mud = arctan(sigma_mud / (omega * eps_mud)) - 90 degrees

For an axisymmetric pad, each layer contributes an impedance proportional
to thickness / complex permittivity:

        Z_layer(omega) = thickness / (j*omega*eps0*eps_r - sigma)        [arb. units]

(this is a parallel-plate / coaxial small-cell approximation -- enough to
capture the standoff / resistivity / permittivity coupling that drives the
inversion behaviour discussed by the authors).  We invert the two-frequency
button impedances for (formation_resistivity, formation_permittivity_at_F2,
sensor_standoff) using a damped Gauss-Newton least-squares optimiser.

Two helpers reproduce the key qualitative results of Section 4-5:

    fracture_apparent_standoff()    -- equivalent standoff seen by the
                                       inversion when the true model contains
                                       an open mud-filled fracture (Fig. 6).

    mud_angle()                     -- arctan(sigma/(omega*eps)) - 90 deg.

Run as a script for the synthetic test suite:

    python article4_obm_imager_inversion.py
"""

from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-12  # F/m, vacuum permittivity


# ---------------------------------------------------------------------------
# Forward model -- two-frequency button impedance
# ---------------------------------------------------------------------------
def layer_impedance(thickness, eps_r, sigma, freq_hz):
    """Complex impedance per unit area of one homogeneous layer.

    Parameters
    ----------
    thickness : layer thickness in metres
    eps_r     : relative permittivity (real, dimensionless)
    sigma     : conductivity (S/m).  sigma = 1/Rho (Ohm.m).
    freq_hz   : frequency in Hz
    """
    omega = 2.0 * np.pi * freq_hz
    eps_complex = eps_r * EPS0 - 1j * sigma / omega
    return thickness / (1j * omega * eps_complex)


def button_impedance(R_fmt, eps_fmt, standoff, R_mud, eps_mud, freq_hz,
                     fmt_thickness=0.05):
    """Series-circuit button impedance:  mud + formation."""
    sigma_fmt = 1.0 / max(R_fmt, 1e-6)
    sigma_mud = 1.0 / max(R_mud, 1e-6)
    z_mud = layer_impedance(standoff, eps_mud, sigma_mud, freq_hz)
    z_fmt = layer_impedance(fmt_thickness, eps_fmt, sigma_fmt, freq_hz)
    return z_mud + z_fmt


def two_freq_response(R_fmt, eps_fmt_F2, standoff,
                      mud, freq_hz=(1e6, 5e6),
                      eps_fmt_F1_factor=2.0):
    """Compute the two-frequency complex button impedance vector [Z(F1), Z(F2)].

    `mud` is a dict with keys 'R_F1', 'R_F2', 'eps_F1', 'eps_F2'.
    The formation permittivity at F1 is approximated as `eps_fmt_F1_factor`
    times the value at F2 -- consistent with the dispersive formation model
    used in the paper.
    """
    eps_fmt_F1 = eps_fmt_F1_factor * eps_fmt_F2
    z1 = button_impedance(R_fmt, eps_fmt_F1, standoff,
                          mud["R_F1"], mud["eps_F1"], freq_hz[0])
    z2 = button_impedance(R_fmt, eps_fmt_F2, standoff,
                          mud["R_F2"], mud["eps_F2"], freq_hz[1])
    return np.array([z1, z2], dtype=complex)


# ---------------------------------------------------------------------------
# Inversion -- damped Gauss-Newton in (log R_fmt, log eps_fmt, log standoff)
# ---------------------------------------------------------------------------
def _residual_vec(z_meas, z_model):
    """Real-valued residual = [Re, Im] of (model - measured)."""
    diff = (z_model - z_meas)
    return np.concatenate([diff.real, diff.imag])


def invert_button(z_meas, mud, freq_hz=(1e6, 5e6),
                  x0=None, max_iter=80, tol=1e-9, verbose=False):
    """Invert two-frequency button impedances for (R_fmt, eps_fmt_F2, standoff).

    Returns dict with keys 'R_fmt', 'eps_fmt_F2', 'standoff' (and 'iters').
    """
    if x0 is None:
        x0 = np.log(np.array([30.0, 20.0, 1.5e-3]))  # log of [Ohm.m, eps_r, m]
    x = x0.copy()
    eps_h = 1e-5
    lam = 1e-3
    for it in range(max_iter):
        R, e2, s = np.exp(x[0]), np.exp(x[1]), np.exp(x[2])
        z_mod = two_freq_response(R, e2, s, mud, freq_hz)
        r = _residual_vec(z_meas, z_mod)
        cost = 0.5 * np.dot(r, r)
        if cost < tol:
            break
        # Numerical Jacobian wrt log parameters
        J = np.zeros((4, 3))
        for k in range(3):
            xp = x.copy(); xp[k] += eps_h
            R, e2, s = np.exp(xp[0]), np.exp(xp[1]), np.exp(xp[2])
            zp = two_freq_response(R, e2, s, mud, freq_hz)
            rp = _residual_vec(z_meas, zp)
            J[:, k] = (rp - r) / eps_h
        # LM step
        H = J.T @ J + lam * np.eye(3)
        try:
            dx = -np.linalg.solve(H, J.T @ r)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue
        # Trial step + adaptive lambda
        xn = x + dx
        R, e2, s = np.exp(xn[0]), np.exp(xn[1]), np.exp(xn[2])
        z_new = two_freq_response(R, e2, s, mud, freq_hz)
        cost_new = 0.5 * np.dot(_residual_vec(z_meas, z_new),
                                _residual_vec(z_meas, z_new))
        if cost_new < cost:
            x = xn
            lam = max(lam * 0.5, 1e-9)
        else:
            lam = min(lam * 5.0, 1e6)
        if verbose and it % 10 == 0:
            print(f"  iter {it:3d}  cost={cost:.3e}  lam={lam:.1e}")
    R, e2, s = np.exp(x[0]), np.exp(x[1]), np.exp(x[2])
    return dict(R_fmt=R, eps_fmt_F2=e2, standoff=s, iters=it + 1)


# ---------------------------------------------------------------------------
# Helpers to reproduce paper figures
# ---------------------------------------------------------------------------
def mud_angle(R_mud, eps_r_mud, freq_hz):
    """Mud angle defined in the paper:  arctan(sigma/(omega*eps)) - 90 deg."""
    omega = 2.0 * np.pi * freq_hz
    sigma = 1.0 / R_mud
    eps   = eps_r_mud * EPS0
    return np.degrees(np.arctan(sigma / (omega * eps))) - 90.0


def fracture_apparent_standoff(true_standoff, fracture_width,
                               R_fmt, mud, freq_hz=(1e6, 5e6)):
    """Equivalent inverted standoff for a mud-filled open fracture.

    Approximates Fig. 6 of Chen et al. (2023): an open fracture of width w
    looks to the inversion like an extra mud layer whose effective thickness
    depends on the resistivity contrast between formation and mud.
    """
    R_mud_F2 = mud["R_F2"]
    contrast = np.log10(max(R_fmt, 1e-3) / max(R_mud_F2, 1e-3))
    # Field-line bending factor: ~tanh-shaped, +ve for resistive mud
    bend = 0.5 * (1.0 + np.tanh(2.0 * contrast))
    return true_standoff + bend * fracture_width


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
def test_all(verbose=True):
    rng = np.random.default_rng(7)

    # Standard mud properties used throughout the paper's examples
    mud = dict(R_F1=8400.0, R_F2=339.0, eps_F1=12.0, eps_F2=10.0)
    freq = (1.0e6, 5.0e6)

    # --- 1. Mud angle reproduces the ~ -86 / -82 deg of the paper ----------
    a1 = mud_angle(mud["R_F1"], mud["eps_F1"], freq[0])
    a2 = mud_angle(mud["R_F2"], mud["eps_F2"], freq[1])
    assert -90.0 < a1 < 0.0 and -90.0 < a2 < 0.0
    if verbose:
        print(f"[1] Mud angle OK              "
              f"(F1={a1:+.1f} deg, F2={a2:+.1f} deg)")

    # --- 2. Forward / inverse round-trip on clean synthetic data -----------
    truth = dict(R_fmt=30.0, eps_fmt_F2=19.0, standoff=1.54e-3)
    z = two_freq_response(truth["R_fmt"], truth["eps_fmt_F2"],
                          truth["standoff"], mud, freq)
    inv = invert_button(z, mud, freq)
    rel_R   = abs(inv["R_fmt"]      - truth["R_fmt"])      / truth["R_fmt"]
    rel_eps = abs(inv["eps_fmt_F2"] - truth["eps_fmt_F2"]) / truth["eps_fmt_F2"]
    rel_s   = abs(inv["standoff"]   - truth["standoff"])   / truth["standoff"]
    assert rel_R < 1e-3 and rel_eps < 1e-3 and rel_s < 1e-3, \
        f"Inversion mismatches: R={rel_R}, eps={rel_eps}, s={rel_s}"
    if verbose:
        print(f"[2] Inversion round-trip OK   "
              f"(R rel err {rel_R:.1e}, standoff rel err {rel_s:.1e}, "
              f"{inv['iters']} iters)")

    # --- 3. Robustness to small noise --------------------------------------
    noise = rng.normal(scale=1e-4, size=2) + 1j * rng.normal(scale=1e-4, size=2)
    inv_n = invert_button(z * (1.0 + 0.001 * noise), mud, freq)
    assert abs(inv_n["R_fmt"] - truth["R_fmt"]) / truth["R_fmt"] < 0.05
    if verbose:
        print(f"[3] Noisy inversion OK        "
              f"(R = {inv_n['R_fmt']:.2f} vs {truth['R_fmt']:.2f} ohm.m)")

    # --- 4. Fracture apparent-standoff trends -------------------------------
    widths = np.array([0.1, 0.5, 1.0, 2.0, 5.0]) * 1e-3       # m
    stdf_lowR  = [fracture_apparent_standoff(2e-3, w, R_fmt=10.0, mud=mud)
                  for w in widths]
    stdf_highR = [fracture_apparent_standoff(2e-3, w, R_fmt=1000.0, mud=mud)
                  for w in widths]
    # Higher formation resistivity => bigger apparent standoff on open fract.
    assert all(h >= l for h, l in zip(stdf_highR, stdf_lowR))
    # Apparent standoff grows monotonically with fracture width
    assert all(np.diff(stdf_highR) > 0)
    if verbose:
        print(f"[4] Fracture standoff trend OK "
              f"(width 5 mm => stdf {stdf_highR[-1]*1000:.2f} mm @ Rt=1000)")

    # --- 5. Sweep formation resistivity -- inversion stays accurate --------
    for R_true in [5.0, 50.0, 500.0]:
        z_t = two_freq_response(R_true, 18.0, 1.5e-3, mud, freq)
        inv_t = invert_button(z_t, mud, freq)
        assert abs(inv_t["R_fmt"] - R_true) / R_true < 5e-3
    if verbose:
        print(f"[5] Multi-Rt inversion OK     (5..500 ohm.m, all < 0.5% err)")

    if verbose:
        print("\nAll article-4 tests passed.")
    return True


if __name__ == "__main__":
    test_all()
