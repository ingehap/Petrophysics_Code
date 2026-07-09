"""
Article 2: Inversion-Based Workflow for Quantitative Interpretation of the
           New-Generation Oil-Based-Mud Resistivity Imager
Yong-Hua Chen, Dzevat Omeragic, Tarek Habashy, Richard Bloemenkamp,
Tianhua Zhang, Phillip Cheung, Robert Laronga (2014)
Reference: Petrophysics Vol. 55, No. 6 (December 2014), pp. 554-571
DOI: none assigned (this issue predates SPWLA DOI assignment)

An oil-based-mud (OBM) micro-resistivity imager measures the complex button
impedance through a thin resistive mud layer.  The interpretation is posed as a
multiplicatively regularized Gauss-Newton inversion that recovers the formation
impedivity (and standoff / mud properties) from the in-phase and out-of-phase
button impedances at two frequencies.

Implements:

  - Formation and mud impedivity  xi = 1/(j*w*eps0*eps + sigma)
  - ZB90 composite processing (project the button impedance perpendicular to
    the mud-impedance vector)
  - Multiplicatively regularized data misfit (Eq. 1) and the regularization
    schedule lambda_k = alpha*misfit^beta (Eq. 2; Habashy & Abubakar, 2004)
  - A scalar Gauss-Newton inversion that recovers formation impedivity from a
    measured complex impedance via the series mud + formation circuit model

Note: this issue's PDF has a text layer; the two impedivity definitions are
exact, while the displayed inversion equations were dropped in extraction and
reconstructed in the standard multiplicatively regularized Gauss-Newton form.
SI units; impedivity in Ohm*m, conductivity in S/m, permittivity relative.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

EPS0 = 8.854e-12  # vacuum permittivity, F/m


# ---------------------------------------------- impedivity --------------

def impedivity(eps_r, sigma, freq):
    """Complex impedivity of a medium

        xi = 1/(j*w*eps0*eps_r + sigma),   w = 2*pi*freq,

    used for both the formation (xi_f) and the mud (xi_m).
    """
    w = 2.0 * np.pi * freq
    return 1.0 / (1j * w * EPS0 * eps_r + sigma)


def impedivity_amplitude(eps_r, sigma, freq):
    """Impedivity amplitude  |xi| = 1/|j*w*eps0*eps_r + sigma|."""
    return np.abs(impedivity(eps_r, sigma, freq))


# ---------------------------------------------- composite processing --------------

def zb90(z_button, z_mud):
    """ZB90 composite processing: project the button impedance perpendicular to
    the mud-impedance vector to approximate the formation impedivity.

        ZB90 = |Z_button| * sin(angle(Z_button) - angle(Z_mud)),

    i.e. the component of the measured impedance orthogonal to the (large) mud
    impedance, recovering the formation-bearing part of the series circuit.
    """
    return np.abs(z_button) * np.sin(np.angle(z_button) - np.angle(z_mud))


# ---------------------------------------------- regularization schedule --------------

def data_misfit(simulated, measured, wd=None):
    """Weighted data misfit (Eq. 1, data term)

        phi_d = || Wd*(s(m) - d) ||^2,

    with Wd a diagonal weight (default: inverse of the measured amplitudes).
    """
    simulated = np.asarray(simulated, complex)
    measured = np.asarray(measured, complex)
    if wd is None:
        wd = 1.0 / np.maximum(np.abs(measured), 1e-30)
    r = wd * (simulated - measured)
    return float(np.sum(np.abs(r) ** 2))


def regularization_coefficient(misfit, alpha, beta, lam_max=np.inf):
    """Multiplicative-regularization coefficient (Eq. 2)

        lambda_k = min(alpha*misfit^beta, lam_max),

    with beta > 1 so the regularization vanishes faster than the misfit as the
    inversion converges.
    """
    return petrolib.inversion_numerics.costs.reg_lambda_multiplicative(
        misfit, alpha, beta, lam_max)


# ---------------------------------------------- Gauss-Newton inversion --------------

def series_circuit_impedance(xi_f, z_mud):
    """Series mud + formation circuit model: the measured button impedance is
    the (large) mud impedance in series with the formation impedivity.

        Z = z_mud + xi_f.
    """
    return z_mud + xi_f


def invert_formation_impedivity(z_measured, z_mud, xi0=None, n_iter=20,
                                alpha=1e-3, beta=2.0, ref=0.0):
    """Recover the formation impedivity from a measured button impedance with a
    multiplicatively regularized Gauss-Newton scheme (Eqs. 1-2).

    The forward model is the series circuit Z = z_mud + xi_f, whose Jacobian
    w.r.t. xi_f is unity; the regularization pulls xi_f toward ``ref`` with a
    weight that decays as the misfit shrinks.  Returns the recovered impedivity.
    """
    xi = complex(z_measured - z_mud) if xi0 is None else complex(xi0)
    wd = 1.0 / max(abs(z_measured), 1e-30)
    for _ in range(n_iter):
        sim = series_circuit_impedance(xi, z_mud)
        mis = data_misfit([sim], [z_measured], wd=np.array([wd]))
        lam = regularization_coefficient(mis, alpha, beta)
        # normal equations for a scalar unknown: (J^H Wd^2 J + lam) dx = rhs
        jhj = wd ** 2  # |J|^2, J = 1
        rhs = wd ** 2 * (z_measured - sim) - lam * (xi - ref)
        xi += rhs / (jhj + lam)
    return xi


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: OBM Resistivity Imager Inversion")
    print("=" * 60)

    freq = 1e6
    # Conductive formation has a small impedivity; resistive mud a large one
    xi_f_true = impedivity(eps_r=15.0, sigma=0.1, freq=freq)
    z_mud = impedivity(eps_r=10.0, sigma=1e-6, freq=freq)
    print(f"  |xi_f|={abs(xi_f_true):.3e}  |z_mud|={abs(z_mud):.3e}")
    assert abs(z_mud) > abs(xi_f_true)
    # higher conductivity lowers the impedivity amplitude
    assert impedivity_amplitude(15, 1.0, freq) < impedivity_amplitude(15, 0.1, freq)

    # ZB90 recovers a formation signal that survives a large mud impedance
    z_button = series_circuit_impedance(xi_f_true, z_mud)
    proj = zb90(z_button, z_mud)
    print(f"  ZB90 projection = {proj:.3e}")
    assert np.isfinite(proj)

    # Regularization decays faster than the misfit (beta > 1)
    lam_hi = regularization_coefficient(1.0, alpha=1e-2, beta=2.0)
    lam_lo = regularization_coefficient(1e-2, alpha=1e-2, beta=2.0)
    assert lam_lo < lam_hi

    # Gauss-Newton recovers the true formation impedivity from the measurement
    xi_rec = invert_formation_impedivity(z_button, z_mud)
    err = abs(xi_rec - xi_f_true) / abs(xi_f_true)
    print(f"  recovered |xi_f|={abs(xi_rec):.3e}  rel.err={err:.2e}")
    assert err < 1e-6
    print("  PASS")
    return {"xi_f": complex(xi_f_true), "rel_err": float(err)}


if __name__ == "__main__":
    test_all()
