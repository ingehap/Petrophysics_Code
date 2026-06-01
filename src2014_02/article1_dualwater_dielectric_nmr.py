"""
Article 1: Solving Complex Dual-Water Equation using Dielectric-NMR-Spectroscopy
           and Conventional Logs
Willy Tan, Ryan Lafferty, Thomas J. Neville (2014)
Reference: Petrophysics Vol. 55, No. 1 (February 2014), pp. 14-23
DOI: none assigned (this issue predates SPWLA DOI assignment)

The dual-water saturation equation has six unknowns; this paper closes them by
combining measurements: a dielectric tool gives the invaded-zone saturation and
water conductivity, NMR clay-bound water gives Qv (hence CEC), and nuclear
spectroscopy gives mineralogy/porosity.  The invaded-zone equation is then
inverted for the clay-dependent cementation exponent m0, which is carried to the
uninvaded zone to solve for Sw.

Implements:

  - Dual-water conductivity (Clavier et al., 1984; Eq. 1) with its excess-clay
    (beta*Qv) and effective-connate-water ((1-alpha*vQH*Qv)*Cw) terms
  - Clay-bound-water saturation  Swb = alpha*vQH*Qv  (Eq. 3)
  - Qv from CEC  Qv = CEC*rho_grain*(1-phi)/phi  (Eq. 4)
  - Inversion for the clay-dependent cementation exponent m0 (Eq. 5)
  - Solving the dual-water equation for water saturation

Note: this issue's PDF dropped the display equations in extraction; the two
conductivity terms are transcribed verbatim (with the worked example) and the
dual-water form is reconstructed in standard Clavier et al. (1984) form.  The
dielectric tool is treated as a black box (no CRIM law appears).  Conductivities
in S/m, Qv in meq/cm^3.
"""

import numpy as np


# ---------------------------------------------- dual-water terms --------------

def excess_clay_conductivity(beta, qv):
    """Excess clay (counterion) conductivity term  beta*Qv (S/m).

    Worked example: beta = 4.604 (S/m)/(meq/cm^3), Qv = 1.0 -> 4.6 S/m.
    """
    return beta * qv


def effective_water_conductivity(cw, alpha, vqh, qv):
    """Effective connate-water conductivity term  (1 - alpha*vQH*Qv)*Cw (S/m).

    Worked example: Cw = 5.81 S/m, alpha = 1.0, vQH = 0.319 cm^3/meq, Qv = 1.0
    -> 3.96 S/m.
    """
    return (1.0 - alpha * vqh * qv) * cw


def dual_water_conductivity(phi, sw, m0, n, cw, qv, alpha, vqh, beta):
    """Dual-water total conductivity (Eq. 1)

        Ct = phi^m0*[Sw^n*(1-alpha*vQH*Qv)*Cw + Sw^(n-1)*beta*Qv],

    the effective-connate-water term plus the excess-clay term, reducing to
    Co = phi^m0*[(1-alpha*vQH*Qv)*Cw + beta*Qv] at Sw = 1.
    """
    sw = np.asarray(sw, float)
    cwf = effective_water_conductivity(cw, alpha, vqh, qv)
    return phi ** m0 * (sw ** n * cwf + sw ** (n - 1) * beta * qv)


# ---------------------------------------------- clay-bound water --------------

def clay_bound_water_saturation(alpha, vqh, qv):
    """Clay-bound-water saturation from Qv (Eq. 3)  Swb = alpha*vQH*Qv."""
    return alpha * vqh * qv


def qv_from_clay_bound_water(swb, alpha, vqh):
    """Qv from the clay-bound-water saturation (inverse of Eq. 3)

        Qv = Swb/(alpha*vQH),

    the NMR path: clay-bound water (T2 below the cutoff) gives Qv directly.
    """
    return swb / (alpha * vqh)


def clay_bound_water_porosity_nmr(t2, amplitudes, t2_cutoff=3.0):
    """Clay-bound-water (CBW) porosity from an NMR T2 distribution

        CBW = sum(A_i : T2_i < cutoff),

    summing the porosity-calibrated amplitudes below the clay-bound-water cutoff
    (Martin & Dacy, 2004; the paper uses a practical T2 < 3 msec cutoff).
    """
    t2 = np.asarray(t2, float)
    a = np.asarray(amplitudes, float)
    return float(a[t2 < t2_cutoff].sum())


def qv_from_cec(cec, rho_grain, phi):
    """Cation exchange capacity per pore volume from CEC (Eq. 4)

        Qv = CEC*rho_grain*(1 - phi)/phi,

    with CEC in meq/g, grain density in g/cm^3 and Qv in meq/cm^3.
    """
    return cec * rho_grain * (1.0 - phi) / phi


def cec_from_qv(qv, rho_grain, phi):
    """CEC log from the Qv log (inverse of Eq. 4)

        CEC = Qv*phi/(rho_grain*(1 - phi)),

    the cation-exchange-capacity curve the paper derives from Qv.
    """
    return qv * phi / (rho_grain * (1.0 - phi))


# ---------------------------------------------- inversion --------------

def invert_m0(cxo, sxo, n, cmfe, qv, alpha, vqh, beta, phi):
    """Invert the invaded-zone dual-water equation for the cementation exponent
    m0 (Eq. 5)

        m0 = ln(Cxo/[Sxo^n*Cwf + Sxo^(n-1)*beta*Qv])/ln(phi),

    with the effective mud-filtrate conductivity Cmfe in place of Cw.
    """
    cwf = effective_water_conductivity(cmfe, alpha, vqh, qv)
    bracket = sxo ** n * cwf + sxo ** (n - 1) * beta * qv
    return np.log(cxo / bracket) / np.log(phi)


def solve_water_saturation(ct, phi, m0, n, cw, qv, alpha, vqh, beta,
                           n_iter=100, tol=1e-10):
    """Solve the dual-water equation for Sw by bisection on [1e-4, 1]."""
    lo, hi = 1e-4, 1.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        c = dual_water_conductivity(phi, mid, m0, n, cw, qv, alpha, vqh, beta)
        if c < ct:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Dual-Water Dielectric-NMR-Spectroscopy")
    print("=" * 60)

    # Worked example: the two conductivity terms reproduce the paper's numbers
    cw, alpha, vqh, beta, qv = 5.81, 1.0, 0.319, 4.604, 1.0
    clay = excess_clay_conductivity(beta, qv)
    water = effective_water_conductivity(cw, alpha, vqh, qv)
    print(f"  excess-clay term = {clay:.2f} S/m   effective-water term = {water:.2f} S/m")
    assert np.isclose(clay, 4.6, atol=0.05) and np.isclose(water, 3.96, atol=0.05)

    # Qv from CEC, and round-trip to clay-bound-water saturation
    qv_calc = qv_from_cec(cec=0.04, rho_grain=2.66, phi=0.25)
    swb = clay_bound_water_saturation(alpha, vqh, qv_calc)
    print(f"  Qv={qv_calc:.3f} meq/cm3   Swb={swb:.3f}")
    assert qv_calc > 0 and 0 < swb < 1
    # Qv <-> CEC and Qv <-> Swb invert cleanly (the NMR and spectroscopy paths)
    assert np.isclose(cec_from_qv(qv_calc, 2.66, 0.25), 0.04)
    assert np.isclose(qv_from_clay_bound_water(swb, alpha, vqh), qv_calc)

    # NMR clay-bound water: only T2 below the 3-msec cutoff counts toward CBW
    t2 = np.array([1.0, 2.5, 5.0, 50.0, 500.0])
    amps = np.array([0.02, 0.03, 0.05, 0.08, 0.07])
    cbw = clay_bound_water_porosity_nmr(t2, amps, t2_cutoff=3.0)
    print(f"  NMR CBW porosity = {cbw:.3f}")
    assert np.isclose(cbw, 0.05)  # only the 1.0 and 2.5 msec bins

    # m0 inversion round-trips: build Cxo from a known m0, then recover it
    phi, sxo, n, m0_true = 0.25, 0.7, 1.67, 2.05
    cxo = dual_water_conductivity(phi, sxo, m0_true, n, cw, qv, alpha, vqh, beta)
    m0_rec = invert_m0(cxo, sxo, n, cw, qv, alpha, vqh, beta, phi)
    print(f"  recovered m0 = {m0_rec:.3f}  (true {m0_true})")
    assert np.isclose(m0_rec, m0_true)

    # Solve Sw from Ct and check it round-trips through the forward model
    ct = dual_water_conductivity(phi, 0.40, m0_true, n, cw, qv, alpha, vqh, beta)
    sw = solve_water_saturation(ct, phi, m0_true, n, cw, qv, alpha, vqh, beta)
    print(f"  recovered Sw = {sw:.3f}")
    assert np.isclose(sw, 0.40, atol=1e-3)
    print("  PASS")
    return {"excess_clay": float(clay), "Qv": float(qv_calc), "m0": float(m0_rec),
            "Sw": float(sw)}


if __name__ == "__main__":
    test_all()
