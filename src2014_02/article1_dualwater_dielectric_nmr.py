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
  - Wet-rock conductivity Co (Sw = 1) and Qv from the Co-Cw method (the
    independent excess-conductivity Qv the paper validates 1:1 against NMR)
  - Inversion for the clay-dependent cementation exponent m0 (Eq. 5)
  - Solving the dual-water equation for water saturation
  - The end-to-end six-step joint-inversion workflow (dielectric + NMR +
    spectroscopy + microresistivity)
  - The Fig. 7 input-sensitivity study (Sxo is the dominant control on Sw)
  - The Appendix n-vs-m trade-off (cementation factor for a target Sw)
  - Lithology classification from the m0 log (clean ~1.8 -> shaly ~2.2)

Note: this issue's PDF dropped the display equations in extraction; the two
conductivity terms are transcribed verbatim (with the worked example) and the
dual-water form is reconstructed in standard Clavier et al. (1984) form.  The
dielectric tool is treated as a black box (no CRIM law appears).  Conductivities
in S/m, Qv in meq/cm^3.
"""

import numpy as np

# Saturation exponent: the paper notes n = 2.0 is a good generic starting value,
# while the case study's six-core average (used in its final Sw computation) is
# n = 1.67 (Tan et al., 2014).
DEFAULT_SATURATION_EXPONENT = 2.0
CASE_STUDY_SATURATION_EXPONENT = 1.67

# Reference connate-water salinity and conductivity of the worked example
# (formation water 15-23 ppt NaCl-equivalent; reference 19 ppt -> Cw = 5.81 S/m
# at 60 degC).
REFERENCE_SALINITY_PPT = 19.0
REFERENCE_CW_SI = 5.81


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


def co_wet_conductivity(phi, m0, cw, qv, alpha, vqh, beta):
    """Wet-rock conductivity Co - the Sw = 1 limit of the dual-water Eq. 1

        Co = phi^m0*[(1 - alpha*vQH*Qv)*Cw + beta*Qv],

    the quantity measured in the Co-Cw experiment the paper uses to derive a
    second, independent Qv (compared 1:1 against the NMR Qv on its core set).
    """
    return dual_water_conductivity(phi, 1.0, m0, n=1.0, cw=cw, qv=qv,
                                   alpha=alpha, vqh=vqh, beta=beta)


def qv_from_co_cw(co, phi, m0, cw, alpha, vqh, beta):
    """Qv from the Co-Cw (excess-conductivity) method.

    Inverting the wet-rock conductivity Co = phi^m0*[(1 - alpha*vQH*Qv)*Cw +
    beta*Qv] for Qv (it is linear in Qv) gives

        Qv = (Co/phi^m0 - Cw)/(beta - alpha*vQH*Cw),

    the clay-cation concentration the paper reports agreeing ~1:1 with the NMR
    Qv (their cross-validation of the two independent paths).
    """
    return (co / phi ** m0 - cw) / (beta - alpha * vqh * cw)


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


def m0_lithology(m0):
    """Classify lithology from the dual-water cementation exponent m0.

    The paper's m0 log spans ~1.8 in clean sand, increasing toward ~2.2 in the
    shalier sand (m0 rises with clay content).  Splitting at the 2.0 midpoint of
    that 1.8-2.2 range, returns "clean" for m0 < 2.0 and "shaly" otherwise.
    """
    return "clean" if m0 < 2.0 else "shaly"


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


# ---------------------------------------------- joint workflow --------------

def dielectric_water_salinity_ok(salinity_ppt, limit_ppt=50.0):
    """Dielectric-tool validity check (Step 1): the dielectric-derived invaded-
    zone water conductivity Cmfe is reliable only when the effective water
    salinity is below ~50 ppt."""
    return bool(salinity_ppt < limit_ppt)


def solve_sw_workflow(rt, phi, qv, cw, n, sxo, cmfe, cxo, alpha, vqh, beta):
    """End-to-end six-step dual-water joint-inversion workflow (Tan et al., 2014).

    Combines the independent measurements to evaluate Sw in the uninvaded zone:

      - Step 1 (dielectric): invaded-zone Sxo and water conductivity Cmfe,
      - Step 2 (NMR): clay-cation concentration Qv,
      - Step 3 (spectroscopy/density): porosity phi,
      - Step 4 (microresistivity): invaded-zone conductivity Cxo,
      - Step 5: invert the invaded-zone dual-water equation for m0 (Eq. 5),
      - Step 6: carry m0 to the uninvaded zone and solve Eq. 1 for Sw, using the
        true formation conductivity Ct = 1/Rt.

    Returns (Sw, m0).
    """
    m0 = invert_m0(cxo, sxo, n, cmfe, qv, alpha, vqh, beta, phi)
    ct = 1.0 / rt
    sw = solve_water_saturation(ct, phi, m0, n, cw, qv, alpha, vqh, beta)
    return sw, m0


def sw_parameter_sensitivity(rt, phi, qv, cw, n, sxo, cmfe, cxo,
                             alpha, vqh, beta, frac=0.05):
    """Sensitivity of the inverted uninvaded-zone Sw to the workflow inputs.

    Re-runs ``solve_sw_workflow`` with each of the invaded-zone water saturation
    Sxo, the clay-cation concentration Qv, the connate-water salinity (via Cw)
    and the saturation exponent n perturbed by a relative ``frac``, and returns a
    dict of the absolute change |dSw| for each.  This reproduces the headline of
    the paper's Fig. 7 sensitivity study - the dielectric invaded-zone saturation
    Sxo is the dominant control on the computed water saturation (the paper's full
    reported ordering is Sxo > Qv > salinity > n, but the relative weighting of
    the lesser terms depends on the chosen perturbation ranges and operating
    point, so only the Sxo dominance is robust to a uniform perturbation).
    """
    base, _ = solve_sw_workflow(rt, phi, qv, cw, n, sxo, cmfe, cxo,
                                alpha, vqh, beta)

    def sw_with(**override):
        args = dict(rt=rt, phi=phi, qv=qv, cw=cw, n=n, sxo=sxo, cmfe=cmfe,
                    cxo=cxo, alpha=alpha, vqh=vqh, beta=beta)
        args.update(override)
        return solve_sw_workflow(**args)[0]

    return {
        "Sxo":      abs(sw_with(sxo=sxo * (1.0 + frac)) - base),
        "Qv":       abs(sw_with(qv=qv * (1.0 + frac)) - base),
        "salinity": abs(sw_with(cw=cw * (1.0 + frac)) - base),
        "n":        abs(sw_with(n=n * (1.0 + frac)) - base),
    }


def archie_cementation_for_sw(sw, n, rt, rw, phi, a=1.0):
    """Cementation factor m that yields a target Sw for an assumed saturation
    exponent n (Appendix n-vs-m trade-off)

        m = log(a*Rw/(Sw^n*Rt))/log(phi),

    from the clean-Archie relation  Sw^n = a*Rw/(phi^m*Rt).  Because n and m are
    competing factors toward a fixed Sw, increasing n decreases the inverted m
    (e.g. for Sw = 0.8, Rt = 20, phi = 0.20, Rw = 0.565: n=1.5->m=2.00,
    n=2.5->m=1.87, n=3.5->m=1.73).
    """
    return np.log10(a * rw / (sw ** n * rt)) / np.log10(phi)


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

    # Co-Cw path: build the wet-rock conductivity from a known Qv, then recover
    # Qv from it (the second, independent Qv the paper checks against NMR)
    co = co_wet_conductivity(0.25, m0=2.0, cw=cw, qv=qv, alpha=alpha, vqh=vqh, beta=beta)
    qv_cocw = qv_from_co_cw(co, 0.25, m0=2.0, cw=cw, alpha=alpha, vqh=vqh, beta=beta)
    print(f"  Co={co:.3f} S/m   Qv(Co-Cw)={qv_cocw:.3f}")
    assert np.isclose(qv_cocw, qv)

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

    # Six-step workflow: build a synthetic invaded zone (Sxo at a known m0),
    # then recover m0 and the uninvaded-zone Sw end-to-end
    assert dielectric_water_salinity_ok(19.0) and not dielectric_water_salinity_ok(60.0)
    sxo, cmfe = 0.8, 6.0
    cxo = dual_water_conductivity(phi, sxo, m0_true, n, cmfe, qv, alpha, vqh, beta)
    rt = 1.0 / ct
    sw_wf, m0_wf = solve_sw_workflow(rt, phi, qv, cw, n, sxo, cmfe, cxo,
                                     alpha, vqh, beta)
    print(f"  workflow: m0={m0_wf:.3f}  Sw={sw_wf:.3f}")
    assert np.isclose(m0_wf, m0_true) and np.isclose(sw_wf, 0.40, atol=1e-3)

    # Fig. 7 sensitivity: the dielectric Sxo is the dominant control on Sw
    sens = sw_parameter_sensitivity(rt, phi, qv, cw, n, sxo, cmfe, cxo,
                                    alpha, vqh, beta)
    print("  Sw sensitivity:", {k: round(v, 4) for k, v in sens.items()})
    assert max(sens, key=sens.get) == "Sxo" and all(v >= 0 for v in sens.values())

    # Appendix n-vs-m trade-off reproduces the paper's tabulated pairs
    m_15 = archie_cementation_for_sw(0.8, n=1.5, rt=20.0, rw=0.565, phi=0.20)
    m_25 = archie_cementation_for_sw(0.8, n=2.5, rt=20.0, rw=0.565, phi=0.20)
    m_35 = archie_cementation_for_sw(0.8, n=3.5, rt=20.0, rw=0.565, phi=0.20)
    print(f"  n-m trade-off @ Sw=0.8: n=1.5->m={m_15:.2f}  n=2.5->m={m_25:.2f}  n=3.5->m={m_35:.2f}")
    assert np.isclose(m_15, 2.00, atol=0.02) and np.isclose(m_25, 1.87, atol=0.02)
    assert np.isclose(m_35, 1.73, atol=0.02) and m_15 > m_25 > m_35

    # m0 log lithology split (clean sand ~1.8 -> shalier sand ~2.2)
    assert m0_lithology(1.8) == "clean" and m0_lithology(2.2) == "shaly"
    assert CASE_STUDY_SATURATION_EXPONENT == 1.67
    print("  PASS")
    return {"excess_clay": float(clay), "Qv": float(qv_calc), "m0": float(m0_rec),
            "Sw": float(sw)}


if __name__ == "__main__":
    test_all()
