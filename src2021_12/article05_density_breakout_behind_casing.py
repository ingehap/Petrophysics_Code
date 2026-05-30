"""
Article 5: Identification of Breakout Behind Casing: Methodology to Obtain
           Openhole-Equivalent Caliper Measurements Through Slotted Liner
           Using the Density Tool
Mosse, Pell, Neville (2021)
DOI: 10.30632/PJV62N6-2021a5

A three-detector gamma-gamma (litho-density) tool is run inside an uncemented
slotted liner to recover an openhole-equivalent caliper.  Shallow windows
characterize completion hardware and annulus-fluid density; deeper windows,
corrected for casing and a variable annulus density, are differenced against a
pre-existing openhole density to infer annulus thickness (borehole breakout /
fines accumulation).

Implements:

  - Radial response  J = (rho_app - rho_fm)/(rho_ann - rho_fm)   (Eq. 1)
  - tanh radial-response model  J(h) = tanh(lambda * h)          (Eq. 2)
  - Casing-corrected density rho_cc                              (Eq. 3)
  - Casing + nominal-cement-corrected density rho_CH             (Eq. 4)
  - Annulus-thickness change from rho_CH (exact + Taylor, Eq. 5)
  - Completion / fluid classification by annulus density

Note: the journal's Eqs. 1-5 and the lambda table are image-rendered and were
not in the text; the forms here are faithful standard reconstructions
constrained by the paper's prose, boundary conditions, and the quoted
geometrical terms (C_SS3 = 0.52, C_LS3 = 1.78).  Densities in g/cc, lengths in
inches.
"""

import numpy as np

# Geometry defaults (8.75-in bit, 7-in liner)
H0_NOMINAL = 0.875          # nominal annulus thickness (in)
CSTK = 0.317               # liner thickness (in)
C_SS3 = 0.52               # short-spacing geometrical term (Eq. 5, quoted)
C_LS3 = 1.78               # long-spacing geometrical term (Eq. 5, quoted)

# Annulus-fluid densities (g/cc)
RHO_GAS = 0.20
RHO_WATER = 1.00
RHO_SOLIDS = 1.80
RHO_CEMENT = 1.90


# ---------------------------------------------- Eq. 1: radial J ---------

def radial_response(rho_apparent, rho_fm, rho_ann):
    """J = (rho_app - rho_fm)/(rho_ann - rho_fm)  (Eq. 1).  J in [0, 1]."""
    return (rho_apparent - rho_fm) / (rho_ann - rho_fm)


# ---------------------------------------------- Eq. 2: tanh model -------

def J_tanh(h_ann, lam):
    """Modeled radial response  J(h) = tanh(lambda * h)  (Eq. 2)."""
    return np.tanh(lam * np.asarray(h_ann, float))


# ---------------------------------------------- Eqs. 3-4: densities -----

def casing_corrected_density(rho_fm, rho_ann, h_ann, lam):
    """Casing-corrected density  rho_cc = rho_fm + J(h)*(rho_ann - rho_fm) (Eq. 3)."""
    return rho_fm + J_tanh(h_ann, lam) * (rho_ann - rho_fm)


def ch_corrected_density(rho_fm, rho_ann, h_ann, lam, h0=H0_NOMINAL):
    """Casing + nominal-cement-corrected density rho_CH (Eq. 4).

    Normalized so rho_CH = rho_fm at the nominal thickness h0 and
    rho_CH -> rho_ann as h -> infinity:
        rho_CH = rho_fm + (rho_ann - rho_fm) * (J(h) - J0)/(1 - J0).
    """
    J0 = J_tanh(h0, lam)
    Jh = J_tanh(h_ann, lam)
    return rho_fm + (rho_ann - rho_fm) * (Jh - J0) / (1.0 - J0)


# ---------------------------------------------- Eq. 5: thickness --------

def annulus_thickness_exact(rho_CH, rho_fm, rho_ann, lam, h0=H0_NOMINAL):
    """Invert Eq. 4 exactly for annulus thickness h_ann (in)."""
    J0 = J_tanh(h0, lam)
    y = (rho_CH - rho_fm) / (rho_ann - rho_fm)
    Jh = np.clip(J0 + y * (1.0 - J0), -0.999999, 0.999999)
    return np.arctanh(Jh) / lam


def annulus_thickness_change_taylor(rho_CH, rho_fm, rho_ann, C):
    """First-order Taylor estimate of annulus-thickness change (Eq. 5).

    delta_h = C * (rho_CH - rho_fm)/(rho_ann - rho_fm),  C = C_SS3 or C_LS3.
    """
    return C * (rho_CH - rho_fm) / (rho_ann - rho_fm)


def coefficient_C(lam, h0=H0_NOMINAL):
    """Geometrical sensitivity C = (1 - J0)/(lambda * sech^2(lambda*h0))."""
    J0 = np.tanh(lam * h0)
    sech2 = 1.0 - J0 ** 2
    return (1.0 - J0) / (lam * sech2)


# ---------------------------------------------- completion class --------

def classify_annulus(rho_ann):
    """Discrete completion/fluid class from annulus density (g/cc)."""
    if rho_ann < 0.6:
        return "gas"
    if rho_ann < 0.95:
        return "slots"
    if rho_ann < 1.4:
        return "water"
    return "solids"


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Density Breakout Behind Casing")
    print("=" * 60)

    rho_fm = 2.55              # openhole formation density
    rho_ann = RHO_WATER

    # Pick lambda so the long-spacing geometrical term reproduces C_LS3 = 1.78
    lam = 0.30
    for _ in range(80):        # simple bisection-free Newton-ish tune
        C = coefficient_C(lam)
        lam *= (C / C_LS3) ** 0.5
    print(f"  tuned lambda           = {lam:.3f} 1/in  (C = {coefficient_C(lam):.2f})")
    assert abs(coefficient_C(lam) - C_LS3) < 0.05

    # Radial response is bounded in [0, 1]
    rho_app = casing_corrected_density(rho_fm, rho_ann, H0_NOMINAL, lam)
    J = radial_response(rho_app, rho_fm, rho_ann)
    assert 0.0 <= J <= 1.0

    # Nominal water annulus -> rho_CH == rho_fm -> recovered thickness == h0
    rho_CH_nom = ch_corrected_density(rho_fm, rho_ann, H0_NOMINAL, lam)
    h_nom = annulus_thickness_exact(rho_CH_nom, rho_fm, rho_ann, lam)
    print(f"  rho_CH at nominal      = {rho_CH_nom:.4f} (expect {rho_fm})")
    print(f"  recovered h (nominal)  = {h_nom:.3f} in (expect {H0_NOMINAL})")
    assert abs(rho_CH_nom - rho_fm) < 1e-9
    assert abs(h_nom - H0_NOMINAL) < 1e-6

    # Breakout: true annulus 1.6 in -> positive thickness change
    h_true = 1.6
    rho_CH_bo = ch_corrected_density(rho_fm, rho_ann, h_true, lam)
    h_exact = annulus_thickness_exact(rho_CH_bo, rho_fm, rho_ann, lam)
    dh_taylor = annulus_thickness_change_taylor(rho_CH_bo, rho_fm, rho_ann, C_LS3)
    print(f"  breakout: rho_CH       = {rho_CH_bo:.3f} g/cc")
    print(f"  exact h / change       = {h_exact:.3f} in / {h_exact - H0_NOMINAL:+.3f} in")
    print(f"  Taylor delta_h (Eq. 5) = {dh_taylor:+.3f} in")
    assert abs(h_exact - h_true) < 1e-6
    assert (h_exact - H0_NOMINAL) > 0.5
    assert dh_taylor > 0.0

    # Completion classification
    for rho, expect in [(0.2, "gas"), (1.0, "water"), (1.8, "solids")]:
        assert classify_annulus(rho) == expect
    print(f"  annulus classes        = gas/water/solids OK")
    print("  PASS")
    return {"lambda": lam, "h_breakout": h_exact, "dh_taylor": dh_taylor}


if __name__ == "__main__":
    test_all()
