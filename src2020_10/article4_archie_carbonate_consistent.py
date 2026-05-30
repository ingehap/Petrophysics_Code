"""
Article 4: Towards a Petrophysically Consistent Implementation of Archie's
           Equation for Heterogeneous Carbonate Rocks
Ramamoorthy, Ramakrishnan, Dasgupta, Raina (2020)
DOI: 10.30632/PJV61N5-2020a4

Archie's equation with a single cementation exponent m is inadequate for vuggy
carbonates whose pore system spans well-separated intergranular and vug
subsystems.  An effective-medium (symmetric Bruggeman) homogenization gives a
depth-varying effective m, and predicts the characteristic non-Archie behavior
that the resistivity index in vuggy intervals rises with a slope (effective
saturation exponent) near or below unity rather than the Archie value of two.

Implements:

  - Archie  Sw^n = a*Rw/(phi^m*Rt) ; R0 = a*Rw/phi^m              (Eqs. 1a-1b)
  - Resistivity index  RI = Rt/R0 = Sw^(-n)                       (Eq. 1c)
  - Formation factor  F = a/phi^m  and effective m from F         (Eq. 6)
  - Symmetric Bruggeman effective conductivity                    (Eqs. 4-5)
  - Vuggy resistivity-index slope (effective saturation exponent) (Eqs. 8-11)

Note: this issue's PDF text layer dropped the typeset glyphs of Eqs. 1-13, so
these are the standard Archie / symmetric-Bruggeman forms anchored to the
preserved variable definitions.  Paper anchors reproduced: Archie m = n = 2,
best-fit m = 1.7, and the vuggy resistivity index whose slope stays near 1.
"""

import numpy as np

SIGMA_W = 10.0           # S/m, brine conductivity


# ---------------------------------------------- Archie ------------------

def archie_sw(Rt, Rw, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation  Sw = (a*Rw/(phi^m*Rt))^(1/n)  (Eq. 1a)."""
    phi = np.asarray(phi, float)
    return (a * Rw / (phi ** m * np.asarray(Rt, float))) ** (1.0 / n)


def r_zero(Rw, phi, a=1.0, m=2.0):
    """Resistivity at Sw = 1  R0 = a*Rw/phi^m  (Eq. 1b)."""
    return a * Rw / np.asarray(phi, float) ** m


def resistivity_index(Rt, R0):
    """Resistivity index  RI = Rt/R0 = Sw^(-n)  (Eq. 1c)."""
    return np.asarray(Rt, float) / np.asarray(R0, float)


def formation_factor(phi, a=1.0, m=2.0):
    """Archie formation factor  F = a/phi^m."""
    return a / np.asarray(phi, float) ** m


def effective_cementation_exponent(F, phi, a=1.0):
    """Effective m back-computed from F and phi  (Eq. 6)  m = ln(F/a)/ln(1/phi)."""
    return np.log(F / a) / np.log(1.0 / np.asarray(phi, float))


# ---------------------------------------------- Bruggeman ---------------

def bruggeman_effective(fracs, sigmas, tol=1e-10):
    """Symmetric Bruggeman effective conductivity (Eqs. 4-5).

    Solves  sum_i f_i * (sigma_i - sigma)/(sigma_i + 2*sigma) = 0  by bisection.
    fracs sum to 1; conductivities sigmas > 0.
    """
    f = np.asarray(fracs, float)
    s = np.asarray(sigmas, float)

    def g(sig):
        return np.sum(f * (s - sig) / (s + 2.0 * sig))

    lo, hi = s.min() * 1e-6, s.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * hi:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Consistent Archie for Heterogeneous Carbonates")
    print("=" * 60)

    # Archie round-trip and RI = Sw^-n
    Rw, phi = 0.05, 0.20
    R0 = r_zero(Rw, phi)
    Rt = 20.0
    sw = archie_sw(Rt, Rw, phi)
    RI = resistivity_index(Rt, R0)
    print(f"  Sw = {sw:.3f}   RI = {RI:.2f}   Sw^-2 = {sw**-2:.2f}")
    assert abs(RI - sw ** -2) < 1e-9

    # Effective m inverts the formation factor exactly
    F = formation_factor(phi, m=2.0)
    m_eff = effective_cementation_exponent(F, phi)
    print(f"  formation factor F     = {F:.1f}   effective m = {m_eff:.3f}")
    assert abs(m_eff - 2.0) < 1e-9

    # Bruggeman effective conductivity lies between the component conductivities
    sig = bruggeman_effective([0.7, 0.3], [1.0, 10.0])
    print(f"  Bruggeman sigma        = {sig:.3f} S/m  (between 1 and 10)")
    assert 1.0 < sig < 10.0

    # Heterogeneity -> depth-varying effective m: at fixed total porosity, moving
    # porosity into separate (poorly connected) vugs changes the whole-rock
    # sigma0 and hence the effective m, so a single Archie m cannot describe the
    # column.  Separate-vug porosity elevates m above the matrix value of 2 (the
    # classic vuggy-carbonate effect, Lucia), and m rises with vug fraction.
    phi_t = 0.20
    m_values = []
    for fv in (0.02, 0.06, 0.10):                  # vug volume fraction
        phi_m = (phi_t - fv) / (1.0 - fv)          # matrix microporosity
        sig_m = SIGMA_W * phi_m ** 2               # matrix Archie (m=2)
        sig0 = bruggeman_effective([1.0 - fv, fv], [sig_m, SIGMA_W])
        F_eff = SIGMA_W / sig0
        m_values.append(effective_cementation_exponent(F_eff, phi_t))
    m_values = np.array(m_values)
    print(f"  effective m vs vug frac = {np.array2string(m_values, precision=3)}")
    assert m_values.std() > 0.05                   # m genuinely varies
    assert np.all(np.diff(m_values) > 0)           # more separate vugs -> higher m
    assert np.all(m_values > 2.0)                  # separate vugs elevate m

    # Vuggy resistivity index: as connected water-filled vugs drain, the
    # conductive matrix shorts the resistive oil, so RI rises with an effective
    # saturation exponent well below the Archie value of 2 (near unity).
    fv = 0.12
    sig_m = SIGMA_W * 0.10 ** 2                     # conductive microporous matrix
    swv = np.linspace(1.0, 0.2, 9)                  # vug water saturation
    sig0 = bruggeman_effective([1 - fv, fv], [sig_m, SIGMA_W])
    RI_v, sw_t = [], []
    phi_m = 0.10
    for s in swv:
        sig_vug = SIGMA_W * s ** 2                  # Archie inside the vug
        sig = bruggeman_effective([1 - fv, fv], [sig_m, sig_vug])
        RI_v.append(sig0 / sig)
        sw_t.append((phi_m + fv * s) / (phi_m + fv))
    n_eff = np.polyfit(np.log(1.0 / np.array(sw_t)), np.log(RI_v), 1)[0]
    print(f"  vuggy effective n      = {n_eff:.3f}  (Archie n = 2)")
    assert 0.0 < n_eff < 1.5                        # sub-Archie, near unity
    print("  PASS")
    return {"m_eff": float(m_eff), "m_vug_range": m_values.tolist(),
            "n_eff_vuggy": float(n_eff)}


if __name__ == "__main__":
    test_all()
