"""
Article 5: Dielectric Permittivity: A Petrophysical Parameter for Shales
Matthew Josh (2014)
Reference: Petrophysics Vol. 55, No. 4 (August 2014), pp. 319-332
DOI: none assigned (this issue predates SPWLA DOI assignment)

A regular contribution.  Broadband dielectric measurements on shale pastes and
preserved plugs are correlated with shale petrophysical parameters: the
real permittivity of the paste at 10 MHz predicts cation exchange capacity,
specific surface area and P-wave velocity, and the Debye / Cole-Cole dispersion
models describe the frequency response.

Implements:

  - Complex permittivity and the equivalent imaginary part (polarization +
    conduction loss)  eps''_eq = eps''_pol + sigma/(omega*eps0)
  - Debye and Cole-Cole relaxation models
  - CEC from paste permittivity at 10 MHz  (Eq. 1)
  - Specific surface area from permittivity  (Eqs. 4, 5)
  - P-wave velocity from paste permittivity at 10 MHz  (Eq. 6)
  - Analytic specific surface area from clay mineralogy  (Eq. A1)

Note: this issue's PDF has a text layer; the empirical correlations (Eqs. 1, 4,
5, 6, A1) survived verbatim, while the Debye / Cole-Cole / complex-permittivity
forms are named but not displayed and reconstructed in standard form (Debye,
1929; Cole & Cole, 1941).  CEC in cmol/kg, SSA in m^2/g, Vp in m/s.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

EPS0 = 8.854e-12  # vacuum permittivity, F/m


# ---------------------------------------------- complex permittivity --------------

def equivalent_imaginary_permittivity(eps_pol_imag, sigma, freq):
    """Equivalent imaginary relative permittivity (polarization + conduction loss)

        eps''_eq = eps''_pol + sigma/(omega*eps0),   omega = 2*pi*freq.

    The conduction term dominates at low frequency and gives a 1:1 down-slope in
    log(eps''_eq) vs log(f).
    """
    omega = 2.0 * np.pi * np.asarray(freq, float)
    return eps_pol_imag + sigma / (omega * EPS0)


def debye(freq, eps_inf, eps_s, tau):
    """Debye complex relative permittivity (Debye, 1929)

        eps* = eps_inf + (eps_s - eps_inf)/(1 + i*omega*tau).
    """
    return petrolib.em_dielectric.debye(freq, eps_inf=eps_inf, eps_s=eps_s, tau=tau)


def cole_cole(freq, eps_inf, eps_s, tau, alpha):
    """Cole-Cole complex relative permittivity (Cole & Cole, 1941)

        eps* = eps_inf + (eps_s - eps_inf)/(1 + (i*omega*tau)^(1-alpha)),

    the distributed-relaxation generalization of Debye (alpha = 0 recovers it).
    """
    return petrolib.em_dielectric.cole_cole(
        freq, eps_inf=eps_inf, eps_s=eps_s, tau=tau, alpha=alpha
    )


# ---------------------------------------------- empirical correlations --------------

def cec_from_permittivity(eps_paste_10mhz):
    """CEC from the real paste permittivity at 10 MHz (Eq. 1, R^2 = 0.926)

        CEC = -15.912 + 0.4504*eps'_r(paste @ 10 MHz)   [cmol/kg].
    """
    return -15.912 + 0.4504 * np.asarray(eps_paste_10mhz, float)


def ssa_from_permittivity_10mhz(eps_paste_10mhz):
    """Specific surface area from the real paste permittivity at 10 MHz
    (Eq. 4, R^2 = 0.932)

        SSA = -124.26 + 3.2372*eps'_r(paste @ 10 MHz)   [m^2/g].
    """
    return -124.26 + 3.2372 * np.asarray(eps_paste_10mhz, float)


def ssa_from_permittivity_1ghz(eps_imag_paste_1ghz):
    """Specific surface area from the imaginary paste permittivity at 1 GHz
    (Eq. 5, R^2 = 0.921)

        SSA = -75.209 + 24.806*eps''_r(paste @ 1 GHz)   [m^2/g].
    """
    return -75.209 + 24.806 * np.asarray(eps_imag_paste_1ghz, float)


def vp_from_permittivity(eps_paste_10mhz):
    """P-wave velocity from the real paste permittivity at 10 MHz
    (Eq. 6, R^2 = 0.865)

        Vp = 8986.1 - 93.83*eps'_r(paste @ 10 MHz)   [m/s].
    """
    return 8986.1 - 93.83 * np.asarray(eps_paste_10mhz, float)


def analytic_ssa(frac_smectite, frac_illite, frac_kaolinite):
    """Analytic specific surface area from clay mineralogy (Eq. A1)

        An.SSA = 700*%smectite + 60*%illite + 40*%kaolinite   [m^2/g],

    using the end-member surface areas (smectite ~700, illite 60, kaolinite 40).
    Fractions are weight fractions (0-1).
    """
    return 700.0 * frac_smectite + 60.0 * frac_illite + 40.0 * frac_kaolinite


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Dielectric Permittivity of Shales")
    print("=" * 60)

    # Conduction loss dominates the equivalent imaginary permittivity at low f
    e_lo = equivalent_imaginary_permittivity(5.0, sigma=0.03, freq=1e4)
    e_hi = equivalent_imaginary_permittivity(5.0, sigma=0.03, freq=1e8)
    print(f"  eps''_eq(10kHz)={e_lo:.1f}  eps''_eq(100MHz)={e_hi:.2f}")
    assert e_lo > e_hi

    # Debye/Cole-Cole: real part drops from static to high-frequency limit
    d = debye(np.array([1e3, 1e12]), eps_inf=4.0, eps_s=40.0, tau=1e-7)
    assert np.isclose(d[0].real, 40.0, atol=1e-3) and np.isclose(d[1].real, 4.0, atol=1e-3)
    cc = cole_cole(1e7, 4.0, 40.0, 1e-7, alpha=0.0)
    assert np.isclose(cc, debye(1e7, 4.0, 40.0, 1e-7))  # alpha=0 recovers Debye

    # Empirical correlations on a representative paste permittivity
    eps_paste = 75.0
    cec = cec_from_permittivity(eps_paste)
    ssa = ssa_from_permittivity_10mhz(eps_paste)
    vp = vp_from_permittivity(eps_paste)
    print(f"  CEC={cec:.2f} cmol/kg  SSA={ssa:.1f} m2/g  Vp={vp:.0f} m/s")
    assert 2.9 <= cec <= 20 and 11 <= ssa <= 130 and 1000 < vp < 9000
    # higher permittivity -> higher CEC/SSA, lower velocity
    assert cec_from_permittivity(90) > cec and vp_from_permittivity(90) < vp

    # Analytic SSA dominated by smectite
    a_ssa = analytic_ssa(0.09, 0.50, 0.41)
    print(f"  analytic SSA = {a_ssa:.1f} m2/g")
    assert np.isclose(a_ssa, 700 * 0.09 + 60 * 0.50 + 40 * 0.41)
    print("  PASS")
    return {"CEC": float(cec), "SSA": float(ssa), "Vp": float(vp)}


if __name__ == "__main__":
    test_all()
