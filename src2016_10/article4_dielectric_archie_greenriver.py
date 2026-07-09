"""
Article 4: Advanced Log Interpretation in Field Development
Merkel, Lessenger (2016)
Reference: Petrophysics Vol. 57, No. 5 (October 2016), pp. 479-491
DOI: none assigned (this issue predates SPWLA DOI assignment)

In the Monument Butte field (Green River Formation, Uinta Basin), dielectric
logs give a salinity-independent bulk volume water (BVW) via the complex
refractive index method (CRIM).  A Pickett plot of CRIM BVW against resistivity
yields the Archie cementation exponent m and water resistivity Rw; a second
Pickett plot then fixes the saturation exponent n from the irreducible-BVW
intercept.  Spatially varying m and n (rather than the default m = n = 2) are
applied to a shaly-sand model in wells lacking dielectric/NMR data.

Implements:

  - Electromagnetic skin depth  delta = sqrt(2*rho/(omega*mu))  (Eq. 1)
  - CRIM dielectric mixing law and BVW from the measured permittivity
  - Pickett-plot fit for Archie m and Rw (CRIM BVW vs resistivity)
  - Archie water saturation and bulk volume water

Note: this issue's PDF has a text layer; the skin-depth relation (Eq. 1) and the
CRIM/Pickett workflow are transcribed from the body, while the typeset
dielectric-mixing glyphs were dropped and reconstructed in standard CRIM form.
Resistivity in ohm-m, frequency in Hz; dielectric constants relative
(dimensionless).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

MU0 = 4.0e-7 * np.pi          # permeability of free space (H/m)

# Typical relative permittivities (real part) used in CRIM
EPS_WATER = 56.0
EPS_MATRIX = 4.65
EPS_HC = 2.2


# ---------------------------------------------- skin depth --------------

def skin_depth(rho, freq, mu_r=1.0):
    """Electromagnetic skin depth  delta = sqrt(2*rho/(omega*mu))  (Eq. 1)

    with omega = 2*pi*freq and mu = mu_r*mu0 (Jordan & Balmain, 1968).  The Lower
    Green River carries a relative magnetic permeability around 10.
    """
    return petrolib.em_dielectric.skin_depth(rho, freq, mu_r=mu_r)


# ---------------------------------------------- CRIM dielectric --------------

def crim_permittivity(phi, sw, eps_water=EPS_WATER, eps_matrix=EPS_MATRIX, eps_hc=EPS_HC):
    """Complex refractive index method (CRIM) mixing law

        sqrt(eps) = (1-phi)*sqrt(eps_matrix) + phi*Sw*sqrt(eps_water)
                    + phi*(1-Sw)*sqrt(eps_hc),

    returning the effective relative permittivity of the rock.
    """
    return petrolib.em_dielectric.crim(
        phi, sw, eps_w=eps_water, eps_hc=eps_hc, eps_matrix=eps_matrix
    )


def crim_bvw(eps_measured, phi, eps_water=EPS_WATER, eps_matrix=EPS_MATRIX, eps_hc=EPS_HC):
    """Bulk volume water from the measured permittivity via CRIM

        BVW = phi*Sw = (sqrt(eps) - (1-phi)*sqrt(eps_m) - phi*sqrt(eps_hc))
                       / (sqrt(eps_w) - sqrt(eps_hc)),

    the salinity-independent water volume the dielectric tool resolves.
    """
    return petrolib.em_dielectric.bvw_from_permittivity(
        eps_measured, phi, eps_w=eps_water, eps_hc=eps_hc, eps_matrix=eps_matrix
    )


# ---------------------------------------------- Archie / Pickett --------------

def pickett_m_rw(bvw, rt):
    """Fit Archie m and Rw from a Pickett plot of CRIM BVW vs. resistivity.

    On the R0 (100% water) line  Rt = Rw * BVW^(-m), so
        log10(Rt) = log10(Rw) - m*log10(BVW).
    A least-squares line through (log BVW, log Rt) gives slope -m and
    intercept log10(Rw).  Returns (m, Rw).
    """
    lf = petrolib.inversion_numerics.fitting.fit_line(
        bvw, rt, xform="log10", yform="log10")
    return -lf.slope, 10.0 ** lf.intercept


def archie_sw(rt, rw, phi, m=2.0, n=2.0):
    """Archie water saturation  Sw = (Rw/(phi^m * Rt))^(1/n)."""
    return petrolib.saturation_resistivity.archie_sw(rt, rw, phi=phi, m=m, n=n)


def bulk_volume_water(phi, sw):
    """Bulk volume water  BVW = phi*Sw."""
    return petrolib.saturation_resistivity.bulk_volume_water(phi, sw)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Dielectric/CRIM Archie Parameters (Green River)")
    print("=" * 60)

    # Skin depth increases with resistivity (here with mu_r ~ 10)
    d = skin_depth(50.0, 1.0e9, mu_r=10.0)
    print(f"  skin depth             = {d * 100:.2f} cm")
    assert d > 0 and skin_depth(100.0, 1e9, 10.0) > skin_depth(10.0, 1e9, 10.0)

    # CRIM is self-consistent: BVW recovered from the forward permittivity = phi*Sw
    phi, sw = 0.18, 0.45
    eps = crim_permittivity(phi, sw)
    bvw = crim_bvw(eps, phi)
    print(f"  CRIM eps / BVW         = {eps:.2f} / {bvw:.4f}")
    assert np.isclose(bvw, phi * sw)

    # Pickett plot recovers the m and Rw used to synthesize an R0 line
    m_true, rw_true = 1.3, 0.23
    bvw_pts = np.array([0.02, 0.05, 0.10, 0.20])
    rt_pts = rw_true * bvw_pts ** (-m_true)       # 100% water (Sw = 1, BVW = phi)
    m_fit, rw_fit = pickett_m_rw(bvw_pts, rt_pts)
    print(f"  Pickett m / Rw         = {m_fit:.3f} / {rw_fit:.3f}")
    assert np.isclose(m_fit, m_true) and np.isclose(rw_fit, rw_true)

    # Lower m (1.3) yields a higher Sw than the default m = 2 for the same Rt
    sw_default = archie_sw(20.0, rw_true, phi, m=2.0, n=2.0)
    sw_fit = archie_sw(20.0, rw_true, phi, m=m_fit, n=2.0)
    print(f"  Sw (m=2 / m=1.3)       = {sw_default:.3f} / {sw_fit:.3f}")
    assert sw_fit < sw_default
    assert np.isclose(bulk_volume_water(phi, sw_fit), phi * sw_fit)
    print("  PASS")
    return {"BVW": float(bvw), "m": float(m_fit), "Rw": float(rw_fit), "Sw": float(sw_fit)}


if __name__ == "__main__":
    test_all()
