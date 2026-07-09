"""
Article 3: Experimental Study of the Effects of Wettability and Fluid Saturation
           on NMR and Dielectric Measurements in Limestone
Lalitha Venkataramanan, Martin D. Hurlimann, Jeffrey A. Tarvin, Kamilla Fellah,
Diana Acero-Allard, Nikita V. Seleznev (2014)
Reference: Petrophysics Vol. 55, No. 6 (December 2014), pp. 572-586
DOI: none assigned (this issue predates SPWLA DOI assignment)

Oolitic Edwards limestone with a bimodal (intergranular ~10 um, intragranular
~1 um) pore system is characterized by MICP, NMR and broadband dielectric.  The
mercury injection / extrusion gives the pore-throat distribution and a
pore-body / pore-throat ratio via Land trapping; NMR surface relaxation maps T2
to pore size; and the complex-refractive-index (CRIM) model converts the
high-frequency permittivity to water saturation.

Implements:

  - Washburn pore-throat radius  r = -2*sigma*cos(theta)/Pc  (Eq. 1)
  - Land trapping constant and the disconnected-saturation quadratic (Eqs. 3-7)
  - NMR surface relaxation  1/T2 = rho*(A/V) + 1/T2B  (Eq. 8); spherical form
    1/T2 = 3*rho/r
  - CRIM water saturation from the high-frequency permittivity

Note: this issue's PDF has a text layer; the Washburn relation (Eq. 1), the
Land quadratic coefficients (Eqs. 5-7) and the surface-relaxation form (Eq. 8)
are transcribed from the body, while the dropped glyphs are reconstructed in
standard form (Washburn, 1921; Land, 1968; Seleznev et al., 2004).  SI units;
T2 in seconds, radii in metres, relaxivity in m/s.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- MICP pore throat --------------

def washburn_radius(pc, sigma=0.480, theta_deg=140.0):
    """Pore-throat radius from capillary pressure (Washburn; Eq. 1)

        r = -2*sigma*cos(theta)/Pc,

    with the mercury-air surface tension sigma = 0.480 N/m and contact angle
    theta ~ 140 deg, so the numerator is positive.  Pc in Pa, r in m.

    The leading minus folds the obtuse mercury angle's negative cosine to a
    positive radius; the library expresses this as the |cos(theta)| (mercury)
    convention (absolute=True), which matches ``-2*sigma*cos(theta)/Pc`` for
    theta in [90, 180] deg.
    """
    return petrolib.capillary_pressure.washburn_radius(
        pc, sigma=sigma, theta_deg=theta_deg, absolute=True)


# ---------------------------------------------- Land trapping --------------

def land_constant(s_res, s_i=1.0):
    """Land trapping constant from a residual / initial nonwetting saturation
    (Land, 1968; Eq. 3)

        1/S_res - 1/S_i = C.
    """
    return petrolib.relperm_wettability.land_c(s_i_max=s_i, s_r_max=s_res)


def disconnected_saturation(land_c, s_res, s_imb):
    """Disconnected (trapped) saturation at the end of extrusion from the Land
    quadratic  a*Sdc^2 + b*Sdc + k = 0  (Eqs. 5-7)

        a = C,  b = -C*(S_res + S_imb),  k = S_imb*(C*S_res - 1) + S_res.

    Returns the physical (smaller, non-negative) root.
    """
    a = land_c
    b = -land_c * (s_res + s_imb)
    k = s_imb * (land_c * s_res - 1.0) + s_res
    disc = np.sqrt(b * b - 4 * a * k)
    roots = np.array([(-b - disc) / (2 * a), (-b + disc) / (2 * a)])
    physical = roots[(roots >= 0) & (roots <= 1)]
    return float(physical.min()) if physical.size else float(roots.min())


# ---------------------------------------------- NMR surface relaxation --------------

def t2_surface_relaxation(rho, surface_to_volume, t2_bulk):
    """NMR transverse relaxation time from surface relaxation (Eq. 8)

        1/T2 = rho*(A/V) + 1/T2B,

    with the surface relaxivity rho (m/s), surface-to-volume ratio A/V (1/m)
    and bulk relaxation T2B (s).
    """
    return petrolib.nmr.t2_apparent(t2_bulk=t2_bulk, rho=rho, s_over_v=surface_to_volume)


def t2_spherical_pore(rho, radius, t2_bulk=np.inf):
    """T2 for a spherical pore of radius r, where A/V = 3/r (Eq. 8)

        1/T2 = 3*rho/r + 1/T2B.
    """
    return petrolib.nmr.t2_apparent(t2_bulk=t2_bulk, rho=rho, s_over_v=3.0 / radius)


def pore_radius_from_t2(rho, t2, t2_bulk=np.inf):
    """Invert the spherical surface-relaxation form for the pore radius

        r = 3*rho/(1/T2 - 1/T2B).
    """
    return petrolib.nmr.pore_radius_from_t2(t2, rho=rho, shape_factor=3.0, t2_bulk=t2_bulk)


# ---------------------------------------------- CRIM dielectric --------------

def crim_permittivity(sw, phi, eps_w, eps_hc, eps_matrix):
    """Complex-refractive-index (CRIM) mixing permittivity

        sqrt(eps) = Sw*phi*sqrt(eps_w) + (1-Sw)*phi*sqrt(eps_hc)
                    + (1-phi)*sqrt(eps_matrix).
    """
    return petrolib.em_dielectric.crim(
        phi, sw, eps_w=eps_w, eps_hc=eps_hc, eps_matrix=eps_matrix
    )


def crim_water_saturation(eps, phi, eps_w, eps_hc, eps_matrix):
    """Water saturation from CRIM, solving the mixing law for Sw

        Sw = [sqrt(eps) - (1-phi)*sqrt(eps_m) - phi*sqrt(eps_hc)]
             / [phi*(sqrt(eps_w) - sqrt(eps_hc))].
    """
    return petrolib.em_dielectric.sw_from_permittivity(
        eps, phi, eps_w=eps_w, eps_hc=eps_hc, eps_matrix=eps_matrix, clip=False
    )


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: NMR & Dielectric in Limestone")
    print("=" * 60)

    # Washburn: higher injection pressure resolves smaller throats
    r_hi = washburn_radius(1e6)
    r_lo = washburn_radius(1e7)
    print(f"  r(1 MPa)={r_hi*1e6:.3f} um   r(10 MPa)={r_lo*1e6:.3f} um")
    assert r_hi > r_lo > 0

    # Land trapping: residual < initial gives a positive constant and trapping
    c = land_constant(s_res=0.35, s_i=1.0)
    sdc = disconnected_saturation(c, s_res=0.35, s_imb=0.5)
    print(f"  Land C={c:.3f}  disconnected S={sdc:.3f}")
    assert c > 0 and 0 <= sdc <= 0.5

    # NMR: micropores (small r) relax faster (shorter T2) than macropores
    rho = 6.6e-6  # m/s
    t2_micro = t2_spherical_pore(rho, 1e-6, t2_bulk=3.0)
    t2_macro = t2_spherical_pore(rho, 10e-6, t2_bulk=3.0)
    print(f"  T2(1um)={t2_micro*1e3:.1f} ms   T2(10um)={t2_macro*1e3:.1f} ms")
    assert t2_micro < t2_macro
    # round-trip the pore radius
    assert np.isclose(pore_radius_from_t2(rho, t2_micro, 3.0), 1e-6)

    # CRIM: round-trip water saturation through the mixing law
    sw_true, phi = 0.30, 0.22
    eps = crim_permittivity(sw_true, phi, eps_w=76.0, eps_hc=2.2, eps_matrix=7.5)
    sw_rec = crim_water_saturation(eps, phi, 76.0, 2.2, 7.5)
    print(f"  eps={eps:.2f}  Sw_recovered={sw_rec:.3f}")
    assert np.isclose(sw_rec, sw_true)
    print("  PASS")
    return {"Land_C": float(c), "Sdc": float(sdc), "Sw": float(sw_rec)}


if __name__ == "__main__":
    test_all()
