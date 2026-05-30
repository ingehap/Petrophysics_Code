"""
Article 1: Lessons Learned From Casedhole Formation Evaluation Along
           Unconventional Horizontal Wells
Sullivan, Wang, Bolshakov, Song, Lazorek, Tohidi, Seth (2020)
DOI: 10.30632/PJV61N3-2020a1

A case study of through-the-bit casedhole logging (spectral gamma ray,
pulsed-neutron spectroscopy, multipole sonic) in unconventional laterals,
calibrated against an analog openhole vertical well and cuttings.  Sigma gives
water saturation, the fast-neutron cross section gives gas, and a
modified-ANNIE (M-ANNIE) VTI model converts sonic stiffnesses to anisotropic
elastic / mechanical properties.

Implements:

  - Spectral gamma ray  gAPI = 4*Th + 8*U + 16*K                  (Eq. 1, verbatim)
  - M-ANNIE VTI engineering moduli from stiffnesses Cij           (Eqs. 2-5)
  - Sigma water saturation from the porosity balance
  - Acoustic impedance Z = rho*Vp and the gas cutoff
  - Elemental dry-weight correction (+2 Ca, -3 Fe, -2 Al wt%)

Note: this issue's PDF text layer kept the equation numbers and definitions but
dropped the typeset glyphs (Eq. 1 survived verbatim), so the M-ANNIE VTI forms
are the standard transversely-isotropic stiffness-to-engineering-modulus
relations.  Paper anchors: SGR coefficients 4/8/16, the +2/-3/-2 wt% elemental
shifts, and the 0.3 Mrayl acoustic-impedance gas cutoff.
"""

import numpy as np

AI_GAS_CUTOFF_MRAYL = 0.3        # paper's color-map gas flag


# ---------------------------------------------- spectral GR -------------

def spectral_gr(th_ppm, u_ppm, k_wtpct):
    """Spectral gamma ray  gAPI = 4*Th + 8*U + 16*K  (Eq. 1, Ellis & Singer)."""
    return 4.0 * th_ppm + 8.0 * u_ppm + 16.0 * k_wtpct


# ---------------------------------------------- M-ANNIE VTI -------------

def vti_moduli(C11, C33, C13, C44, C66):
    """VTI engineering moduli from stiffness coefficients (Eqs. 2-5).

    Returns (E_vert, E_horz, nu_vert, nu_horz) where (C12 = C11 - 2*C66):
      E_vert = C33 - 2*C13^2/(C11 + C12)
      E_horz = (C11 - C12)*(C11*C33 - 2*C13^2 + C12*C33)/(C11*C33 - C13^2)
      nu_vert(31) = C13/(C11 + C12)
      nu_horz(12) = (C12*C33 - C13^2)/(C11*C33 - C13^2)
    """
    C12 = C11 - 2.0 * C66
    E_v = C33 - 2.0 * C13 ** 2 / (C11 + C12)
    E_h = (C11 - C12) * (C11 * C33 - 2.0 * C13 ** 2 + C12 * C33) \
        / (C11 * C33 - C13 ** 2)
    nu_v = C13 / (C11 + C12)
    nu_h = (C12 * C33 - C13 ** 2) / (C11 * C33 - C13 ** 2)
    return E_v, E_h, nu_v, nu_h


# ---------------------------------------------- saturation / AI ---------

def sigma_water_saturation(sigma_log, phi, sigma_ma, sigma_w, sigma_hc):
    """Water saturation from the sigma porosity balance (clipped to [0,1])."""
    phi = np.asarray(phi, float)
    num = np.asarray(sigma_log, float) - sigma_ma * (1.0 - phi) - phi * sigma_hc
    return np.clip(num / (phi * (sigma_w - sigma_hc)), 0.0, 1.0)


def acoustic_impedance(rho_kg_m3, vp_m_s):
    """Acoustic impedance  Z = rho*Vp  (Mrayl)."""
    return np.asarray(rho_kg_m3, float) * np.asarray(vp_m_s, float) / 1e6


def is_gas(impedance_mrayl, cutoff=AI_GAS_CUTOFF_MRAYL):
    """Flag low-impedance (gas) intervals below the cutoff."""
    return np.asarray(impedance_mrayl, float) < cutoff


def correct_elements(ca, fe, al):
    """Apply the casing/cement dry-weight shifts: +2 Ca, -3 Fe, -2 Al (wt%)."""
    return ca + 2.0, fe - 3.0, al - 2.0


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Casedhole FE Along Horizontal Wells")
    print("=" * 60)

    # Spectral GR: a hot organic shale reads high; verify the 4/8/16 weights
    gr = spectral_gr(12.0, 8.0, 3.0)
    print(f"  spectral GR            = {gr:.0f} API")
    assert abs(gr - (4 * 12 + 8 * 8 + 16 * 3)) < 1e-9
    assert spectral_gr(15.0, 8.0, 3.0) < 250.0      # within the formation range

    # M-ANNIE VTI: an isotropic limit (C11=C33, C13=C12, C44=C66) gives
    # E_vert = E_horz and nu_vert = nu_horz
    C11 = C33 = 60.0; C44 = C66 = 24.0
    C12 = C11 - 2 * C66                              # = 12
    C13 = C12                                        # isotropic
    Ev, Eh, nv, nh = vti_moduli(C11, C33, C13, C44, C66)
    print(f"  VTI E_v/E_h            = {Ev:.2f} / {Eh:.2f} GPa")
    print(f"  VTI nu_v/nu_h          = {nv:.3f} / {nh:.3f}")
    assert abs(Ev - Eh) < 1e-6 and abs(nv - nh) < 1e-6
    # an anisotropic shale (stiffer horizontally) has E_horz > E_vert
    Ev2, Eh2, _, _ = vti_moduli(75.0, 55.0, 20.0, 22.0, 28.0)
    assert Eh2 > Ev2

    # Sigma water saturation: a saline wet zone reads higher Sw than a gas zone
    sw_wet = sigma_water_saturation(28.0, 0.06, 9.0, 60.0, 12.0)
    sw_gas = sigma_water_saturation(12.0, 0.06, 9.0, 60.0, 12.0)
    print(f"  Sw wet / gas           = {sw_wet:.2f} / {sw_gas:.2f}")
    assert sw_wet > sw_gas

    # Acoustic impedance: gas sand is lower-impedance than brine sand
    z_gas = acoustic_impedance(2050.0, 2600.0)
    z_brine = acoustic_impedance(2350.0, 3600.0)
    print(f"  AI gas / brine         = {z_gas:.2f} / {z_brine:.2f} Mrayl")
    assert z_gas < z_brine

    # Elemental correction shifts
    ca, fe, al = correct_elements(10.0, 5.0, 8.0)
    assert (ca, fe, al) == (12.0, 2.0, 6.0)
    print("  PASS")
    return {"gr": gr, "E_v": Ev, "E_h": Eh, "sw_wet": float(sw_wet)}


if __name__ == "__main__":
    test_all()
