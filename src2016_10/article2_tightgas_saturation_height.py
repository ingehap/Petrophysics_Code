"""
Article 2: How Pore-Scale Attributes May Be Used to Derive Robust Drainage and
           Imbibition Water-Saturation Models in Complex Tight-Gas Reservoirs
Merletti, Gramin, Salunke, Hamman, Spain, Shabro, Armitage, Torres-Verdin,
Salter, Dacy (2016)
Reference: Petrophysics Vol. 57, No. 5 (October 2016), pp. 447-464
DOI: none assigned (this issue predates SPWLA DOI assignment)

Drainage and imbibition saturation-height models are built for a tight-gas
reservoir (Almond Formation) from SCAL data.  Capillary-pressure tests are
converted from air-mercury to air-brine and from laboratory to reservoir
conditions, corrected for clay-bound water, then fit with a Thomeer drainage
model and a modified Brooks-Corey imbibition model.  Trapped-gas saturation
(the imbibition endmember) follows Land's empirical model.

Implements:

  - Air-mercury -> air-brine capillary-pressure conversion (Eq. 1)
  - Clay-bound-water saturation correction (Eqs. 2-3)
  - Laboratory -> reservoir capillary-pressure conversion (Eqs. 4-5)
  - Thomeer drainage water-saturation model (Pe, G, Swirr)
  - Land trapped-gas model and Swgt = 1 - Sgt (Eqs. 8, 10)
  - Modified Brooks-Corey imbibition Pc / Sw (Eqs. 11-13)
  - Buoyancy capillary pressure (height above free-water level)

Note: this issue's PDF has a text layer; the conversion and Brooks-Corey
relations (Eqs. 1-13) are transcribed from the body, while the typeset
display-equation glyphs were dropped and reconstructed in standard form.  Pc in
psi, IFT in dyne/cm, densities in g/cm^3, saturations/porosities as fractions.
"""

import numpy as np

GRAV_PSI_PER_FT = 0.433       # psi/ft per unit specific gravity (water column)


# ---------------------------------------------- Pc conversions --------------

def pc_air_brine_from_mercury(pc_am, ift_ab=72.0, theta_ab=0.0,
                              ift_am=480.0, theta_am=140.0):
    """Air-mercury -> air-brine capillary pressure (Eq. 1)

        Pc_ab = Pc_am * (IFT_ab*cos(theta_ab)) / (IFT_am*cos(theta_am)).
    """
    return pc_am * (ift_ab * abs(np.cos(np.radians(theta_ab)))
                    / (ift_am * abs(np.cos(np.radians(theta_am)))))


def cbw_saturation_correction(swp, phi_t, phi_e):
    """Clay-bound-water correction of the pseudo-wetting/nonwetting saturation
    (Eqs. 2-3)

        Swp_corr = Swp*phi_e/phi_t + (phi_t - phi_e)/phi_t      (wetting)
        Swnp_corr = 1 - Swp_corr                                (nonwetting),

    restoring the volume occupied by clay-bound water (phi_t - phi_e).
    """
    swp_corr = swp * phi_e / phi_t + (phi_t - phi_e) / phi_t
    return swp_corr, 1.0 - swp_corr


def pc_lab_to_reservoir(pc_lab, ift_res=47.0, theta_res=0.0,
                        ift_lab=72.0, theta_lab=0.0):
    """Laboratory -> reservoir air-brine capillary pressure (Eqs. 4-5)

        Pc_res = Pc_lab * (IFT_res*cos(theta_res)) / (IFT_lab*cos(theta_lab)).
    """
    return pc_lab * (ift_res * np.cos(np.radians(theta_res))
                     / (ift_lab * np.cos(np.radians(theta_lab))))


# ---------------------------------------------- drainage (Thomeer) --------------

def thomeer_sw(pc, pe, g, swirr):
    """Thomeer (1960) drainage water-saturation model

        Sw = 1 - (1 - Swirr)*exp(-G/log10(Pc/Pe))   for Pc >= Pe,  else Sw = 1,

    with entry pressure Pe, pore-geometrical factor G and irreducible water
    saturation Swirr.
    """
    pc = np.asarray(pc, float)
    snw = np.where(pc > pe, (1.0 - swirr) * np.exp(-g / np.log10(pc / pe)), 0.0)
    return 1.0 - snw


# ---------------------------------------------- imbibition (trapped gas) --------------

def land_trapped_gas(sgi, sgt_max):
    """Land (1968) trapped-gas saturation from initial gas saturation (Eq. 8)

        1/Sgt - 1/Sgi = 1/Sgt_max - 1   (Sgi_max = 1)  ->  Sgt = Sgi/(1 + C*Sgi),

    with Land's constant C = 1/Sgt_max - 1.  (The paper averages Land and Jerauld
    for the Almond samples; Land matches carbonates, Jerauld clastics.)
    """
    c = 1.0 / sgt_max - 1.0
    return sgi / (1.0 + c * sgi)


def swgt(sgt):
    """Water saturation at trapped gas  Swgt = 1 - Sgt  (Eq. 10)."""
    return 1.0 - sgt


def normalized_sw(sw, swirr, sw_gt):
    """Normalized water saturation between Swirr and Swgt (Eq. 11)

        Swn = (Sw - Swirr)/(Swgt - Swirr).
    """
    return (sw - swirr) / (sw_gt - swirr)


def brooks_corey_imbibition_pc(swn, a, b):
    """Modified Brooks-Corey imbibition capillary pressure (Eq. 12)

        Pc = A * Swn^(-B),

    where A is the capillary pressure at Swgt (Swn = 1) and B is the curvature.
    """
    return a * np.asarray(swn, float) ** (-b)


def brooks_corey_imbibition_sw(pc, a, b, swirr, sw_gt):
    """Modified Brooks-Corey imbibition water saturation (Eq. 13)

        Sw = Swirr + (Swgt - Swirr)*(A/Pc)^(1/B),

    the inverse of Eqs. 11-12 (clipped to the [Swirr, Swgt] data range).
    """
    pc = np.asarray(pc, float)
    swn = (a / pc) ** (1.0 / b)
    sw = swirr + (sw_gt - swirr) * swn
    return np.clip(sw, swirr, sw_gt)


# ---------------------------------------------- saturation height --------------

def buoyancy_pc(height_above_fwl, sg_water, sg_gas):
    """Buoyancy capillary pressure at a height above the free-water level

        Pc = 0.433*(SGw - SGg)*HAFWL    [psi],

    the equilibrium relation Pc = (rho_w - rho_g)*g*h converted to psi/ft.
    """
    return GRAV_PSI_PER_FT * (sg_water - sg_gas) * np.asarray(height_above_fwl, float)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Tight-Gas Drainage/Imbibition Sw Models")
    print("=" * 60)

    # Air-mercury -> air-brine conversion lowers Pc (mercury IFT is much higher)
    pc_ab = pc_air_brine_from_mercury(1000.0)
    print(f"  Pc air-brine from 1000 psi Hg = {pc_ab:.1f} psi")
    assert 0 < pc_ab < 1000.0

    # CBW correction raises the wetting saturation and the two parts sum to one
    swp_c, swnp_c = cbw_saturation_correction(0.30, phi_t=0.12, phi_e=0.10)
    assert swp_c > 0.30 and np.isclose(swp_c + swnp_c, 1.0)

    # Lab -> reservoir conversion scales Pc by the IFT ratio (47/72)
    assert np.isclose(pc_lab_to_reservoir(100.0), 100.0 * 47.0 / 72.0)

    # Thomeer drainage: Sw = 1 below entry pressure and decreases above it
    assert np.isclose(thomeer_sw(5.0, pe=10.0, g=0.3, swirr=0.2), 1.0)
    sw_lo = thomeer_sw(100.0, pe=10.0, g=0.3, swirr=0.2)
    sw_hi = thomeer_sw(1000.0, pe=10.0, g=0.3, swirr=0.2)
    print(f"  Thomeer Sw @100/1000 psi = {sw_lo:.3f} / {sw_hi:.3f}")
    assert 0.2 <= sw_hi < sw_lo < 1.0

    # Land trapping: Sgt = Sgt_max when Sgi = 1, and is smaller for smaller Sgi
    assert np.isclose(land_trapped_gas(1.0, 0.35), 0.35)
    assert land_trapped_gas(0.8, 0.35) < 0.35
    sgt = land_trapped_gas(0.8, 0.35)
    print(f"  Land Sgt (Sgi=0.8)     = {sgt:.3f}")

    # Brooks-Corey imbibition Pc and Sw are mutual inverses
    sw_g = swgt(sgt)
    swn = normalized_sw(0.6, 0.2, sw_g)
    pc = brooks_corey_imbibition_pc(swn, a=30.0, b=0.5)
    sw_back = brooks_corey_imbibition_sw(pc, a=30.0, b=0.5, swirr=0.2, sw_gt=sw_g)
    assert np.isclose(sw_back, 0.6)

    # Buoyancy capillary pressure increases with height above the FWL
    assert buoyancy_pc(200.0, 1.05, 0.20) > buoyancy_pc(50.0, 1.05, 0.20) > 0
    print("  PASS")
    return {"Pc_ab": float(pc_ab), "Thomeer_Sw": float(sw_hi), "Sgt": float(sgt)}


if __name__ == "__main__":
    test_all()
