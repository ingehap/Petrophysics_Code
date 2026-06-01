"""
Article 3: An Evaluation of Spontaneous Imbibition of Water into Oil-Wet
           Carbonate Reservoir Cores Using Nanofluid
Abbas Roustaei (2014)
Reference: Petrophysics Vol. 55, No. 1 (February 2014), pp. 31-37
DOI: none assigned (this issue predates SPWLA DOI assignment)

Spontaneous imbibition of water into oil-wet carbonate cores is enhanced with a
silica-nanoparticle fluid that alters the rock wettability toward water-wet.
Young's law relates the contact angle to the interfacial tensions, and the
imbibition oil recovery is tracked against time for brine, surfactant and
nanofluid.

Implements:

  - Young's law contact angle  cos(theta) = (sigma_so - sigma_sw)/sigma_wo (Eq. 1)
  - Wettability classification from the contact angle
  - The Young-Laplace capillary pressure (the capillary force the paper turns
    "from a barrier to a driving force" by wettability alteration)
  - The recovery-curve-shape test for capillary- vs gravity-dominated imbibition
  - A first-order spontaneous-imbibition recovery curve toward a final plateau
  - The critical-micelle-concentration (CMC) effectiveness check for the
    surfactant, and the nanofluid's incremental recovery over the surfactant

Note: this issue's PDF dropped the displayed Young's law (Eq. 1) in extraction;
it is reconstructed in standard form.  No dimensionless-time, Bond-number or
Amott-index relation appears (the recovery analysis is qualitative), so the
recovery is modelled with the standard first-order capillary-imbibition curve
and the capillary/gravity discrimination follows the paper's stated rule (the
shape of the recovery response: curved -> capillary, linear -> gravity).
Measured anchors: nanofluid raises oil-water IFT 2.65 -> 9.21 mN/m, drives the
oil-phase contact angle to ~134 deg (water/air contact angle ~21 deg, strongly
water-wet), and lifts final recovery above ~50% IOIP (brine ~4.3%, surfactant
~46%).
"""

import numpy as np

# Optimum SiO2-nanoparticle concentration (in 1 wt% C12TAB) for the strongest
# wettability alteration and highest recovery (Roustaei, 2014).
OPTIMUM_NANOPARTICLE_CONC_GL = 3.0  # g/L

# Measured critical micelle concentration of the C12TAB surfactant (0.4-0.5 wt%).
CMC_C12TAB_WT_PCT = 0.45


# ---------------------------------------------- Young's law --------------

def cos_contact_angle(sigma_so, sigma_sw, sigma_wo):
    """Cosine of the oil-phase contact angle from Young's law (Eq. 1)

        cos(theta) = (sigma_so - sigma_sw)/sigma_wo.

    Raising the water-oil IFT (the nanofluid effect) lowers cos(theta) toward 0,
    i.e. shifts the system from oil-wet toward water-wet.
    """
    return (sigma_so - sigma_sw) / sigma_wo


def young_contact_angle(sigma_so, sigma_sw, sigma_wo):
    """Oil-phase contact angle from Young's law (Eq. 1)

        cos(theta) = (sigma_so - sigma_sw)/sigma_wo,

    with the solid-oil, solid-water and water-oil interfacial tensions.
    Returns theta in degrees.
    """
    cos_t = cos_contact_angle(sigma_so, sigma_sw, sigma_wo)
    return np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))


def work_of_adhesion(sigma_wo, contact_angle_deg):
    """Work of adhesion between water and the rock (Young-Dupre)

        W = sigma_wo*(1 + cos(theta)),

    the energy to separate water from the surface; larger W means stronger
    water-wetting, the favourable direction for spontaneous imbibition.
    """
    return sigma_wo * (1.0 + np.cos(np.radians(contact_angle_deg)))


def wettability_class(contact_angle_deg):
    """Classify wettability from the (water-phase) contact angle

        < 75 deg : water-wet,  75-105 deg : intermediate,  > 105 deg : oil-wet.
    """
    a = contact_angle_deg
    if a < 75:
        return "water-wet"
    if a <= 105:
        return "intermediate"
    return "oil-wet"


# ---------------------------------------------- capillary force --------------

def capillary_pressure(sigma_wo, contact_angle_deg, pore_radius):
    """Young-Laplace capillary pressure in a cylindrical pore

        Pc = 2*sigma_wo*cos(theta)/r,

    the capillary force driving (or resisting) spontaneous imbibition.  This is
    the paper's central mechanism: wettability alteration toward water-wet
    (theta < 90 deg) makes cos(theta) positive, turning the capillary force
    "from a barrier to a driving force", and because raising the oil-water IFT
    (the nanofluid effect, 2.65 -> 9.21 mN/m) raises sigma_wo, the imbibition
    capillary pressure increases and more oil is displaced.  Pc > 0 imbibes
    water; Pc < 0 (oil-wet, theta > 90 deg) opposes it.  sigma_wo in N/m (or
    mN/m), r in m (or mm) consistently; returns Pc in the matching pressure unit.
    """
    return 2.0 * sigma_wo * np.cos(np.radians(contact_angle_deg)) / pore_radius


def imbibition_mechanism(recovery, atol=0.02):
    """Classify the dominant spontaneous-imbibition driving force from the shape
    of a cumulative recovery-vs-time response (the paper's qualitative test)

        curved response  -> capillary-dominated,
        linear response  -> gravity-dominated.

    A straight-line fit residual below ``atol`` (fraction of IOIP) flags a
    linear, gravity-dominated response; a larger residual flags curvature, i.e.
    a capillary-dominated response.  Returns "capillary" or "gravity".
    """
    r = np.asarray(recovery, float)
    t = np.arange(r.size, dtype=float)
    slope, intercept = np.polyfit(t, r, 1)
    residual = float(np.sqrt(np.mean((r - (slope * t + intercept)) ** 2)))
    return "gravity" if residual < atol else "capillary"


# ---------------------------------------------- imbibition recovery --------------

def imbibition_recovery(t, final_recovery, rate):
    """First-order spontaneous-imbibition oil recovery toward a final plateau

        R(t) = R_final*(1 - exp(-rate*t)),

    the standard capillary-imbibition approach to the ultimate recovery R_final
    (fraction of IOIP), with a characteristic rate (1/time).
    """
    return final_recovery * (1.0 - np.exp(-rate * np.asarray(t, float)))


def above_cmc(concentration_wt_pct, cmc=CMC_C12TAB_WT_PCT):
    """Whether a surfactant concentration exceeds the critical micelle
    concentration (CMC).

    The paper measured the C12TAB CMC at 0.4-0.5 wt% and ran all imbibition
    tests above it (Standnes & Austad, 2003): only above the CMC do surfactant
    micelles form and the wettability-altering / IFT action operate fully.
    """
    return bool(concentration_wt_pct > cmc)


def incremental_recovery(recovery_nanofluid, recovery_surfactant):
    """Incremental oil recovery of the nanofluid over the plain surfactant

        dR = R_nanofluid - R_surfactant,

    the paper's headline result that adding SiO2 nanoparticles to the surfactant
    recovers ~10% IOIP more oil than the surfactant alone (nanofluid > ~50% vs
    surfactant ~46% IOIP).
    """
    return recovery_nanofluid - recovery_surfactant


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Nanofluid Spontaneous Imbibition")
    print("=" * 60)

    # Young's law: raising the water-oil IFT (nanofluid: 2.65 -> 9.21 mN/m)
    # drives cos(theta) toward zero, so the contact angle increases
    theta_low_ift = young_contact_angle(sigma_so=0.030, sigma_sw=0.028, sigma_wo=0.00265)
    theta_high_ift = young_contact_angle(sigma_so=0.030, sigma_sw=0.028, sigma_wo=0.00921)
    print(f"  theta(low IFT)={theta_low_ift:.1f} deg   theta(high IFT)={theta_high_ift:.1f} deg")
    assert theta_high_ift > theta_low_ift  # higher IFT -> larger contact angle

    # cos(theta) decreases as the water-oil IFT rises (paper's linear trend)
    assert (cos_contact_angle(0.030, 0.028, 0.00921)
            < cos_contact_angle(0.030, 0.028, 0.00265))

    # Work of adhesion grows as the surface becomes more water-wet (smaller theta)
    w_ww = work_of_adhesion(0.00921, 21.0)    # strongly water-wet (nanofluid)
    w_ow = work_of_adhesion(0.00921, 134.0)   # oil-wet
    print(f"  work of adhesion: water-wet={w_ww:.4f}  oil-wet={w_ow:.4f} J/m2")
    assert w_ww > w_ow

    # Wettability classification spans the range
    assert wettability_class(21) == "water-wet"
    assert wettability_class(95) == "intermediate"
    assert wettability_class(134) == "oil-wet"

    # Capillary force flips sign with wettability: oil-wet (theta>90) resists
    # imbibition (Pc<0), water-wet (theta<90, nanofluid) drives it (Pc>0), and
    # raising the oil-water IFT (2.65 -> 9.21 mN/m) increases the driving Pc
    pc_oilwet = capillary_pressure(sigma_wo=0.00265, contact_angle_deg=134.0, pore_radius=5e-6)
    pc_ww_low = capillary_pressure(sigma_wo=0.00265, contact_angle_deg=21.0, pore_radius=5e-6)
    pc_ww_high = capillary_pressure(sigma_wo=0.00921, contact_angle_deg=21.0, pore_radius=5e-6)
    print(f"  Pc: oil-wet={pc_oilwet:.1f}  water-wet(low IFT)={pc_ww_low:.1f}  water-wet(high IFT)={pc_ww_high:.1f} Pa")
    assert pc_oilwet < 0 < pc_ww_low < pc_ww_high   # barrier -> driving force

    # Imbibition recovery rises monotonically to the plateau; nanofluid (higher
    # final recovery) beats surfactant which beats brine
    t = np.linspace(0, 10, 50)
    r_brine = imbibition_recovery(t, final_recovery=0.043, rate=1.0)
    r_surf = imbibition_recovery(t, final_recovery=0.46, rate=0.5)
    r_nano = imbibition_recovery(t, final_recovery=0.52, rate=0.5)
    print(f"  final recovery: brine={r_brine[-1]:.2f}  surf={r_surf[-1]:.2f}  nano={r_nano[-1]:.2f}")
    assert r_nano[-1] > r_surf[-1] > r_brine[-1]
    assert np.all(np.diff(r_nano) >= 0) and r_nano[0] == 0.0

    # CMC check: the imbibition surfactant concentration (1 wt%) is above the
    # measured CMC (~0.45 wt%); a trace dose is below it
    assert above_cmc(1.0) and not above_cmc(0.1)
    assert OPTIMUM_NANOPARTICLE_CONC_GL == 3.0
    # Nanofluid adds ~10% IOIP over the surfactant alone (paper's headline);
    # at the (final-recovery) plateaus the increment is exactly 0.52 - 0.46
    assert np.isclose(incremental_recovery(0.52, 0.46), 0.06)
    d_rec = incremental_recovery(r_nano[-1], r_surf[-1])
    print(f"  incremental recovery (nano - surf) = {d_rec*100:.1f}% IOIP")
    assert d_rec > 0 and np.isclose(d_rec, 0.06, atol=5e-3)

    # Mechanism test: the curved (exponential) imbibition response is capillary-
    # dominated, a straight-line gravity drainage response is gravity-dominated
    assert imbibition_mechanism(r_nano) == "capillary"
    assert imbibition_mechanism(np.linspace(0.0, 0.5, 50)) == "gravity"

    print("  PASS")
    return {"theta_nanofluid": float(theta_high_ift), "R_nano": float(r_nano[-1]),
            "work_adhesion_ww": float(w_ww), "Pc_water_wet": float(pc_ww_high)}


if __name__ == "__main__":
    test_all()
