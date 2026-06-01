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
  - A first-order spontaneous-imbibition recovery curve toward a final plateau

Note: this issue's PDF dropped the displayed Young's law (Eq. 1) in extraction;
it is reconstructed in standard form.  No dimensionless-time, Bond-number or
Amott-index relation appears (the recovery analysis is qualitative), so the
recovery is modelled with the standard first-order capillary-imbibition curve.
Measured anchors: nanofluid raises oil-water IFT 2.65 -> 9.21 mN/m, drives the
oil-phase contact angle to ~134 deg, and lifts final recovery above ~50% IOIP.
"""

import numpy as np


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


# ---------------------------------------------- imbibition recovery --------------

def imbibition_recovery(t, final_recovery, rate):
    """First-order spontaneous-imbibition oil recovery toward a final plateau

        R(t) = R_final*(1 - exp(-rate*t)),

    the standard capillary-imbibition approach to the ultimate recovery R_final
    (fraction of IOIP), with a characteristic rate (1/time).
    """
    return final_recovery * (1.0 - np.exp(-rate * np.asarray(t, float)))


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

    # Imbibition recovery rises monotonically to the plateau; nanofluid (higher
    # final recovery) beats surfactant which beats brine
    t = np.linspace(0, 10, 50)
    r_brine = imbibition_recovery(t, final_recovery=0.043, rate=1.0)
    r_surf = imbibition_recovery(t, final_recovery=0.46, rate=0.5)
    r_nano = imbibition_recovery(t, final_recovery=0.52, rate=0.5)
    print(f"  final recovery: brine={r_brine[-1]:.2f}  surf={r_surf[-1]:.2f}  nano={r_nano[-1]:.2f}")
    assert r_nano[-1] > r_surf[-1] > r_brine[-1]
    assert np.all(np.diff(r_nano) >= 0) and r_nano[0] == 0.0
    print("  PASS")
    return {"theta_nanofluid": float(theta_high_ift), "R_nano": float(r_nano[-1]),
            "work_adhesion_ww": float(w_ww)}


if __name__ == "__main__":
    test_all()
