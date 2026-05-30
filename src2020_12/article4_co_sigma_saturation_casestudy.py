"""
Article 4: New Generation of Pulsed-Neutron Multidetector Comparison in a
           Challenging Multistack Clastic Reservoir - A Case Study in a Brown
           Field, Malaysia
Johare, Mohd Amin, Prasodjo, Afandi, Din (2020)
DOI: 10.30632/PJV61N6-2020a4

A field case study comparing a new-generation multidetector pulsed-neutron tool
against an older tool in a multistacked clastic reservoir, evaluated by
carbon/oxygen (C/O) ratio (saturation independent of formation-water salinity)
and by sigma (capture cross section, salinity dependent).  The paper publishes
no numbered equations; this module implements the standard pulsed-neutron
saturation relations the comparison relies on.

Implements:

  - C/O-ratio oil saturation from water/oil response endpoints (salinity-free)
  - Sigma (capture) water saturation from the porosity balance
  - Near/far multidetector ratio as a qualitative gas/quality indicator

Note: this issue's PDF text layer drops typeset glyphs and the paper itself has
no numbered equations, so these are standard-form pulsed-neutron saturation
relations the case study applies.  Sigma in capture units (c.u.).
"""

import numpy as np


# ---------------------------------------------- C/O saturation ----------

def co_oil_saturation(cor, cor_water, cor_oil):
    """Oil saturation from the C/O ratio by linear interpolation.

        So = (COR - COR_water) / (COR_oil - COR_water)
    COR_water / COR_oil are the wet and oil-filled response lines (porosity and
    lithology dependent, salinity independent).  Clipped to [0, 1].
    """
    so = (np.asarray(cor, float) - cor_water) / (cor_oil - cor_water)
    return np.clip(so, 0.0, 1.0)


# ---------------------------------------------- sigma saturation --------

def sigma_water_saturation(sigma_log, phi, sigma_ma, sigma_w, sigma_hc):
    """Water saturation from sigma via the volumetric porosity balance.

        Sw = (Sigma_log - Sigma_ma*(1-phi) - phi*Sigma_hc)
             / (phi*(Sigma_w - Sigma_hc))
    Clipped to [0, 1].
    """
    phi = np.asarray(phi, float)
    num = np.asarray(sigma_log, float) - sigma_ma * (1.0 - phi) - phi * sigma_hc
    sw = num / (phi * (sigma_w - sigma_hc))
    return np.clip(sw, 0.0, 1.0)


# ---------------------------------------------- multidetector -----------

def detector_ratio(near_count, far_count):
    """Near/far count-rate ratio (porosity / gas qualitative indicator)."""
    return np.asarray(near_count, float) / np.asarray(far_count, float)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: C/O and Sigma Saturation (Malaysia)")
    print("=" * 60)

    # C/O saturation: salinity-independent oil indicator
    cor_w, cor_o = 0.18, 0.42
    so_oil = co_oil_saturation(0.38, cor_w, cor_o)
    so_wet = co_oil_saturation(0.19, cor_w, cor_o)
    print(f"  So (oil sand / wet)    = {so_oil:.2f} / {so_wet:.2f}")
    assert so_oil > 0.7 and so_wet < 0.1

    # Sigma saturation in a saline clastic: high sigma (saline water) -> high Sw
    sw_wet = sigma_water_saturation(28.0, 0.25, sigma_ma=8.0, sigma_w=60.0, sigma_hc=22.0)
    sw_oil = sigma_water_saturation(16.0, 0.25, sigma_ma=8.0, sigma_w=60.0, sigma_hc=22.0)
    print(f"  Sw (saline wet / oil)  = {sw_wet:.2f} / {sw_oil:.2f}")
    assert sw_wet > sw_oil
    # the two saturation answers (1-So vs Sw) agree to within the model spread
    assert abs((1.0 - so_oil) - sw_oil) < 0.35

    # Multidetector ratio responds to a gas (low-HI) zone
    r_liquid = detector_ratio(1000.0, 400.0)
    r_gas = detector_ratio(1000.0, 600.0)         # gas raises far counts
    print(f"  near/far ratio liquid / gas = {r_liquid:.2f} / {r_gas:.2f}")
    assert r_gas < r_liquid
    print("  PASS")
    return {"so_oil": float(so_oil), "sw_wet": float(sw_wet),
            "sw_oil": float(sw_oil)}


if __name__ == "__main__":
    test_all()
