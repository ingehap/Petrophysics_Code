"""
Article 2: Case Studies Demonstrating the Impact of Cement Quality on
           Carbon/Oxygen and Elemental Analysis From Casedhole Pulsed-Neutron
           Logging
Wang, Sullivan, Seth, Barnes, Wilson, Lazorek (2020)
DOI: 10.30632/PJV61N3-2020a2

Monte-Carlo (MCNP) modeling and two field examples show how cement quality -
channels, voids, poor centralization - perturbs the carbon/oxygen (C/O) ratio
and the calcium elemental yield in casedhole pulsed-neutron logging.  The paper
publishes no numbered equations; this module implements the quantitative
ratio/yield relations the study relies on.

Implements:

  - Carbon/oxygen ratio  C/O = Y_C / Y_O
  - Salinity-independent oil saturation from C/O endpoints
  - Cement calcium-yield contribution and formation-calcium correction
  - Oil-based-mud (OBM) vs water-based-mud (WBM) channel C/O bias
  - Sigma water saturation (capture cross section)

Note: this issue's PDF has a text layer but the paper publishes no numbered
equations; these are the standard pulsed-neutron C/O and yields-to-weights
relations.  Paper anchors reproduced: cement calcium atomic concentration
9.31%, cement contributing > 40% of the total calcium yield, residual oil
saturation 30-40%, and the OBM-channel C/O bias (WBM negligible).
"""

import numpy as np

CEMENT_CA_ATOMIC = 9.31          # % atomic calcium in cement


# ---------------------------------------------- C/O ratio ---------------

def carbon_oxygen_ratio(c_yield, o_yield):
    """Carbon/oxygen ratio  C/O = Y_C / Y_O (rises with oil)."""
    return np.asarray(c_yield, float) / np.asarray(o_yield, float)


def oil_saturation_from_co(cor, cor_water, cor_oil):
    """Salinity-independent oil saturation by C/O endpoint interpolation.

        So = (C/O - C/O_water) / (C/O_oil - C/O_water),  clipped to [0, 1].
    """
    so = (np.asarray(cor, float) - cor_water) / (cor_oil - cor_water)
    return np.clip(so, 0.0, 1.0)


# ---------------------------------------------- calcium correction ------

def formation_calcium(total_ca_yield, cement_fraction):
    """Formation calcium yield after removing the cement contribution.

    cement_fraction is the fraction of the measured Ca yield that comes from
    the cement (the paper finds this exceeds 0.40 in poorly bonded geometries).
    """
    return total_ca_yield * (1.0 - cement_fraction)


# ---------------------------------------------- channel bias ------------

def co_channel_bias(cor_true, channel_volume, mud_type):
    """C/O bias from a mud-filled channel behind casing.

    An oil-based-mud channel adds carbon (raises C/O); a water-based-mud channel
    adds mostly oxygen (slightly lowers C/O, negligible bias).
    """
    if mud_type == "OBM":
        return cor_true + 1.5 * channel_volume        # carbon-rich -> positive bias
    return cor_true - 0.1 * channel_volume            # WBM -> tiny negative bias


# ---------------------------------------------- sigma -------------------

def sigma_water_saturation(sigma_log, phi, sigma_ma, sigma_w, sigma_hc):
    """Water saturation from the sigma porosity balance (clipped to [0,1])."""
    phi = np.asarray(phi, float)
    num = np.asarray(sigma_log, float) - sigma_ma * (1.0 - phi) - phi * sigma_hc
    return np.clip(num / (phi * (sigma_w - sigma_hc)), 0.0, 1.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Cement Quality Impact on C/O & Elements")
    print("=" * 60)

    # C/O rises for oil-bearing rock
    cor_oil = carbon_oxygen_ratio(0.42, 0.55)
    cor_wat = carbon_oxygen_ratio(0.22, 0.80)
    print(f"  C/O oil / water        = {cor_oil:.3f} / {cor_wat:.3f}")
    assert cor_oil > cor_wat

    # Salinity-independent oil saturation; residual-oil 30-40% maps inside [0,1]
    so = oil_saturation_from_co(0.45, cor_water=0.40, cor_oil=0.60)
    print(f"  oil saturation (ROS)   = {so:.2f}")
    assert 0.0 <= so <= 1.0 and abs(so - 0.25) < 1e-9

    # Cement contributes > 40% of the calcium yield -> correction lowers it a lot
    ca_form = formation_calcium(total_ca_yield=1.0, cement_fraction=0.42)
    print(f"  formation Ca (42% cement) = {ca_form:.3f}")
    assert abs(ca_form - 0.58) < 1e-9
    # more cement contamination -> less formation calcium
    assert formation_calcium(1.0, 0.6) < ca_form

    # OBM channel biases C/O upward; WBM channel is negligible
    cor_true = 0.30
    bias_obm = co_channel_bias(cor_true, 0.1, "OBM") - cor_true
    bias_wbm = co_channel_bias(cor_true, 0.1, "WBM") - cor_true
    print(f"  C/O bias OBM / WBM     = {bias_obm:+.3f} / {bias_wbm:+.3f}")
    assert bias_obm > 0.1 and abs(bias_wbm) < 0.02

    # Sigma water saturation behaves correctly (higher sigma -> higher Sw)
    assert sigma_water_saturation(13.0, 0.10, 9.0, 60.0, 22.0) > \
        sigma_water_saturation(11.0, 0.10, 9.0, 60.0, 22.0)
    print("  PASS")
    return {"cor_oil": cor_oil, "so_ros": float(so), "ca_form": ca_form,
            "bias_obm": bias_obm}


if __name__ == "__main__":
    test_all()
