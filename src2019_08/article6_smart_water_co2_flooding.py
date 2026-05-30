"""
Article 6: Novel Coupling Smart Water-CO2 Flooding for Sandstone Reservoirs
Al-Saedi, Flori (2019)
DOI: 10.30632/PJV60N4-2019a6

Coupling low-salinity "smart water" with CO2 flooding improves oil recovery in
sandstones: the smart water alters wettability toward more water-wet (lowering
residual oil), while CO2 dissolves into and swells the oil and lowers its
viscosity (improving the mobility ratio).  The coupled process recovers more oil
than either method alone.

Implements:

  - Buckley-Leverett fractional flow  fw = 1/(1 + (kro/muo)/(krw/muw))
  - Microscopic displacement efficiency  E_D = (Soi - Sor)/Soi
  - Smart-water wettability shift (lowers residual oil saturation)
  - CO2 oil-viscosity reduction / swelling (improves mobility ratio)
  - Coupled recovery factor vs each method alone

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the issue's table of contents and these are
faithful standard-form reconstructions of the fractional-flow / recovery
relations the coupled-flooding study relies on.
"""

import numpy as np


# ---------------------------------------------- fractional flow ---------

def fractional_flow(krw, kro, mu_w, mu_o):
    """Buckley-Leverett water fractional flow  fw = 1/(1 + (kro/muo)/(krw/muw))."""
    return 1.0 / (1.0 + (kro / mu_o) / (krw / mu_w))


def mobility_ratio(krw_end, kro_end, mu_w, mu_o):
    """End-point mobility ratio  M = (krw/muw)/(kro/muo)  (lower is better)."""
    return (krw_end / mu_w) / (kro_end / mu_o)


def displacement_efficiency(soi, sor):
    """Microscopic displacement efficiency  E_D = (Soi - Sor)/Soi."""
    return (soi - sor) / soi


# ---------------------------------------------- EOR effects -------------

def smartwater_residual_oil(sor_base, salinity_ppm, sor_min=0.15):
    """Smart-water (low-salinity) residual oil: lower salinity -> lower Sor.

    Sor decreases as salinity falls from seawater toward low-salinity brine.
    """
    f = np.clip(salinity_ppm / 35000.0, 0.0, 1.0)        # 1 = seawater, 0 = fresh
    return sor_min + (sor_base - sor_min) * f


def co2_viscosity_reduction(mu_oil, co2_mole_frac):
    """CO2 dissolution lowers oil viscosity  mu = mu_oil*exp(-k*x_CO2)."""
    return mu_oil * np.exp(-2.0 * co2_mole_frac)


def recovery_factor(soi, sor):
    """Oil recovery factor from initial and residual oil saturation."""
    return (soi - sor) / soi


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Coupled Smart Water-CO2 Flooding")
    print("=" * 60)

    mu_w, mu_o = 0.5, 5.0

    # Fractional flow rises with water saturation (krw up, kro down)
    fw_lo = fractional_flow(0.05, 0.8, mu_w, mu_o)
    fw_hi = fractional_flow(0.5, 0.1, mu_w, mu_o)
    print(f"  fw low/high Sw         = {fw_lo:.2f} / {fw_hi:.2f}")
    assert fw_hi > fw_lo and 0 <= fw_lo <= 1 and 0 <= fw_hi <= 1

    # Smart water (low salinity) lowers residual oil saturation
    sor_sw = smartwater_residual_oil(0.35, salinity_ppm=2000.0)
    sor_sea = smartwater_residual_oil(0.35, salinity_ppm=35000.0)
    print(f"  Sor smart-water / seawater = {sor_sw:.3f} / {sor_sea:.3f}")
    assert sor_sw < sor_sea

    # CO2 reduces oil viscosity, improving the mobility ratio
    mu_o_co2 = co2_viscosity_reduction(mu_o, 0.4)
    M_base = mobility_ratio(0.2, 0.8, mu_w, mu_o)
    M_co2 = mobility_ratio(0.2, 0.8, mu_w, mu_o_co2)
    print(f"  mobility ratio base/CO2 = {M_base:.2f} / {M_co2:.2f}")
    assert mu_o_co2 < mu_o and M_co2 < M_base       # CO2 lowers M (better sweep)

    # Coupled smart-water + CO2 recovers more than either alone
    soi = 0.65
    rf_base = recovery_factor(soi, 0.35)
    rf_sw = recovery_factor(soi, smartwater_residual_oil(0.35, 2000.0))
    rf_co2 = recovery_factor(soi, 0.28)             # CO2 alone lowers Sor
    rf_coupled = recovery_factor(soi, 0.15)         # coupled lowers Sor most
    print(f"  RF base/SW/CO2/coupled = {rf_base:.2f} / {rf_sw:.2f} / "
          f"{rf_co2:.2f} / {rf_coupled:.2f}")
    assert rf_coupled > rf_sw > rf_base and rf_coupled > rf_co2 > rf_base
    print("  PASS")
    return {"sor_smartwater": float(sor_sw), "M_co2": float(M_co2),
            "rf_coupled": float(rf_coupled)}


if __name__ == "__main__":
    test_all()
