"""
Article 1 (Tutorial): Capillary Pressure Tutorial Part 2 - The Path Out of the
                      Jungle
Thomas (2018)
DOI: 10.30632/PJV59N5-2018t1

Part 2 of the capillary-pressure tutorial shows how to use laboratory capillary-
pressure data: convert it from the lab fluid system to reservoir conditions
(through the interfacial-tension and contact-angle scaling), build the
saturation-height function above the free-water level, and normalize curves
across rocks with the Leverett J-function.

Implements:

  - Lab-to-reservoir capillary-pressure conversion (sigma*cos(theta) scaling)
  - Saturation-height function  h = Pc/((rho_w - rho_hc)*g)
  - Leverett J-function  J = Pc*sqrt(k/phi)/(sigma*cos(theta))
  - Drainage/imbibition hysteresis (offset irreducible saturations)

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the capillary-pressure conversion relations the tutorial teaches.  SI units.
"""

import numpy as np

G_ACCEL = 9.81


# ---------------------------------------------- conversion --------------

def lab_to_reservoir_pc(pc_lab, sigma_lab, theta_lab, sigma_res, theta_res):
    """Convert lab Pc to reservoir Pc via the |sigma*cos theta| ratio.

        Pc_res = Pc_lab * |sigma_res*cos(theta_res)| / |sigma_lab*cos(theta_lab)|
    Magnitudes are used so a non-wetting lab system (e.g. air-mercury, theta>90)
    converts to a positive reservoir capillary pressure.
    """
    return (np.asarray(pc_lab, float)
            * abs(sigma_res * np.cos(np.radians(theta_res)))
            / abs(sigma_lab * np.cos(np.radians(theta_lab))))


def saturation_height(pc, rho_w, rho_hc):
    """Height above the free-water level  h = Pc/((rho_w - rho_hc)*g)  (m)."""
    return np.asarray(pc, float) / ((rho_w - rho_hc) * G_ACCEL)


def leverett_j(pc, k, phi, sigma, theta_deg):
    """Leverett J-function  J = Pc*sqrt(k/phi)/(sigma*cos(theta))."""
    return (np.asarray(pc, float) * np.sqrt(k / phi)
            / (sigma * np.cos(np.radians(theta_deg))))


def imbibition_sw(drainage_sw, sor=0.25):
    """Imbibition saturation with trapping  Sw_imb = Sw_drain + Sor*(1 - Sw_drain).

    Hysteresis: the imbibition curve sits above the drainage curve because of
    residual non-wetting-phase trapping.
    """
    return drainage_sw + sor * (1.0 - np.asarray(drainage_sw, float))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1 (Tutorial): Capillary Pressure, Part 2")
    print("=" * 60)

    # Lab (air-mercury, high sigma) to reservoir (oil-water) lowers Pc
    pc_res = lab_to_reservoir_pc(1e6, sigma_lab=0.480, theta_lab=140.0,
                                 sigma_res=0.030, theta_res=30.0)
    print(f"  Pc lab 1e6 -> reservoir = {pc_res:.0f} Pa")
    assert pc_res < 1e6 and pc_res > 0

    # Saturation-height rises with Pc above the FWL
    assert saturation_height(1e5, 1000.0, 700.0) > saturation_height(1e4, 1000.0, 700.0)

    # Leverett J collapses two rocks (same throat radius) onto one J
    sigma, theta = 0.03, 30.0
    pc = 2 * sigma * np.cos(np.radians(theta)) / 2e-6
    jA = leverett_j(pc, 1e-13, 0.20, sigma, theta)
    jB = leverett_j(pc, 4e-13, 0.20, sigma, theta)
    print(f"  Leverett J rockA/rockB = {jA:.3f} / {jB:.3f}")
    assert jB > jA > 0                            # different k -> J differs at same Pc

    # Imbibition curve sits above drainage (hysteresis / trapping)
    sw_d = np.array([0.2, 0.4, 0.6])
    sw_i = imbibition_sw(sw_d, sor=0.25)
    print(f"  Sw drainage/imbibition = {sw_d} / {np.array2string(sw_i, precision=2)}")
    assert np.all(sw_i > sw_d)
    print("  PASS")
    return {"pc_res": float(pc_res), "J_rockA": float(jA)}


if __name__ == "__main__":
    test_all()
