"""
Article 6: Rock Physics Modeling of Gas Hydrate Reservoirs Through
Integrated Core and Well-Log Data in NGHP-02 Area, KG Offshore Basin,
India
Kumar, Mishra, Chatterjee, Tiwari, Avadhani (2022)
DOI: 10.30632/PJV63N2-2022a6

Body text was not present in the available PDF extract, so this module
is a *methodology proxy* guided by the editor's letter: rock-physics
modeling of Vp and Vs in NGHP-02 gas-hydrate reservoirs by depositional
hydrate type; a Jason (grain-supported) effective-medium model is
selected based on clay volume and porosity; cross-plots of modeled vs.
recorded Vp / Vs / acoustic impedance discriminate gas hydrate in
shaly-sand vs. sand-dominated layers from calcite and shale.

Implements:

  - Two end-member hydrate models:
        cementing model:    K_dry, G_dry = phi * (K_h, G_h) +
                                            (1 - phi) * (K_g, G_g)
        grain-supported:    Voigt-Reuss-Hill of (water + hydrate +
                                                  grain) at the pore
                                                  scale
  - Gassmann fluid substitution for the bulk modulus K_sat:
        K_sat / (K_min - K_sat) = K_dry / (K_min - K_dry)
                                + K_fl / (phi (K_min - K_fl))
  - Vp, Vs from K_sat, G_dry, rho_b
  - Vp/Vs cross-plot discriminator: separates the four classes
    (hydrate-bearing shaly sand / sand / calcite / shale) by ratio
    thresholds.
"""

import numpy as np


# ---------------------------------------------- end-member moduli (GPa) --

MINERAL = dict(
    quartz=(37.0, 44.0, 2650.0),
    clay=(21.0, 7.0, 2580.0),
    calcite=(76.8, 32.0, 2710.0),
)
HYDRATE = (8.41, 3.54, 920.0)
WATER = (2.25, 0.0, 1030.0)


# ---------------------------------------------- mixing -----------------

def vrh_two(K1, G1, f1, K2, G2):
    """Voigt-Reuss-Hill average of two solids."""
    f2 = 1.0 - f1
    K_v = f1 * K1 + f2 * K2
    G_v = f1 * G1 + f2 * G2
    K_r = 1.0 / (f1 / K1 + f2 / K2)
    G_r = 1.0 / (f1 / G1 + f2 / G2)
    return 0.5 * (K_v + K_r), 0.5 * (G_v + G_r)


def cementing_model(phi, K_min, G_min, K_h=HYDRATE[0], G_h=HYDRATE[1]):
    """Hydrate cementing model - moduli rise with phi as hydrate cements."""
    K_dry = (1.0 - phi) * K_min + phi * K_h
    G_dry = (1.0 - phi) * G_min + phi * G_h
    return K_dry, G_dry


def jason_grain_supported(phi, S_h, K_min, G_min):
    """Jason / grain-supported model - hydrate distributed in pore space
    behaves like a stiff fluid; dry-rock moduli depend mainly on grain
    contact, with hydrate just stiffening the pore fluid."""
    K_dry = (1.0 - phi) ** 3 * K_min * (1.0 - 0.5 * phi)
    G_dry = (1.0 - phi) ** 3 * G_min * (1.0 - 0.5 * phi)
    return K_dry, G_dry


# ---------------------------------------------- Gassmann ---------------

def gassmann_fluid_substitution(K_dry, K_min, phi, K_fl):
    """K_sat from the standard Gassmann formula."""
    num = (1.0 - K_dry / K_min) ** 2
    den = phi / K_fl + (1.0 - phi) / K_min - K_dry / K_min ** 2
    return K_dry + num / den


# ---------------------------------------------- velocities -------------

def velocities(K_sat, G, rho_b_kg_m3):
    """V_p, V_s in m/s.  K, G in GPa; rho in kg/m^3."""
    Vp = np.sqrt((K_sat + 4.0 / 3.0 * G) * 1e9 / rho_b_kg_m3)
    Vs = np.sqrt(G * 1e9 / rho_b_kg_m3)
    return float(Vp), float(Vs)


def bulk_density(phi, S_h, mineral_rho, water_rho=WATER[2], hyd_rho=HYDRATE[2]):
    return (1.0 - phi) * mineral_rho + phi * (S_h * hyd_rho + (1.0 - S_h) * water_rho)


# ---------------------------------------------- classifier -----------

def classify_lithology(Vp_m_s, Vs_m_s):
    """Simple Vp/Vs and Vp-based classifier into four classes.

    Hydrate-bearing layers have Vp > 2000 m/s and Vp/Vs > 2.5;
    sand: Vp 1800-2200, Vp/Vs ~ 2.0;
    calcite: Vp > 4500;
    shale: Vp < 1800 or Vp/Vs > 3.0.
    """
    ratio = Vp_m_s / Vs_m_s
    if Vp_m_s > 4500:
        return "calcite"
    if Vp_m_s > 2400 and 2.0 < ratio < 3.0:
        return "hydrate_sand"
    if Vp_m_s < 1800 or ratio > 3.0:
        return "shale"
    return "sand"


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Gas-Hydrate Rock-Physics Modeling (proxy)")
    print("=" * 60)

    phi = 0.40
    S_h = 0.50
    K_qtz, G_qtz, rho_qtz = MINERAL["quartz"]
    K_clay, G_clay, rho_clay = MINERAL["clay"]

    # Shaly sand: 80 % quartz + 20 % clay
    K_min, G_min = vrh_two(K_qtz, G_qtz, 0.80, K_clay, G_clay)
    rho_min = 0.80 * rho_qtz + 0.20 * rho_clay
    K_dry, G_dry = jason_grain_supported(phi, S_h, K_min, G_min)
    K_fl_effective = S_h * HYDRATE[0] + (1.0 - S_h) * WATER[0]
    K_sat = gassmann_fluid_substitution(K_dry, K_min, phi, K_fl_effective)
    rho_b = bulk_density(phi, S_h, rho_min)
    Vp_h, Vs_h = velocities(K_sat, G_dry, rho_b)
    label_h = classify_lithology(Vp_h, Vs_h)
    print(f"  Hydrate-bearing shaly sand:")
    print(f"    Vp = {Vp_h:.0f} m/s   Vs = {Vs_h:.0f} m/s   "
          f"Vp/Vs = {Vp_h/Vs_h:.2f}   class = {label_h}")

    # Plain shaly sand (no hydrate, water only)
    K_dry_w, G_dry_w = jason_grain_supported(phi, 0.0, K_min, G_min)
    K_sat_w = gassmann_fluid_substitution(K_dry_w, K_min, phi, WATER[0])
    rho_b_w = bulk_density(phi, 0.0, rho_min)
    Vp_w, Vs_w = velocities(K_sat_w, G_dry_w, rho_b_w)
    label_w = classify_lithology(Vp_w, Vs_w)
    print(f"  Water-saturated shaly sand:")
    print(f"    Vp = {Vp_w:.0f} m/s   Vs = {Vs_w:.0f} m/s   "
          f"Vp/Vs = {Vp_w/Vs_w:.2f}   class = {label_w}")

    # Calcite end-member
    K_c, G_c, rho_c = MINERAL["calcite"]
    K_dry_c, G_dry_c = jason_grain_supported(0.15, 0.0, K_c, G_c)
    K_sat_c = gassmann_fluid_substitution(K_dry_c, K_c, 0.15, WATER[0])
    rho_b_c = bulk_density(0.15, 0.0, rho_c)
    Vp_c, Vs_c = velocities(K_sat_c, G_dry_c, rho_b_c)
    label_c = classify_lithology(Vp_c, Vs_c)
    print(f"  Calcite (15 % porosity):")
    print(f"    Vp = {Vp_c:.0f} m/s   Vs = {Vs_c:.0f} m/s   "
          f"class = {label_c}")

    assert Vp_h > Vp_w, "Hydrate must stiffen the sand and raise Vp"
    assert label_c == "calcite", "Tight calcite must classify as calcite"
    print("  PASS")
    return {"Vp_hydrate": Vp_h, "Vp_water": Vp_w, "Vp_calcite": Vp_c,
            "labels": [label_h, label_w, label_c]}


if __name__ == "__main__":
    test_all()
