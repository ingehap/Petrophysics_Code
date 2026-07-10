"""
Article 5: Corrosion Behavior and Mechanism Analysis of Oilwell Cement
Under CO2 and H2S Conditions
Zhou, Zeng, Sun, Zhou, Lei, Wan, Luo, Wu, Zhang, Xiao (2022)
DOI: 10.30632/PJV63N5-2022a5

The paper exposes Class-G oilwell cement cylinders (25 mm dia x 30 mm,
density 1.90 g/cm^3, 15 % silica fume) to 12 % CO2 + 5 ppm H2S in
distilled water at 150 C and 75 MPa for 7, 14, and 30 days, and reports

    Day  7: k ~ 3e-4 mD,  partially uncorroded
    Day 14: k ~ 8e-3 mD,  partial corrosion
    Day 30: k ~ 6.46e-2 mD,  fully corroded; tensile strength 9.8 MPa

A ~ 200x permeability increase over 30 days.

This module implements:

  - The single labelled gas-Darcy form for steady-state permeability
    measurement (Eq. 1):

        k = (2 * Q * P0 * mu * L) / (A * (P1^2 - P2^2))

  - A diffusion-limited reaction-front model: corrosion depth x_f(t)
    advances as sqrt(t), x_f = K * sqrt(t).
  - A direct exponential-in-time permeability evolution k(t) = k_init *
    exp(B*t) fitted to the paper's three measurements - the simplest
    closed-form model that captures the ~200x rise in 30 days.
  - Tensile-strength loss as a linear function of corrosion fraction
    (rim-area-of-cylinder geometry).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# --------------------------------------------- gas Darcy (Eq. 1) ---------

def gas_darcy_permeability(Q_kg_s, P0_Pa, mu_Pa_s, L_m, A_m2, P1_Pa, P2_Pa):
    """k = (2 Q P0 mu L) / (A (P1^2 - P2^2))  (Eq. 1).  k in m^2."""
    return 2.0 * Q_kg_s * P0_Pa * mu_Pa_s * L_m \
           / (A_m2 * (P1_Pa ** 2 - P2_Pa ** 2))


# --------------------------------------------- corrosion-front geometry --

def corrosion_depth_mm(t_days, K_mm_per_sqrt_day=2.5):
    """Diffusion-limited reaction-front depth:  x_f = K sqrt(t)."""
    return petrolib.integrity_drilling.corrosion_front_depth(t_days, K_mm_per_sqrt_day)


def corrosion_fraction(t_days, plug_radius_mm=12.5, K=2.5):
    """Cylindrical-rim fraction that has been corroded."""
    x_f = corrosion_depth_mm(t_days, K)
    if x_f >= plug_radius_mm:
        return 1.0
    return float(1.0 - ((plug_radius_mm - x_f) / plug_radius_mm) ** 2)


# --------------------------------------------- permeability evolution ---

def permeability_mD(t_days, k_init_mD=6.0e-5, B_per_day=0.230):
    """Empirical exponential-in-time growth of plug permeability.

        k(t) = k_init * exp(B * t)

    Defaults are fitted to the paper's three measurements:
        k(7)  = 3.00e-4 mD
        k(14) = 1.50e-3 mD  (paper measured 8.0e-3; model underpredicts)
        k(30) = 6.00e-2 mD
    so the 7-to-30-day ratio is exp(B * 23) ~ 200x as reported.
    """
    return float(k_init_mD * np.exp(B_per_day * t_days))


# --------------------------------------------- tensile strength ---------

def tensile_strength_MPa(t_days, S0_MPa=18.0, S_min_MPa=9.5,
                         plug_radius_mm=12.5, K=2.5):
    """Strength loss as a linear function of corrosion fraction."""
    f = corrosion_fraction(t_days, plug_radius_mm, K)
    return float((1.0 - f) * S0_MPa + f * S_min_MPa)


# --------------------------------------------- tests ---------------------

def test_all():
    print("=" * 60)
    print("Article 5: Oilwell Cement Corrosion (CO2 + H2S)")
    print("=" * 60)

    # Eq. 1 sanity check
    Q = 5e-7
    k_m2 = gas_darcy_permeability(Q_kg_s=Q, P0_Pa=101_325.0, mu_Pa_s=1.8e-5,
                                  L_m=0.030, A_m2=np.pi * 0.0125 ** 2,
                                  P1_Pa=2e6, P2_Pa=1e6)
    print(f"  Eq. 1 sanity:  k for Q={Q:.0e}, dP=1->2 MPa = {k_m2 * 9.87e15:8.3f} mD")

    # Reaction-front geometry
    K = 12.5 / np.sqrt(25.0)   # mm/sqrt(day) - fully corroded ~day 25
    print(f"  Reaction-front coefficient K = {K:.3f} mm / sqrt(day)")

    print("\n  Days   corrosion frac    k (mD)         tens (MPa)")
    for t in (7.0, 14.0, 30.0):
        f = corrosion_fraction(t, K=K)
        k_t = permeability_mD(t)
        s_t = tensile_strength_MPa(t, K=K)
        print(f"    {t:4.0f}      {f:5.3f}        {k_t:.3e}       {s_t:5.2f}")

    k7 = permeability_mD(7.0)
    k30 = permeability_mD(30.0)
    s30 = tensile_strength_MPa(30.0, K=K)
    ratio = k30 / k7
    print(f"\n  k(30d) / k(7d) = {ratio:6.1f}x  (paper reports ~200x)")
    print(f"  k(7d)  = {k7:.3e} mD  (paper ~ 3.0e-4)")
    print(f"  k(30d) = {k30:.3e} mD  (paper ~ 6.5e-2)")
    print(f"  Tensile strength at 30 d = {s30:.2f} MPa  (paper ~ 9.8 MPa)")

    assert ratio > 100.0, "Permeability should rise ~100x or more by 30 d"
    assert 9.0 < s30 < 11.0, "Tensile strength should fall to ~10 MPa"
    print("  PASS")
    return {"k_7d_mD": k7, "k_30d_mD": k30, "ratio": ratio,
            "tensile_30d_MPa": s30}


if __name__ == "__main__":
    test_all()
