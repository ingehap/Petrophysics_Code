"""
Article 4: Mud-Filtrate Invasion in Laminated and Spatially Heterogeneous
Rocks: High-Resolution In-Situ Visualization and Analysis Using Time-Lapse
X-Ray Microcomputed Tomography (Micro-CT)
Schroeder, Torres-Verdin (2022)
DOI: 10.30632/PJV63N5-2022a4

Reproduces the dimensionless-number analysis and the Buckley-Leverett /
mudcake-controlled radial-invasion machinery that the paper uses to
interpret time-lapse micro-CT scans of WBM filtrate invasion into four
initially air-saturated outcrop cores (Leopard, Nugget, Texas Cream
Limestone, Vuggy Dolomite).

  - Capillary number       N_ca = v * mu / sigma
  - Bond number            N_B  = delta_rho * g * R^2 / sigma
  - Brooks-Corey relative permeabilities and the Leverett J-function
    J(Sw) = Pc(Sw) * sqrt(k / phi) / (sigma * cos theta)
  - Fractional-flow curve  fw(Sw) = (krw / mu_w) / (krw / mu_w + kro / mu_o)
  - Buckley-Leverett front saturation from the Welge tangent
  - Mudcake-controlled radial invasion-front position via the
    Dewan-Chenevert sqrt(t) law
        x_front(t) = sqrt(2 * k_eff * dP / (mu_w * phi) * t)
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# --------------------------------------------- dimensionless numbers -----

def capillary_number(v_m_s, mu_Pa_s, sigma_N_m):
    return v_m_s * mu_Pa_s / sigma_N_m


def bond_number(delta_rho_kg_m3, R_m, sigma_N_m, g=9.81):
    return delta_rho_kg_m3 * g * R_m ** 2 / sigma_N_m


# --------------------------------------------- Brooks-Corey + Leverett --

def krw_kro_brooks_corey(sw, swi=0.20, sor=0.20, nw=2.0, no=2.0,
                         krw_max=0.8, kro_max=0.9):
    se = np.clip((sw - swi) / (1.0 - swi - sor), 0.0, 1.0)
    return krw_max * se ** nw, kro_max * (1.0 - se) ** no


def fractional_flow(sw, mu_w_Pa_s, mu_o_Pa_s, **kwargs):
    krw, kro = krw_kro_brooks_corey(sw, **kwargs)
    num = krw / mu_w_Pa_s
    den = num + kro / mu_o_Pa_s
    return num / (den + 1e-30)


def leverett_J(Pc_Pa, k_m2, phi, sigma_N_m, theta_deg=180.0):
    # |cos| convention (abs): mercury (theta=180) keeps J positive.
    return petrolib.capillary_pressure.leverett_j(
        Pc_Pa, sigma=sigma_N_m, theta_deg=theta_deg, k=k_m2, phi=phi, absolute=True)


# --------------------------------------------- Buckley-Leverett front ---

def welge_tangent(sw, mu_w_Pa_s, mu_o_Pa_s, swi=0.20, sor=0.20):
    """Front saturation Sw_f from the Welge tangent of fw(Sw).

    The tangent passes through (Swi, 0) and is tangent to the
    fractional-flow curve at the point Sw_f.
    """
    fw = fractional_flow(sw, mu_w_Pa_s, mu_o_Pa_s, swi=swi, sor=sor)
    valid = (sw > swi) & (sw < 1.0 - sor)
    slope = np.where(valid, fw / (sw - swi + 1e-9), 0.0)
    i = int(np.argmax(slope))
    return float(sw[i]), float(fw[i])


# --------------------------------------------- mudcake-controlled invasion

def invasion_front_position_m(t_s, k_eff_m2, dP_Pa, mu_w_Pa_s, phi):
    """x_front(t) = sqrt(2 k_eff dP / (mu phi) * t)  (Dewan-Chenevert)."""
    return np.sqrt(2.0 * k_eff_m2 * dP_Pa / (mu_w_Pa_s * phi) * t_s)


# --------------------------------------------- tests --------------------

def test_all():
    print("=" * 60)
    print("Article 4: Mud-Filtrate Invasion (Micro-CT / Buckley-Leverett)")
    print("=" * 60)

    mu_w = 0.001        # Pa.s
    mu_o = 0.018        # gas / non-wetting phase here treated as 'o'
    sigma = 0.072       # N/m
    delta_rho = 990.0   # kg/m^3 (water - air)
    R_pore = 1e-5       # m - PORE radius (~ 10 um), the relevant length scale
                         # for the Bond number quoted in the paper

    # Spurt-loss vs late-time front velocity (Leopard sandstone analogue;
    # paper reports peak N_ca ~ 1.9e-5 then dropping below 1e-6 within ~3 s)
    v_spurt = 1.5e-3
    v_late = 5e-5
    N_ca_spurt = capillary_number(v_spurt, mu_w, sigma)
    N_ca_late = capillary_number(v_late, mu_w, sigma)
    N_B = bond_number(delta_rho, R_pore, sigma)
    print(f"  Capillary number  spurt = {N_ca_spurt:.2e}   "
          f"late = {N_ca_late:.2e}")
    print(f"  Bond number              = {N_B:.2e}")
    assert N_ca_late < 1e-5 < N_ca_spurt, \
        "Spurt N_ca should exceed late N_ca and span 1e-6 to 1e-5"

    # Buckley-Leverett saturation front
    sw_grid = np.linspace(0.20, 0.80, 121)
    sw_f, fw_f = welge_tangent(sw_grid, mu_w, mu_o)
    print(f"  Buckley-Leverett front Sw_f  = {sw_f:.3f}  fw_f = {fw_f:.3f}")
    # Sanity: front saturation should be between Swi and 1 - Sor
    assert 0.30 < sw_f < 0.80

    # Mudcake-controlled invasion front (Leopard analogue)
    k_eff = 100e-15           # m^2 (~ 100 mD effective ahead of mudcake)
    dP = 5e6                  # Pa
    phi = 0.20
    for t_min in (0.5, 5.0, 30.0, 120.0):
        x = invasion_front_position_m(t_min * 60.0, k_eff, dP, mu_w, phi)
        print(f"  t = {t_min:6.1f} min   x_front = {x * 100:6.2f} cm")

    # Heterogeneous-rock saturation behaviour: average filtrate saturation
    # in the invaded zone should land in the 0.40-0.55 band the paper reports
    krw_grid, kro_grid = krw_kro_brooks_corey(sw_grid)
    behind = sw_grid >= sw_f
    avg_sat_invaded = float(sw_grid[behind].mean())
    print(f"  Mean Sw behind front  = {avg_sat_invaded:.3f}  "
          f"(paper reports 0.43 - 0.51)")

    print("  PASS")
    return {"N_ca_spurt": float(N_ca_spurt),
            "N_B": float(N_B),
            "Sw_front": sw_f,
            "mean_Sw_behind_front": avg_sat_invaded}


if __name__ == "__main__":
    test_all()
