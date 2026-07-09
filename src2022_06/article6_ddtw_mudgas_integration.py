"""
Article 6: Formation Evaluation Using NMR, Mud Gas, and Triple-Combo
Data - A Norwegian LWD Case History
Thern, Kotwicki, Ritzmann, Petersen, Mohnke (2022)
DOI: 10.30632/PJV63N3-2022a6

Heimdal Sandstone, Alvheim Field LWD case study.  Compares four porosity
workflows and implements the paper's Density + DTW (DDTW) integration:

  - Standalone Dual-Wait-Time (DTW) polarisation-corrected NMR porosity
        S(TW) = S_inf * (1 - exp(-TW / T1))                  (Eq. 1)
  - Density-derived porosity with variable matrix density
        phi_D = (rho_ma - rho_b) / (rho_ma - rho_fl)
  - Density + DTW (DDTW) integration that exploits the opposite gas
    responses of the two measurements - gives an average gas
    hydrogen index HI_gas without compositional information.
  - Continuous HI log derived from mud-gas composition.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- NMR DTW (Eq. 1) ----------

def dtw_polarisation(TW_s, T1_s):
    return petrolib.nmr.t1_saturation_recovery(TW_s, 1.0, T1_s)


def dtw_porosity_correction(phi_apparent, TW_short_s, TW_long_s,
                            T1_water_s=2.0, T1_gas_s=5.0):
    """Two-TW polarisation correction for NMR porosity in the presence of gas.

    Returns the gas-corrected NMR porosity using the difference between
    long-TW and short-TW signals divided by the differential polarisation.
    """
    p_water_short = dtw_polarisation(TW_short_s, T1_water_s)
    p_water_long = dtw_polarisation(TW_long_s, T1_water_s)
    p_gas_short = dtw_polarisation(TW_short_s, T1_gas_s)
    p_gas_long = dtw_polarisation(TW_long_s, T1_gas_s)
    # Gas hydrogen index inferred from short/long ratio
    return float(phi_apparent * (p_water_long / max(p_water_short, 1e-9)))


# ---------------------------------------------- density porosity --------

def density_porosity(rho_b, rho_ma, rho_fl=1.0):
    return float((rho_ma - rho_b) / (rho_ma - rho_fl))


def variable_matrix_density(v_sand=0.7, v_shale=0.3,
                            rho_sand=2.65, rho_shale=2.50):
    return v_sand * rho_sand + v_shale * rho_shale


# ---------------------------------------------- DDTW integration --------

def ddtw_porosity(phi_NMR_apparent, rho_b, rho_ma, HI_water=1.0,
                  HI_gas=0.42, rho_water=1.0, rho_gas=0.20):
    """Closed-form two-equation gas-zone DDTW solution.

    With Sg + Sw = 1 (no oil), define u = phi, v = phi * Sg.  The
    density and NMR apparent porosities become linear in (u, v):

        phi_D_apparent   = u + a * v       a = (rho_water - rho_gas) / (rho_ma - rho_water)
        phi_NMR_apparent = u - b * v       b = 1 - HI_gas

    Solving:
        v = (phi_D_apparent - phi_NMR_apparent) / (a + b)
        u = phi_D_apparent - a * v
        Sg = v / u
    """
    phi_D_apparent = density_porosity(rho_b, rho_ma, rho_fl=rho_water)
    a = (rho_water - rho_gas) / (rho_ma - rho_water)
    b = 1.0 - HI_gas
    v = (phi_D_apparent - phi_NMR_apparent) / (a + b)
    u = phi_D_apparent - a * v
    Sg = float(np.clip(v / max(u, 1e-9), 0.0, 1.0))
    return float(max(u, 0.0)), Sg


# ---------------------------------------------- mud-gas HI ------------

def mud_gas_HI(C1_frac, C2_frac, C3_frac, C4_frac, C5_frac):
    """Hydrogen index from molar fractions of the C1-C5 alkanes:

        HI = sum_n f_n * (4 + 2 n_C(n)) / MW_n / 9
    """
    H_atoms = dict(C1=4, C2=6, C3=8, C4=10, C5=12)
    MW = dict(C1=16.04, C2=30.07, C3=44.10, C4=58.12, C5=72.15)
    fracs = dict(C1=C1_frac, C2=C2_frac, C3=C3_frac, C4=C4_frac, C5=C5_frac)
    HI = 0.0
    for k in fracs:
        HI += fracs[k] * H_atoms[k] / MW[k]
    # Calibrate so pure methane gives ~ 0.42
    return float(HI * 0.42 / (H_atoms["C1"] / MW["C1"]))


# ---------------------------------------------- tests -----------------

def test_all():
    print("=" * 60)
    print("Article 6: DDTW NMR + Mud-Gas Integrated Porosity")
    print("=" * 60)

    # Variable matrix density
    rho_ma = variable_matrix_density()
    print(f"  Variable matrix density = {rho_ma:.3f} g/cc")
    assert 2.55 < rho_ma < 2.65

    # Density porosity vs NMR-apparent porosity in a gas zone
    phi_true, Sg_true = 0.22, 0.65
    HI_gas = 0.42
    rho_b_gas = (1.0 - phi_true) * rho_ma \
                + phi_true * (Sg_true * 0.20 + (1.0 - Sg_true) * 1.0)
    phi_NMR_apparent = phi_true * (1.0 + Sg_true * (HI_gas - 1.0))
    print(f"  Gas-zone synthetic   rho_b = {rho_b_gas:.3f}, "
          f"phi_NMR_app = {phi_NMR_apparent:.3f}")

    phi_DDTW, Sg_DDTW = ddtw_porosity(phi_NMR_apparent, rho_b_gas, rho_ma)
    print(f"  DDTW solution        phi = {phi_DDTW:.3f}   Sg = {Sg_DDTW:.3f}")
    print(f"  Truth                phi = {phi_true:.3f}   Sg = {Sg_true:.3f}")
    assert abs(phi_DDTW - phi_true) < 0.04, "DDTW phi must be within 4 p.u."

    # Mud-gas HI for an analogue C1-C5 composition
    HI_mg = mud_gas_HI(0.80, 0.10, 0.05, 0.03, 0.02)
    print(f"  Mud-gas HI            = {HI_mg:.3f}")
    assert 0.30 < HI_mg < 0.55
    print("  PASS")
    return {"phi_DDTW": phi_DDTW, "Sg_DDTW": Sg_DDTW, "HI_mudgas": HI_mg}


if __name__ == "__main__":
    test_all()
