"""
Wang and Ehlig-Economides (2023), Petrophysics 64(6): 970-977.
CO2 solubility in saline water for CO2 trapping, accounting for pressure,
temperature, salinity, and an often-neglected factor (e.g., dissolved CH4 or
the activity-coefficient correction). Implements a Duan-Sun-style solubility
model in simplified form.
"""
import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


def henry_co2(T_K):
    """Approximate Henry's-law constant for CO2 in pure water (MPa/molality)."""
    return petrolib.geochem_fluids.solubility.henry_constant_co2(T_K)


def activity_coeff_salt(m_nacl, T_K):
    """Salting-out activity coefficient (Setschenow-style)."""
    k_s = 0.11 + 1e-4 * (T_K - 298)
    return np.exp(2 * k_s * m_nacl)


def co2_solubility(P_MPa, T_K, m_nacl, m_ch4=0.0):
    """CO2 solubility in mol/kg H2O. Includes a CH4 competition correction."""
    return petrolib.geochem_fluids.solubility.co2_solubility_brine(P_MPa, T_K, m_nacl, m_ch4=m_ch4)


def trapping_capacity(porosity, sw, rho_brine, m_co2, M_co2=44e-3):
    """kg CO2 dissolved per m^3 reservoir rock."""
    return porosity * sw * rho_brine * m_co2 * M_co2


def test_all():
    P = np.array([5, 10, 20, 30])      # MPa
    T = 350.0
    m_nacl = 1.0
    sols = [co2_solubility(p, T, m_nacl) for p in P]
    sols_ch4 = [co2_solubility(p, T, m_nacl, m_ch4=0.5) for p in P]
    cap = trapping_capacity(0.20, 0.8, 1050, sols[2])
    print("Wang & Ehlig-Economides CO2 solubility:")
    for p, s, sc in zip(P, sols, sols_ch4):
        print(f"  P={p:>4} MPa  CO2 sol = {s:.3f}  with CH4 = {sc:.3f} mol/kg")
    print(f"  trapping cap @20MPa: {cap:.2f} kg/m^3")
    assert sols[1] > sols[0] > 0
    assert all(sc < s for s, sc in zip(sols, sols_ch4))
    print("  PASS")


if __name__ == "__main__":
    test_all()
