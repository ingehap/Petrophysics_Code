"""
Article 3: The Compressibility Factor (Z) of Shale Gas at the Core Scale
Tran, Sakhaee-Pour (2019)
DOI: 10.30632/PJV60N4-2019a3

The real-gas compressibility (deviation) factor Z relates pressure, volume and
temperature for gas in shale.  At the core / nanopore scale, confinement shifts
the apparent critical properties (lower critical temperature and pressure), so
the confined Z departs from the bulk-correlation value - which matters for
gas-in-place and material-balance calculations in shale.

Implements:

  - Pseudo-reduced pressure / temperature
  - Beggs-Brill explicit Z-factor correlation
  - Real-gas density  rho = P*M/(Z*R*T)
  - Confinement-shifted critical properties (core-scale Z)

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the issue's table of contents and these are
faithful standard-form reconstructions (Beggs-Brill Z plus a confinement
critical-property shift) of the core-scale Z method the paper develops.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

R_GAS = 8.314            # J/mol/K


# ---------------------------------------------- reduced properties ------

def pseudo_reduced(P, T, Ppc, Tpc):
    """Pseudo-reduced pressure and temperature  Ppr = P/Ppc, Tpr = T/Tpc."""
    return petrolib.geochem_fluids.pvt.pseudo_reduced(P, T, Ppc, Tpc)


def z_beggs_brill(Ppr, Tpr):
    """Beggs & Brill (1973) explicit compressibility factor Z."""
    return petrolib.geochem_fluids.pvt.z_beggs_brill(Ppr, Tpr)


def gas_density(P_pa, T_K, Z, M_kg_mol=0.016):
    """Real-gas density  rho = P*M/(Z*R*T)  (kg/m^3).  M default = methane."""
    return petrolib.geochem_fluids.pvt.gas_density(P_pa, T_K, m_kg_mol=M_kg_mol, z=Z)


# ---------------------------------------------- confinement -------------

def confined_critical(Tpc, Ppc, pore_nm, sigma_lj_nm=0.38):
    """Confinement-shifted critical temperature/pressure in a nanopore.

    Critical properties decrease with confinement (Zarragoicoechea-Kuz type):
        dTc*/Tc = a1*(sigma/rp) - a2*(sigma/rp)^2
    with rp the pore radius; same shift applied (approximately) to Pc.
    """
    rp = pore_nm / 2.0
    x = sigma_lj_nm / rp
    shift = 1.0 - (0.9409 * x - 0.2415 * x ** 2)
    return Tpc * shift, Ppc * shift


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Compressibility Factor Z of Shale Gas")
    print("=" * 60)

    Tpc, Ppc = 190.6, 4.60e6                    # methane critical T(K), P(Pa)
    T = 350.0

    # Z -> 1 at low pressure
    ppr0, tpr = pseudo_reduced(1e5, T, Ppc, Tpc)
    z0 = z_beggs_brill(ppr0, tpr)
    print(f"  Z at ~1 bar            = {z0:.3f}  (expect ~1)")
    assert abs(z0 - 1.0) < 0.02

    # Z dips below 1 at moderate reduced pressure, recovers at high pressure
    P_mid = 20e6
    ppr, tpr = pseudo_reduced(P_mid, T, Ppc, Tpc)
    z_mid = z_beggs_brill(ppr, tpr)
    print(f"  Z at 20 MPa (Ppr={ppr:.2f}) = {z_mid:.3f}")
    assert 0.6 < z_mid < 1.0

    # Real-gas density is positive and rises with pressure
    rho_lo = gas_density(10e6, T, z_beggs_brill(*pseudo_reduced(10e6, T, Ppc, Tpc)))
    rho_hi = gas_density(30e6, T, z_beggs_brill(*pseudo_reduced(30e6, T, Ppc, Tpc)))
    print(f"  gas density 10/30 MPa  = {rho_lo:.1f} / {rho_hi:.1f} kg/m^3")
    assert rho_hi > rho_lo > 0

    # Confinement lowers the critical properties, shifting the core-scale Z
    Tpc_c, Ppc_c = confined_critical(Tpc, Ppc, pore_nm=4.0)
    print(f"  confined Tpc/Ppc       = {Tpc_c:.1f} K / {Ppc_c/1e6:.2f} MPa")
    assert Tpc_c < Tpc and Ppc_c < Ppc
    z_conf = z_beggs_brill(*pseudo_reduced(P_mid, T, Ppc_c, Tpc_c))
    print(f"  bulk Z / confined Z    = {z_mid:.3f} / {z_conf:.3f}")
    assert abs(z_conf - z_mid) > 1e-3            # confinement changes Z
    print("  PASS")
    return {"Z_20MPa": float(z_mid), "Z_confined": float(z_conf),
            "Tpc_confined": float(Tpc_c)}


if __name__ == "__main__":
    test_all()
