"""
Article 2: Permeability Modeling in Clay-Rich Carbonate Reservoir
Storebo, Meireles, Fabricius (2022)
DOI: 10.30632/PJV63N2-2022a2

Compares four Kozeny-equation routes for permeability in the Lower
Cretaceous Sola / Tuxen marly chalks (Well Boje-2C).  Implements:

  - Base Kozeny permeability  k = c * phi^3 / S_phi^2          (Eq. 1)
  - Mortensen et al. (1998) shielding factor c(phi)            (Eq. 2)
  - Ternary calcite / silicate / pyrite mixing law for porosity (Eqs. 3-4)
  - Pore-space SSA from mass fractions of silicates + carbonate (Eq. 5)
  - SSA from spectral GR:  Sb = x * rho_b * (Th + K) + y        (Eq. 6)
  - SSA from Sw + pseudo water-film thickness pwft              (Eqs. 7-8)
  - SSA from NMR T2 of the water peak:  S_phi = 1 / (rho * T2)   (Eqs. 9-10)
  - Flow-zone-indicator FZI                                     (Eqs. 11-13)
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Eq. 1: Kozeny ----------

def kozeny_permeability(phi, S_phi_per_m, c_shield=0.20):
    """k [m^2] = c * phi^3 / S_phi^2  (Eq. 1).  S_phi in 1/m."""
    return c_shield * phi ** 3 / S_phi_per_m ** 2


# ---------------------------------------------- Eq. 2: Mortensen c(phi) -

def mortensen_shielding(phi):
    """c(phi) = (4*cos(phi*pi/3))^2 / 8.  Approximation valid for phi >= 0.20."""
    return float((4.0 * np.cos(phi * np.pi / 3.0)) ** 2 / 8.0)


# ---------------------------------------------- Eq. 3-4: porosity ----

def porosity_from_density(rho_b, mass_calc, mass_sil, mass_py,
                          rho_calc=2.71, rho_sil=2.65, rho_py=5.0,
                          rho_fluid=1.0):
    """Ternary mineral density + bulk-mixing law for porosity (Eqs. 3-4)."""
    rho_g = mass_calc * rho_calc + mass_sil * rho_sil + mass_py * rho_py
    return float(petrolib.porosity_lithology.density_porosity(rho_b, rho_g, rho_fluid))


# ---------------------------------------------- Eq. 5: pore SSA ----

def pore_ssa_from_mineralogy(mass_sil, mass_carb, ssa_sil=20.0,
                             ssa_carb=2.0, rho_g=2.71):
    """Pore-space SSA from solid SSAs (Eq. 5).  Returns 1/m using
    grain-density to convert from m^2/g to 1/m."""
    ssa_solid_m2_g = mass_sil * ssa_sil + mass_carb * ssa_carb
    return ssa_solid_m2_g * rho_g * 1000.0     # 1/m


# ---------------------------------------------- Eq. 6: SSA from GR ---

def ssa_from_spectral_gr(rho_b, Th_ppm, K_pct, x=0.05, y=0.10):
    """Sb = x * rho_b * (Th + K) + y  (Eq. 6)."""
    return float(x * rho_b * (Th_ppm + K_pct) + y)


# ---------------------------------------------- Eqs. 7-8: pwft ----

def ssa_from_pwft(phi, Sw, pwft_nm=10.0):
    """S_phi = phi * Sw / pwft  (Eqs. 7-8).  Returns 1/m."""
    return float(phi * Sw / (pwft_nm * 1e-9))


# ---------------------------------------------- Eqs. 9-10: NMR T2 ---

def ssa_from_nmr_t2(T2_water_ms, surface_relaxivity_um_s):
    """S_phi = 1 / (rho * T2_water)  (Eqs. 9-10).  Returns 1/m."""
    rho_m_s = surface_relaxivity_um_s * 1e-6
    return float(1.0 / (rho_m_s * T2_water_ms * 1e-3))


# ---------------------------------------------- Eqs. 11-13: FZI ----

def void_ratio(phi):
    return phi / (1.0 - phi)


def flow_zone_indicator(c_shield, void_eps):
    """FZI = (1 / S_phi) * sqrt(c / (1 + eps))  (Eq. 11)."""
    return float(np.sqrt(c_shield / (1.0 + void_eps)))


def fzi_from_vp(Vp_m_s, a=0.5, b=-2.0):
    """log(FZI) = a * log(Vp) + b  (Eq. 13)."""
    return float(10.0 ** (a * np.log10(Vp_m_s) + b))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Kozeny Permeability for Clay-Rich Chalk")
    print("=" * 60)

    # Boje-2C analogue: 26% phi, 78% calcite, 20% silicates, 2% pyrite
    rho_b = 2.1
    phi = porosity_from_density(rho_b, mass_calc=0.78, mass_sil=0.20,
                                mass_py=0.02)
    c = mortensen_shielding(phi)
    print(f"  phi (Eqs. 3-4)            = {phi:.3f}")
    print(f"  c(phi)  Mortensen (Eq. 2) = {c:.3f}")
    assert 0.20 < phi < 0.40

    # Four SSA estimators
    S_min = pore_ssa_from_mineralogy(0.20, 0.78)
    S_gr = ssa_from_spectral_gr(rho_b, Th_ppm=4.0, K_pct=0.5)
    S_pwft = ssa_from_pwft(phi, Sw=0.40, pwft_nm=10.0)
    S_nmr = ssa_from_nmr_t2(T2_water_ms=20.0, surface_relaxivity_um_s=3.5)
    print(f"  S_phi  mineralogy      = {S_min:.2e} 1/m")
    print(f"  S_phi  spectral GR     = {S_gr:.2e} (relative units)")
    print(f"  S_phi  pwft  10 nm     = {S_pwft:.2e} 1/m")
    print(f"  S_phi  NMR T2 = 20 ms  = {S_nmr:.2e} 1/m")

    # Kozeny permeability with NMR-derived SSA (the paper's preferred route)
    k_m2 = kozeny_permeability(phi, S_nmr, c_shield=c)
    k_mD = k_m2 / 0.9869e-15
    print(f"  Kozeny k (NMR + ro_BET) = {k_mD:7.3f} mD")
    assert 0.001 < k_mD < 1.0, "k must fall in chalk band 0.001-1 mD"

    # FZI route (Eqs. 11-13)
    eps = void_ratio(phi)
    FZI = flow_zone_indicator(c, eps)
    FZI_Vp = fzi_from_vp(3000.0, a=0.5, b=-2.0)
    print(f"  FZI (Eqs. 11-12)         = {FZI:.4f}")
    print(f"  FZI from Vp = 3000 m/s   = {FZI_Vp:.4f}")
    print("  PASS")
    return {"phi": phi, "c": c, "k_mD": k_mD, "FZI": FZI}


if __name__ == "__main__":
    test_all()
