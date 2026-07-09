"""
Article 1: Thermal Maturity-Adjusted Log Interpretation (TMALI) in Organic
           Shales
Craddock, Miles, Lewis, Pomerantz (2019)
DOI: 10.30632/PJV60N5-2019a1

Standard log interpretation in organic shales assumes fixed kerogen properties,
but the kerogen skeletal density, hydrogen index, carbon fraction and neutron
response all change with thermal maturity (vitrinite reflectance Ro).  TMALI
computes maturity-adjusted kerogen endpoints and propagates them through the
volumetric log responses (density, neutron, sigma) so TOC, matrix density,
porosity and water saturation are not biased by a wrong kerogen endpoint.

Implements:

  - Element molar fractions  n_i = (w_i/m_i)/sum(w_i/m_i)          (Eq. 1)
  - Electron density  rho_e = 2*rho*(sum Z)/(sum A)               (Eq. 3)
  - Apparent log density  rho_a = 1.0704*rho_e - 0.1883           (Eq. 4)
  - Kerogen hydrogen index  HI_k = nH/(M_k*0.11)                  (Eq. 6)
  - TOC -> kerogen volume  phi_k = TOC*F_k*(rho_ma/rho_k)         (Eq. 10)
  - Bulk density and density porosity                            (Eqs. 11-12)
  - Maturity-adjusted kerogen density vs Ro

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are the standard
log-petrophysics forms anchored to those definitions.  Paper anchors: kerogen
density 1.1-1.6 g/cm^3 (rho_k ~ 1.21 at Ro 0.7%, ~1.33 at 1.1%, ~1.42 at 1.4%).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- composition -------------

def molar_fractions(mass_fracs, atomic_masses):
    """Element molar fractions  n_i = (w_i/m_i)/sum(w_i/m_i)  (Eq. 1)."""
    moles = np.asarray(mass_fracs, float) / np.asarray(atomic_masses, float)
    return moles / moles.sum()


def electron_density(rho, sum_Z, sum_A):
    """Electron density  rho_e = 2*rho*(sum Z)/(sum A)  (Eq. 3)."""
    return petrolib.nuclear.electron_density_index(sum_Z, sum_A, rho)


def apparent_density(rho_e):
    """Apparent log density from electron density  rho_a = 1.0704*rho_e - 0.1883  (Eq. 4)."""
    return petrolib.nuclear.rhob_from_rhoe(rho_e)


# ---------------------------------------------- kerogen -----------------

def kerogen_hydrogen_index(nH, M_k):
    """Kerogen hydrogen index  HI_k = nH/(M_k*0.11)  (Eq. 6).

    nH = hydrogen atoms per kerogen unit, M_k = molecular weight, 0.11 = the
    hydrogen molar density of water (mol/cm^3).
    """
    return nH / (M_k * 0.11)


def maturity_kerogen_density(Ro):
    """Maturity-adjusted kerogen skeletal density (g/cm^3) vs vitrinite Ro (%).

        rho_k = 1.0 + 0.3*Ro,  clipped to [1.1, 1.6]
    Reproduces the paper's ~1.21 (Ro 0.7%), ~1.33 (1.1%), ~1.42 (1.4%).
    """
    return np.clip(1.0 + 0.3 * np.asarray(Ro, float), 1.1, 1.6)


def toc_to_kerogen_volume(toc, F_k, rho_ma, rho_k):
    """Kerogen volume fraction from TOC  phi_k = TOC*F_k*(rho_ma/rho_k)  (Eq. 10).

    toc as a mass fraction; F_k = reciprocal of the kerogen carbon mass fraction.
    """
    return toc * F_k * (rho_ma / rho_k)


# ---------------------------------------------- density / porosity ------

def bulk_density(phi, rho_f, rho_ma):
    """Volumetric bulk density  rho_b = phi*rho_f + (1-phi)*rho_ma  (Eq. 11)."""
    return petrolib.porosity_lithology.bulk_density(phi, rho_ma, rho_f)


def density_porosity(rho_b, rho_ma, rho_f):
    """Density porosity  phi_D = (rho_ma - rho_b)/(rho_ma - rho_f)  (Eq. 12)."""
    return petrolib.porosity_lithology.density_porosity(rho_b, rho_ma, rho_f)


def sigma_water_saturation(sigma_b, sigma_ma, sigma_w, sigma_hc, phi):
    """Sigma water saturation (degenerate at low salinity)  (Eq. 16)."""
    return petrolib.nuclear.sw_from_sigma(
        sigma_b, phi, sigma_ma=sigma_ma, sigma_w=sigma_w, sigma_hc=sigma_hc
    )


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Thermal Maturity-Adjusted Log Interpretation")
    print("=" * 60)

    # Molar fractions of a simple kerogen unit sum to 1 (by mole, hydrogen is
    # abundant despite carbon dominating the mass)
    n = molar_fractions([0.80, 0.08, 0.02, 0.02, 0.08], [12.0, 1.0, 14.0, 32.0, 16.0])
    assert abs(n.sum() - 1.0) < 1e-12 and np.all(n > 0)

    # Apparent-density calibration: limestone (Z/A->1) reads back its density
    rho_e = electron_density(2.71, sum_Z=50.0, sum_A=100.0)   # CaCO3
    rho_a = apparent_density(rho_e)
    print(f"  limestone rho_e/rho_a  = {rho_e:.3f} / {rho_a:.3f}")
    assert abs(rho_a - 2.71) < 0.01

    # Maturity adjustment reproduces the paper's kerogen densities
    rk = maturity_kerogen_density(np.array([0.7, 1.1, 1.4]))
    print(f"  kerogen density vs Ro  = {np.array2string(rk, precision=2)}")
    assert abs(rk[0] - 1.21) < 0.02 and abs(rk[2] - 1.42) < 0.02
    assert np.all(rk >= 1.1) and np.all(rk <= 1.6)

    # Using a too-low (immature) kerogen density at high maturity biases the
    # kerogen volume from TOC
    phik_correct = toc_to_kerogen_volume(0.10, F_k=1.25, rho_ma=2.68,
                                         rho_k=maturity_kerogen_density(1.4))
    phik_wrong = toc_to_kerogen_volume(0.10, F_k=1.25, rho_ma=2.68, rho_k=1.21)
    print(f"  kerogen vol correct/wrong = {phik_correct:.3f} / {phik_wrong:.3f}")
    assert phik_wrong > phik_correct                    # low rho_k overstates phi_k

    # Density porosity round-trips with the bulk-density mixing law
    rb = bulk_density(0.12, 1.0, 2.68)
    assert abs(density_porosity(rb, 2.68, 1.0) - 0.12) < 1e-9

    # Sigma Sw behaves and is higher in saline water
    assert sigma_water_saturation(20.0, 9.0, 60.0, 22.0, 0.10) > \
        sigma_water_saturation(14.0, 9.0, 60.0, 22.0, 0.10)
    print("  PASS")
    return {"rho_a": float(rho_a), "rk_14": float(rk[2]),
            "phik_correct": float(phik_correct)}


if __name__ == "__main__":
    test_all()
