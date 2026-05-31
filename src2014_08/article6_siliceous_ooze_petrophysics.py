"""
Article 6: Petrophysical Analysis of Siliceous-Ooze Sediments, More Basin,
           Norwegian Sea
Ahmed Awadalkarim, Morten Kanne Sorensen, Ida Lykke Fabricius (2014)
Reference: Petrophysics Vol. 55, No. 4 (August 2014), pp. 333-348
DOI: none assigned (this issue predates SPWLA DOI assignment)

A regular contribution.  Opal-A-rich siliceous-ooze sediments have an unusually
low grain density and high structural-water content, which biases the standard
density, neutron and acoustic log interpretation.  This module implements the
opal-corrected grain density, density porosity, hydrogen-index neutron
correction and Biot-coefficient relations the paper uses to evaluate them.

Implements:

  - Opal fraction from the measured grain density  (Eq. 1)
  - Non-opal volume from gamma ray  (Eq. 2) and corrected grain density (Eq. 3)
  - Structural-water moles in SiO2.nH2O  (Eq. 4)
  - Corrected bulk density  (Eq. 7) and density porosity  (Eq. 8)
  - Hydrogen index of the log and true neutron porosity  (Eqs. 9-11)
  - Biot coefficient  beta = 1 - K_dry/K_o  (Eq. 12)

Note: this issue's PDF has a text layer; Eqs. 1, 3, 7, 9, 10 survived, while the
Eq. 2/4/8/11/12 bodies were dropped in extraction and reconstructed from the
surviving variable definitions in standard form (Biot & Willis, 1957).  Opal-A
grain density 2.16 g/cm^3, formation water 1.025 g/cm^3, mineral modulus K_o = 5
GPa.  Densities in g/cm^3, moduli in GPa.
"""

import numpy as np

M_SIO2 = 60.09   # g/mol
M_H2O = 18.02    # g/mol


# ---------------------------------------------- opal fraction --------------

def opal_fraction_from_grain_density(rho_g_core, rho_g_opal=2.16, rho_g_rest=2.78):
    """Opal volume fraction from the measured grain density (Eq. 1)

        rho_g_core = rho_g_opal*x + rho_g_rest*(1 - x)  ->
        x = (rho_g_rest - rho_g_core)/(rho_g_rest - rho_g_opal).
    """
    return (rho_g_rest - rho_g_core) / (rho_g_rest - rho_g_opal)


def non_opal_volume_from_gr(gr_log, gr_min, gr_max):
    """Non-opal volume fraction from the gamma-ray log (Eq. 2, linear form)

        V_non_opal = (GR_log - GR_min)/(GR_max - GR_min);

    the opal volume fraction is 1 - V_non_opal.
    """
    return np.clip((np.asarray(gr_log, float) - gr_min) / (gr_max - gr_min), 0, 1)


def corrected_grain_density(opal_fraction, rho_g_opal=2.16, rho_g_rest=2.78):
    """Grain density from the opal volume fraction log (Eq. 3)

        rho_g = rho_g_opal*x + rho_g_rest*(1 - x).
    """
    x = opal_fraction
    return rho_g_opal * x + rho_g_rest * (1.0 - x)


# ---------------------------------------------- structural water --------------

def structural_water_moles(water_mass_fraction):
    """Moles of structural water n in opal SiO2.nH2O (Eq. 4)

        n = (f/(1 - f))*(M_SiO2/M_H2O),

    with f the structural-water mass fraction (f = 0.035 gives n = 0.121).
    """
    f = water_mass_fraction
    return (f / (1.0 - f)) * (M_SIO2 / M_H2O)


# ---------------------------------------------- density porosity --------------

def corrected_bulk_density(phi, rho_grain, rho_fluid=1.025):
    """Bulk density from porosity and the corrected grain density (Eq. 7)

        rho_b = (1 - phi)*rho_grain + phi*rho_fluid.
    """
    return (1.0 - phi) * rho_grain + phi * rho_fluid


def density_porosity(rho_b, rho_grain, rho_fluid=1.025):
    """Density porosity (Eq. 8)

        phi = (rho_grain - rho_b)/(rho_grain - rho_fluid).

    Using a quartz grain density (2.65) instead of the opal-corrected value
    overestimates porosity by ~5-10 p.u.
    """
    return (rho_grain - rho_b) / (rho_grain - rho_fluid)


# ---------------------------------------------- neutron / hydrogen index --------------

def hydrogen_index_log(phi_neutron, hi_solid=0.113, hi_water=1.0):
    """Apparent log hydrogen index from neutron porosity (Eqs. 9-10)

        HI_log = phi_n*HI_water + (1 - phi_n)*HI_solid,

    HI_solid being the matrix hydrogen index (calcite ~0.113; opal-bearing
    solids carry structural water, ~0.19).
    """
    return phi_neutron * hi_water + (1.0 - phi_neutron) * hi_solid


def true_neutron_porosity(hi_log, hi_solid, hi_water=1.0):
    """True neutron porosity by inverting the hydrogen-index mix (Eq. 11)

        phi_n_true = (HI_log - HI_solid)/(HI_water - HI_solid).
    """
    return (hi_log - hi_solid) / (hi_water - hi_solid)


# ---------------------------------------------- Biot coefficient --------------

def biot_coefficient(k_dry, k_mineral=5.0):
    """Biot effective-stress coefficient (Biot & Willis, 1957; Eq. 12)

        beta = 1 - K_dry/K_o,

    with the dry-frame bulk modulus K_dry and mineral bulk modulus K_o.  beta < 1
    indicates a cemented/stiff frame; beta -> 1 is normal compaction.
    """
    return 1.0 - k_dry / k_mineral


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Siliceous-Ooze (Opal-A) Petrophysics")
    print("=" * 60)

    # Opal fraction from grain density: core value 2.43 -> ~0.57
    x = opal_fraction_from_grain_density(2.43)
    print(f"  opal volume fraction = {x:.3f}")
    assert np.isclose(x, 0.565, atol=0.01)
    # round-trip through the corrected grain density
    assert np.isclose(corrected_grain_density(x), 2.43)

    # Gamma-ray non-opal volume is bounded in [0, 1]
    v = non_opal_volume_from_gr(60.0, gr_min=20.0, gr_max=120.0)
    assert np.isclose(v, 0.4)

    # Structural water: f = 3.5 wt% -> n = 0.121
    n = structural_water_moles(0.035)
    print(f"  structural water moles n = {n:.3f}")
    assert np.isclose(n, 0.121, atol=1e-3)

    # Density porosity: opal grain density round-trips, quartz overestimates phi
    rho_g = corrected_grain_density(x)
    rho_b = corrected_bulk_density(0.52, rho_g)
    phi_opal = density_porosity(rho_b, rho_g)
    phi_quartz = density_porosity(rho_b, 2.65)
    print(f"  phi(opal)={phi_opal:.3f}  phi(quartz)={phi_quartz:.3f}")
    assert np.isclose(phi_opal, 0.52) and phi_quartz > phi_opal + 0.05

    # Hydrogen index: true neutron porosity is below the apparent (water HI)
    hi = hydrogen_index_log(0.50, hi_solid=0.19)
    phi_true = true_neutron_porosity(hi, hi_solid=0.19)
    assert np.isclose(phi_true, 0.50)
    # using calcite HI (0.113) for an opal solid overestimates the porosity
    # (the wireline neutron reads ~3-4 p.u. above the true value)
    phi_biased = true_neutron_porosity(hi, hi_solid=0.113)
    assert phi_biased > 0.50

    # Biot coefficient near 1 for a soft frame, below 1 when cemented
    b = biot_coefficient(k_dry=0.3, k_mineral=5.0)
    print(f"  Biot coefficient = {b:.3f}")
    assert 0.9 < b < 1.0 and biot_coefficient(2.5) < b
    print("  PASS")
    return {"opal_x": float(x), "n_water": float(n), "phi_opal": float(phi_opal),
            "biot": float(b)}


if __name__ == "__main__":
    test_all()
