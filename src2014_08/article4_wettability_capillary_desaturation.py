"""
Article 4: Impact of Wettability on Residual Oil Saturation and Capillary
           Desaturation Curves
K. J. Humphry, B. M. J. M. Suijkerbuijk, H. A. van der Linde, S. G. J. Pieterse,
S. K. Masalmeh (2014)
Reference: Petrophysics Vol. 55, No. 4 (August 2014), pp. 313-318
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best of the 2013 SCA Symposium.  Capillary desaturation curves (CDC) - residual
oil saturation as a function of capillary or Bond number - are measured on cores
of varying wettability.  More water-wet rock traps more oil and desaturates only
at higher capillary numbers; mixed/oil-wet rock traps less and desaturates more
gradually.

Implements:

  - Capillary number  N_Ca = v_b*mu_b/gamma  (Eq. 1)
  - Bond number  N_Bo = drho*a*k/gamma  (Eq. 2)
  - Trapping number  N_T = N_Ca + N_Bo
  - Dimensionless imbibition time (Ma et al., 1999)  (Eq. 3)
  - Capillary desaturation curve  Sor(N) with a critical (onset) number

Note: this issue's PDF has a text layer; the Eq. 1-3 bodies were dropped in
extraction, but the variable definitions survived and are reconstructed in
standard form (Ma et al., 1999).  The CDC has no closed form in the paper and is
modelled here with the standard logistic-in-log-N transition between the initial
and irreducible residual oil saturations.  SI units; saturations as fractions.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- dimensionless numbers --------------

def capillary_number(brine_velocity, brine_viscosity, ift):
    """Capillary number (Eq. 1)

        N_Ca = v_b*mu_b/gamma,

    with the in-pore brine velocity v_b, brine viscosity mu_b and brine-oil
    interfacial tension gamma.
    """
    return petrolib.relperm_wettability.capillary_number(
        mu=brine_viscosity, v=brine_velocity, sigma=ift)


def bond_number(delta_density, acceleration, permeability, ift):
    """Bond number (Eq. 2)

        N_Bo = drho*a*k/gamma,

    with the brine-oil density difference drho, gravitational acceleration a,
    brine permeability k and interfacial tension gamma.
    """
    return petrolib.relperm_wettability.bond_number(
        drho=delta_density, k=permeability, sigma=ift, g=acceleration)


def trapping_number(n_ca, n_bo):
    """Total trapping number  N_T = N_Ca + N_Bo, combining viscous and
    gravitational mobilization of the trapped phase."""
    return petrolib.relperm_wettability.trapping_number(n_ca, n_bo)


def dimensionless_imbibition_time(t, k, phi, ift, mu_w, mu_o, l_c):
    """Dimensionless time for spontaneous imbibition (Ma et al., 1999; Eq. 3)

        t_d = t*sqrt(k/phi)*(gamma/sqrt(mu_w*mu_o))*(1/L_c^2),

    the scaling that collapses imbibition recovery curves of different cores.
    """
    return t * np.sqrt(k / phi) * (ift / np.sqrt(mu_w * mu_o)) / l_c ** 2


# ---------------------------------------------- capillary desaturation curve --------------

def capillary_desaturation(n_trap, sor_initial, sor_irreducible, n_critical,
                           width=1.0):
    """Capillary desaturation curve: residual oil saturation vs trapping number

        Sor(N) = Sor_irr + (Sor_init - Sor_irr)/(1 + (N/N_crit)^(1/width)),

    a smooth transition that stays at the plateau Sor_init below the critical
    (onset) number N_crit and falls toward Sor_irr as N rises above it.
    """
    # The library exponent is 1/width; sor_max=plateau, sor_min=irreducible floor.
    return petrolib.relperm_wettability.capillary_desaturation(
        n_trap, sor_max=sor_initial, sor_min=sor_irreducible,
        n_crit=n_critical, exponent=1.0 / width)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Wettability & Capillary Desaturation Curves")
    print("=" * 60)

    # Capillary number rises with velocity/viscosity, falls with IFT
    nca = capillary_number(1e-5, 1e-3, 0.03)
    print(f"  N_Ca = {nca:.2e}")
    assert nca > 0 and capillary_number(1e-5, 1e-3, 0.06) < nca

    # Bond number rises with permeability and density contrast
    nbo = bond_number(150.0, 9.81, 320e-3 * 9.869e-13, 0.03)
    assert nbo > 0 and trapping_number(nca, nbo) > nca

    # Dimensionless imbibition time is positive and scales with sqrt(k/phi)
    td1 = dimensionless_imbibition_time(100.0, 1e-13, 0.20, 0.03, 1e-3, 2e-3, 0.05)
    td2 = dimensionless_imbibition_time(100.0, 4e-13, 0.20, 0.03, 1e-3, 2e-3, 0.05)
    assert td2 > td1 > 0

    # CDC: plateau below the critical number, desaturation above it
    n = np.logspace(-7, -2, 50)
    sor = capillary_desaturation(n, sor_initial=0.5, sor_irreducible=0.15,
                                 n_critical=1e-5)
    print(f"  Sor(low N)={sor[0]:.3f}  Sor(high N)={sor[-1]:.3f}")
    assert np.isclose(sor[0], 0.5, atol=0.02)         # trapped plateau
    assert sor[-1] < 0.2                               # mobilized
    assert np.all(np.diff(sor) <= 1e-9)                # monotonically decreasing
    # a more water-wet rock (higher critical number) traps oil to higher N
    sor_ww = capillary_desaturation(1e-4, 0.5, 0.15, n_critical=1e-4)
    sor_ow = capillary_desaturation(1e-4, 0.5, 0.15, n_critical=1e-5)
    assert sor_ww > sor_ow
    print("  PASS")
    return {"N_Ca": float(nca), "N_Bo": float(nbo), "Sor_high": float(sor[-1])}


if __name__ == "__main__":
    test_all()
