"""
Article 4: Fluid Typing and Pore Size in Organic Shale Using 2D NMR in Saturated
           Kerogen Isolates
Singer, Chen, Hirasaki (2016)
Reference: Petrophysics Vol. 57, No. 6 (December 2016), pp. 604-619
DOI: none assigned (this issue predates SPWLA DOI assignment)

Low-field 2D T1-T2 NMR on saturated kerogen isolates types fluids and sizes
pores.  The relaxation rate splits into bulk and surface terms; in the
fast-diffusion regime the surface rate is the relaxivity times the
surface-to-volume ratio (6/d for spheres), so the pore diameter follows.  A T2
cutoff splits intragranular (absorbed) from intergranular pores, the BET surface
area is partitioned by porosity^(2/3), and the T1/T2 ratio types the fluid
(heptane reads higher than water).

Implements:

  - Surface relaxation rate  1/T_S = 1/T_obs - 1/T_bulk
  - Surface-relaxivity relation  1/T_S = rho*(S/V)  and pore diameter d = 6/(S/V)
  - BET surface-area partition by porosity^(2/3)
  - T1/T2 fluid typing (heptane vs water)

Note: this issue's PDF has a text layer but the typeset equations (Eqs. 1-24)
were dropped, so the relations are faithful standard-form reconstructions; the
T2 cutoff (1.5 ms) and the heptane-over-water T1/T2 contrast are transcribed.
Relaxivity in um/s, times in s, diameters in um.
"""

import numpy as np

T2_CUTOFF_S = 1.5e-3         # intragranular / intergranular split


# ---------------------------------------------- relaxation --------------

def surface_rate(t_observed, t_bulk):
    """Surface relaxation rate  1/T_S = 1/T_obs - 1/T_bulk  (Eqs. 3-4)."""
    return 1.0 / np.asarray(t_observed, float) - 1.0 / t_bulk


def surface_to_volume(rho, t_surface):
    """Surface-to-volume ratio from the surface relaxation  S/V = 1/(rho*T_S)."""
    return 1.0 / (rho * np.asarray(t_surface, float))


def pore_diameter(rho, t_surface):
    """Spherical pore diameter  d = 6/(S/V) = 6*rho*T_S  (Eqs. 13-16)."""
    return 6.0 * rho * np.asarray(t_surface, float)


def bet_partition(phi_small, phi_large, s_bet):
    """Partition BET surface area by porosity^(2/3) (Eqs. 9-12)

        SBET,< = SBET * phi<^(2/3)/(phi<^(2/3) + phi>^(2/3)),  and the complement.
    """
    a = phi_small ** (2.0 / 3.0)
    b = phi_large ** (2.0 / 3.0)
    s_small = s_bet * a / (a + b)
    return s_small, s_bet - s_small


def fluid_type(t1t2, cutoff=2.0):
    """Type the fluid from the T1/T2 ratio: heptane (hydrocarbon) reads higher."""
    return "hydrocarbon" if t1t2 >= cutoff else "water"


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: 2D NMR Kerogen Fluid Typing")
    print("=" * 60)

    # Surface rate from the observed and bulk times (heptane bulk 1.91 s)
    ts = 1.0 / surface_rate(0.5, 1.91)
    assert ts > 0

    # Pore diameter from relaxivity and surface relaxation time
    rho2 = 8.1e-6           # m/s (intergranular heptane)
    d = pore_diameter(rho2, 0.05)
    print(f"  pore diameter          = {d * 1e6:.2f} um")
    assert d > 0 and np.isclose(surface_to_volume(rho2, 0.05) * d, 6.0)

    # BET partition sums back to the total surface area
    s_small, s_large = bet_partition(0.04, 0.06, 14.0)
    print(f"  SBET small / large     = {s_small:.2f} / {s_large:.2f} m^2/g")
    assert np.isclose(s_small + s_large, 14.0) and s_small < s_large

    # T1/T2 fluid typing
    assert fluid_type(14.0) == "hydrocarbon" and fluid_type(1.2) == "water"
    print("  PASS")
    return {"pore_d_um": float(d * 1e6), "SBET_small": float(s_small)}


if __name__ == "__main__":
    test_all()
