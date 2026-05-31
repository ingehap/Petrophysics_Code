"""
Article 2: Effects of Bitumen Extraction on the 2D NMR Response of Saturated
           Kerogen Isolates
Chen, Singer, Kuang, Vargas, Hirasaki (2017)
Reference: Petrophysics Vol. 58, No. 5 (October 2017), pp. 470-484
DOI: none assigned (this issue predates SPWLA DOI assignment)

Low-field 2D T1-T2 relaxometry of heptane-saturated kerogen pellets, before and
after toluene bitumen extraction, is interpreted with surface-relaxation theory.
In the surface-dominated (fast-diffusion) regime the relaxation rate is
proportional to the pore surface-to-volume ratio, the surface relaxivity follows
from the BET surface area, and the pore diameter follows from the relaxivity and
relaxation time.  The T1/T2 ratio splits intragranular from intergranular pores.

Implements:

  - Pellet bulk volume and swelling percent
  - Relaxation-rate decomposition  1/T = 1/T_bulk + 1/T_surface
  - Surface relaxation  T = 1/(rho*(S/V))  and pore diameter  d = 6*rho*T
  - Fast-diffusion validity  d*rho/D0 << 1  and the Archie formation factor

Note: this issue's PDF has a text layer; the equation set is the cleanest in the
issue but the typeset glyphs were dropped, so the relations are faithful
standard-form reconstructions.  SI: relaxivity rho in m/s, time in s, length in
m, S/V in 1/m.
"""

import numpy as np

T_BULK_HEPTANE = 1.918       # bulk heptane T1 = T2 (s)
D0_HEPTANE = 3.43e-9         # bulk diffusivity (m^2/s = 3.43 um^2/ms)


# ---------------------------------------------- volume --------------

def bulk_volume(diameters, heights):
    """Pellet bulk volume  BV = sum_i (pi/4)*Di^2*Hi  (Eq. 1)."""
    d = np.asarray(diameters, float)
    h = np.asarray(heights, float)
    return float(np.sum(np.pi / 4.0 * d ** 2 * h))


def swelling_percent(bv_saturated, bv_dried):
    """Swelling percent  = (BV_sat - BV_dried)/BV_dried*100  (Eq. 2)."""
    return (bv_saturated - bv_dried) / bv_dried * 100.0


# ---------------------------------------------- relaxation --------------

def combined_relaxation_time(t_bulk, t_surface, t_diffusion=np.inf):
    """Total relaxation time from parallel rates  1/T = 1/T_bulk + 1/T_surf + 1/T_diff."""
    return 1.0 / (1.0 / t_bulk + 1.0 / t_surface + 1.0 / t_diffusion)


def surface_relaxation_time(rho, s_over_v):
    """Surface-dominated relaxation time  T = 1/(rho*(S/V))  (Eqs. 6-7)."""
    return 1.0 / (rho * np.asarray(s_over_v, float))


def pore_diameter(rho, t):
    """Spherical pore diameter  d = 6*rho*T  (S/V = 6/d)  (Eqs. 10-11)."""
    return 6.0 * rho * np.asarray(t, float)


def fast_diffusion_ratio(d, rho, d0=D0_HEPTANE):
    """Fast-diffusion validity ratio  d*rho/D0 (must be << 1)."""
    return d * rho / d0


def formation_factor(phi_micro, m):
    """Archie formation factor of the microporosity  F = phi_micro^(-m)  (Eq. 14)."""
    return np.asarray(phi_micro, float) ** (-m)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Kerogen 2D NMR & Bitumen Extraction")
    print("=" * 60)

    # Four pellets; saturation swells the bulk volume
    bv = bulk_volume([0.01, 0.01], [0.005, 0.004])
    assert bv > 0 and swelling_percent(1.1 * bv, bv) > 0

    # Combined relaxation time is shorter than either contribution alone
    t = combined_relaxation_time(T_BULK_HEPTANE, 0.05)
    assert t < min(T_BULK_HEPTANE, 0.05)

    # Intragranular pore diameter: rho1,mu=0.7 um/s, T1,mu=7.5 ms -> ~31 nm
    d1 = pore_diameter(0.7e-6, 7.5e-3)
    print(f"  intragranular d1       = {d1 * 1e9:.1f} nm")
    assert np.isclose(d1 * 1e9, 31.5, atol=1.0)

    # Surface relaxation: larger S/V -> shorter T2
    assert surface_relaxation_time(1.1e-6, 1e6) < surface_relaxation_time(1.1e-6, 1e5)

    # Fast-diffusion regime holds (ratio << 1); high cementation exponent
    ratio = fast_diffusion_ratio(d1, 0.7e-6)
    print(f"  fast-diffusion ratio   = {ratio:.2e}")
    assert ratio < 0.1
    assert formation_factor(0.1, 6.3) > 1.0
    print("  PASS")
    return {"d1_nm": float(d1 * 1e9), "fast_diff_ratio": float(ratio)}


if __name__ == "__main__":
    test_all()
