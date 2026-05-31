"""
Article 3: Coupling Multiphase Hydrodynamic and NMR Pore-Scale Modeling for
           Advanced Characterization of Saturated Rocks
Evseev, Dinariev, Hurlimann, Safonov (2015)
Reference: Petrophysics Vol. 56, No. 1 (February 2015), pp. 32-44
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best Papers of the 2014 SCA Symposium.  Density-functional hydrodynamics models
the pore-scale fluid distribution, and the NMR response is simulated by solving
the (generalized) Bloch-Torrey equation with surface-relaxation boundary
conditions - coupling magnetization dynamics to the pore geometry and
wettability.  In the fast-diffusion limit the surface relaxation rate is the
relaxivity times the surface-to-volume ratio; restricted diffusion gives the
Mitra short-time apparent diffusion coefficient.

Implements:

  - Magnetization decay  M(t) = M0*exp(-t/T2)
  - Fast-diffusion surface relaxation  1/T2 = 1/T2bulk + rho*(S/V)
  - Mitra short-time restricted diffusion  D(t)/D0 = 1 - (4/(9 sqrt(pi)))*(S/V)*sqrt(D0*t)
  - 1D Bloch-Torrey simulation of a slab pore with surface relaxation

Note: this issue's PDF has a text layer; the Bloch-Torrey / surface-relaxation
physics is transcribed from the body, while the typeset glyphs were dropped and
reconstructed in standard form (Callaghan, 1991; Mitra et al., 1992).  Times in
s, T2 in s, relaxivity in m/s, S/V in 1/m, diffusion in m^2/s.
"""

import numpy as np


# ---------------------------------------------- relaxation --------------

def magnetization_decay(m0, time, t2):
    """Transverse magnetization decay  M(t) = M0*exp(-t/T2)."""
    return m0 * np.exp(-np.asarray(time, float) / t2)


def surface_relaxation_t2(t2_bulk, rho, sv):
    """Fast-diffusion surface relaxation time

        1/T2 = 1/T2bulk + rho*(S/V),

    the Bloch-Torrey surface-relaxation boundary condition in the
    well-mixed (fast-diffusion) limit.
    """
    return 1.0 / (1.0 / t2_bulk + rho * sv)


def mitra_restricted_diffusion(d0, time, sv):
    """Mitra short-time restricted apparent diffusion coefficient

        D(t)/D0 = 1 - (4/(9*sqrt(pi)))*(S/V)*sqrt(D0*t),

    the early-time decrease of the apparent diffusion coefficient that encodes
    the pore surface-to-volume ratio.
    """
    return d0 * (1.0 - (4.0 / (9.0 * np.sqrt(np.pi))) * sv * np.sqrt(d0 * np.asarray(time, float)))


# ---------------------------------------------- Bloch-Torrey 1D --------------

def bloch_torrey_1d(length, n_cells, diffusion_coeff, t2_bulk, rho, total_time, n_steps):
    """1D Bloch-Torrey simulation of a slab pore with surface relaxation.

    Solves dm/dt = D d2m/dx2 - m/T2bulk on [0, length] with surface-relaxation
    flux D dm/dx = -+ rho*m at the two walls (explicit finite differences).
    Returns (times, total_magnetization) where the total magnetization is the
    spatial average; its decay defines an effective T2.
    """
    dx = length / n_cells
    dt = total_time / n_steps
    # explicit-scheme stability guard
    if diffusion_coeff * dt / dx ** 2 > 0.5:
        dt = 0.4 * dx ** 2 / diffusion_coeff
        n_steps = int(np.ceil(total_time / dt))
    m = np.ones(n_cells)
    times = np.zeros(n_steps + 1)
    total = np.zeros(n_steps + 1)
    total[0] = m.mean()
    for k in range(1, n_steps + 1):
        lap = np.zeros_like(m)
        lap[1:-1] = (m[2:] - 2 * m[1:-1] + m[:-2]) / dx ** 2
        # surface relaxation at the walls: D*dm/dx = -rho*m (one-sided)
        lap[0] = (m[1] - m[0]) / dx ** 2 - (rho / diffusion_coeff) * m[0] / dx
        lap[-1] = (m[-2] - m[-1]) / dx ** 2 - (rho / diffusion_coeff) * m[-1] / dx
        m = m + dt * (diffusion_coeff * lap - m / t2_bulk)
        times[k] = k * dt
        total[k] = m.mean()
    return times, total


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: NMR Pore-Scale Modeling (Bloch-Torrey)")
    print("=" * 60)

    # Magnetization decays exponentially with T2
    assert np.isclose(magnetization_decay(1.0, 0.5, 0.5), np.exp(-1.0))

    # Surface relaxation shortens T2 below the bulk value; stronger for larger S/V
    t2 = surface_relaxation_t2(t2_bulk=3.0, rho=1e-5, sv=1e4)
    print(f"  surface-relaxation T2  = {t2:.4f} s")
    assert t2 < 3.0 and surface_relaxation_t2(3.0, 1e-5, 5e4) < t2

    # Mitra: apparent diffusion decreases at short times, more for larger S/V
    d_lo = mitra_restricted_diffusion(2.5e-9, 1e-3, sv=5e4)
    d_hi_sv = mitra_restricted_diffusion(2.5e-9, 1e-3, sv=2e5)
    print(f"  Mitra D(t) sv 5e4/2e5  = {d_lo:.3e} / {d_hi_sv:.3e}")
    assert d_lo < 2.5e-9 and d_hi_sv < d_lo

    # 1D Bloch-Torrey: total magnetization decays; higher relaxivity decays faster
    t, mag = bloch_torrey_1d(length=1e-4, n_cells=40, diffusion_coeff=2.5e-9,
                             t2_bulk=3.0, rho=5e-5, total_time=0.5, n_steps=2000)
    t2b, magb = bloch_torrey_1d(1e-4, 40, 2.5e-9, 3.0, 2e-4, 0.5, 2000)
    print(f"  Bloch-Torrey M(end) rho 5e-5/2e-4 = {mag[-1]:.3f} / {magb[-1]:.3f}")
    assert mag[0] > mag[-1] > 0 and magb[-1] < mag[-1]
    print("  PASS")
    return {"T2_surface": float(t2), "M_end": float(mag[-1])}


if __name__ == "__main__":
    test_all()
