"""
Article 6: Quantifying Interfacial Interactions Between Minerals and
Reservoir/Fracturing Fluids
Silveira de Araujo, Heidari (2022)
DOI: 10.30632/PJV63N6-2022a6

A genuine all-atom MD run (RASPA/CLAYFF/SPC-E/GAFF) is far outside the
scope of this repository, but the paper's *analysis* pipeline can be
faithfully reproduced on a synthetic Brownian trajectory.  This module:

  - Generates 2-D Brownian trajectories for an ensemble of "ions" /
    "waters" inside a slit nanopore with two reflecting walls.
  - Adds an optional adsorbed "sticky" near-wall layer that traps a
    fraction of particles and reduces their long-time diffusivity.
  - Computes a per-particle density profile across the slit (the
    quantity the paper plots as density vs distance to surface).
  - Extracts the self-diffusion coefficient from the long-time slope
    of the mean-square displacement (Eq. 1):

        D = lim_{t -> inf} <|r(t) - r(0)|^2> / (2 d t)

    where d is the dimensionality of the diffusion (= 1 for in-plane
    component, = 2 for the in-slit isotropic case used here).
  - Counts "hydrogen-bond-like" contacts as a proxy for mineral-fluid
    interaction strength.
"""

import numpy as np


# --------------------------------------------- Brownian slit-pore -------

def langevin_slit(n_particles=400, n_steps=2000, dt=1.0,
                  D_bulk=1.0e-9, slit_width_nm=3.0, sticky_frac=0.0,
                  D_sticky_factor=0.05, seed=0):
    """Langevin/Brownian trajectories for `n_particles` confined between two
    reflecting walls at +/- slit_width/2.

    Returns positions (n_steps, n_particles, 2) in nm.  Step variance is
    set so that the bulk-particle subset asymptotes to a measured
    self-diffusion of D_bulk (in m^2/s).
    """
    rng = np.random.default_rng(seed)
    half = slit_width_nm / 2.0
    pos = np.zeros((n_steps, n_particles, 2))
    pos[0, :, 0] = rng.uniform(-half, half, n_particles)
    pos[0, :, 1] = 0.0
    sticky = rng.random(n_particles) < sticky_frac
    D = np.where(sticky, D_bulk * D_sticky_factor, D_bulk)
    # Step std (nm) so that <r^2> grows as 2 d D t with positions in nm
    sigma_nm = np.sqrt(2.0 * D * dt) * 1e9  # convert m -> nm
    for it in range(1, n_steps):
        step = rng.standard_normal((n_particles, 2)) * sigma_nm[:, None]
        bias = np.zeros_like(step)
        bias[sticky, 0] = np.sign(pos[it - 1, sticky, 0]) * 0.005
        new = pos[it - 1] + step + bias
        new[:, 0] = np.where(new[:, 0] >  half,  2 * half - new[:, 0],
                     np.where(new[:, 0] < -half, -2 * half - new[:, 0],
                              new[:, 0]))
        pos[it] = new
    return pos, sticky


# --------------------------------------------- density profile ----------

def density_profile(positions, n_bins=40, slit_width_nm=3.0):
    """Mean density along the slit-normal direction."""
    edges = np.linspace(-slit_width_nm / 2, slit_width_nm / 2, n_bins + 1)
    hist = np.zeros(n_bins)
    flat = positions[:, :, 0].flatten()
    h, _ = np.histogram(flat, bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), h / h.sum()


# --------------------------------------------- MSD / D (Eq 1) -----------

def mean_squared_displacement(positions, axis="both"):
    """<|r(t)-r(0)|^2> over all particles for each time origin t.

    `axis` controls which Cartesian component(s) contribute:
        "both": both in-slit components (default - bounded x + free y)
        "x":    slit-normal only (saturates at ~ slit_width^2 / 12)
        "y":    slit-parallel only (linear growth - the unconfined dir)
    """
    disp = positions - positions[0:1]
    if axis == "both":
        return (disp ** 2).sum(-1).mean(-1)
    if axis == "x":
        return (disp[:, :, 0] ** 2).mean(-1)
    if axis == "y":
        return (disp[:, :, 1] ** 2).mean(-1)
    raise ValueError(axis)


def diffusion_from_msd(msd, dt, d_dim=1, fit_window=(0.3, 0.9)):
    """Linear fit of MSD vs t over the late-time window (avoids ballistic
    onset and wall-bounce saturation).  Returns D in m^2/s.

    `msd` is assumed to be in nm^2 (positions in nm); we convert to m^2
    via the 1e-18 factor before dividing by 2 * d_dim.

    For confined slit pores the diffusion analysis is usually done on the
    unconfined component only (d_dim = 1) because the slit-normal MSD
    saturates at slit_width^2 / 12.
    """
    n = len(msd)
    lo = int(fit_window[0] * n)
    hi = int(fit_window[1] * n)
    t = np.arange(n) * dt
    slope_nm2_s, _ = np.polyfit(t[lo:hi], msd[lo:hi], 1)
    return float(slope_nm2_s * 1e-18 / (2 * d_dim))


# --------------------------------------------- "H-bond" contacts --------

def near_wall_contacts(positions, slit_width_nm=3.0,
                       contact_threshold_nm=0.30):
    """Fraction of all (particle, time) samples within contact_threshold of
    either wall.  Crude proxy for an averaged mineral-fluid H-bond count."""
    half = slit_width_nm / 2.0
    dist_to_wall = half - np.abs(positions[:, :, 0])
    return float((dist_to_wall < contact_threshold_nm).mean())


# --------------------------------------------- tests --------------------

def test_all():
    print("=" * 60)
    print("Article 6: MD-Style Analysis of Mineral/Fluid Interface")
    print("=" * 60)

    dt = 5e-12  # 5 ps timestep -> total 10 ns for 2000 steps
    pos_clean, _ = langevin_slit(n_particles=400, n_steps=2000, dt=dt,
                                 D_bulk=1.0e-9, sticky_frac=0.0, seed=0)
    pos_sticky, sticky = langevin_slit(n_particles=400, n_steps=2000, dt=dt,
                                       D_bulk=1.0e-9,
                                       sticky_frac=0.30,
                                       D_sticky_factor=0.05, seed=0)
    # Fit MSD on the unconfined slit-parallel direction
    msd_c = mean_squared_displacement(pos_clean, axis="y")
    msd_s = mean_squared_displacement(pos_sticky, axis="y")
    D_c = diffusion_from_msd(msd_c, dt, d_dim=1)
    D_s = diffusion_from_msd(msd_s, dt, d_dim=1)

    print(f"  Self-diffusion D  (no surface)       = {D_c:.3e} m^2/s")
    print(f"  Self-diffusion D  (30% sticky layer) = {D_s:.3e} m^2/s")
    print(f"  Reduction ratio                       = {D_s / D_c:.2f}")
    bins_x, dens_c = density_profile(pos_clean)
    _, dens_s = density_profile(pos_sticky)
    print(f"  Wall-contact fraction (clean)        = "
          f"{near_wall_contacts(pos_clean):.3f}")
    print(f"  Wall-contact fraction (sticky)       = "
          f"{near_wall_contacts(pos_sticky):.3f}")

    # Sticky case should peak density at the walls (the paper's signature
    # for adsorbed methanol/citric acid on illite)
    edge_dens = dens_s[0] + dens_s[-1]
    bulk_dens = dens_s[len(dens_s) // 2]
    print(f"  Sticky density edge / bulk           = {edge_dens / bulk_dens:.2f}")

    # Sanity
    assert D_c > 5e-10, "Clean Brownian D should be ~ D_bulk"
    assert D_s < D_c, "Sticky population must reduce mean D"
    assert edge_dens > bulk_dens, "Sticky case must peak density at walls"
    print("  PASS")
    return {"D_clean": D_c, "D_sticky": D_s, "ratio": D_s / D_c,
            "edge_bulk_ratio": edge_dens / bulk_dens}


if __name__ == "__main__":
    test_all()
