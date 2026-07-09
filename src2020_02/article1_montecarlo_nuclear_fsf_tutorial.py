"""
Article 1 (Tutorial): Simulation of Borehole Nuclear Measurements - A Practical
                      Tutorial Guide for Implementation of Monte Carlo Methods
                      and Approximations Based on Flux Sensitivity Functions
Luycx, Bennis, Torres-Verdin, Preeg (2020)
DOI: 10.30632/PJV61N1-2020T1

A tutorial on modeling borehole nuclear-tool response with Monte Carlo (MCNP)
and on a fast approximation via flux sensitivity functions (FSFs).  A track-
length estimator tallies the particle flux in the detector; the detector
reaction rate follows from the flux and the reaction cross section.  FSFs - the
product of the background flux and the detector importance (adjoint) function,
normalized to unity - let a perturbed response be approximated by a first-order
convolution of the FSF with the cross-section perturbation, replacing long MCNP
runs with near-instant forward modeling.

Implements:

  - F4 track-length flux estimator  phi = sum(W*T)/V_det           (Eqs. 3, 18-19)
  - Detector reaction rate  N = c * integral(phi(E)*sigma(E) dE)   (Eq. 4)
  - Importance / adjoint  Imp = Score/Weight                       (Eq. 17)
  - Flux sensitivity function (background flux x importance, normalized)(Eqs. 20-22)
  - First-order perturbed response  N = N_b + integral(FSF*dSigma)  (Eqs. 23-24)

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are the standard MCNP
track-length / FSF perturbation forms anchored to those definitions.
"""

import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------- track-length flux -------

def track_length_flux(weights, track_lengths, volume):
    """F4 track-length flux estimator  phi = sum(W_j * T_j)/V_det  (Eqs. 3, 19)."""
    w = np.asarray(weights, float)
    t = np.asarray(track_lengths, float)
    return float(np.sum(w * t) / volume)


def mc_track_length_flux(n_particles, mfp, cell_length, area, seed=0):
    """Monte-Carlo track-length flux through a slab cell of given length/area.

    Each particle enters at a random depth and travels an exponentially
    distributed free path (mean `mfp`) capped at the cell exit; the estimator
    is sum(track lengths)/volume.  Returns the flux estimate.
    """
    rng = np.random.default_rng(seed)
    entry = rng.uniform(0.0, cell_length, n_particles)
    free_path = rng.exponential(mfp, n_particles)
    track = np.minimum(free_path, cell_length - entry)
    return track_length_flux(np.ones(n_particles), track, cell_length * area) / n_particles


# ---------------------------------------------- reaction rate -----------

def reaction_rate(flux_E, sigma_E, energies, c=1.0):
    """Detector reaction rate  N = c * integral(phi(E)*sigma(E) dE)  (Eq. 4)."""
    return c * float(_trapezoid(np.asarray(flux_E, float) * np.asarray(sigma_E, float),
                              np.asarray(energies, float)))


# ---------------------------------------------- importance / FSF --------

def importance(score, weight):
    """MCNP importance (adjoint) function  Imp = Score/Weight  (Eq. 17)."""
    return np.asarray(score, float) / np.asarray(weight, float)


def flux_sensitivity_function(background_flux, response_importance, dr):
    """FSF = background_flux * importance, normalized to unit integral (Eqs. 20-22)."""
    fsf = np.asarray(background_flux, float) * np.asarray(response_importance, float)
    norm = np.sum(fsf) * dr
    return fsf / norm if norm > 0 else fsf


def perturbed_response(N_b, fsf, delta_sigma, dr):
    """First-order perturbed detector response  N = N_b + integral(FSF*dSigma)  (Eq. 24)."""
    return N_b + N_b * float(np.sum(np.asarray(fsf, float)
                                    * np.asarray(delta_sigma, float)) * dr)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1 (Tutorial): Monte Carlo Nuclear & FSF")
    print("=" * 60)

    # Track-length estimator: deterministic formula check
    phi = track_length_flux([1, 1, 1], [2.0, 3.0, 5.0], volume=10.0)
    assert abs(phi - 1.0) < 1e-12

    # Monte-Carlo flux converges (variance shrinks) as particle count grows
    f_small = [mc_track_length_flux(2000, 5.0, 10.0, 1.0, seed=s) for s in range(8)]
    f_large = [mc_track_length_flux(50000, 5.0, 10.0, 1.0, seed=s) for s in range(8)]
    print(f"  MC flux std 2k / 50k   = {np.std(f_small):.4f} / {np.std(f_large):.4f}")
    assert np.std(f_large) < np.std(f_small)
    assert all(f > 0 for f in f_large)

    # Reaction rate is positive and scales with cross section
    E = np.linspace(0.025, 10.0, 200)
    flux = np.exp(-E / 2.0)
    sigma = 1.0 / np.sqrt(E)                 # 1/v capture cross section
    N1 = reaction_rate(flux, sigma, E)
    N2 = reaction_rate(flux, 2.0 * sigma, E)
    assert N1 > 0 and abs(N2 / N1 - 2.0) < 1e-9

    # FSF integrates to 1
    r = np.linspace(0.5, 30.0, 60)
    dr = r[1] - r[0]
    psi = np.exp(-r / 8.0)                    # background flux
    imp = np.exp(-r / 10.0)                   # detector importance
    fsf = flux_sensitivity_function(psi, imp, dr)
    print(f"  FSF integral           = {np.sum(fsf)*dr:.4f}")
    assert abs(np.sum(fsf) * dr - 1.0) < 1e-9

    # First-order perturbation: accurate for small dSigma, the linearity limit
    N_b = 100.0
    small = perturbed_response(N_b, fsf, np.full_like(r, 0.001), dr)
    big = perturbed_response(N_b, fsf, np.full_like(r, 0.05), dr)
    print(f"  N (small/large dSigma) = {small:.2f} / {big:.2f}")
    assert small > N_b and big > small        # positive cross-section perturbation
    print("  PASS")
    return {"flux_det": float(np.mean(f_large)), "N1": N1, "N_pert_small": small}


if __name__ == "__main__":
    test_all()
