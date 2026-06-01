"""
A Method to Characterize the Emulsions in Oil-Water Production Wells
Using Multifrequency Dielectric Technique

Reference:
    Albenayyan, N., Hassan, A., El-Husseiny, A., and Mahmoud, M. (2026).
    A Method to Characterize the Emulsions in Oil-Water Production Wells
    Using Multifrequency Dielectric Technique. Petrophysics, 67(3), 594-618.
    DOI: 10.30632/PJV67N3-2026a9   (cover paper)

The paper combines multifrequency dielectric measurements with microscopic
imaging to characterise water-in-oil (W/O) and oil-in-water (O/W) emulsions
across a wide frequency range, correlating the dielectric response with
droplet-scale features to identify emulsion type, stability, and time
evolution.

This module implements the standard effective-medium dielectric mixing models
used to interpret such measurements:
    - Bruggeman / Hanai-Bruggeman asymmetric mixing.
    - Maxwell-Garnett mixing for dilute dispersions.
    - An emulsion-type discriminator (W/O vs O/W) from the low-frequency
      permittivity / conductivity, since a continuous water phase (O/W) is
      far more conductive and polarisable than a continuous oil phase (W/O).
    - A simple coalescence / phase-separation kinetics model for stability vs
      time, and a droplet-size estimate from the dielectric relaxation time.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

EPS0 = 8.8541878128e-12  # F/m


# ---------------------------------------------------------------------------
# 1. Effective-medium mixing models
# ---------------------------------------------------------------------------

def maxwell_garnett(eps_host: complex, eps_incl: complex,
                    vol_incl: float) -> complex:
    """
    Maxwell-Garnett effective permittivity for a dilute dispersion of
    spherical inclusions (permittivity eps_incl, volume fraction vol_incl)
    in a host (eps_host):

        (e_eff - e_h)/(e_eff + 2 e_h) = f * (e_i - e_h)/(e_i + 2 e_h)
    """
    f = vol_incl
    num = (eps_incl - eps_host) / (eps_incl + 2.0 * eps_host)
    beta = f * num
    return eps_host * (1.0 + 2.0 * beta) / (1.0 - beta)


def bruggeman(eps_host: complex, eps_incl: complex, vol_incl: float,
              n_iter: int = 200) -> complex:
    """
    Hanai-Bruggeman asymmetric effective permittivity (valid to high volume
    fractions), solved by fixed-point iteration of

        (e_i - e_eff)/(e_i - e_h) * (e_h/e_eff)**(1/3) = 1 - f
    """
    eps_eff = (1.0 - vol_incl) * eps_host + vol_incl * eps_incl  # init
    target = 1.0 - vol_incl
    for _ in range(n_iter):
        ratio = (eps_incl - eps_eff) / (eps_incl - eps_host) * \
            (eps_host / eps_eff) ** (1.0 / 3.0)
        # Nudge eps_eff so that |ratio| -> target.
        eps_eff = eps_eff * (1.0 + 0.3 * (abs(ratio) - target))
    return eps_eff


# ---------------------------------------------------------------------------
# 2. Emulsion-type discrimination (W/O vs O/W)
# ---------------------------------------------------------------------------

@dataclass
class EmulsionState:
    eps_real_lowf: float   # low-frequency real permittivity
    conductivity: float    # low-frequency conductivity, S/m
    water_cut: float       # water volume fraction (0-1)


def emulsion_type(state: EmulsionState,
                  cond_threshold: float = 1e-2) -> str:
    """
    Discriminate emulsion type from the low-frequency response.

    An oil-in-water (O/W) emulsion has a continuous (conductive, high-
    permittivity) water phase, whereas a water-in-oil (W/O) emulsion has a
    continuous (resistive, low-permittivity) oil phase. A continuous-water
    system shows conductivity well above the threshold.
    """
    if state.conductivity >= cond_threshold and state.eps_real_lowf > 20.0:
        return "oil-in-water (O/W)"
    return "water-in-oil (W/O)"


def inversion_point_proximity(water_cut: float,
                              critical_wc: float = 0.5) -> float:
    """
    Distance of the current water cut from the phase-inversion point
    (the W/O -> O/W transition), normalised to [0, 1]; small values flag an
    unstable emulsion near inversion.
    """
    return abs(water_cut - critical_wc) / max(critical_wc, 1e-9)


# ---------------------------------------------------------------------------
# 3. Stability kinetics and droplet size
# ---------------------------------------------------------------------------

def separation_fraction(t_min: np.ndarray, rate_const: float) -> np.ndarray:
    """
    First-order phase-separation (coalescence) kinetics; fraction of the
    dispersed phase that has separated after time t:

        S(t) = 1 - exp(-k t)

    A larger rate constant k => less stable emulsion.
    """
    return 1.0 - np.exp(-rate_const * np.asarray(t_min, float))


def droplet_radius_from_relaxation(tau: float, eps_cont: float,
                                   sigma_cont: float) -> float:
    """
    Estimate the dispersed-droplet radius from the interfacial (Maxwell-Wagner)
    relaxation time of the emulsion:

        tau = (eps0 * (2 e_cont + e_drop)) / (2 sigma_cont)
        => a proxy droplet length-scale  r ~ sqrt(tau * sigma_cont / eps0)

    Returns a relative droplet length scale (arbitrary consistent units).
    """
    return math.sqrt(max(tau, 0.0) * sigma_cont / (EPS0 * eps_cont))


# ---------------------------------------------------------------------------
# 4. Convenience: full workflow example
# ---------------------------------------------------------------------------

def example_workflow():
    """Run a complete example and print key results."""
    print("=" * 64)
    print("MF Dielectric Characterisation of Oil-Water Emulsions")
    print("Ref: Albenayyan et al., Petrophysics 67(3) 2026 (cover)")
    print("=" * 64)

    eps_water = complex(70.0, -40.0)   # brine at low frequency
    eps_oil = complex(2.3, -0.001)     # dead oil

    print("\nEffective permittivity vs water cut (W/O, oil continuous):")
    for wc in (0.1, 0.3, 0.5):
        mg = maxwell_garnett(eps_oil, eps_water, wc)
        bg = bruggeman(eps_oil, eps_water, wc)
        print(f"  wc={wc:.1f}  Maxwell-Garnett e'={mg.real:6.2f}  "
              f"Bruggeman e'={bg.real:6.2f}")

    states = [
        EmulsionState(eps_real_lowf=8.0,  conductivity=1e-4, water_cut=0.25),
        EmulsionState(eps_real_lowf=55.0, conductivity=0.4,  water_cut=0.70),
    ]
    print("\nEmulsion-type discrimination:")
    for s in states:
        print(f"  wc={s.water_cut:.2f}  e'={s.eps_real_lowf:5.1f}  "
              f"sigma={s.conductivity:.0e} -> {emulsion_type(s)}  "
              f"(inv-proximity {inversion_point_proximity(s.water_cut):.2f})")

    t = np.array([0, 10, 30, 60, 120])
    sep = separation_fraction(t, rate_const=0.02)
    print(f"\nPhase-separation over time (k=0.02/min): "
          f"{np.array2string(sep, precision=2)}")

    r = droplet_radius_from_relaxation(tau=1e-7, eps_cont=70.0, sigma_cont=0.4)
    print(f"Droplet length-scale proxy from relaxation: {r:.3e}")

    return states


if __name__ == "__main__":
    example_workflow()
