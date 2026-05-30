"""
Article 2: Analysis of Lateral Fluid Gradients From DFA Measurements and
           Simulation of Reservoir Fluid Mixing Processes Over Geologic Time
Chen, Kristensen, Johansen, Achourov, Betancourt, Mullins (2021)
DOI: 10.30632/PJV62N1-2021a1

Downhole fluid analysis (DFA) measures optical density (asphaltene content) and
GOR versus depth.  Reservoir-connectivity and fluid-mixing questions are
addressed with the Flory-Huggins-Zuo (FHZ) equation of state, whose
gravity-dominated solution predicts an equilibrium asphaltene gradient, and
with a 1D diffusion model that simulates how an initial compositional contrast
relaxes toward equilibrium over geologic time.

Implements:

  - FHZ gravity term  OD(h2)/OD(h1) = exp[ v_a g (rho_f - rho_a)(h2-h1)/(RT) ]
  - Vertical asphaltene (optical-density) profile vs depth
  - 1D diffusive mixing:  erfc step-front profile and the column
    homogenization time  t ~ H^2/(pi^2 D)
  - Equilibrium vs disequilibrium (still-mixing) diagnosis

Note: this issue's source PDF has no usable text layer, so the FHZ gravity term
and the diffusion relations are faithful standard-form reconstructions of the
methods the paper applies (the full FHZ EOS adds solubility and entropy terms;
the gravity term dominates the asphaltene-nanoaggregate distribution).
SI units internally; depths in metres, T in kelvin.
"""

import numpy as np
from scipy.special import erfc

R_GAS = 8.314          # J/mol/K
G_ACCEL = 9.81         # m/s^2


# ---------------------------------------------- FHZ gravity term --------

def fhz_od_ratio(dz_m, v_a_m3mol=0.004, rho_fluid=700.0, rho_asph=1200.0, T=380.0):
    """Asphaltene optical-density ratio OD(z2)/OD(z1) over a depth step dz.

    Gravity (Boltzmann) term of the Flory-Huggins-Zuo EOS, depth positive
    downward:
        OD(z2)/OD(z1) = exp[ v_a g (rho_asph - rho_fluid)(z2 - z1) / (R T) ]
    v_a is the asphaltene (nanoaggregate) molar volume in m^3/mol (~0.002-0.01),
    rho in kg/m^3.  Asphaltene denser than the bulk fluid -> OD increases
    downward (ratio > 1 for dz > 0); a lighter asphaltene reverses the gradient.
    """
    expo = v_a_m3mol * G_ACCEL * (rho_asph - rho_fluid) * np.asarray(dz_m, float) \
        / (R_GAS * T)
    return np.exp(expo)


def asphaltene_profile(depth_m, od_ref, depth_ref, **kw):
    """Optical-density (asphaltene) profile vs depth from the FHZ gravity term."""
    depth = np.asarray(depth_m, float)
    return od_ref * fhz_od_ratio(depth - depth_ref, **kw)


# ---------------------------------------------- diffusive mixing --------

def diffusion_front(z_m, t_s, D=1e-9, c_top=0.0, c_bot=1.0, z0=0.0):
    """Concentration profile of a diffusing step front after time t.

        c(z,t) = c_top + (c_bot - c_top) * 0.5 * erfc( (z - z0) / (2 sqrt(D t)) )
    z positive downward, z0 the initial interface depth.  Returns c(z).
    """
    z = np.asarray(z_m, float)
    arg = (z - z0) / (2.0 * np.sqrt(max(D * t_s, 1e-30)))
    return c_top + (c_bot - c_top) * 0.5 * erfc(arg)


def mixing_time(height_m, D=1e-9):
    """Time to homogenize a column of height H by diffusion  t ~ H^2/(pi^2 D).

    The slowest-decaying Fourier mode of 1D diffusion in a sealed column of
    height H decays with time constant H^2/(pi^2 D); returns that constant (s).
    """
    return height_m ** 2 / (np.pi ** 2 * D)


def is_equilibrium(od_meas, depth, od_ref, depth_ref, rtol=0.05, **kw):
    """True if the measured OD profile matches the FHZ equilibrium gradient.

    A column whose DFA gradient matches FHZ is connected and at equilibrium;
    a systematic misfit indicates an active (incomplete) mixing process.
    """
    pred = asphaltene_profile(depth, od_ref, depth_ref, **kw)
    return bool(np.all(np.abs(np.asarray(od_meas, float) - pred) <= rtol * pred))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: DFA Lateral Fluid Gradients & Mixing")
    print("=" * 60)

    # FHZ: asphaltene (OD) increases downward because asphaltene is denser
    depth = np.array([3000.0, 3050.0, 3100.0, 3150.0])
    od = asphaltene_profile(depth, od_ref=0.30, depth_ref=3000.0)
    print(f"  OD vs depth            = {np.array2string(od, precision=4)}")
    assert np.all(np.diff(od) > 0)            # monotonically increasing downward
    assert abs(od[0] - 0.30) < 1e-9

    # Reversing the density contrast (light asphaltene) flips the gradient
    od_inv = asphaltene_profile(depth, 0.30, 3000.0, rho_asph=600.0)
    assert np.all(np.diff(od_inv) < 0)

    # Diffusion front: at the interface c = mean; spreads with sqrt(D t)
    z = np.linspace(-20.0, 20.0, 41)
    early = diffusion_front(z, t_s=mixing_time(5.0))      # short column-scale
    late = diffusion_front(z, t_s=mixing_time(50.0))      # long time
    mid = np.argmin(np.abs(z))
    print(f"  front c at interface   = {early[mid]:.3f}  (expect 0.5)")
    assert abs(early[mid] - 0.5) < 1e-9
    # the later/longer-time front is smoother (smaller max gradient)
    assert np.max(np.abs(np.diff(late))) < np.max(np.abs(np.diff(early)))

    # Mixing time scales as H^2 / D : doubling height quadruples the time
    t1 = mixing_time(10.0); t2 = mixing_time(20.0)
    print(f"  mixing time 10m / 20m  = {t1:.2e} / {t2:.2e} s")
    assert abs(t2 / t1 - 4.0) < 1e-9

    # Equilibrium test: a clean FHZ profile passes; a perturbed one fails
    assert is_equilibrium(od, depth, 0.30, 3000.0)
    assert not is_equilibrium(od * np.array([1.0, 1.2, 0.8, 1.3]),
                              depth, 0.30, 3000.0)
    # Geologic-time context: a 50-m column mixes in ~10^11 s (~1000s of years+)
    print(f"  50 m column mixing time = {mixing_time(50.0):.2e} s")
    print("  PASS")
    return {"od_grad": od, "t_mix_50m_s": mixing_time(50.0)}


if __name__ == "__main__":
    test_all()
