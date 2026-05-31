"""
Article 3: The Dynamics of Reservoir Fluids and their Substantial Systematic
           Variations
Oliver C. Mullins, Julian Y. Zuo, Kang Wang, Paul S. Hammond, Ilaria De Santo,
Hadrien Dumont, Vinay K. Mishra, Li Chen, Andrew E. Pomerantz, Chengli Dong,
Hani Elshahawi, Douglas J. Seifert (2014)
Reference: Petrophysics Vol. 55, No. 2 (April 2014), pp. 96-112
DOI: none assigned (this issue predates SPWLA DOI assignment)

Special Issue on Deepwater.  Reservoir-fluid gradients are set by the interplay
of diffusion, buoyancy-driven gravity currents and the gravitational
(Boltzmann) distribution of asphaltene particles, whose sizes follow the
Yen-Mullins model.  This module implements those transport/equilibrium relations.

Implements:

  - 1D diffusion length / time  x^2 = 2*D*t  (Eq. 1)
  - Steady-state gravity-current velocity  V = k*drho*g*sin(theta)/(phi*mu) (Eq. 2)
  - Asphaltene gravitational (Boltzmann) distribution  A_h/A_o = exp(...) (Eq. 3)
  - Asphaltene gradient half-height
  - Yen-Mullins particle sizes (molecule / nanoaggregate / cluster)

Note: this issue's PDF dropped the display equations in extraction; Eq. 1
survived (x^2 = 2 D t) and Eqs. 2-3 are reconstructed from the surrounding text
and nomenclature (Mullins, 2010; Zuo et al., 2010).  SI units; concentrations
relative.
"""

import numpy as np

KB = 1.380649e-23  # Boltzmann constant, J/K
G = 9.81           # m/s^2

# Yen-Mullins asphaltene particle diameters (m)
YEN_MULLINS = {"molecule": 1.5e-9, "nanoaggregate": 2.0e-9, "cluster": 5.0e-9}


# ---------------------------------------------- diffusion --------------

def diffusion_length(diffusion, time):
    """1D diffusion length (Eq. 1)

        x = sqrt(2*D*t).
    """
    return np.sqrt(2.0 * diffusion * np.asarray(time, float))


def diffusion_time(distance, diffusion):
    """Time to diffuse a distance (Eq. 1, inverted)

        t = x^2/(2*D).
    """
    return np.asarray(distance, float) ** 2 / (2.0 * diffusion)


# ---------------------------------------------- gravity current --------------

def gravity_current_velocity(permeability, delta_density, dip_deg, porosity,
                             viscosity):
    """Steady-state buoyancy-driven gravity-current velocity (Eq. 2)

        V = k*drho*g*sin(theta)/(phi*mu),

    with permeability k (m^2), density inversion drho (kg/m^3), dip theta,
    porosity phi and viscosity mu (Pa*s).
    """
    return (permeability * delta_density * G * np.sin(np.radians(dip_deg))
            / (porosity * viscosity))


# ---------------------------------------------- asphaltene gradient --------------

def asphaltene_boltzmann(height, ref_height, particle_volume, delta_density,
                         temperature):
    """Asphaltene gravitational (Boltzmann) concentration ratio (Eq. 3)

        A_h/A_o = exp[-V*drho*g*(h - h_o)/(kB*T)],

    the gravity-only limit of the FHZ EoS for low-GOR oils, with the particle
    volume V (m^3) and density contrast drho (kg/m^3).
    """
    energy = particle_volume * delta_density * G * (np.asarray(height, float)
                                                    - ref_height)
    return np.exp(-energy / (KB * temperature))


def asphaltene_half_height(particle_diameter, delta_density, temperature):
    """Height over which the asphaltene concentration drops by half

        h_1/2 = kB*T*ln(2)/(V*drho*g),   V = (pi/6)*d^3.

    ~20 m for 5-nm clusters; ~cm-scale for hypothetical larger aggregates.
    """
    v = np.pi / 6.0 * particle_diameter ** 3
    return KB * temperature * np.log(2.0) / (v * delta_density * G)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Dynamics of Reservoir Fluids")
    print("=" * 60)

    # Diffusion: methane over 15 m takes ~1 Ma (order of magnitude).  The
    # in-rock effective diffusion is the free-fluid value (~1e-6 cm^2/s) reduced
    # by tortuosity (~1e-7 cm^2/s = 1e-11 m^2/s).
    d_methane = 1e-7 * 1e-4  # effective D in rock, cm^2/s -> m^2/s
    t = diffusion_time(15.0, d_methane)
    years = t / (365.25 * 24 * 3600)
    print(f"  diffusion time over 15 m = {years:.2e} yr")
    assert 1e5 < years < 1e7
    # round-trip the length
    assert np.isclose(diffusion_length(d_methane, t), 15.0)

    # Gravity current: a moderate density inversion drives a finite velocity
    v = gravity_current_velocity(permeability=100 * 9.869e-16, delta_density=10.0,
                                 dip_deg=10.0, porosity=0.20, viscosity=1e-3)
    print(f"  gravity-current velocity = {v:.2e} m/s")
    assert v > 0
    # steeper dip and higher permeability speed it up
    assert gravity_current_velocity(100 * 9.869e-16, 10.0, 20.0, 0.20, 1e-3) > v

    # Asphaltene Boltzmann: concentration falls with height above reference
    a = asphaltene_boltzmann(height=20.0, ref_height=0.0,
                             particle_volume=np.pi / 6 * (2e-9) ** 3,
                             delta_density=200.0, temperature=350.0)
    print(f"  A(20 m)/A(0) = {a:.3f}")
    assert 0 < a < 1

    # Half-height: clusters (5 nm) settle on a ~10-m scale, much shorter than
    # nanoaggregates (2 nm)
    h_cluster = asphaltene_half_height(YEN_MULLINS["cluster"], 200.0, 350.0)
    h_nano = asphaltene_half_height(YEN_MULLINS["nanoaggregate"], 200.0, 350.0)
    print(f"  half-height: cluster={h_cluster:.1f} m  nanoaggregate={h_nano:.0f} m")
    assert h_cluster < h_nano and YEN_MULLINS["cluster"] > YEN_MULLINS["nanoaggregate"]
    print("  PASS")
    return {"diffusion_yr": float(years), "V_gravity": float(v),
            "half_height_cluster": float(h_cluster)}


if __name__ == "__main__":
    test_all()
