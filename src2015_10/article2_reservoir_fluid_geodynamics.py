"""
Article 2: Differing Equilibration Times of GOR, Asphaltenes and Biomarkers as
           Determined by Charge History and Reservoir Fluid Geodynamics
Wang, Kauerauf, Zuo, Chen, Dong, Elshahawi, Mullins (2015)
Reference: Petrophysics Vol. 56, No. 5 (October 2015), pp. 440-456
DOI: none assigned (this issue predates SPWLA DOI assignment)

Reservoir fluid geodynamics (RFG): different crude-oil components (GOR,
asphaltenes, biomarkers) equilibrate over geologic time at different rates, set
by charge history and diffusion.  Asphaltene gradients are modeled with the
Flory-Huggins-Zuo (FHZ) equation of state, whose dominant term is the
gravitational (Boltzmann) term on the asphaltene particle (Yen-Mullins
molecule/nanoaggregate/cluster sizes).  Diffusive mixing over a length L takes a
time ~ L^2/D, so small fast-diffusing molecules equilibrate long before the
large asphaltene clusters.

Implements:

  - Gravitational force on an asphaltene particle  F = m*g  (Eq. 1)
  - FHZ asphaltene (optical-density) gravity gradient (Eq. 2, dominant term)
  - Diffusion length over time  L = sqrt(D*t)
  - Diffusive equilibration time  tau = L^2/D

Note: this issue's PDF has a text layer; the FHZ gravity term (Eq. 2) and the
diffusion relations are transcribed from the body, while the typeset glyphs were
dropped and reconstructed in standard form.  SI units: volumes in m^3, density
in kg/m^3, height in m, temperature in K, D in m^2/s, time in s.
"""

import numpy as np

KB = 1.380649e-23             # Boltzmann constant, J/K
G_EARTH = 9.81                # m/s^2


# ---------------------------------------------- gravity --------------

def gravity_force(mass, g=G_EARTH):
    """Gravitational force on an asphaltene particle  F = m*g  (Eq. 1)."""
    return mass * g


# ---------------------------------------------- FHZ asphaltene gradient --------------

def fhz_gravity_gradient(od_ref, h_ref, h, particle_volume, delta_rho, temperature):
    """FHZ asphaltene optical-density gradient, gravitational term (Eq. 2)

        OD(h) = OD_ref * exp[ Va*delta_rho*g*(h_ref - h) / (kB*T) ],

    where Va is the asphaltene-particle volume (Yen-Mullins: molecule <
    nanoaggregate < cluster), delta_rho the asphaltene-oil density difference,
    and h increases upward.  Asphaltene content (OD) increases downward; the
    entropy and solubility terms are smaller and omitted here.
    """
    exponent = particle_volume * delta_rho * G_EARTH * (h_ref - h) / (KB * temperature)
    return od_ref * np.exp(exponent)


# ---------------------------------------------- diffusion --------------

def diffusion_length(diffusion_coeff, time):
    """Characteristic diffusion length  L = sqrt(D*t)."""
    return np.sqrt(diffusion_coeff * np.asarray(time, float))


def equilibration_time(length, diffusion_coeff):
    """Diffusive equilibration time over a length  tau = L^2/D,

    so a smaller diffusion coefficient (larger asphaltene cluster) needs far
    longer to equilibrate than a fast-diffusing gas molecule (GOR).
    """
    return length ** 2 / diffusion_coeff


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Reservoir Fluid Geodynamics")
    print("=" * 60)

    # Newton's gravity term
    assert np.isclose(gravity_force(1e-21), 1e-21 * 9.81)

    # FHZ: asphaltene optical density increases downward (h < h_ref)
    va = (4.0 / 3.0) * np.pi * (1.0e-9) ** 3      # ~2 nm nanoaggregate radius
    od_top = 1.0
    od_down = fhz_gravity_gradient(od_top, h_ref=0.0, h=-100.0, particle_volume=va,
                                   delta_rho=60.0, temperature=350.0)
    od_up = fhz_gravity_gradient(od_top, h_ref=0.0, h=100.0, particle_volume=va,
                                 delta_rho=60.0, temperature=350.0)
    print(f"  OD down/ref/up         = {od_down:.4f} / {od_top:.4f} / {od_up:.4f}")
    assert od_down > od_top > od_up

    # Larger particles (clusters) give a steeper gradient than nanoaggregates
    va_cluster = (4.0 / 3.0) * np.pi * (3.0e-9) ** 3
    od_cluster = fhz_gravity_gradient(od_top, 0.0, -100.0, va_cluster, 60.0, 350.0)
    assert od_cluster > od_down

    # Diffusion length grows as sqrt(D*t); ~ tens of m over 50 Myr at D=1e-7 cm2/s
    d = 1.0e-11                                   # m^2/s  (1e-7 cm^2/s)
    t50 = 50e6 * 3.156e7                          # 50 Myr in seconds
    L = diffusion_length(d, t50)
    print(f"  diffusion length 50Myr = {L:.1f} m")
    assert L > 0

    # Equilibration time: fast-diffusing GOR equilibrates before slow asphaltenes
    tau_gor = equilibration_time(50.0, 1.0e-9)    # gas, high D
    tau_asph = equilibration_time(50.0, 1.0e-12)  # cluster, low D
    print(f"  tau GOR / asphaltene   = {tau_gor:.2e} / {tau_asph:.2e} s")
    assert tau_asph > tau_gor > 0
    print("  PASS")
    return {"OD_down": float(od_down), "L_50Myr": float(L), "tau_ratio": float(tau_asph / tau_gor)}


if __name__ == "__main__":
    test_all()
