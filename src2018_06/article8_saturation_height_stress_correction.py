"""
Article 8: Saturation-Height Modeling: Assessing Capillary Pressure Stress
           Corrections
Hulea (2018)
DOI: 10.30632/PJV59N3-2018a7  (inferred - see note)

Laboratory capillary-pressure curves are measured on unstressed (or partially
stressed) core, but the reservoir is under net overburden stress, which closes
pore throats, lowers permeability and porosity, and therefore raises capillary
pressure.  This *methodology proxy* implements the saturation-height workflow the
paper assesses: the Leverett J-function, the saturation-height function above
the free-water level, a net-stress correction to permeability/porosity that
rescales capillary pressure, and a Brooks-Corey saturation curve.

Implements:

  - Leverett J-function  J = Pc*sqrt(k/phi)/(sigma*cos(theta))
  - Saturation height  h = Pc/((rho_w - rho_hc)*g)
  - Net-stress permeability/porosity correction and the rescaled Pc
  - Brooks-Corey saturation  Sw = Swirr + (1 - Swirr)*(Pe/Pc)^lambda

Note: this article's body was beyond this issue's machine extraction (the source
text ended at journal p372), so - as with the other methodology proxies in this
repository - the relations below are the standard saturation-height formulas the
paper assesses, not formulas transcribed from it.  The DOI suffix (a7) is
inferred from the issue's confirmed pattern.  SI units.
"""

import numpy as np

G_ACCEL = 9.81


# ---------------------------------------------- saturation height --------------

def leverett_j(pc, k, phi, sigma, theta_deg):
    """Leverett J-function  J = Pc*sqrt(k/phi)/(sigma*cos(theta))."""
    return np.asarray(pc, float) * np.sqrt(k / phi) / (sigma * np.cos(np.radians(theta_deg)))


def saturation_height(pc, rho_w, rho_hc):
    """Height above the free-water level  h = Pc/((rho_w - rho_hc)*g)  (m)."""
    return np.asarray(pc, float) / ((rho_w - rho_hc) * G_ACCEL)


# ---------------------------------------------- stress correction --------------

def stressed_permeability(k0, net_stress, c_k=2.0e-8):
    """Permeability under net stress  k = k0*exp(-c_k*stress)  (exponential closure)."""
    return k0 * np.exp(-c_k * np.asarray(net_stress, float))


def stressed_porosity(phi0, net_stress, c_phi=2.0e-9):
    """Porosity under net stress  phi = phi0*exp(-c_phi*stress)."""
    return phi0 * np.exp(-c_phi * np.asarray(net_stress, float))


def stress_corrected_pc(pc_lab, k0, phi0, k_stress, phi_stress):
    """Rescale lab Pc to reservoir stress by holding the J-function constant

        Pc_res = Pc_lab*sqrt((k0/phi0)/(k_stress/phi_stress)).

    Stress closes throats (k, phi fall), so Pc rises at a given saturation.
    """
    return pc_lab * np.sqrt((k0 / phi0) / (k_stress / phi_stress))


def brooks_corey_sw(pc, pc_entry, lam=2.0, swirr=0.1):
    """Brooks-Corey saturation  Sw = Swirr + (1 - Swirr)*(Pe/Pc)^lambda  (Pc >= Pe)."""
    pc = np.asarray(pc, float)
    sw = swirr + (1.0 - swirr) * (pc_entry / pc) ** lam
    return np.clip(sw, swirr, 1.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Saturation-Height Stress Corrections (proxy)")
    print("=" * 60)

    # J-function and saturation height both rise with capillary pressure
    assert leverett_j(2e5, 1e-13, 0.2, 0.03, 30.0) > leverett_j(1e5, 1e-13, 0.2, 0.03, 30.0)
    assert saturation_height(1e5, 1000.0, 700.0) > saturation_height(1e4, 1000.0, 700.0)

    # Net stress closes pore throats: k and phi fall
    k0, phi0, stress = 1e-13, 0.20, 3.0e7              # 30 MPa net stress
    ks, phis = stressed_permeability(k0, stress), stressed_porosity(phi0, stress)
    print(f"  k0/k_stress            = {k0:.2e} / {ks:.2e} m^2")
    assert ks < k0 and phis < phi0

    # ... so the stress-corrected capillary pressure is higher than the lab value
    pc_res = stress_corrected_pc(1e5, k0, phi0, ks, phis)
    print(f"  Pc lab/reservoir       = 1.0e5 / {pc_res:.3e} Pa")
    assert pc_res > 1e5

    # Brooks-Corey: Sw = 1 at the entry pressure and decreases as Pc rises
    assert np.isclose(brooks_corey_sw(1e4, pc_entry=1e4), 1.0)
    sw = brooks_corey_sw(np.array([1e4, 5e4, 2e5]), pc_entry=1e4, lam=2.0, swirr=0.15)
    print(f"  Brooks-Corey Sw        = {np.array2string(sw, precision=3)}")
    assert sw[0] > sw[1] > sw[2] and sw[-1] >= 0.15
    print("  PASS")
    return {"Pc_reservoir": float(pc_res), "Sw_curve": sw.tolist()}


if __name__ == "__main__":
    test_all()
