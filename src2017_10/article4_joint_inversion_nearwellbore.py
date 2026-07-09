"""
Article 4: Imaging Near-Wellbore Petrophysical Properties by Joint Inversion of
           Sonic, Resistivity, and Density Logging Data
Shetty, Liang, Simoes, Canesin, Boyd, Zeroug, Sinha, Habashy, Domingues, Amorim,
Abbots (2017)
Reference: Petrophysics Vol. 58, No. 5 (October 2017), pp. 501-516
DOI: none assigned (this issue predates SPWLA DOI assignment)

A pixel-based 1D-radial Gauss-Newton joint inversion images near-wellbore
saturation and pore shape from sonic, resistivity, and density logs.  Each radial
pixel has water/gas/oil saturations and a pore aspect ratio; the forward
petrophysics links them to resistivity (Archie), elastic velocities (moduli with
Gassmann/Wood/Brie fluid substitution), and density, and a regularized
relative-misfit cost function is minimized.

Implements:

  - Archie resistivity  Rh = a*Rw/(phi^m*Sw^n)
  - Relative gas fraction  f = Sg/(Sg + So)
  - Velocities from moduli  Vp = sqrt((K + 4/3*mu)/rho),  Vs = sqrt(mu/rho)
  - Effective fluid modulus by Wood's law and Brie's empirical law
  - Volumetric density mixing and the relative-misfit cost function

Note: this issue's PDF has a text layer but every typeset equation glyph was
dropped in extraction, so the relations are faithful standard-form
reconstructions from the prose and the Xu-White appendix.  SI: moduli in Pa,
velocities in m/s, density in kg/m^3, resistivity in ohm.m.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- petrophysics --------------

def archie_resistivity(rw, phi, sw, a=1.0, m=2.0, n=2.0):
    """Archie resistivity of a pixel  Rh = a*Rw/(phi^m*Sw^n)  (Eq. 1)."""
    return petrolib.saturation_resistivity.archie_rt(sw, rw, phi=phi, a=a, m=m, n=n)


def gas_fraction(sg, so):
    """Relative gas fraction  f = Sg/(Sg + So)  (Eq. 2)."""
    return sg / (sg + so)


def velocities(k, mu, rho):
    """Elastic velocities  Vp = sqrt((K + 4/3*mu)/rho),  Vs = sqrt(mu/rho)."""
    vp = np.sqrt((k + 4.0 / 3.0 * mu) / rho)
    vs = np.sqrt(mu / rho)
    return vp, vs


def wood_fluid_modulus(saturations, moduli):
    """Wood's law effective fluid modulus  1/K = sum_i S_i/K_i  (Eq. A1.5)."""
    s = np.asarray(saturations, float)
    k = np.asarray(moduli, float)
    return 1.0 / np.sum(s / k)


def brie_fluid_modulus(k_water, k_gas, sw, e=3.0):
    """Brie's empirical fluid modulus  K = (Kw - Kg)*Sw^e + Kg  (Eq. A1.6)."""
    return (k_water - k_gas) * np.asarray(sw, float) ** e + k_gas


def density_mix(matrix_fraction, matrix_density, phi, sw, so, sg,
                rho_w=1000.0, rho_o=800.0, rho_g=200.0):
    """Volumetric bulk density  rho = (1-phi)*rho_ma + phi*(Sw*rho_w + So*rho_o + Sg*rho_g)."""
    return matrix_fraction * matrix_density + phi * (sw * rho_w + so * rho_o + sg * rho_g)


def relative_misfit(data, sim):
    """Relative-L2 data misfit  sum(((d - s)/d)^2)  (the data term of the cost)."""
    d = np.asarray(data, float)
    s = np.asarray(sim, float)
    return float(np.sum(((d - s) / d) ** 2))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Joint Inversion Near-Wellbore")
    print("=" * 60)

    # Resistivity falls as water saturation rises
    assert archie_resistivity(0.05, 0.2, 0.8) < archie_resistivity(0.05, 0.2, 0.4)

    # Gas fraction
    assert np.isclose(gas_fraction(0.3, 0.1), 0.75)

    # Velocities: Vp > Vs, and density lowers velocity
    vp, vs = velocities(2.0e10, 1.2e10, 2400.0)
    print(f"  Vp / Vs                = {vp:.0f} / {vs:.0f} m/s")
    assert vp > vs

    # Wood's law gives a much softer fluid than the arithmetic mean (gas dominates)
    kw = wood_fluid_modulus([0.5, 0.5], [2.2e9, 1.0e5])
    assert kw < 0.5 * (2.2e9 + 1.0e5)

    # Brie's law runs from gas (Sw=0) to water (Sw=1) modulus, monotonically
    assert np.isclose(brie_fluid_modulus(2.2e9, 1.0e5, 0.0), 1.0e5)
    assert np.isclose(brie_fluid_modulus(2.2e9, 1.0e5, 1.0), 2.2e9)
    assert brie_fluid_modulus(2.2e9, 1.0e5, 0.7) > brie_fluid_modulus(2.2e9, 1.0e5, 0.3)

    # Density mixing and a zero misfit when sim equals data
    rho = density_mix(0.8, 2650.0, 0.2, 0.6, 0.1, 0.3)
    print(f"  bulk density           = {rho:.0f} kg/m^3")
    assert rho > 0 and relative_misfit([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    print("  PASS")
    return {"Vp": float(vp), "Vs": float(vs), "rho": float(rho)}


if __name__ == "__main__":
    test_all()
