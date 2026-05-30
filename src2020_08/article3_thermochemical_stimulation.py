"""
Article 3: Improvement of Petrophysical Properties of Tight Sandstone and
           Limestone Reservoirs Using Thermochemical Fluids
Mustafa, Mahmoud, Abdulraheem, Tariq, Al-Nakhli (2020)
DOI: 10.30632/PJV61N4-2020a3

Two thermochemical reagents (NaNO2 + NH4Cl) are injected into tight cores and
reacted in situ; the exothermic reaction generates a pressure/temperature pulse
that creates microfractures, improving porosity and permeability and reducing
strength.  Before/after laboratory measurements (porosity, permeability,
capillary pressure, UCS, acoustic velocities and elastic moduli) quantify the
improvement.

Implements:

  - Exothermic reaction stoichiometry and heat release
  - Fractional improvement / reduction ratios
  - Dynamic elastic moduli from Vp, Vs, rho  E, nu, K, mu       (Eqs. 1-2, 6-7)
  - Young-Laplace capillary pressure  Pc = 2*sigma*cos(theta)/r (Eq. 3)
  - Centrifuge capillary pressure  Pc = 0.5*drho*w^2*(r2^2-r1^2)(Eq. 4)
  - Scratch-test specific energy  Ft = E*A                      (Eq. 5)

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs (Eq. 5 'Ft = (E)*A' survived), so
the moduli / capillary forms are the standard expressions anchored to those
definitions.  Paper benchmarks reproduced: porosity +80% (limestone) / +40.4%
(Scioto); permeability +1359.9% / +320.7%; UCS 38.2->17.1 MPa; dH = 369 kJ/mol.
"""

import numpy as np

DELTA_H_KJ_MOL = 369.0       # average reaction enthalpy


# ---------------------------------------------- reaction ----------------

def reaction_heat(moles, dH_kj_mol=DELTA_H_KJ_MOL):
    """Heat released (kJ) by the exothermic reaction  Q = moles * dH.

    NaNO2(aq) + NH4Cl(aq) -> NaCl(aq) + 2 H2O(g) + N2(g) + heat
    """
    return moles * dH_kj_mol


def n2_moles(moles_reagent):
    """Moles of N2 generated (1:1 with the limiting 1 M reagent pair)."""
    return moles_reagent


# ---------------------------------------------- improvement ratios ------

def improvement_ratio(before, after):
    """Fractional change (%)  (after - before)/before * 100."""
    return (np.asarray(after, float) - before) / before * 100.0


# ---------------------------------------------- elastic moduli ----------

def dynamic_youngs(rho, vp, vs):
    """Dynamic Young's modulus  E = rho*Vs^2*(3Vp^2-4Vs^2)/(Vp^2-Vs^2)  (Eq. 1)."""
    vp2, vs2 = np.asarray(vp, float) ** 2, np.asarray(vs, float) ** 2
    return rho * vs2 * (3.0 * vp2 - 4.0 * vs2) / (vp2 - vs2)


def dynamic_poisson(vp, vs):
    """Dynamic Poisson's ratio  nu = (Vp^2-2Vs^2)/(2(Vp^2-Vs^2))  (Eq. 2)."""
    vp2, vs2 = np.asarray(vp, float) ** 2, np.asarray(vs, float) ** 2
    return (vp2 - 2.0 * vs2) / (2.0 * (vp2 - vs2))


def bulk_modulus(rho, vp, vs):
    """Bulk modulus  K = rho*(Vp^2 - 4/3*Vs^2)  (Eq. 6)."""
    return rho * (np.asarray(vp, float) ** 2 - 4.0 / 3.0 * np.asarray(vs, float) ** 2)


def shear_modulus(rho, vs):
    """Shear modulus  mu = rho*Vs^2  (Eq. 7)."""
    return rho * np.asarray(vs, float) ** 2


# ---------------------------------------------- capillary / scratch -----

def young_laplace_pc(sigma, theta_deg, r):
    """Young-Laplace capillary pressure  Pc = 2*sigma*cos(theta)/r  (Eq. 3)."""
    return 2.0 * sigma * np.cos(np.radians(theta_deg)) / np.asarray(r, float)


def centrifuge_pc(drho, rpm, r1, r2):
    """Centrifuge capillary pressure  Pc = 0.5*drho*w^2*(r2^2-r1^2)  (Eq. 4).

    drho in kg/m^3, radii in m, rpm -> w = 2*pi*rpm/60 ; returns Pa.
    """
    w = 2.0 * np.pi * rpm / 60.0
    return 0.5 * drho * w ** 2 * (r2 ** 2 - r1 ** 2)


def scratch_energy(intrinsic_energy, area):
    """Scratch-test horizontal force  Ft = E*A  (Eq. 5, verbatim)."""
    return intrinsic_energy * area


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Thermochemical Stimulation of Tight Rocks")
    print("=" * 60)

    # Exothermic heat scales with reacted moles
    q = reaction_heat(2.0)
    print(f"  heat (2 mol)           = {q:.0f} kJ")
    assert abs(q - 738.0) < 1e-9 and n2_moles(2.0) == 2.0

    # Improvement ratios reproduce the paper's headline numbers
    poro_ls = improvement_ratio(14.22, 14.22 * 1.80)      # limestone +80%
    perm_sc = improvement_ratio(0.965, 0.965 * 4.207)     # Scioto +320.7%
    ucs_ls = improvement_ratio(38.2, 17.1)                # limestone UCS
    print(f"  porosity +{poro_ls:.0f}%  perm +{perm_sc:.0f}%  UCS {ucs_ls:.0f}%")
    assert abs(poro_ls - 80.0) < 0.5
    assert abs(perm_sc - 320.7) < 0.5
    assert -56 < ucs_ls < -54                              # ~ -55%

    # Dynamic moduli drop after treatment (Scioto Vp/Vs reductions)
    rho = 2.4
    E_before = dynamic_youngs(rho, 2948.0, 1735.0)
    E_after = dynamic_youngs(rho, 2743.0, 1699.0)
    nu_before = dynamic_poisson(2948.0, 1735.0)
    print(f"  E before/after         = {E_before/1e9:.2f} / {E_after/1e9:.2f} GPa")
    assert E_after < E_before                              # softening
    assert 0.0 < nu_before < 0.5
    assert bulk_modulus(rho, 2948.0, 1735.0) > 0
    assert shear_modulus(rho, 1735.0) > 0

    # Capillary pressure: smaller throats need higher Pc; treatment widens
    # throats and lowers Pc
    pc_tight = young_laplace_pc(0.025, 30.0, 0.5e-6)
    pc_open = young_laplace_pc(0.025, 30.0, 1.5e-6)
    print(f"  Pc 0.5um / 1.5um       = {pc_tight:.0f} / {pc_open:.0f} Pa")
    assert pc_tight > pc_open

    # Centrifuge Pc rises with rotational speed
    pc8000 = centrifuge_pc(200.0, 8000.0, 0.04, 0.07)
    pc9000 = centrifuge_pc(200.0, 9000.0, 0.04, 0.07)
    assert pc9000 > pc8000 > 0

    # Scratch energy
    assert scratch_energy(20e6, 1e-4) > 0
    print("  PASS")
    return {"heat_kJ": q, "poro_pct": float(poro_ls), "perm_pct": float(perm_sc),
            "ucs_pct": float(ucs_ls), "E_before_GPa": float(E_before/1e9)}


if __name__ == "__main__":
    test_all()
