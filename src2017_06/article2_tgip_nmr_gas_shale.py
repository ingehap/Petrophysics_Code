"""
Article 2: A Novel Determination of Total Gas-In-Place (TGIP) for Gas Shale From
           Magnetic Resonance Logs
Kausik, Kleinberg, Rylander, Lewis, Sibbit, Westacott (2017)
Reference: Petrophysics Vol. 58, No. 3 (June 2017), pp. 232-241
DOI: none assigned (this issue predates SPWLA DOI assignment)

Magnetic resonance counts hydrogen protons directly, so the total gas-in-place
(free + adsorbed) follows from the gas-filled MR porosity without needing a
hydrogen-index or Langmuir model.  This module implements the supporting
relations: the hydrogen index of a fluid, the mean protons per molecule and
molecular weight of a gas mixture, the gas specific gravity, the moles-to-scf
conversion, and the TGIP per unit formation volume.

Implements:

  - Hydrogen index  HI = (rho*n/M)/(rho_w*n_w/M_w)
  - Mean protons per molecule  ng = sum(m_i*alpha_i)/sum(m_i)
  - Gas mixture molecular weight and specific gravity  yg = MWg/MWair
  - Moles-to-scf conversion  Vscf = 0.8305e6*nu  and TGIP per m^3

Note: this issue's PDF has a text layer; Eq. 15 and the appendix mixture
relations survived, while the proton-counting equations lost their glyphs and
are faithful standard-form reconstructions.  Densities in g/cm^3.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

MW_AIR = 28.97
SCF_PER_MOLE = 0.8305        # Eq. 15 constant (scf per mole, x1e6 cm^3/m^3)


# ---------------------------------------------- hydrogen index --------------

def hydrogen_index(rho, n_protons, mol_weight, rho_w=1.0, n_w=2.0, m_w=18.02):
    """Hydrogen index  HI = (rho*n/M)/(rho_w*n_w/M_w)  (Eqs. 5-6)."""
    return petrolib.nmr.hydrogen_index(rho, n_protons, mol_weight, rho_w=rho_w, n_w=n_w, m_w=m_w)


def mean_protons(component_amounts, protons_per_component):
    """Mean protons per molecule  ng = sum(m_i*alpha_i)/sum(m_i)  (Eq. 21)."""
    m = np.asarray(component_amounts, float)
    a = np.asarray(protons_per_component, float)
    return float((m * a).sum() / m.sum())


def mixture_molecular_weight(fractions, molecular_weights):
    """Gas mixture molecular weight  MWg = sum(C_i*MW_i)  (Eq. 22)."""
    return float(petrolib.geochem_fluids.pvt.mixture_mw(fractions, molecular_weights))


def gas_gravity(mw_gas, mw_air=MW_AIR):
    """Gas specific gravity  yg = MWg/MWair  (Eq. 23)."""
    return mw_gas / mw_air


# ---------------------------------------------- gas in place --------------

def moles_to_scf(nu_moles_per_cm3):
    """Convert moles of gas per cm^3 of formation to scf/m^3  Vscf = 0.8305e6*nu (Eq. 15)."""
    return SCF_PER_MOLE * 1.0e6 * np.asarray(nu_moles_per_cm3, float)


def tgip_scf_per_m3(phi_mr_gas, molar_density_mol_per_cm3):
    """Total gas-in-place per m^3 from the gas-filled MR porosity (Eqs. 14-16)

        TGIP = Vscf(phi_MR(gas) * molar_density),

    where molar_density is moles of gas per cm^3 of gas-filled pore space.
    """
    nu = phi_mr_gas * molar_density_mol_per_cm3
    return moles_to_scf(nu)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: TGIP From Magnetic Resonance Logs")
    print("=" * 60)

    # Hydrogen index of water is 1 by definition
    assert np.isclose(hydrogen_index(1.0, 2.0, 18.02), 1.0)

    # Downhole methane (rho ~ 0.25 g/cc) has HI ~ 0.5
    hi_ch4 = hydrogen_index(0.25, 4.0, 16.04)
    print(f"  HI methane (0.25 g/cc) = {hi_ch4:.3f}")
    assert 0.4 < hi_ch4 < 0.7

    # Mean protons of a 90% CH4 / 10% C2H6 gas
    ng = mean_protons([0.9, 0.1], [4.0, 6.0])
    assert np.isclose(ng, 4.2)

    # Mixture molecular weight and gas gravity
    mwg = mixture_molecular_weight([0.9, 0.1], [16.04, 30.07])
    yg = gas_gravity(mwg)
    print(f"  MWg / gas gravity      = {mwg:.2f} / {yg:.3f}")
    assert 16.0 < mwg < 18.0 and 0.5 < yg < 0.65

    # Moles-to-scf is linear; TGIP positive and scales with gas porosity
    assert np.isclose(moles_to_scf(2.0), 2.0 * SCF_PER_MOLE * 1e6)
    tgip = tgip_scf_per_m3(0.05, 1.0e-2)
    assert tgip > 0 and tgip_scf_per_m3(0.10, 1.0e-2) > tgip
    print("  PASS")
    return {"HI_methane": float(hi_ch4), "MWg": float(mwg), "TGIP": float(tgip)}


if __name__ == "__main__":
    test_all()
