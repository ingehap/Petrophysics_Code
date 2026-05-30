"""
Article 7: Downhole Fluid Analysis and Gas Chromatography; a Powerful Combination
           for Reservoir Evaluation
Mullins, Forsythe, Pomerantz, Wilkinson, Winkelman, Mishra, Canas, Chen,
Jackson, Betancourt, Zuo, Kauerauf, Peters (2018)
DOI: 10.30632/PJV59N5-2018a6

Downhole fluid analysis (DFA) and laboratory gas chromatography together
characterize reservoir fluids: gas/oil ratio and composition from chromatography,
and the asphaltene (optical-density) gradient with depth from DFA, interpreted
with the Flory-Huggins-Zuo equation of state to assess reservoir connectivity
and equilibrium.

Implements:

  - Gas/oil ratio from a C1-C7+ composition
  - FHZ asphaltene optical-density gradient with depth
  - Equilibrium (connectivity) check against the FHZ trend
  - Gas chromatography mole-fraction normalization

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the DFA / FHZ relations the paper applies.  Depths in m, T in K.
"""

import numpy as np

R_GAS = 8.314
G_ACCEL = 9.81


# ---------------------------------------------- composition -------------

def normalize_composition(peak_areas):
    """Normalize gas-chromatograph peak areas to mole fractions (sum to 1)."""
    a = np.asarray(peak_areas, float)
    return a / a.sum()


def gas_oil_ratio(mole_fracs, light_idx, scale=1.33e4):
    """GOR proxy = scale * (sum of light-component mole fraction)/(rest).

    light_idx selects the gas components (C1-C5); scale converts to scf/STB.
    """
    mf = np.asarray(mole_fracs, float)
    light = mf[light_idx].sum()
    return scale * light / (1.0 - light)


# ---------------------------------------------- FHZ asphaltene ----------

def fhz_od_ratio(dz_m, v_a=0.004, rho_fluid=700.0, rho_asph=1200.0, T=380.0):
    """Asphaltene optical-density ratio over a depth step (FHZ gravity term).

        OD(z2)/OD(z1) = exp[ v_a*g*(rho_asph - rho_fluid)*dz/(R*T) ]
    """
    expo = v_a * G_ACCEL * (rho_asph - rho_fluid) * np.asarray(dz_m, float) / (R_GAS * T)
    return np.exp(expo)


def asphaltene_profile(depths, od_ref, depth_ref, **kw):
    """Optical-density (asphaltene) profile vs depth from the FHZ gravity term."""
    depths = np.asarray(depths, float)
    return od_ref * fhz_od_ratio(depths - depth_ref, **kw)


def is_connected(od_meas, depths, od_ref, depth_ref, rtol=0.05, **kw):
    """Connected & equilibrated if the OD profile matches the FHZ trend."""
    pred = asphaltene_profile(depths, od_ref, depth_ref, **kw)
    return bool(np.all(np.abs(np.asarray(od_meas, float) - pred) <= rtol * pred))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: DFA and Gas Chromatography")
    print("=" * 60)

    # Composition normalization
    mf = normalize_composition([60, 12, 8, 5, 4, 11.0])    # C1, C2, C3, C4, C5, C6+
    assert abs(mf.sum() - 1.0) < 1e-12

    # GOR: a gassier fluid (more light ends) has a higher GOR
    gor_gassy = gas_oil_ratio(normalize_composition([80, 8, 4, 3, 2, 3.0]), [0, 1, 2, 3, 4])
    gor_oily = gas_oil_ratio(normalize_composition([40, 8, 6, 5, 5, 36.0]), [0, 1, 2, 3, 4])
    print(f"  GOR gassy / oily       = {gor_gassy:.0f} / {gor_oily:.0f} scf/STB")
    assert gor_gassy > gor_oily

    # FHZ: asphaltene optical density increases downward (denser asphaltene)
    depths = np.array([3000.0, 3050.0, 3100.0])
    od = asphaltene_profile(depths, od_ref=0.30, depth_ref=3000.0)
    print(f"  OD vs depth            = {np.array2string(od, precision=4)}")
    assert np.all(np.diff(od) > 0)

    # Connectivity: a clean FHZ profile is "connected"; a perturbed one is not
    assert is_connected(od, depths, 0.30, 3000.0)
    assert not is_connected(od * np.array([1.0, 1.3, 0.8]), depths, 0.30, 3000.0)
    print("  PASS")
    return {"GOR_gassy": float(gor_gassy), "od_gradient": od.tolist()}


if __name__ == "__main__":
    test_all()
