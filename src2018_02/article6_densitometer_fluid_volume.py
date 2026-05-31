"""
Article 6: Using a Densitometer for Quantitative Determinations of Fluid Density
           and Fluid Volume in Coreflooding Experiments at Reservoir Conditions
Olsen (2018)
DOI: 10.30632/petro_059_1_a5

An inline densitometer downstream of the core measures the produced-fluid
density against injected volume; assuming non-bypassing ("train") two-phase
flow, the mixture density converts to a produced-water fraction, which is
integrated over injected volume to give cumulative produced water, and produced
oil follows by volume closure at constant pressure and temperature.

Implements:

  - Water fraction from mixture density  WF = (rho - rho_oil)/(rho_water - rho_oil)
  - Cumulative produced water  ProdWater = sum_i WF_i*(Vinj_i - Vinj_{i-1})
  - Produced oil by closure  ProdOil = Vinj - ProdWater
  - Inline mixture density  rho = WF*rho_water + (1 - WF)*rho_oil

Note: this issue's PDF has a text layer; the five working equations lost their
typeset glyphs in extraction but every variable is defined in the prose, so the
relations are faithful standard-form reconstructions.  The water fraction is
written in the physically standard linear-mixing direction (WF = 0 at pure oil,
1 at pure water).  Densities in g/ml, volumes in ml.
"""

import numpy as np


# ---------------------------------------------- density / fraction --------------

def water_fraction(mixture_density, oil_density, water_density):
    """Water volume fraction from the mixture density (Eq. 3)

        WF = (rho_mix - rho_oil)/(rho_water - rho_oil),

    clipped to [0, 1]; the inverse of linear volumetric density mixing.
    """
    wf = (np.asarray(mixture_density, float) - oil_density) / (water_density - oil_density)
    return np.clip(wf, 0.0, 1.0)


def mixture_density(wf, oil_density, water_density):
    """Inline mixture density  rho = WF*rho_water + (1 - WF)*rho_oil."""
    wf = np.asarray(wf, float)
    return wf * water_density + (1.0 - wf) * oil_density


def produced_water(water_fractions, vinj):
    """Cumulative produced water  ProdWater = sum_i WF_i*(Vinj_i - Vinj_{i-1})  (Eq. 2)."""
    wf = np.asarray(water_fractions, float)
    v = np.asarray(vinj, float)
    dv = np.diff(v, prepend=0.0)
    return float(np.sum(wf * dv))


def produced_oil(vinj_total, prod_water):
    """Produced oil by volume closure at constant P,T  ProdOil = Vinj - ProdWater (Eq. 5)."""
    return vinj_total - prod_water


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Densitometer Fluid Volume")
    print("=" * 60)

    oil, water = 0.7821, 1.0015
    # Water fraction is 0 at pure oil, 1 at pure water, 0.5 at the mean density
    assert np.isclose(water_fraction(oil, oil, water), 0.0)
    assert np.isclose(water_fraction(water, oil, water), 1.0)
    assert np.isclose(water_fraction(0.5 * (oil + water), oil, water), 0.5)

    # Round-trip: density -> WF -> density
    rho = mixture_density(0.3, oil, water)
    assert np.isclose(water_fraction(rho, oil, water), 0.3)

    # Inject 1.0 ml at a constant 40% water cut -> 0.4 ml water, 0.6 ml oil
    vinj = np.linspace(0.1, 1.0, 10)
    wf = np.full_like(vinj, 0.4)
    pw = produced_water(wf, vinj)
    po = produced_oil(vinj[-1], pw)
    print(f"  produced water / oil   = {pw:.3f} / {po:.3f} ml")
    assert np.isclose(pw, 0.4) and np.isclose(pw + po, vinj[-1])
    print("  PASS")
    return {"prod_water": pw, "prod_oil": float(po)}


if __name__ == "__main__":
    test_all()
