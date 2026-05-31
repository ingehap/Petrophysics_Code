"""
Article 10: New Perspectives on the Effects of Gas Adsorption on Storage and
            Production of Natural Gas From Shale Formations
Tinni, Sondergeld, Rai (2018)
DOI: 10.30632/petro_059_1_a9  (contents-only - see note)

Gas in shale is stored both as free gas in the pore space and as gas adsorbed on
the kerogen/clay surfaces; the adsorbed fraction follows a Langmuir isotherm and
desorbs as the reservoir is produced down in pressure.  This *methodology proxy*
implements the standard storage relations: the Langmuir adsorption isotherm,
free + adsorbed gas-in-place, the adsorbed fraction of total storage, and the
gas released by a pressure drawdown.

Implements:

  - Langmuir isotherm  V = VL*P/(PL + P)
  - Free gas in place  G_free = phi*(1 - Sw)/Bg
  - Total gas in place  G = G_free + rho_rock*V_adsorbed
  - Gas released between two pressures (production / desorption)

Note: this article's body was not in this issue's machine extraction (contents
page only), so - as with the other methodology proxies in this repository - the
relations below are the standard adsorbed-gas storage formulas the title
describes, not formulas transcribed from the paper.  The DOI is the authoritative
SPWLA/CrossRef value for this issue.
"""

import numpy as np


# ---------------------------------------------- adsorption --------------

def langmuir_isotherm(pressure, vl, pl):
    """Langmuir adsorbed gas content  V = VL*P/(PL + P).

    VL = Langmuir volume (max adsorption), PL = Langmuir pressure (P at VL/2).
    """
    p = np.asarray(pressure, float)
    return vl * p / (pl + p)


def free_gas_in_place(phi, sw, bg):
    """Free gas in place per unit volume  G_free = phi*(1 - Sw)/Bg.

    Bg = gas formation volume factor (reservoir/standard volume).
    """
    return phi * (1.0 - sw) / bg


def total_gas_in_place(phi, sw, bg, rho_rock, pressure, vl, pl):
    """Total gas in place = free gas + adsorbed gas (rho_rock*Langmuir)."""
    return free_gas_in_place(phi, sw, bg) + rho_rock * langmuir_isotherm(pressure, vl, pl)


def adsorbed_fraction(phi, sw, bg, rho_rock, pressure, vl, pl):
    """Adsorbed gas as a fraction of total gas in place."""
    ads = rho_rock * langmuir_isotherm(pressure, vl, pl)
    return ads / (free_gas_in_place(phi, sw, bg) + ads)


def gas_released(rho_rock, p_initial, p_final, vl, pl):
    """Adsorbed gas desorbed between two pressures (drawdown p_initial -> p_final)."""
    return rho_rock * (langmuir_isotherm(p_initial, vl, pl) - langmuir_isotherm(p_final, vl, pl))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 10: Shale Gas Adsorption (proxy)")
    print("=" * 60)

    vl, pl = 100.0, 1000.0      # scf/ton, psi
    # Langmuir: V = VL/2 at P = PL, and saturates toward VL at high pressure
    assert np.isclose(langmuir_isotherm(pl, vl, pl), vl / 2)
    assert langmuir_isotherm(1e6, vl, pl) > 0.99 * vl
    assert langmuir_isotherm(3000.0, vl, pl) > langmuir_isotherm(1000.0, vl, pl)

    # Adsorbed gas is a meaningful fraction of total storage in organic shale
    frac = adsorbed_fraction(0.06, 0.30, 0.004, rho_rock=2.5, pressure=3000.0, vl=vl, pl=pl)
    print(f"  adsorbed fraction      = {frac:.3f}")
    assert 0.0 < frac < 1.0

    # Drawing the pressure down releases (desorbs) adsorbed gas
    released = gas_released(2.5, 4000.0, 500.0, vl, pl)
    print(f"  gas released 4000->500 = {released:.2f}")
    assert released > 0
    print("  PASS")
    return {"adsorbed_fraction": float(frac), "released": float(released)}


if __name__ == "__main__":
    test_all()
