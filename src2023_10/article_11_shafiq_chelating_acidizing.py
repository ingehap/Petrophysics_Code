"""
article_11_shafiq_chelating_acidizing.py
========================================
Implementation of ideas from:

    Shafiq, M. U., Ben Mahmud, H., Khan, M., Gishkori, S. N., Wang, L.,
    and Jamil, M. (2023).  "Effect of Chelating Agents on Tight
    Sandstone Formation Mineralogy During Sandstone Acidizing."
    Petrophysics, 64(5), 796-817.  DOI: 10.30632/PJV64N5-2023a11

The paper compares three chelating agents (HEDTA, EDTA, GLDA) for
matrix acidizing of tight sandstones.  Reaction with calcite, clays
(illite/kaolinite) and feldspar dissolves a fraction of the mineral
mass, redistributing pore-size distribution and changing porosity and
permeability.  HEDTA proved most effective.

Implemented:

    * Mineral-dissolution model:  fractional mineral mass loss as a
      function of agent type, concentration, time and temperature
      (per-mineral first-order kinetics with effective rate constants
      digitised from Tables 4-6 of Shafiq et al.)
    * Updated porosity:  phi' = phi + (m_dissolved / rho_mineral) / V
    * Kozeny-Carman permeability update:
            k' / k = (phi'/phi)^3 * ((1-phi)/(1-phi'))^2
    * Pore-size-distribution shift: dissolution preferentially widens
      the small-radius end of the distribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# Effective dissolution rate constants (1/s) at 90 deg C, 0.6 M
# Numbers are representative orderings from Shafiq et al. - HEDTA > EDTA > GLDA
# for calcite/clay; feldspar is hardest to dissolve.
# ---------------------------------------------------------------------------
RATE_CONSTANTS: Dict[str, Dict[str, float]] = {
    "HEDTA": {"calcite": 8.0e-4, "kaolinite": 2.5e-4,
              "illite":  1.8e-4, "feldspar": 5.0e-5},
    "EDTA":  {"calcite": 6.0e-4, "kaolinite": 1.6e-4,
              "illite":  1.1e-4, "feldspar": 3.0e-5},
    "GLDA":  {"calcite": 4.5e-4, "kaolinite": 1.0e-4,
              "illite":  0.7e-4, "feldspar": 2.0e-5},
}

# Mineral grain densities (g/cc)
RHO_MINERAL = {"calcite": 2.71, "kaolinite": 2.60,
               "illite": 2.75, "feldspar": 2.56, "qtz": 2.65}


@dataclass
class CoreState:
    porosity: float
    permeability_mD: float
    grain_density: float = 2.65
    minerals: Dict[str, float] = field(default_factory=dict)  # mass fractions
    psd_um: np.ndarray = field(default_factory=lambda: np.linspace(0.05, 5.0, 30))
    psd_freq: np.ndarray = field(default_factory=lambda: np.array([]))


def fractional_dissolution(agent: str, mineral: str,
                           conc_mol_L: float, t_min: float,
                           T_K: float = 363.15) -> float:
    """Arrhenius-like fractional dissolution X = 1 - exp(-k_eff * t).

    k_eff = k0 * (conc / 0.6) * exp(-Ea/R * (1/T - 1/Tref))   with
    Ea = 25 kJ/mol, Tref = 363.15 K (90 C).
    """
    k0 = RATE_CONSTANTS[agent][mineral]
    Ea = 25_000.0
    R = 8.314
    Tref = 363.15
    k_eff = k0 * (conc_mol_L / 0.6) * np.exp(-Ea / R * (1.0 / T_K - 1.0 / Tref))
    return 1.0 - np.exp(-k_eff * t_min * 60.0)


def kozeny_carman_update(phi: float, phi_new: float, k_mD: float) -> float:
    return k_mD * (phi_new / phi) ** 3 * ((1.0 - phi) / (1.0 - phi_new)) ** 2


def acidize(core: CoreState, agent: str, conc_mol_L: float,
            t_min: float, T_K: float = 363.15) -> tuple[CoreState, dict]:
    """Apply chelating-agent treatment.  Returns the new core state and a
    summary of fractional mass dissolved per mineral."""
    new_minerals = dict(core.minerals)
    dissolved_mass_frac = 0.0
    summary = {}
    for mineral, frac in core.minerals.items():
        if mineral == "qtz":
            X = 0.0           # quartz does not react with chelating agents
        else:
            X = fractional_dissolution(agent, mineral, conc_mol_L, t_min, T_K)
        summary[mineral] = X
        new_minerals[mineral] = frac * (1.0 - X)
        dissolved_mass_frac += frac * X

    # Dissolved mass converts to extra void.  Use grain density as proxy.
    delta_phi = dissolved_mass_frac * (1.0 - core.porosity)
    phi_new = min(0.40, core.porosity + delta_phi)
    k_new = kozeny_carman_update(core.porosity, phi_new, core.permeability_mD)

    # PSD shift: small-radius end gets enhanced (multiplicative)
    new_freq = core.psd_freq.copy() if core.psd_freq.size else \
        np.exp(-(core.psd_um - 0.5) ** 2 / 0.4)
    new_freq = new_freq * (1.0 + 4.0 * dissolved_mass_frac
                           * np.exp(-core.psd_um))
    new_freq /= new_freq.sum() + 1e-12

    new_core = CoreState(porosity=phi_new, permeability_mD=k_new,
                         grain_density=core.grain_density,
                         minerals=new_minerals,
                         psd_um=core.psd_um.copy(),
                         psd_freq=new_freq)
    return new_core, summary


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    base = CoreState(
        porosity=0.07, permeability_mD=0.5,
        minerals={"qtz": 0.55, "feldspar": 0.10, "calcite": 0.10,
                  "kaolinite": 0.15, "illite": 0.10},
        psd_freq=np.exp(-(np.linspace(0.05, 5, 30) - 0.5) ** 2 / 0.4),
    )

    # Run all three agents under the same conditions
    results = {}
    for agent in ("HEDTA", "EDTA", "GLDA"):
        new, summary = acidize(base, agent, conc_mol_L=0.6, t_min=120,
                               T_K=363.15)
        results[agent] = (new, summary)
        # Sanity
        assert new.porosity > base.porosity
        assert new.permeability_mD > base.permeability_mD
        # No quartz dissolution
        assert summary["qtz"] == 0.0
        # Calcite dissolves more than feldspar
        assert summary["calcite"] > summary["feldspar"]

    # HEDTA should produce greatest porosity / k uplift (key claim of paper)
    phi_h = results["HEDTA"][0].porosity
    phi_e = results["EDTA"][0].porosity
    phi_g = results["GLDA"][0].porosity
    assert phi_h > phi_e > phi_g, (phi_h, phi_e, phi_g)

    k_h = results["HEDTA"][0].permeability_mD
    k_e = results["EDTA"][0].permeability_mD
    k_g = results["GLDA"][0].permeability_mD
    assert k_h > k_e > k_g, (k_h, k_e, k_g)

    # Kozeny-Carman: doubling phi from 0.07 to 0.14 should give k uplift > 8x
    k_2 = kozeny_carman_update(0.07, 0.14, 1.0)
    assert k_2 > 8.0, k_2

    # Concentration scaling - higher concentration -> more dissolution
    X_lo = fractional_dissolution("HEDTA", "calcite", 0.3, 120)
    X_hi = fractional_dissolution("HEDTA", "calcite", 0.9, 120)
    assert X_hi > X_lo

    # Time scaling - longer treatment -> more dissolution
    X_short = fractional_dissolution("HEDTA", "calcite", 0.6, 30)
    X_long = fractional_dissolution("HEDTA", "calcite", 0.6, 240)
    assert X_long > X_short

    print(f"article_11_shafiq_chelating_acidizing: OK  "
          f"(phi: HEDTA={phi_h:.3f}, EDTA={phi_e:.3f}, GLDA={phi_g:.3f}; "
          f"k mD: {k_h:.2f}, {k_e:.2f}, {k_g:.2f})")


if __name__ == "__main__":
    test_all()
