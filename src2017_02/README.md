# Petrophysics February 2017 - Vol. 58, No. 1

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 58, No. 1 (February 2017) - a multiphase-flow / special-core-analysis (SCAL)
special issue with five articles spanning flow regimes during immiscible
displacement, relative-permeability effects in MICP, osmosis as an
oil-mobilization mechanism, micro-CT pore-scale fluid distribution, and a
benchmark of four SCAL simulators.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article5_scal_simulator_comparison.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_flow_regimes_immiscible.py` | Flow Regimes During Immiscible Displacement | Armstrong, McClure, Berrill, Rücker, Schlüter, Berg | 10-18 |
| `article2_micp_relperm_transition.py` | Relative Permeability Effects Overlooked in MICP Measurements; Transition Zones Likely to be Smaller | Maas, Springer, Hebing | 19-27 |
| `article3_osmosis_low_salinity.py` | Wettability Effects on Osmosis as an Oil-Mobilization Mechanism During Low-Salinity Waterflooding | Fredriksen, Rognmo, Sandengen, Fernø | 28-35 |
| `article4_microct_salinity_distribution.py` | Fast X-Ray Micro-CT Study of the Impact of Brine Salinity on the Pore-Scale Fluid Distribution During Waterflooding | Bartels, Rücker, Berg, Mahani, Georgiadis, Fadili, Brussee, Coorn, van der Linde, Hinz, Jacob, Wagner, Henkel, Enzmann, Bonnin, Stampanoni, Ott, Blunt, Hassanizadeh | 36-47 |
| `article5_scal_simulator_comparison.py` | Comparison of Four Numerical Simulators for SCAL Experiments | Lenormand, Lorentzen, Maas, Ruth | 48-56 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 58 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2017_02.pdf`,
> ~12 MB) has a text layer, so titles, authors, and page ranges were read from
> the contents page and bodies; **all five articles have full bodies**. As with
> the other issues, the typeset formula glyphs were dropped in extraction, so the
> numbered formulas are faithful standard-form reconstructions. Articles 3
> (osmosis) and 4 (micro-CT) are experimental/imaging papers; their modules
> implement the standard physics and image-bookkeeping relations they rely on.
> (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Armstrong et al.)**: flow regimes - the Corey relative
  permeabilities (wetting / non-wetting), the capillary number, the Euler
  characteristic (phase connectivity), and the ganglion flux fraction.

- **Article 2 (Maas et al.)**: MICP relative-permeability effects - the
  equilibration shortfall (apparent vs equilibrium saturation), the (phi/K)^0.5
  capillary-pressure scaling, a two-sample Student t statistic, and the
  homogeneity-number sample filter.

- **Article 3 (Fredriksen et al.)** *(experimental paper)*: osmosis in
  low-salinity waterflooding - the van't Hoff osmotic pressure, the
  Stokes-Einstein diffusivity, Fick's diffusive flux, and the capillary-pressure
  convention.

- **Article 4 (Bartels et al.)** *(imaging study)*: micro-CT fluid distribution -
  phase saturation from segmented voxels, a granulometry pore-size distribution,
  the oil fraction resolved by pore size, and the mean oil-occupied pore size
  (wettability-shift signature).

- **Article 5 (Lenormand et al.)**: SCAL simulator comparison - the
  capillary-pressure convention (Pc = P_oil - P_water), the Corey relative
  permeabilities and water fractional flow, the Buckley-Leverett fractional-flow
  derivative, and the Darcy pressure drop.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2017)
Reference: Petrophysics Vol. 58, No. 1, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
