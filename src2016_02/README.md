# Petrophysics February 2016 - Vol. 57, No. 1

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 57, No. 1 (February 2016) - the Best Papers of the 2015 SCA Symposium plus
one regular submission: CO2-brine multiphase flow in sandstone, estimating
saturations in organic shales with 2D NMR, low-permeability measurement
insights, low-salinity waterflooding, and graphical solutions for laminated and
dispersed shaly sands.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article5_shaly_sand_graphical_solutions.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_co2_brine_multiphase_flow.py` | The Impact of Reservoir Conditions and Rock Heterogeneity on CO2-Brine Multiphase Flow in Permeable Sandstone | Krevor, Reynolds, Al-Menhali, Niu | 12-18 |
| `article2_2d_nmr_shale_saturations.py` | Estimating Saturations in Organic Shales Using 2D NMR | Nicot, Vorapalawut, Rousseau, Madariaga, Hamon, Korb | 19-29 |
| `article3_low_permeability_measurements.py` | Low-Permeability Measurements: Insights | Profice, Hamon, Nicot | 30-40 |
| `article4_low_salinity_waterflooding.py` | Low-Salinity Waterflooding: Facts, Inconsistencies and the Way Forward | Hamon | 41-50 |
| `article5_shaly_sand_graphical_solutions.py` | Graphical Solutions for Laminated and Dispersed Shaly Sands | Bootle | 51-59 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 57 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2016_02.pdf`,
> ~10 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly; **all five articles have full bodies**. Articles 1
> and 4 are experimental / review SCA papers with few or no display equations, so
> they are implemented from the standard physics they rely on (Land trapping,
> Leverett/Corey functions; Buckley-Leverett, Welge, Amott). Articles 2, 3 and 5
> carry numbered relations that survived as inline text (the NMRD dispersion
> models, the Klinkenberg/Darcy relations, and the Archie / Waxman-Smits / Juhasz
> / Rh-Rv equations); the typeset display-equation glyphs were dropped and are
> faithful standard-form reconstructions. (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Krevor et al.)**: CO2-brine multiphase flow - the Land (1968)
  trapping constant and initial-residual curve, the Leverett J-function
  (capillary pressure / interfacial-tension scaling that collapses the curves),
  Corey two-phase relative permeability, and the capillary number governing the
  intrinsic-vs-effective relative-permeability distinction.

- **Article 2 (Nicot et al.)**: 2D NMR shale saturations - T1/T2 fluid typing
  (oil high, water low), partitioning a 2D T1-T2 map by a T1/T2 cutoff into oil
  and water volumes for the NMR saturation, and the NMRD dispersion models
  (Eqs. 1-2): water by 2D diffusion near paramagnetic surfaces (logarithmic
  dispersion) and oil by quasi-1D diffusion in kerogen pores (R1 ~ 1/sqrt(omega)).

- **Article 3 (Profice et al.)**: low-permeability measurements - the
  Klinkenberg apparent permeability (Eq. 1) and its k-vs-1/Pm fit for (kl, b),
  the compressible (gas) and incompressible (liquid) Darcy permeabilities
  (Eqs. 2-3), the gas mean free path and Knudsen number for the slip/transition
  regimes (Eqs. 4-5), and the deviation indicator comparing techniques (Eq. 6).

- **Article 4 (Hamon)**: low-salinity waterflooding - the Buckley-Leverett water
  fractional flow with Corey relative permeability, the Welge tangent for the
  shock-front and average-saturation construction, the recovery factor and the
  LSWI incremental recovery (Sorw_HS - Sorw_LS), and the Amott-Harvey
  wettability index.

- **Article 5 (Bootle)**: shaly-sand graphical solutions - the Archie
  saturation (Eq. 3), the Waxman-Smits/Juhasz conductivity and Sw solve (with Qv
  from CEC and clay-dependent variable exponents; Eqs. 2, 4-6, reverting to
  Archie at Vcl = 0), and the laminated-shale Rh (parallel) / Rv (series)
  anisotropy model with a joint solve for the laminated-shale volume and sand
  resistivity.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2016)
Reference: Petrophysics Vol. 57, No. 1, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
