# Petrophysics December 2017 - Vol. 58, No. 6

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 58, No. 6 (December 2017): a digital-log-preparation tutorial and five
articles - driller's depth correction, carbonate pore structure from MICP and
sonic logs, 3D-printed Berea sandstone, the impact of thermal maturity on
kerogen density, and the impact of core-cleaning methods.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article3_carbonate_pore_structure_sonic.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_digital_log_preparation.py` | *Tutorial:* Preparing Your Digital Well Logs for Computer-Based Interpretation | Thomas | 559-563 |
| `article2_drillers_depth_waypoint.py` | Driller's Depth Quality Improvement: Way-Point Methodology | Bolt | 564-575 |
| `article3_carbonate_pore_structure_sonic.py` | Characterization of Pore Structure Variation and Permeability Heterogeneity in Carbonate Rocks Using MICP and Sonic Logs: Puguang Gas Field, China | Huang, Dou, Sun | 576-591 |
| `article4_3d_printing_berea.py` | 3D Printing Berea Sandstone: Testing a New Tool for Petrophysical Analysis of Reservoirs | Ishutov, Hasiuk | 592-602 |
| `article5_kerogen_density_maturity.py` | Experimental Quantification of the Impact of Thermal Maturity on Kerogen Density | Jagadisan, Yang, Heidari | 603-612 |
| `article6_cleaning_methods_porosity.py` | Impact of Different Cleaning Methods on Petrophysical Measurements | Gupta, Rai, Tinni, Sondergeld | 613-622 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the
> December 2017 President's column states the society "will soon begin to assign
> DOI[s]", the PDF carries no article DOIs, and CrossRef has none registered
> for Vol. 58. Articles are therefore cited by volume/issue/page rather than DOI.
> (The next issue, February 2018, is the first to carry DOIs, using the older
> `10.30632/petro_059_1_*` scheme.)
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2017_12.pdf`,
> ~17 MB) has a text layer, so titles, authors, and page ranges were read from
> the contents page and bodies; **all six items have full bodies** (the
> conference abstracts after p.622 are not articles and are not implemented). As
> with the other issues, the typeset formula glyphs were dropped in extraction,
> so the numbered formulas are faithful standard-form reconstructions from the
> surviving variable definitions.

## Implementation notes & substitutions

- **Article 1 (Thomas)** *(tutorial)*: digital-log preprocessing - the
  density-porosity transform (and how gas inflates it), the flushed-zone Archie
  saturation, "squaring" a log into constant-value beds, inflection-point bed
  alignment, and the minimum bed thickness for a deep resistivity tool.

- **Article 2 (Bolt)**: driller's-depth way-point correction - the per-station
  thermal elongation, the drillpipe cross-section and stretch coefficient, the
  elastic stretch under axial load, the summed correction over stations, and the
  quadrature uncertainty.

- **Article 3 (Huang et al.)**: carbonate pore structure from sonic - elastic
  moduli from Vp/Vs/density, the Vp/Vs forms, the Sun (2000) frame flexibility
  factors inverted from the (1-phi)^gamma law, pore-type classification from
  gamma_mu, the MICP tortuosity, and the Leverett J-function.

- **Article 4 (Ishutov & Hasiuk)**: 3D-printed sandstone - porosity from
  segmented voxels, model magnification, the gap-test printed-size calibration
  (160 um design -> ~132 um printed), the printability check, and the
  proxy-vs-natural porosity/throat offsets.

- **Article 5 (Jagadisan et al.)**: kerogen density vs maturity - the
  pyrite/iron-corrected kerogen density, the multimineral matrix density, the
  total porosity, Archie (m=2, n=1.5), and the porosity sensitivity to the
  kerogen density.

- **Article 6 (Gupta et al.)**: core-cleaning effects - bulk and grain volumes
  from weights and densities, the crushed-sample helium porosity, the porosity
  gain after cleaning (bitumen removal lowers grain volume), and a
  solvent-efficiency ranking.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2017)
Reference: Petrophysics Vol. 58, No. 6, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
