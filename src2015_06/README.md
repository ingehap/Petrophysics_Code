# Petrophysics June 2015 - Vol. 56, No. 3

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 56, No. 3 (June 2015): three articles plus a technical note, spanning
heavy-oil reservoir evaluation with NMR in the Long Lake / Kinosis SAGD
projects, real-time downhole fluid-sample contamination prediction, an
asphaltenes tutorial (Yen-Mullins model and the Flory-Huggins-Zuo EoS), and the
Bateman-Konen resistivity-salinity transform.

## Quick start

```bash
pip install numpy

# Run all 4 module tests
python test_all.py

# Or run a single article
python article3_asphaltenes_explained.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_heavyoil_nmr_sagd.py` | Improvement in Heavy-Oil Reservoir Evaluation Using Nuclear Magnetic Resonance: Long Lake and Kinosis SAGD Projects, Alberta, Canada | Cheng, Kotov, Pyke, Hanif | 239-250 |
| `article2_fluid_contamination_prediction.py` | A Breakthrough in Accurate Downhole Fluid Sample Contamination Prediction in Real Time | Zuo, Gisolf, Dumont, Dubost, Pfeiffer, Wang, Mishra, Chen, Mullins, Biagi, Gemelli | 251-265 |
| `article3_asphaltenes_explained.py` | Asphaltenes Explained for the Nonchemist | Mullins, Pomerantz, Andrews, Zuo | 266-275 |
| `article4_bateman_konen_transform.py` | *Technical Note:* The Bateman-Konen Resistivity-Salinity Transform | Kennedy | 282-283 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 56 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2015_06.pdf`,
> ~7 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly; **all four items have full bodies**, including the
> technical note (whose sequel "II" appears in the August 2015 issue, where its
> body was beyond extraction - here the original Bateman-Konen formula is
> implemented from the body). The cover credits a "Manchuk et al." image is a
> cover-art credit only; it is not a separate article in this issue's table of
> contents. The typeset display-equation glyphs were dropped in extraction, so
> the relations are faithful standard-form reconstructions from the surviving
> variable definitions. (This issue has no tutorial section beyond the
> asphaltenes review.)

## Implementation notes & substitutions

- **Article 1 (Cheng et al.)**: heavy-oil NMR (SAGD) - the density total
  porosity, the gamma-ray index and Clavier shale volume, the NMR-visible
  bitumen volume below a 4-ms T2 cutoff (clay-bound-water corrected), the total
  bitumen bulk volume from the density-vs-NMR porosity difference, and the
  bitumen weight fraction.

- **Article 2 (Zuo et al.)**: fluid contamination prediction - the linear binary
  mixing rule for OD, density and shrinkage factor (Eqs. 1, 2, 5), the
  modified-GOR f-function (Eq. 4), the contamination volume fraction from any
  sensor (Eq. 6), and the power-law cleanup vs. pumped volume with virgin-fluid
  endpoint extrapolation (probe-geometry exponent).

- **Article 3 (Mullins et al.)**: asphaltenes explained - the Yen-Mullins
  particle-size hierarchy (molecule / nanoaggregate / cluster), and the
  Flory-Huggins-Zuo optical-density ratio between two depths (Eq. 3) with its
  gravity (Boltzmann/Archimedes) and solubility contributions.

- **Article 4 (Kennedy)** *(technical note)*: Bateman-Konen resistivity-salinity
  - the R75 = 0.0123 + 3647.5/C^0.955 transform (Eq. 1) and its inverse
  (Eqs. 2-3), the Arps temperature conversion, Rw at any temperature from
  salinity, and the modified formation-factor power law F = b + a/phi^m discussed
  in the note's provenance.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2015)
Reference: Petrophysics Vol. 56, No. 3, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
