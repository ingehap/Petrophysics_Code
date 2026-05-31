# Petrophysics October 2017 - Vol. 58, No. 5

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 58, No. 5 (October 2017) - the **"Best of 2017 SPWLA Symposium"** issue:
five articles spanning hydrocarbon saturation in mixed-wet rocks, 2D NMR of
kerogen isolates, gamma-ray tool characterization, near-wellbore joint
inversion, and Permian core-analysis methods.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article2_kerogen_2d_nmr_bitumen.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_mixedwet_saturation_pcm.py` | Improved Assessment of Hydrocarbon Saturation in Mixed-Wet Rocks With Complex Pore Structure | Garcia, Heidari, Rostami | 454-469 |
| `article2_kerogen_2d_nmr_bitumen.py` | Effects of Bitumen Extraction on the 2D NMR Response of Saturated Kerogen Isolates | Chen, Singer, Kuang, Vargas, Hirasaki | 470-484 |
| `article3_gamma_ray_api_characterization.py` | Characterizing Natural Gamma-Ray Tools Without the API Calibration Formation | Moake | 485-500 |
| `article4_joint_inversion_nearwellbore.py` | Imaging Near-Wellbore Petrophysical Properties by Joint Inversion of Sonic, Resistivity, and Density Logging Data | Shetty, Liang, Simoes, Canesin, Boyd, Zeroug, Sinha, Habashy, Domingues, Amorim, Abbots | 501-516 |
| `article5_permian_core_analysis.py` | Lessons Learned in Permian Core Analysis: Comparison Between Retort, GRI, and Routine Methodologies | Blount, Croft, Driskill, Tepper | 517-527 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 58 (the
> first DOIs appear in the February 2018 issue, using the older
> `10.30632/petro_059_1_*` scheme). Articles are therefore cited by
> volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2017_10.pdf`,
> ~16 MB) has a text layer, so titles, authors, and page ranges were read from
> the contents page and bodies. The machine extraction captured the full bodies
> of **articles 1-4** but **truncated article 5 after ~1.5 pages** (only its
> abstract/intro), so **article 5 is implemented as a methodology proxy**. As
> with the other issues, the typeset formula glyphs were dropped in extraction,
> so the numbered formulas are faithful standard-form reconstructions from the
> surviving variable definitions. (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Garcia et al.)**: mixed-wet saturation by Pore Combination
  Modeling - the Archie and Montaron conductivity models, the percolation-
  threshold generalization that reduces to Archie, CRIM mixing of water-wet and
  oil-wet blocks by the oil-wet fraction, and the saturation inversion.

- **Article 2 (Chen et al.)**: kerogen 2D NMR - pellet bulk volume and swelling,
  the bulk/surface/diffusion relaxation-rate decomposition, the surface
  relaxation and pore diameter (d = 6*rho*T), the fast-diffusion validity ratio,
  and the Archie formation factor of the microporosity.

- **Article 3 (Moake)**: gamma-ray tool characterization - the representative
  (centroid) bin energy, the linear source-emission rate and isotope fractions,
  the U-235-into-U-238 weighting (0.04604), and the count rate / tool sensitivity
  (cps per 200 API).

- **Article 4 (Shetty et al.)**: near-wellbore joint inversion - the Archie pixel
  resistivity, the relative gas fraction, the elastic velocities from moduli,
  Wood's and Brie's effective-fluid-modulus laws, the volumetric density mixing,
  and the relative-misfit cost function.

- **Article 5 (Blount et al.)** *(methodology proxy)*: Permian core analysis -
  porosity from grain/bulk volume, retort and Dean-Stark saturations, the
  hydrocarbon pore volume, and the relative discrepancy used to compare the
  retort/GRI/routine methods.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2017)
Reference: Petrophysics Vol. 58, No. 5, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
