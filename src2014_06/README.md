# Petrophysics June 2014 - Vol. 55, No. 3

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 55, No. 3 (June 2014) - a formation-evaluation and rock-physics case study
of the Bazhenov shale, borehole-carbon corrections for accurate TOC from nuclear
spectroscopy, magnetic-resonance core-plug analysis with a three-magnet array
unilateral magnet, a method for predicting permeability of complex carbonate
reservoirs from NMR logging, and a shale-line analysis for shaly-sand porosity
computation and sedimentary interpretation in deepwater sediments.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article1_bazhenov_rock_physics.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_bazhenov_rock_physics.py` | A Case Study about Formation Evaluation and Rock Physics Modeling of the Bazhenov Shale | Kulyapin, Sokolova | 211-218 |
| `article2_borehole_carbon_toc.py` | Borehole Carbon Corrections Enable Accurate TOC Determination from Nuclear Spectroscopy | Miles, Badry | 219-228 |
| `article3_three_magnet_mr_coreplug.py` | Magnetic Resonance Core-Plug Analysis with the Three-Magnet Array Unilateral Magnet | García-Naranjo, Guo, Marica, Liao, Balcom | 229-239 |
| `article4_nmr_carbonate_permeability.py` | Method for Predicting Permeability of Complex Carbonate Reservoirs Using NMR Logging Measurements | Trevizan, Neto, Coutinho, Machado, Rios, Chen, Shao, Romero | 240-252 |
| `article5_shaly_sand_porosity_deepwater.py` | Analysis of Shale for Shaly-Sand Porosity Computation and Sedimentary Interpretation in Deepwater Sediments | Xu | 253-259 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 55 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2014_06.pdf`,
> ~8 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly. Article 5 (Xu) is the best preserved - nearly all
> of its shale-line equations survived - and Article 2's two correction
> equations are complete. Many display equations in Articles 1 and 4 were
> dropped in extraction (leaving only their numbers) and are faithfully
> reconstructed in standard form (Kuster-Toksoz inclusion moduli, the Timur-
> Coates and SDR permeability transforms, the regularized-RBF-with-PCA model).
> Article 3 is an instrumentation paper with no numbered equations; its CPMG /
> gradient-diffusion relaxation and signal-ratio porosity are written in standard
> unilateral-NMR form. The cover features Article 2 (nuclear-spectroscopy gamma-
> ray spectra, Miles & Badry). (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Kulyapin & Sokolova)**: Bazhenov rock physics - the multimineral
  statistical inversion with unit-sum closure (Eq. 1), the effective porosity
  `phi_ef = phi_total/3` (Eq. 2), the block resistivity (Eq. 6) and matrix-block /
  secondary porosity (Eqs. 3-5), the shear-wave splitting index (Eq. 7), and a
  Kuster-Toksoz spherical-inclusion effective-moduli model with the P-wave
  velocity and acoustic impedance.

- **Article 2 (Miles & Badry)**: borehole carbon corrections - the total
  inorganic carbon yield from carbonate mineralogy, the constant borehole
  correction (Eq. 1), the self-calibrating correction (Eq. 2) with a borehole-
  carbon function linear in hole diameter plus standoff/HI crossterms, and the
  TOC weight fraction from the corrected yield.

- **Article 3 (García-Naranjo et al.)**: three-magnet unilateral MR - the Larmor
  frequency, the CPMG magnetization decay, the effective T2 combining bulk,
  surface-relaxation and constant-gradient diffusion terms, and the signal-ratio
  porosity calibrated against a 21% Berea reference plug.

- **Article 4 (Trevizan et al.)**: NMR carbonate permeability - the T2
  logarithmic mean, the FFI/BVI cutoff partition, the Timur-Coates (Eq. 1) and
  SDR (Eq. 2) permeability transforms, a Gaussian radial-basis-function predictor
  with regularized least-squares fitting, and the PCA variance fraction (Eq. 6)
  for input reduction.

- **Article 5 (Xu)**: shaly-sand porosity - the quartz-fluid and quartz-wet-clay
  line slopes (Eqs. 5, 6), the closed-form effective pore-fluid volume (Eq. 4)
  and the full three-component volumetric solve, the wet/dry-shale neutron
  conversion (Eqs. 7, 9) and dry-clay slope (Eq. 10), and the laminated effective
  porosity `phi_e = Vfl/N` (Eq. 11). Real shale-line slopes from the worked Gulf
  of Mexico example (kcl = 0.229) are used in the demo.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2014)
Reference: Petrophysics Vol. 55, No. 3, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
