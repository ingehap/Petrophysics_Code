# Petrophysics August 2016 - Vol. 57, No. 4

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 57, No. 4 (August 2016): five articles spanning NMR relaxometry in shale,
predicting carbonate rock properties from NMR with radial-basis-function
interpolation, drainage capillary pressure and resistivity index from short-wait
porous-plate experiments, spectral gamma-ray measurement while drilling, and a
pure-matrix-GR indicator of rock-matrix radioactivity.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article1_nmr_relaxometry_shale.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_nmr_relaxometry_shale.py` | NMR Relaxometry in Shale and Implications for Logging | Kausik, Fellah, Rylander, Singer, Lewis, Sinclair | 339-350 |
| `article2_carbonate_nmr_rbf.py` | Predicting Carbonate Rock Properties Using NMR Data and Generalized Interpolation-Based Techniques | Kwak, Hursan, Shao, Chen, Balliet, Eid, Guergueb | 351-368 |
| `article3_porous_plate_pc_ri.py` | Drainage Capillary Pressure and Resistivity Index from Short-Wait Porous-Plate Experiments | Dernaika, Wilson, Skjæveland, Ebeltoft | 369-376 |
| `article4_spectral_gr_mwd.py` | Spectral Gamma-Ray Measurement While Drilling | Xu, Huiszoon, Wang, Adolph, Yi, Cavin, Laughlin, Tollefsen, Jacobsen, Boyce | 377-389 |
| `article5_pure_matrix_gr.py` | Pure Matrix GR, an Indicator of Rock Matrix Gamma Radioactivity and its Applications | Wang, Zhao | 390-396 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 57 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2016_08.pdf`,
> ~13 MB) has a text layer, so titles, authors, page ranges, and the article
> bodies were read directly; **all five articles have full bodies**. Many
> numbered relations survived as inline text (the BPP and spin-rotation
> relations in Article 1, the Coates/SDR transforms in Article 2, the
> exponential-decay model in Article 3, the form factor in Article 4, and the
> full pure-matrix-GR derivation in Article 5), while the typeset
> display-equation glyphs were dropped; those are faithful standard-form
> reconstructions from the surviving variable definitions. (This issue has no
> tutorial.)

## Implementation notes & substitutions

- **Article 1 (Kausik et al.)**: NMR relaxometry - the additive
  liquid-hydrocarbon relaxation rate (Eq. 1), the Bloembergen-Purcell-Pound
  spectral-density model and dipolar T1/T2 rates (Eqs. 2-3, with the T1/T2 ratio
  ~1 for fast motion and large for slow motion / bound fluids), and the bulk-gas
  spin-rotation rate with T1 = T2 (Eq. 4).

- **Article 2 (Kwak et al.)**: carbonate NMR + RBF - the NMR pore-size relation
  (Eq. 1), the Coates (Eq. 2) and SDR/Kenyon (Eq. 3) closed-form permeability
  baselines and the T2 log-mean, plus the paper's actual contribution: PCA
  dimensionality reduction of T2 distributions and a radial-basis-function
  generalized-interpolation fit/evaluate (Gaussian / multiquadric). The
  pore-throat models of Clerke (Eq. 4) and Thomeer (Eq. 5) are referenced but
  not reproduced.

- **Article 3 (Dernaika et al.)**: short-wait porous plate - the
  exponential-decay water-saturation (Eq. 1) and 1/RI (Eq. 2) models, an
  equilibrium-extraction routine (the Guggenheim three-point method, predicting
  the equilibrium asymptote and characteristic time from an early transient -
  the "short-wait" idea), and the Archie resistivity index with a
  saturation-exponent fit.

- **Article 4 (Xu et al.)**: spectral GR while drilling - the
  sourceless-gain-regulation form factor (Eq. 1), a weighted-least-squares
  spectral fit recovering K/U/Th concentrations from the standard spectra, and
  the total (SGR) and uranium-free (CGR) gamma-ray logs in API units.

- **Article 5 (Wang & Zhao)**: pure matrix GR - the homogeneous-formation gamma
  flux Psi = n/(rho*mu) (Eqs. 1-2), the matrix/fluid radioactivity split (Eq. 3),
  the GR-log model (Eq. 5), the porosity/density-immune pure matrix GR Nm
  (Eq. 10), the relative error from neglecting fluid radioactivity (Eq. 11), and
  the comparable matrix GR (Eq. 12).

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2016)
Reference: Petrophysics Vol. 57, No. 4, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
