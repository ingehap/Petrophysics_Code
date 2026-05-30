# Petrophysics December 2021 - Vol. 62, No. 6

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 62, No. 6 (December 2021) - the **"Best Papers of the 2021 Symposium"**
issue.  Ten papers spanning data quality for petrophysical machine learning,
variational-autoencoder mineral quantification, eigenvector dip analysis,
deep-learning sedimentary-geometry interpretation, density-tool breakout
behind casing, NanoTag cuttings depth correlation, multistring isolation
evaluation, overbalanced-drilling core damage, integrated tight-gas
characterization, and resistivity-based rock physics.

## Quick start

```bash
pip install numpy

# Run all 10 module tests
python test_all.py

# Or run a single article
python article03_seat_dip_eigenvectors.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article01_data_quality_ml.py` | Data Quality Considerations for Petrophysical Machine-Learning Models | McDonald | 10.30632/PJV62N6-2021a1 |
| `article02_vae_mineral_spectroscopy.py` | Enhanced Mineral Quantification and Uncertainty Analysis From Downhole Spectroscopy Logs Using Variational Autoencoders | Craddock, Srivastava, Datir, Rose, Zhou, Mosse, Venkataramanan | 10.30632/PJV62N6-2021a2 |
| `article03_seat_dip_eigenvectors.py` | Taming the Thunder Horse With Axes and Vectors | Ruehlicke, Uhrin, Veselovsky, Schlaich | 10.30632/PJV62N6-2021a3 |
| `article04_borehole_image_cnn_sedimentary.py` | Deep-Learning-Based Automated Sedimentary Geometry Characterization From Borehole Images | Lefranc, Bayraktar, Kristensen, Driss, Le Nir, Marza, Kherroubi | 10.30632/PJV62N6-2021a4 |
| `article05_density_breakout_behind_casing.py` | Identification of Breakout Behind Casing: Openhole-Equivalent Caliper Through Slotted Liner Using the Density Tool | Mosse, Pell, Neville | 10.30632/PJV62N6-2021a5 |
| `article06_nanotags_cuttings_depth.py` | NanoTags for Improved Cuttings Depth Correlation | Poitzsch, Zhu, Antoniv, Aljabri, Marsala | 10.30632/PJV62N6-2021a6 |
| `article07_multistring_isolation_acoustic.py` | Case Studies on Multistring Isolation Evaluation in P&A Operations | Zhang, Mueller, Bryce, Brockway, Iskander | 10.30632/PJV62N6-2021a7 |
| `article08_overbalanced_drilling_correction.py` | The Impact of Overbalanced Drilling From Exploration/Appraisal Wells to Field Development Plan | Mohammadlou, Reppert, Del Negro, Jones | 10.30632/PJV62N6-2021a8 |
| `article09_tight_gas_neuquen_integrated.py` | An Integrated Petrophysical Characterization of a Siliciclastic Tight Gas Reservoir in Neuquen Basin, Western Argentina | Carrizo, Santiago, Saldungaray | 10.30632/PJV62N6-2021a9 |
| `article10_resistivity_rockphysics_wolfcamp.py` | Enhanced Assessment of Fluid Saturation in the Wolfcamp Formation of the Permian Basin | Dash, Heidari | 10.30632/PJV62N6-2021a10 |
| `test_all.py` | Master test runner | - | - |

> **Note on coverage.** Articles 1-8 are implemented directly against the
> methods described in the paper bodies.  Articles 9-10 were only available
> as table-of-contents entries and the editor's narrative in the source-PDF
> extract used to build this folder (the extract was truncated partway
> through Article 8), so their modules are **methodology proxies** that
> demonstrate the workflows the editor describes.  Separately, throughout the
> issue the *typeset equations were stored as images* and did not survive
> text extraction (only the equation numbers remained), so the numbered
> formulas here are faithful **standard-form reconstructions** of the methods
> the prose describes - not byte-perfect transcriptions.  When the original
> typeset equations / full paper bodies become available, the modules can be
> replaced in place.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (McDonald)**: z-score (Eq. 1) and IQR/box-plot (Eq. 2) outlier
  detection, simple and reference-percentile normalization (Eqs. 3-4, Shier
  2004), precision/recall (Eqs. 5-6), MAE/RMSE (Eqs. 7-8), Gaussian-noise
  injection (Eqs. 9-10), Pearson correlation (Eq. 11), and sentinel->NaN
  cleaning.  The Table 3 confusion matrix reproduces precision 0.704 / recall
  0.909.

- **Article 2 (Craddock et al.)**: the VAE forward model `e = A @ m` with a
  stoichiometric element->mineral sensitivity matrix, the heteroscedastic
  Gaussian negative-log-likelihood cost (Eqs. 1-2), and a non-negative,
  closure-constrained simplex inversion as the encoder analogue.  TensorFlow
  is replaced by numpy; the decoder's element reconstruction is used as the
  QC the paper describes.

- **Article 3 (Ruehlicke et al.)**: the SEAT eigenvector method - dip ->
  pole-to-bedding (R1), orientation/scatter matrix (R3), eigen-decomposition
  with the minimum-eigenvalue eigenvector as the slump-fold symmetry axis
  (R4), Woodcock (R5) and Vollmer (R6) fabric indices, and the paper's
  central tilt-invariance claim (axis trend stable under <40 deg structural
  tilt).

- **Article 4 (Lefranc et al.)**: the sinusoid model of a planar bed on an
  unrolled borehole image (R9) with a least-squares fit recovering apparent
  dip and dip azimuth, plus softmax / cross-entropy / accuracy (R5-R7) and
  the four-level Rubin (1987) bedform hierarchy.  The CNN is represented by
  its analytic geometric core.

- **Article 5 (Mosse et al.)**: the radial response function (Eq. 1), the
  tanh radial-response model (Eq. 2), casing- and nominal-cement-corrected
  densities (Eqs. 3-4), and annulus-thickness inversion (exact + the Taylor
  form, Eq. 5) using the quoted geometrical terms (C_SS3 = 0.52,
  C_LS3 = 1.78), plus completion/fluid classification by annulus density.

- **Article 6 (Poitzsch et al.)**: the volumetric lag-time algebra - upward
  (Eq. 1) and downward (Eq. 3) lag, conventional (Eq. 2) and NanoTag (Eq. 4)
  generation times, annular capacity from hole/pipe diameters, and the
  depth-error = ROP x dt relation.  Reproduces t_d ~ 17 min and a ~2-ft error
  per 2-min slip at 60 ft/hr.

- **Article 7 (Zhang et al.)** *(proprietary inversion -> physics
  demonstrator)*: standard acoustics - impedance Z = rho.v (R1), reflection
  coefficient (R2), transmitted energy (R3) reproducing the ~95% energy loss
  through a single tubing layer, casing thickness resonance (R4), impedance
  classification, and the operational isolation-qualification logic
  (continuous + cumulative footage).

- **Article 8 (Mohammadlou et al.)** *(observational case study -> standard
  relations)*: overbalance pressure (INF-1), mud hydrostatic pressure
  (INF-2), additive porosity correction (INF-4) reproducing the 33% NMR
  undercall, k-phi semilog transform/fit (INF-7), Klinkenberg correction
  (INF-8), fraction-of-original overburden correction (INF-6), and a
  damage flag (phi > 12 p.u. AND k > 100 md).

- **Article 9 (Carrizo et al.)** *(proxy)*: integrated tight-gas workflow -
  clay volume (linear + Larionov), density porosity, Archie and Simandoux
  saturation, Winland r35 pore-throat radius, RQI/FZI hydraulic units, and an
  overpressure pore-pressure gradient (up to ~50% above hydrostatic).

- **Article 10 (Dash & Heidari)** *(proxy)*: resistivity-based rock physics -
  Archie baseline, a Waxman-Smits dual-conductivity saturation for organic-
  rich mudrock, a core-free inversion for the cementation exponent from a
  wet zone, and hydrocarbon-pore-volume-per-acre showing the Archie-vs-new
  reserve improvement the editor reports.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2021)
DOI: <doi>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
