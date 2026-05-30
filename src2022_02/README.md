# Petrophysics February 2022 - Vol. 63, No. 1

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 63, No. 1 (February 2022) - a regular issue of six papers covering a new
in-situ Raman composition-logging tool, automated well-log depth matching
(CNN vs cross correlation), an automated log-data-analytics workflow,
ultrasonic logging of creeping shale, sand-injectite reservoir evaluation,
and core-based closed-retort quantification in the Delaware Basin.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article2_cnn_xcorr_depth_matching.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_raman_logging_eor_gas_storage.py` | New Logging Tool for Enhanced Oil Recovery and Gas Storage Monitoring Applications | Andrews, Speck | 10.30632/PJV63N1-2022a1 |
| `article2_cnn_xcorr_depth_matching.py` | Automated Well-Log Depth Matching - 1D Convolutional Neural Networks vs. Classic Cross Correlation | Torres Caceres, Duffaut, Yazidi, Westad, Johansen | 10.30632/PJV63N1-2022a2 |
| `article3_log_analytics_dtw_xcorr.py` | Automated Log Data Analytics Workflow - The Value of Data Access and Management to Reduced Turnaround Time for Log Analysis | Torres Caceres, Duffaut, Westad, Stovas, Johansen, Jenssen | 10.30632/PJV63N1-2022a3 |
| `article4_ultrasonic_creeping_shale.py` | Ultrasonic Logging of Creeping Shale | Diez, Johansen, Larsen | 10.30632/PJV63N1-2022a4 |
| `article5_sand_injectite_thomas_stieber.py` | Evaluating Petrophysical Properties and Volumetrics Uncertainties of Sand Injectite Reservoirs - Norwegian North Sea | Kotwicki, Baig, Johansen, Leirdal, Aftret, Sandstad, Anthonsen, Gianotten, Hansen, Firinu | 10.30632/PJV63N1-2022a5 |
| `article6_closed_retort_core_quant.py` | Investigating Delaware Basin Bone Spring and Wolfcamp Observations Through Core-Based Quantification: Case Study in the Integrated Workflow, Including Closed Retort Comparisons | Perry, Zumberge, Cheng | 10.30632/PJV63N1-2022a6 |
| `test_all.py` | Master test runner | - | - |

> **Note on coverage.** Every module is implemented directly against the
> methodology described in the corresponding paper body.  In the source-PDF
> extract used to build this folder the *typeset equations were stored as
> images* and did not survive text extraction (only the equation numbers
> remained), so the numbered formulas here are faithful **standard-form
> reconstructions** of the methods the prose describes - not byte-perfect
> transcriptions of the printed glyphs.  Where a paper publishes no
> equations at all (Article 6 is a descriptive case study), the relations
> are implemented as the standard petrophysical proxies the text invokes.
> When the original typeset equations/figure coefficients become available,
> the constants can be replaced in place.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (Andrews & Speck)**: linear Raman forward model
  `X = G * M @ rho` (Eq. 3), ideal-gas number density
  `rho = f * P / (k_B T)` (Eq. 4), Lorentz-Lorenz molar-refractivity
  excitation-volume correction, a `Sum(f) = 1`-constrained
  composition-plus-gain inversion (Eqs. 5-6), Beer-Lambert cross-absorption
  (Eqs. 7-8), and ideal-gas per-zone/per-component flow allocation
  (Eqs. 1-2).  The synthetic test plants a five-gas composition and recovers
  both the mole fractions and the optical gain to machine precision.

- **Article 2 (Torres Caceres et al.)**: normalized cross-correlation
  alignment (Eq. 1) benchmarked against a compact, pure-numpy 1D CNN
  (conv over the stacked reference/test channels -> ReLU -> average-pooled
  positional bins -> flatten -> linear, Eqs. 2-5) trained with MSE on
  synthetic bulk-shifted windows.  Positional (rather than global) pooling
  preserves the sign of the shift; labels are scaled to [-1, 1] for stable
  training.  Quality metrics: Pearson (Eq. 6), Euclidean distance (Eq. 7),
  Ind1%/Ind4% indicators (Eqs. 8-9).  TensorFlow is replaced by a small
  numpy net so the folder needs only numpy.

- **Article 3 (Torres Caceres et al.)**: cross-correlation depth matching
  with an optional stretch/squeeze factor alpha (Eq. 1), a constrained
  dynamic-time-warping matcher with a Sakoe-Chiba band (Eqs. 2-4), and the
  Appendix-1 QC metrics - Pearson (A1.1), trace energy (A1.2), residual
  energy (A1.3), predictability `P = 1 - RE/TE` (A1.4) and Euclidean
  distance (A1.5).  scipy / dtaidistance are replaced by direct numpy
  implementations.

- **Article 4 (Diez et al.)**: pulse-echo group delay
  `tau = -dphi/dw` with `phi = arg(S_P/S_N)` (Eq. 1), the thickness-resonance
  frequency `f_min = 0.95 * v_p / (2 d)` with the S1-mode negative-group-
  velocity correction (Eq. 2), and PE/PC empirical impedance calibrations
  (Eqs. 3-4) **fit to the numerical anchor pairs quoted in the paper body**
  (the figure-rendered regression coefficients were not in the extract).
  Plus the pitch-catch attenuation rate `alpha = (E_T - E)/L` and the
  normal-incidence reflection coefficient, which reproduce the paper's
  R ~ -0.95 (kerosene gap) -> -0.82 (bonded shale).

- **Article 5 (Kotwicki et al.)**: shale-corrected effective porosity
  (Eq. 1), Herron mineralogical permeability (Eq. 2), CT-scan porosity and
  grain-density volumetric mixing (Eqs. 3, 5), constant-BVW saturation
  (Eq. 4), sand counting with the `Fsd >= 0.30` cutoff (Eq. 6), net
  thickness for bulk and Thomas-Stieber (fractional FNTG) methods
  (Eqs. 7-8), and HVOLH for both (Eqs. 9-10), plus a Thomas-Stieber FNTG
  helper and a Poupon-inversion -> Archie sand-phase saturation.  The
  synthetic three-facies log confirms Thomas-Stieber recovers more
  hydrocarbon pore volume than bulk analysis in the breccia.  *(The paper's
  uncertainty treatment is deterministic - method-vs-CT differences and
  min/max ranges - not a Monte Carlo simulation.)*

- **Article 6 (Perry et al.)** *(no published equations - descriptive case
  study)*: standard petrophysical proxies for the text's uncalibrated
  crossplots - Boyle's-law density porosity, fluid-summation porosity,
  crushing fluid loss `intact_NMR - crushed_NMR`, mass-balance water/oil
  saturations, NMR T2 free/bound-water partition at the 10 ms cutoff,
  Schmoker TOC-from-density (coefficients exposed as tunable parameters),
  and the open -> closed retort collection-efficiency correction
  (~80% -> ~95%).

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2022)
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
