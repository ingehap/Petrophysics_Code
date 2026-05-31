# Petrophysics April 2015 - Vol. 56, No. 2

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 56, No. 2 (April 2015): five articles spanning automatic quantification of
wireline/LWD pressure-test quality, steady-state stress-dependent permeability
of tight oil rocks, permeability estimation in the McMurray formation from
high-resolution data, microresistivity curve extraction from borehole-microimager
data, and a new method to estimate porosity from NMR data with short relaxation
times.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article5_nmr_short_t2_porosity.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_pressure_test_quality.py` | Automatically Quantifying Wireline and LWD Pressure-Test Quality | Proett, Musharfi, Gill, Ma, Meridji, Eyuboglu | 101-115 |
| `article2_stress_dependent_permeability.py` | Steady-State Stress-Dependent Permeability Measurements of Tight Oil-Bearing Rocks | Chhatre, Braun, Sinha, Determan, Passey, Zirkle, Wood, Boros, Berry, Leonardi, Kudva | 116-124 |
| `article3_mcmurray_permeability_upscaling.py` | Estimation of Permeability in the McMurray Formation Using High-Resolution Data Sources | Manchuk, Garner, Deutsch | 125-139 |
| `article4_microresistivity_extraction.py` | Microresistivity Curve Extraction from Borehole Microimager Data | Roslin | 140-146 |
| `article5_nmr_short_t2_porosity.py` | New Method to Estimate Porosity More Accurately from NMR Data with Short Relaxation Times | Venkataramanan, Gruber, LaVigne, Habashy, Iglesias, Cohorn, Anand, Rampurawala, Jain, Heaton, Akkurt, Rylander, Lewis | 147-157 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 56 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2015_04.pdf`,
> ~7 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly; **all five articles have full bodies** and their
> numbered relations survived as inline text (the drawdown-mobility/LSR
> relations, the Darcy/Terzaghi/exponential-decline relations, the Vsh-k
> log-linear transform, the log-log microresistivity calibration, and the ILT
> cost function / bias-correction relations). The typeset display-equation
> glyphs were dropped in extraction, so they are faithful standard-form
> reconstructions. The cover features Article 3 (Manchuk et al.). (This issue
> has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Proett et al.)**: pressure-test quality - the pseudosteady
  hemispherical drawdown mobility (Cpf*q/dP; Eq. 1), the pretest flow rate, the
  least-squares regression slope/intercept and residual standard deviation for
  pressure/temperature stability (Eqs. 2-5), and a relative radius of
  investigation from the mobility and drawdown duration (Eqs. 14-15).

- **Article 2 (Chhatre et al.)**: stress-dependent permeability - Darcy's law
  (Eq. 2), the Terzaghi net confining stress (sleeve - Biot*pore pressure;
  Eq. 3), the exponential permeability-stress decline k = k0*exp(-gamma*NCS)
  (Eq. 5), and a fit of (k0, gamma) from measured k vs. NCS.

- **Article 3 (Manchuk et al.)**: McMurray permeability upscaling - the
  log-linear shale-volume permeability transform (clean-sand and Vsh=0.25
  anchors; Eqs. 2-4), permeability averaging (arithmetic / geometric /
  harmonic), and the flow-based effective-permeability bounds (harmonic <=
  geometric <= arithmetic).

- **Article 4 (Roslin)**: microresistivity curve extraction - histogram (min-max)
  scaling of the button-image data, the per-depth median image curve, the
  log-log microresistivity calibration Rmicro = 10^(A + B*log10(Rmedian)) (Eq. 1)
  and the (A, B) fit against wireline resistivity.

- **Article 5 (Venkataramanan et al.)**: NMR short-T2 porosity - the NMR kernel
  and Tikhonov (regularized) T2 inversion (Eq. 1), the porosity bias from a
  Dirac-delta scan (Eq. 2), the correction factor Cf = 1/(1 + B) and corrected
  porosity (Eqs. 3-4), the SNR-weighted factor (Eq. 5), and the total porosity
  (Eq. 6).

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2015)
Reference: Petrophysics Vol. 56, No. 2, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
