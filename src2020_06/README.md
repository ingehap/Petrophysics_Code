# Petrophysics June 2020 - Vol. 61, No. 3

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 61, No. 3 (June 2020) - a regular issue of five papers spanning casedhole
formation evaluation along unconventional horizontal wells, the impact of cement
quality on carbon/oxygen and elemental pulsed-neutron analysis, reliable
relative-permeability measurement in tight gas sands, an analytical
relative-permeability-from-resistivity model for fractal porous media, and
neural-network estimation of reservoir porosity from drilling parameters.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article4_relperm_resistivity_fractal.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_casedhole_horizontal_fe.py` | Lessons Learned From Casedhole Formation Evaluation Along Unconventional Horizontal Wells | Sullivan, Wang, Bolshakov, Song, Lazorek, Tohidi, Seth | 10.30632/PJV61N3-2020a1 |
| `article2_cement_quality_co_pulsed_neutron.py` | Case Studies Demonstrating the Impact of Cement Quality on Carbon/Oxygen and Elemental Analysis From Casedhole Pulsed-Neutron Logging | Wang, Sullivan, Seth, Barnes, Wilson, Lazorek | 10.30632/PJV61N3-2020a2 |
| `article3_relperm_tight_gas_sand.py` | Reliable Measurement of Saturation-Dependent Relative Permeability in Tight Gas Sand Formations | Gonzalez, Tandon, Heidari, Gramin, Merle | 10.30632/PJV61N3-2020a3 |
| `article4_relperm_resistivity_fractal.py` | Evaluation of Relative Permeability From Resistivity Data for Fractal Porous Media | Shi, Meng, Liu, Zhang, Wang | 10.30632/PJV61N3-2020a4 |
| `article5_porosity_drilling_ann.py` | Estimation of Reservoir Porosity From Drilling Parameters Using Artificial Neural Networks | Al-AbdulJabbar, Al-Azani, Elkatatny | 10.30632/PJV61N3-2020a5 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2020_06.pdf`)
> has a text layer, so the article titles, authors, page ranges, DOIs, equation
> numbers, variable definitions, and many numeric constants were read directly
> from the paper bodies. The PDF-to-text conversion dropped most typeset formula
> *glyphs* (keeping the equation numbers and surrounding prose) - only Article 1's
> Eq. 1 (the spectral-gamma API relation) survived verbatim - so the other
> numbered formulas are **faithful standard-form reconstructions** built from the
> preserved variable definitions, using the standard textbook expressions each
> paper cites (VTI moduli, centrifuge Pc, Corey-Brooks, SDR, fractal / Archie /
> Brooks-Corey, ANN R/RMSE).

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions. A few practical notes:

- **Article 1 (Sullivan et al.)** *(case study)*: the spectral gamma ray
  γAPI = 4·Th + 8·U + 16·K (Eq. 1, verbatim), the M-ANNIE VTI stiffness-to-
  engineering-modulus relations for E_vert/E_horz/ν_vert/ν_horz (Eqs. 2-5,
  verified against the isotropic limit), sigma water saturation, the acoustic
  impedance Z = ρ·Vp gas indicator, and the +2 Ca / −3 Fe / −2 Al wt% elemental
  corrections.

- **Article 2 (Wang et al.)** *(MCNP modeling + case study, no equations)*: the
  carbon/oxygen ratio and salinity-independent oil saturation, the cement
  calcium-yield contribution (> 40%) and formation-calcium correction, the
  OBM-vs-WBM channel C/O bias, and sigma water saturation.

- **Article 3 (Gonzalez et al.)**: the centrifuge capillary pressure
  Pc = ½·Δρ·ω²·(LR²−(LR−L)²) (Eq. 1), the modified Corey-Brooks gas relative
  permeability (Eq. 2, exponent ng in the paper's 0.5-3.75 range), the SDR (NMR
  T2) brine relative permeability (Eqs. 3-4), and the Klinkenberg gas-slippage
  correction.

- **Article 4 (Shi et al.)**: the pore-size fractal PDF (Eq. 1, integrates to 1),
  the pore fractal dimension Df = De − ln φ/ln(rmin/rmax) (Eq. 2, gives 2.767 for
  the base case), the Archie resistivity index (Eq. 11), the fractal /
  Brooks-Corey wetting and nonwetting relative permeabilities (Eqs. 22, 24, with
  λ = De − Df), and the relative-permeability-from-resistivity relationship
  (Eq. 23): higher resistivity index → lower wetting-phase rel-perm.

- **Article 5 (Al-AbdulJabbar et al.)** *(no equations)*: a feed-forward tanh
  ANN predicting porosity from six drilling parameters (ROP, WOB, RPM, torque,
  GPM, SPP), scored by the correlation coefficient R and RMSE. The compact numpy
  net reaches R ≈ 0.98 / RMSE ≈ 0.01 on synthetic data (paper: R ≈ 0.94-0.96,
  RMSE ≈ 0.018-0.035 with a two-hidden-layer × 30-neuron network).

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2020)
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
