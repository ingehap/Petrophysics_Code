# Petrophysics August 2020 - Vol. 61, No. 4

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 61, No. 4 (August 2020) - a compact regular issue of four papers spanning
the flexural attenuation technique for cased-hole annulus evaluation, the effect
of clay minerals and pore-water conductivity on the saturation exponent of
clay-bearing sandstones (digital rock), petrophysical-property improvement of
tight reservoirs using thermochemical fluids, and knowledge-driven hierarchical
clustering for specific-facies detection in well logs.

## Quick start

```bash
pip install numpy

# Run all 4 module tests
python test_all.py

# Or run a single article
python article3_thermochemical_stimulation.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_flexural_attenuation_casing.py` | A Study of the Flexural Attenuation Technique Through Laboratory Measurements and Numerical Simulations | Sirevaag, Johansen, Larsen, Holt | 10.30632/PJV61N4-2020a1 |
| `article2_saturation_exponent_clay_digitalrock.py` | Effects of Clay Minerals and Pore-Water Conductivity on Saturation Exponent of Clay-Bearing Sandstones Based on Digital Rock | Fan, Pan, Guo, Lei | 10.30632/PJV61N4-2020a2 |
| `article3_thermochemical_stimulation.py` | Improvement of Petrophysical Properties of Tight Sandstone and Limestone Reservoirs Using Thermochemical Fluids | Mustafa, Mahmoud, Abdulraheem, Tariq, Al-Nakhli | 10.30632/PJV61N4-2020a3 |
| `article4_kdhc_facies_clustering.py` | Detecting Specific Facies in Well-Log Data Sets Using Knowledge-Driven Hierarchical Clustering | Emelyanova, Peyaud, Dance, Pervukhina | 10.30632/PJV61N4-2020a4 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2020_08.pdf`)
> has a text layer, so the article titles, authors, page ranges, DOIs, equation
> numbers, variable definitions, and many numeric constants were read directly
> from the paper bodies. The PDF-to-text conversion dropped most typeset formula
> *glyphs* (keeping the equation numbers and surrounding prose) - only Article 3's
> Eq. 5 (`Ft = E*A`) and its reaction equation survived verbatim - so the other
> numbered formulas are **faithful standard-form reconstructions** built from the
> preserved variable definitions, using the standard textbook expressions each
> paper cites (plane-wave / Snell, Archie, Waxman-Smits, dynamic moduli,
> Young-Laplace, silhouette / F1).

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions. A few practical notes:

- **Article 1 (Sirevaag et al.)**: the plane-wave phase shift / phase velocity
  (Eqs. 1-2), Snell's optimal incidence angle sin θ = vf/vφ (Eq. 3) - which
  reproduces the paper's 30° from vf = 1325 m/s and vφ = 2650 m/s - the
  amplitude-ratio attenuation 20·log₁₀(A₁/A₂) and coefficient α (Eq. 4), the
  third-interface-echo annulus thickness x_a = s_a·cos θ (Eqs. 5-7), and the
  cosine eccentricity fit Δt(αz) = A·cos(αz+φ)+t_avg (Eq. 8).

- **Article 2 (Fan et al.)**: Archie formation factor and resistivity index
  (Eqs. 1-2), Waxman-Smits saturated conductivity C₀ = (Cw + B·Qv)/F* (Eqs. 3-4),
  cation mobility B(Cw) (Eq. 5), Qv from CEC (Eq. 6), and a partial-saturation
  Waxman-Smits conductivity whose log I vs log Sw slope gives the apparent
  saturation exponent. Reproduces the paper's finding that clay lowers the
  apparent n (clean n = 2 → ~1.0 for high-CEC clay) and that higher Cw dilutes
  the clay effect. The finite-element solve (Eq. 7) is represented by this model.

- **Article 3 (Mustafa et al.)**: the exothermic NaNO₂ + NH₄Cl reaction and its
  heat (ΔH = 369 kJ/mol), fractional improvement ratios, the dynamic elastic
  moduli E / ν / K / µ from Vp, Vs, ρ (Eqs. 1-2, 6-7), the Young-Laplace (Eq. 3)
  and centrifuge (Eq. 4) capillary pressures, and the scratch-test specific
  energy Ft = E·A (Eq. 5, verbatim). Reproduces the headline results: porosity
  +80% (limestone) / +40.4% (Scioto), permeability +1359.9% / +320.7%, and the
  UCS 38.2 → 17.1 MPa (−55%) softening.

- **Article 4 (Emelyanova et al.)**: the neutron-density separation ND (Eq. 1),
  the expert baffle rule (ZDN > 2.55 and MLR > 15), the cluster-area, indicator
  (min-max), purity (P = A_E4/A_C) and decision (D = K/N, stop at 1) metrics
  (Eqs. 2-6), and the F1 score (Eq. 7). The per-node spectral-clustering splitter
  is represented by a k-means + silhouette proxy; reproduces the ~0.98 reservoir
  F1 and a clean two-facies split.

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
