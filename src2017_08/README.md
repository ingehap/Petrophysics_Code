# Petrophysics August 2017 - Vol. 58, No. 4

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 58, No. 4 (August 2017): six articles spanning Bakken NMR relaxometry,
wettability from the NMR T1/T2 ratio, an analytical centrifuge capillary-pressure
model, the impact of measurement errors on pressure-gradient estimation,
sample-contamination quantification, and a fast-neutron gamma density method.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article3_centrifuge_capillary_pressure.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_bakken_nmr_relaxometry.py` | High- and Low-Field NMR Relaxometry and Diffusometry of the Bakken Petroleum System | Kausik, Fellah, Feng, Simpson | 341-351 |
| `article2_wettability_nmr_t1t2.py` | Laboratory and Downhole Wettability from NMR T1/T2 Ratio | Valori, Hursan, Ma | 352-365 |
| `article3_centrifuge_capillary_pressure.py` | An Analytical Model for Analysis of Capillary Pressure Measurements by Centrifuge | Andersen, Skjæveland, Standnes | 366-375 |
| `article4_pressure_gradient_errors.py` | The Impact of Depth and Pressure Measurement Errors on the Estimation of Pressure Gradients | Bowers, Schnacke, Hermance | 376-396 |
| `article5_contamination_quantification.py` | Advances in Quantification of Miscible Contamination in Hydrocarbon and Water Samples From Downhole to Surface Laboratories | Zuo, Gisolf, Pfeiffer, Achourov, Chen, Mullins, Edmundson, Partouche | 397-410 |
| `article6_fast_neutron_gamma_density.py` | A Method of Determining Formation Density Based on Fast-Neutron Gamma Coupled Field Theory | Zhang, Zhang, Liu, Wu, Wu, Jia, Ti, Li | 411-425 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 58 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2017_08.pdf`,
> ~15 MB) has a text layer, so titles, authors, and page ranges were read from
> the contents page and bodies. The machine extraction captured the full bodies
> of **articles 1-4** and the method of **article 5** (its appendix tail was
> truncated), but **article 6 was beyond the extraction (after p408)** and is
> implemented as a **methodology proxy**. As with the other issues, the typeset
> formula glyphs were dropped in extraction, so the numbered formulas are
> faithful standard-form reconstructions. (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Kausik et al.)**: Bakken NMR - the BPP spectral density, the
  T1/T2 ratio, component classification (kerogen / bitumen / clay-bound water /
  free oil / free water) by published cutoffs, and the hydrogen-index porosity
  correction.

- **Article 2 (Valori et al.)**: wettability from NMR - the bulk/surface
  relaxation split, the pore-volume-weighted mean T1/T2, the linear
  T1/T2 -> USBM* wettability calibration, and the surface affinity index.

- **Article 3 (Andersen et al.)**: centrifuge capillary pressure - the
  Hassler-Brunner inner-face capillary pressure, the critical rotation speed, the
  exponential saturation history Sw(t), and Corey relative permeabilities with a
  capillary-pressure correlation.

- **Article 4 (Bowers et al.)**: pressure-gradient measurement errors - the
  pressure-depth model, the pressure-on-depth / depth-on-pressure OLS bracket,
  orthogonal (total-least-squares) regression, and the method-of-moments gradient
  corrected for the depth-error variance.

- **Article 5 (Zuo et al.)**: contamination quantification - the exponential
  heavy-end composition, the two-endpoint mass balance and native recovery, the
  power-law optical-density / property cleanup, and the volume-to-weight
  conversion.

- **Article 6 (Zhang et al.)** *(methodology proxy)*: fast-neutron gamma density -
  the inelastic-gamma count attenuation with density, the density inverted from
  the count, the two-detector compensated density, and a spine-and-ribs standoff
  correction.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2017)
Reference: Petrophysics Vol. 58, No. 4, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
