# Petrophysics October 2016 - Vol. 57, No. 5

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 57, No. 5 (October 2016): five articles spanning electromagnetic
look-ahead-while-drilling resistivity, pore-scale drainage/imbibition
water-saturation models in tight gas, first-order error propagation as part of
petrophysical calculation, advanced dielectric/CRIM log interpretation, and
microfracturing for in-situ stress measurement.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article3_foep_error_propagation.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_emla_lookahead_resistivity.py` | Looking Ahead of the Bit While Drilling: From Vision to Reality | Constable, Antonsen, Stalheim, Olsen, Fjell, Dray, Eikenes, Aarflot, Haldorsen, Digranes, Seydoux, Omeragic, Thiel, Davydychev, Denichou, Salim, Frey, Homan, Tan | 426-446 |
| `article2_tightgas_saturation_height.py` | How Pore-Scale Attributes May Be Used to Derive Robust Drainage and Imbibition Water-Saturation Models in Complex Tight-Gas Reservoirs | Merletti, Gramin, Salunke, Hamman, Spain, Shabro, Armitage, Torres-Verdín, Salter, Dacy | 447-464 |
| `article3_foep_error_propagation.py` | On Error Calculation and Use of First-Order Error Propagation as Integral Part of Petrophysical Calculation | Stalheim | 465-478 |
| `article4_dielectric_archie_greenriver.py` | Advanced Log Interpretation in Field Development | Merkel, Lessenger | 479-491 |
| `article5_microfracturing_insitu_stress.py` | How Can Microfracturing Improve Reservoir Management? | Malik, Jones, Boratko | 492-507 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 57 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2016_10.pdf`,
> ~13 MB) has a text layer, so titles, authors, page ranges, and the article
> bodies were read directly; **all five articles have full bodies**. Many
> numbered relations survived as inline ASCII in the body text (the
> capillary-pressure conversions and modified Brooks-Corey imbibition model in
> Article 2, the full FOEP matrix relations in Article 3, the skin-depth
> relation and CRIM workflow in Article 4), while the typeset display-equation
> glyphs were dropped; those are faithful standard-form reconstructions from the
> surviving variable definitions. (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Constable et al.)**: EMLA look-ahead resistivity - a technology /
  case-study paper. Implements the EM skin depth, the ultradeep
  harmonic-resistivity (UHRA/UHRP) attenuation and phase shift from the antenna
  couplings (Eqs. 1-2), and a depth-of-detection-ahead scaling with the
  transmitter-receiver span and skin depth (field range ~5-30 m). The typeset
  coupling combination lost its glyphs, so the attenuation/phase pair uses the
  standard ratio form of propagation-resistivity tools.

- **Article 2 (Merletti et al.)**: tight-gas saturation-height - the
  air-mercury→air-brine (Eq. 1) and lab→reservoir (Eqs. 4-5) capillary-pressure
  conversions, the clay-bound-water correction (Eqs. 2-3), the Thomeer (1960)
  drainage model, the Land (1968) trapped-gas model (Eq. 8) with `Swgt = 1 - Sgt`
  (Eq. 10), and the modified Brooks-Corey imbibition `Pc`/`Sw` (Eqs. 11-13), plus
  a buoyancy capillary-pressure (height-above-free-water-level) helper.

- **Article 3 (Stalheim)**: first-order error propagation - the matrix-form FOEP
  error `σ_f = √(c′Σc)` (Eq. 3), the variance-covariance matrix from standard
  deviations and correlations (Eqs. 8, 14), the relative contribution of each
  input (Eq. 13), and a numerical Jacobian for nonlinear functions (Eq. 12). The
  petrophysical functions (density porosity, Archie Sw, Vsh, effective porosity;
  Eqs. 15-18) are included, with an analytic porosity Jacobian (Eq. 20).

- **Article 4 (Merkel & Lessenger)**: dielectric/CRIM interpretation - the EM
  skin depth (Eq. 1), the complex-refractive-index-method (CRIM) mixing law and
  the salinity-independent bulk-volume water it yields, and a Pickett-plot
  least-squares fit for the Archie cementation exponent m and water resistivity
  Rw (then Archie Sw with the calibrated exponents).

- **Article 5 (Malik et al.)**: microfracturing - a procedural / case-study
  paper. Implements the overburden stress from a density profile, the Eaton-type
  minimum horizontal stress with pore pressure, the Kirsch breakdown and
  reopening pressures, the maximum-horizontal-stress inversion, the net pressure
  (ISIP - closure), and the G-function-time closure pick (Nolte, 1979).

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2016)
Reference: Petrophysics Vol. 57, No. 5, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
