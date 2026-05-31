# Petrophysics December 2014 - Vol. 55, No. 6

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 55, No. 6 (December 2014) - a review and new dry-clay-parameter shaly-sand
method, an inversion-based workflow for the new-generation oil-based-mud
resistivity imager, an experimental NMR/dielectric study of wettability and
fluid saturation in limestone, a pore-scale evaluation of dielectric
measurements in complex pore/grain structures, the physical basis for a
cased-well quantitative gas-saturation method, and reminiscences on the
development of the first commercial array-induction measurement.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article1_shaly_sand_dry_clay.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_shaly_sand_dry_clay.py` | Review of Existing Shaly-Sand Models and Introduction of a New Method Based on Dry-Clay Parameters | Peeters, Holmes | 543-553 |
| `article2_obm_imager_inversion.py` | Inversion-Based Workflow for Quantitative Interpretation of the New-Generation Oil-Based-Mud Resistivity Imager | Chen, Omeragic, Habashy, Bloemenkamp, Zhang, Cheung, Laronga | 554-571 |
| `article3_nmr_dielectric_limestone.py` | Experimental Study of the Effects of Wettability and Fluid Saturation on Nuclear Magnetic Resonance and Dielectric Measurements in Limestone | Venkataramanan, Hürlimann, Tarvin, Fellah, Acero-Allard, Seleznev | 572-586 |
| `article4_porescale_dielectric.py` | Pore-Scale Evaluation of Dielectric Measurements in Formations with Complex Pore and Grain Structures | Chen, Heidari | 587-597 |
| `article5_cased_well_gas_saturation.py` | Physical Basis for a Cased-Well Quantitative Gas-Saturation Analysis Method | Inanc, Gilchrist, Ansari, Chace | 598-617 |
| `article6_array_induction_geometric_factors.py` | Reminiscences on the Development of the First Commercial Array-Induction Measurement | Elkington | 618-623 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 55 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2014_12.pdf`,
> ~15 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly. Article 1 is the most fully recoverable: its
> shaly-sand conductivity equations (Waxman-Smits, dual-water, modified
> Simandoux) and the new dry-clay difference method survived as inline text.
> Article 4's CRIM and directional-mixing relations also survived. For Articles
> 2 and 5 the *defining* relations (the impedivity definitions; the neutron
> slowing-down physics and count-ratio definitions) survived, while several
> typeset display equations were dropped in extraction and are faithfully
> reconstructed in standard form. Article 6 is a historical narrative essay with
> no display equations; it is implemented as the documented signal-processing
> workflow in standard Doll induction-response form. The cover features the
> OBM-imager paper (Chen et al., Article 2). (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Peeters & Holmes)**: shaly-sand models - the volumetric bound-
  water relations (Eqs. 1-3), the Waxman-Smits conductivity (Eq. 9), Juhasz's
  Qv from dry-clay volume (Eq. 10), the new neutron-density difference method
  for shale and dry-clay volumes (Eqs. 18-19) and the resulting Qv (Eq. 20),
  the dual-water (Eq. 22) and modified-Simandoux (Eq. 25) conductivities, the
  bound-water conductivity (Eq. 23), and the apparent-cementation relation
  (Eq. A1-1). The per-clay dry-clay parameter table (Appendix 2) did not survive
  extraction, so the demo uses representative illite-like values as inputs.

- **Article 2 (Chen et al.)**: OBM imager inversion - the exact formation/mud
  impedivity definitions `xi = 1/(j*w*eps0*eps + sigma)`, the ZB90 composite
  processing (projecting the button impedance perpendicular to the mud line),
  the multiplicatively regularized data misfit (Eq. 1) with its `lambda =
  alpha*misfit^beta` schedule (Eq. 2), and a scalar Gauss-Newton inversion that
  recovers the formation impedivity from the series mud+formation circuit.

- **Article 3 (Venkataramanan et al.)**: NMR & dielectric in limestone - the
  Washburn pore-throat radius (Eq. 1), the Land trapping constant and the
  disconnected-saturation quadratic (Eqs. 3-7), the NMR surface-relaxation T2
  (Eq. 8, with the spherical `A/V = 3/r` form), and the CRIM water saturation.

- **Article 4 (Chen & Heidari)**: pore-scale dielectric - the random-walk
  directional mean-square displacement and tortuosity (Eqs. 1-9), the complex
  relative permittivity (Eq. 10), the general and reservoir CRIM mixing laws
  (Eqs. 14-16), and the new directional, tortuosity-weighted model (Eqs. 17, 25)
  with the coefficient `f = alpha*tau^gamma + beta` (Eq. 18). The per-sample
  calibration constants (Eqs. 26-27, Fig. 4) are left as function inputs.

- **Article 5 (Inanc et al.)**: cased-well gas saturation - the elastic
  energy-loss and collision parameter (Eqs. 3-4), lethargy (Eq. 5), average
  lethargy gain (Eq. 6) and moderating power (Eq. 7), the inelastic and capture
  count ratios RIN13/RATO13 (Eqs. 1-2), and the gas saturation as a normalized
  (optionally nonlinear) interpolation between the gas- and liquid-filled
  Dynamic-Gas-Envelope limits (the Monte-Carlo chart step).

- **Article 6 (Elkington)**: array-induction reminiscences - the Doll cumulative
  radial geometric factor, the geometric-factor-weighted apparent conductivity
  (conductivity-domain combination), the weighted addition of array sub-responses
  to synthesize a target geometric factor, the sparse depth-shift vertical filter,
  and the three-measurement step-profile invasion inversion for (Rxo, Rt, Di).

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2014)
Reference: Petrophysics Vol. 55, No. 6, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
