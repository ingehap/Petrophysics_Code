# Petrophysics December 2016 - Vol. 57, No. 6

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 57, No. 6 (December 2016): five articles plus one technical note, spanning
shale fracturing characterization with machine learning, geomechanics of
orthorhombic media, shale Young's moduli from nanoindentation, 2D NMR fluid
typing in kerogen isolates, ultrasonic-image permeability in carbonates, and
gamma-ray log normalization across mixed well geometries.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article2_orthorhombic_geomechanics.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_shale_fracturing_ml.py` | Shale Fracturing Characterization and Optimization by Using Anisotropic Acoustic Interpretation, 3D Fracture Modeling, and Supervised Machine Learning | Gu, Gokaraju, Chen, Quirein | 573-587 |
| `article2_orthorhombic_geomechanics.py` | Geomechanics of Orthorhombic Media | Far, Quirein, Mekic | 588-596 |
| `article3_shale_youngs_nanoindentation.py` | Macroscale Young's Moduli of Shale Based on Nanoindentations | Li, Sakhaee-Pour | 597-603 |
| `article4_2d_nmr_kerogen_fluid_typing.py` | Fluid Typing and Pore Size in Organic Shale Using 2D NMR in Saturated Kerogen Isolates | Singer, Chen, Hirasaki | 604-619 |
| `article5_ultrasonic_permeability_carbonate.py` | Permeability Estimation Using Ultrasonic Borehole Image Logs in Dual-Porosity Carbonate Reservoirs | Menezes de Jesus, Martins Compan, Surmas | 620-637 |
| `article6_gr_normalization_haynesville.py` | *Technical Note:* Normalizing Gamma-Ray Logs Acquired from a Mixture of Vertical and Horizontal Wells in the Haynesville Shale | Xu, Bayer, Wunderle, Bansal | 638-643 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 57 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2016_12.pdf`,
> ~18 MB) has a text layer, so titles, authors, and page ranges were read from
> the contents page and bodies; **all six items have full bodies**. As with the
> other issues, the typeset formula glyphs were dropped in extraction, so the
> numbered formulas are faithful standard-form reconstructions from the surviving
> variable definitions (the orthorhombic-media equations survived most fully).
> (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Gu et al.)**: shale fracturing - the ANNIE and modified-ANNIE VTI
  stiffness closure, the equivalent isotropic Young's modulus from the
  anisotropic moduli, and the return-on-fracturing-investment objective.

- **Article 2 (Far et al.)**: orthorhombic geomechanics - Hooke's law and
  compliance, the orthorhombic horizontal-stress model with pore pressure (which
  reduces to VTI), and shear-wave splitting.

- **Article 3 (Li & Sakhaee-Pour)**: shale Young's moduli - the indentation
  modulus and hardness, Young's modulus from the indentation modulus, and the
  representative (soft-entity-controlled) vs volume-average modulus upscaling.

- **Article 4 (Singer et al.)**: 2D NMR kerogen - the bulk/surface relaxation
  split, the surface-relaxivity / surface-to-volume relation and pore diameter,
  the porosity^(2/3) BET partition, and T1/T2 fluid typing.

- **Article 5 (Menezes de Jesus et al.)**: ultrasonic image permeability - the
  amplitude attenuation and acoustic reflectance, the multi-class image-
  permeability transform (with the paper's fitted coefficients), and the
  calibration objective.

- **Article 6 (Xu et al.)** *(technical note)*: gamma-ray normalization - the
  histogram normalization, the true-stratigraphic-thickness projection, the
  affine mean/standard-deviation matching, and the maximum percent shift.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2016)
Reference: Petrophysics Vol. 57, No. 6, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
