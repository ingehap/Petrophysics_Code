# Petrophysics October 2021 - Vol. 62, No. 5

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 62, No. 5 (October 2021) - the special issue on **"Applications of 3D
Printing and Synthetic Rocks in Petrophysics, Rock Physics, and Rock
Mechanics"**.  Nine papers spanning binder-saturation control of printed-rock
porosity, image-processing petrophysics education, original-size carbonate
pore replication, 3D-printed mudrock micromodels, fractal characterization of
digital rocks, pore-volume compressibility of unconsolidated sands, fluid
effects on the elastic properties of printed anisotropic rock, joint-roughness
shear behavior of printed samples, and near-wellbore perforation fracturing.

## Quick start

```bash
pip install numpy

# Run all 9 module tests
python test_all.py

# Or run a single article
python article5_fractal_digital_rock.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_binder_saturation_porosity.py` | Binder Saturation as a Controlling Factor for Porosity Variation in 3D-Printed Sandstone | Hodder, Craplewe, Ishutov, Chalaturnyk | 10.30632/PJV62N5-2021a1 |
| `article2_image_processing_petrophysics.py` | Enhanced Learning of Fundamental Petrophysical Concepts Through Image Processing and 3D Printing | Alyafei, Al Musleh, Bautista, Idris, Seers | 10.30632/PJV62N5-2021a2 |
| `article3_carbonate_pore_replication.py` | Replication of Carbonate Reservoir Pores at the Original Size Using 3D Printing | Ishutov, Hodder, Chalaturnyk, Zambrano-Narvaez | 10.30632/PJV62N5-2021a3 |
| `article4_3dprint_mudrock_micromodel.py` | 3D Printing Mudrocks: Experiments in Validating Clay as a Build Material for 3D Printing Porous Micromodels | Hasiuk, Harding | 10.30632/PJV62N5-2021a4 |
| `article5_fractal_digital_rock.py` | Fractal Characterization and Petrophysical Analysis of 3D Dynamic Digital Rocks of Sandstone | Zhao, Luo, Li, Wu, Mao, Ostadhassan | 10.30632/PJV62N5-2021a5 |
| `article6_pore_volume_compressibility.py` | Pore Volume Compressibility of Unconsolidated Sand Reservoirs: Insights Gained Using Laboratory-Created Sand Pack Analogs | Hathon, Myers, Arya | 10.30632/PJV62N5-2021a6 |
| `article7_3dprint_anisotropic_elastic.py` | Effect of Fluids on the Elastic Properties of 3D-Printed Anisotropic Rock Models | Dande, Stewart, Dyaur | 10.30632/PJV62N5-2021a7 |
| `article8_joint_roughness_shear.py` | The Effect of Joint Roughness on Shear Behavior of 3D-Printed Samples Containing a Non-Persistent Joint | Fereshtenejad, Kim, Song | 10.30632/PJV62N5-2021a8 |
| `article9_perforation_fracture_morphology.py` | Research of Near-Wellbore Fracture Morphology, Formation Mechanism, and Propagation Law for Different Perforation Modes During the Perforation Process | Wang, Li, Xu, Jia, Zhang | 10.30632/PJV62N5-2021a9 |
| `test_all.py` | Master test runner | - | - |

> **Note on coverage.** Articles 1-8 are implemented directly against the
> methods described in the paper bodies.  Article 9 was only partly present
> in the source-PDF text extract used to build this folder (the extract was
> truncated partway through its results, and the paper - an experimental
> perforation study - transcribes no equations), so its module is a
> **methodology proxy** that encodes the perforation-mode taxonomy the paper
> describes plus the standard near-wellbore stress relations.  Separately,
> throughout this issue the *typeset equations were stored as images* and did
> not survive text extraction (only the equation numbers remained), and several
> papers are experimental notes with few or no equations; so the formulas here
> are faithful **standard-form reconstructions** of the methods the prose
> describes - not byte-perfect transcriptions.  When the original typeset
> equations / full paper bodies become available, the modules can be replaced
> in place.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (Hodder et al.)**: printed cylinder volume (Eq. 2), binder
  volume from burnout, binder volume fraction (Eq. 4), binder saturation
  level (Eq. 5), and the theoretical porosity trend phi = phi0*(1-S).
  Reproduces the paper's 36/34/32% porosity at 10/15/20% saturation and the
  4-vol%-binder -> 10%-saturation worked example.

- **Article 2 (Alyafei et al.)**: the image-processing pipeline - porosity as
  pore-pixel fraction, phase saturation, irreducible/residual saturation,
  displacement efficiency (Eq. 1), equivalent grain radius, and the contact-
  angle wettability rule.  Reproduces the worked numbers (porosity 27.01%,
  S_wir 0.332, S_or 0.226, E_D ~ 66%).

- **Article 3 (Ishutov et al.)**: pore-size scaling (1:1 vs 5x), equivalent
  spherical pore diameter, cylinder bulk volume, and the scaffolding print-
  time speedup.  (Technical note - publishes no equations or error metrics.)

- **Article 4 (Hasiuk & Harding)**: the Washburn pore-throat-diameter relation
  (Eq. 1) matching the paper's two anchor points (a few psi -> tens of microns;
  33,000 psi -> single-digit nm), the Boyle's-law grain-volume helper,
  porosity from bulk/grain volumes, and the firing dimensional/mass loss.

- **Article 5 (Zhao et al.)**: box-counting fractal dimension (Eq. 3,
  validated against a Sierpinski carpet -> log8/log3 = 1.893), the three
  permeability power laws K(phi)/K(D)/K(Su) (Eqs. 10-12), Archie formation
  factor and cementation-exponent inversion (Eq. 13), and gliding-box
  lacunarity (Eq. 5).

- **Article 6 (Hathon et al.)**: the uniaxial compaction coefficient (Eq. 1),
  pore-volume compressibility Cp = Cm/phi (Eq. 2), the Trask sorting
  coefficient (Eq. 3), and a peaked Cm-vs-effective-stress demonstrator
  (Regions A/B/C).  (Paper defers its forward model to a future publication.)

- **Article 7 (Dande et al.)**: saturated bulk density (Eq. 1), velocity from
  traveltime, isotropic moduli K/G/E/nu from Vp/Vs/rho, Thomsen epsilon/gamma
  anisotropy, Gassmann fluid substitution, and Vp/Vs and impedance.
  Reproduces the worked moduli (rho=790, Vp=2000, Vs=700 -> G=0.39, K=2.65 GPa,
  nu=0.43) and the air-saturated epsilon ~ 0.26.

- **Article 8 (Fereshtenejad et al.)**: Z2 root-mean-square slope of a joint
  profile, the Tse & Cruden (1979) Z2 -> JRC correlation (the relation the
  paper explicitly uses), Barton-Bandis peak shear strength, Mohr-Coulomb,
  and secant shear stiffness - reproducing the qualitative tau-vs-JRC and
  tau-vs-normal-stress trends.

- **Article 9 (Wang et al.)** *(proxy)*: the perforation-mode classification
  (spiral / directional / fixed-plane / interlaced), the three-microfracture-
  type taxonomy, and the standard near-wellbore Kirsch hoop-stress and
  tensile breakdown-pressure relations governing fracture initiation.

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
