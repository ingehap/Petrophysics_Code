# Petrophysics December 2019 - Vol. 60, No. 6

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 60, No. 6 (December 2019) - a **"Best of the 2019 Symposium, Part 2"**
section (articles 1-4) followed by regular submissions (articles 5-10).  Topics
span sonic-slowness deconvolution, ultrasonic LWD caliper/imaging, deducing
permittivity from LWD resistivity, an improved crushed-rock (GRI+) workflow, NMR
light-hydrocarbon/pore-size/tortuosity evaluation, magnetic-susceptibility
effects on NMR, ANN formation-top picking, supervised classifiers for vuggy
facies, gas-hydrate joint elastic-electrical inversion, and a micro/nanofluidic
transport review.

## Quick start

```bash
pip install numpy

# Run all 10 module tests
python test_all.py

# Or run a single article
python article3_lwd_permittivity.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_sonic_inversion_deconvolution.py` | Inversion of High-Resolution High-Quality Sonic Compressional and Shear Logs for Unconventional Reservoirs | Lei, Zeroug, Bose, Prioul, Donald | 10.30632/PJV60N6-2019a1 |
| `article2_ultrasonic_lwd_imaging.py` | New 4.75-in. Ultrasonic LWD Technology Provides High-Resolution Caliper and Imaging in Oil-Based and Water-Based Muds | Li, Lee, Coates, Jin, Wong | 10.30632/PJV60N6-2019a2 |
| `article3_lwd_permittivity.py` | Deducing Electrical Permittivity of Formations From LWD Resistivity Measurements | Stalheim | 10.30632/PJV60N6-2019a3 |
| `article4_crushed_rock_gri_plus.py` | Crushed-Rock Analysis Workflow Based on Advanced Fluid Characterization for Improved Interpretation of Core Data | Nikitin, Durand, McMullen, Blount, Driskill, Hows | 10.30632/PJV60N6-2019a4 |
| `article5_nmr_lighthc_chalk.py` | NMR Evaluation of Light-Hydrocarbon Composition, Pore Size, and Tortuosity in Organic-Rich Chalks | Chen, Singer, Wang, Vinegar, Nguyen, Hirasaki | 10.30632/PJV60N6-2019a5 |
| `article6_nmr_magnetic_susceptibility.py` | Influence of Magnetic Susceptibility Contrast on NMR Studies—Experimental Analysis From Siliciclastic Reservoirs | Sarkar, Chatterjee, Lal, Kumar, Deo | 10.30632/PJV60N6-2019a6 |
| `article7_ann_formation_tops.py` | New Robust Model to Estimate Formation Tops in Real Time Using Artificial Neural Networks (ANN) | Elkatatny, Al-AbdulJabbar, Mahmoud | 10.30632/PJV60N6-2019a7 |
| `article8_ml_vuggy_facies_classifiers.py` | A Comparative Study of Three Supervised Machine-Learning Algorithms for Classifying Carbonate Vuggy Facies in the Kansas Arbuckle Formation | Deng, Xu, Jobe, Xu | 10.30632/PJV60N6-2019a8 |
| `article9_gashydrate_inverse_rockphysics.py` | Joint Interpretation of Elastic and Electrical Data for Petrophysical Properties of Gas-Hydrate-Bearing Sediments Using Inverse Rock Physics Modeling Method | Pan, Li, Zhang, Chen, Cai, Geng | 10.30632/PJV60N6-2019a9 |
| `article10_micronanofluidic_transport_review.py` | Review of Micro/Nanofluidic Insights on Fluid Transport Controls in Tight Rocks | Mehmani, Kelly, Torres-Verdín | 10.30632/PJV60N6-2019a10 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2019_12.pdf`,
> 20 MB) has a text layer, so the article titles, authors, page ranges, and DOIs
> were read directly from the contents page and paper bodies. The machine text
> extraction captured the full bodies of **articles 1-6** but **truncated at
> journal page 823** (mid Article 6's author bios), so **articles 7-10 were
> available only as table-of-contents entries**. Articles 7-10 are therefore
> implemented as **methodology proxies** of the standard, well-established
> methods their titles describe; and, as with the other issues, the typeset
> formula glyphs were dropped in extraction, so the numbered formulas are
> faithful standard-form reconstructions built from the preserved variable
> definitions.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions. A few practical notes:

- **Article 1 (Lei et al.)**: the aperture response kernel, the average-slowness
  convolution d_N = conv(F_N, s) (Eqs. 1, 8), the stacked multiaperture system
  D = G·S (Eqs. 10-12), the Moore-Penrose deconvolution S = G⁺·D (Eq. 13), and
  the QC mismatch (Eq. 14) - recovers a high-resolution slowness log from blurred
  multiaperture logs.

- **Article 2 (Li et al.)** *(hardware/field paper):* the pulse-echo standoff
  = c_mud·t/2, borehole radius/caliper, acoustic impedance Z = ρ·c with the
  reflection coefficient, and a cosine least-squares fit recovering tool
  eccentering from the four azimuthal standoffs.

- **Article 3 (Stalheim)**: the lossy-medium complex wavenumber (Eqs. 1-3, 7-8),
  σ = 2k_r·k_i/(ωμμ₀) (Eq. 4) and ε_r = (k_r²−k_i²)/(ω²μ₀ε₀) (Eq. 5) inverted from
  it, the wavelength λ = 2π/k_r (Eq. 13), and the CRIM mixing / water-saturation
  relations (Eqs. 14, 20) - round-trips σ and ε.

- **Article 4 (Nikitin et al.)**: the fluid-summation and bulk/grain porosities
  (Eqs. 1-2), the conventional crushed-rock water saturation (Eq. 3), and the
  GRI+ water saturation (Eq. 4) with the NMR-derived crushing-loss factor
  β_crush - shows the legacy method understates Sw (paper reports ~30%).

- **Article 5 (Chen et al.)**: the HI porosity rescaling (Eq. 1), the apparent
  T2 = (1/T2 + 1/T2D)⁻¹ with the diffusion term (γ·G·TE)²·D/12 (Eqs. 2-3), the
  diffusion length (Eq. 5), the short-time Padé restricted-diffusion ratio with
  the spherical-pore radius (Eq. 6), and the tortoisity τ = D₀/D(∞) (Eq. 7).

- **Article 6 (Sarkar et al.)**: the three-mechanism relaxation
  1/T2 = 1/T2B + ρ(S/V) + (γ·G·TE)²·D/12 (Eqs. 1-2) and the internal gradient
  recovered from the slope of 1/T2 vs TE² (Eq. 3) - reproduces a planted gradient
  in the paper's 72-510 Gauss/cm range - plus the structural/diffusion/dephasing
  length scales.

- **Article 7 (Elkatatny et al.)** *(methodology proxy):* a feed-forward tanh
  ANN estimating a formation-top depth marker from six drilling parameters,
  scored by R, RMSE, and AAPE (R ≈ 0.98, AAPE < 1% on synthetic data).

- **Article 8 (Deng et al.)** *(methodology proxy):* three supervised
  classifiers - logistic regression, k-nearest-neighbor, and a bagged
  decision-stump ensemble (random-forest analogue) - for vuggy-facies
  classification, with a confusion matrix and accuracy / precision / recall / F1.

- **Article 9 (Pan et al.)** *(methodology proxy):* inverse rock-physics modeling
  jointly inverting a velocity-vs-hydrate-saturation stiffening model and an
  Archie resistivity model for hydrate saturation - more robust than the noisier
  single measurement alone.

- **Article 10 (Mehmani et al.)** *(review, methodology proxy):* the gas mean
  free path and Knudsen number, the flow-regime classification
  (continuum / slip / transition / free-molecular), the Klinkenberg slip-corrected
  apparent permeability, a Beskok-Karniadakis apparent-permeability enhancement,
  and the capillary number.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2019)
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
