# Petrophysics October 2020 - Vol. 61, No. 5

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 61, No. 5 (October 2020) - a regular issue of seven papers spanning
nanoindentation of shale cuttings, classification of shale adsorption-isotherm
curves, automatic wellbore cave-in detection by unsupervised clustering, a
petrophysically consistent Archie's equation for heterogeneous carbonates,
wettability and water-blockage in organic-rich tight rocks, neural-network
prediction of sonic transit times from drilling parameters, and an integrated
multiphysics rock-classification workflow.

## Quick start

```bash
pip install numpy

# Run all 7 module tests
python test_all.py

# Or run a single article
python article6_sonic_transit_drilling_nn.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_nanoindentation_shale.py` | Nanoindentation of Shale Cuttings and Its Application to Core Measurements | Esatyana, Sakhaee-Pour, Sadooni, Al-Kuwari | 10.30632/PJV61N5-2020a1 |
| `article2_adsorption_isotherm_classification.py` | Classification of Adsorption Isotherm Curves for Shale Based on Pore Structure | Tian, Chen, Yan, Deng, He | 10.30632/PJV61N5-2020a2 |
| `article3_cavein_clustering_detection.py` | Automatic Detection of Anomalous Density Measurements due to Wellbore Cave-in | Sen, Ong, Kainkaryam, Sharma | 10.30632/PJV61N5-2020a3 |
| `article4_archie_carbonate_consistent.py` | Towards a Petrophysically Consistent Implementation of Archie's Equation for Heterogeneous Carbonate Rocks | Ramamoorthy, Ramakrishnan, Dasgupta, Raina | 10.30632/PJV61N5-2020a4 |
| `article5_wettability_water_blockage.py` | Revisiting the Concept of Wettability for Organic-Rich Tight Rocks: Application in Formation Damage–Water Blockage | Mukherjee, Dang, Rai, Sondergeld | 10.30632/PJV61N5-2020a5 |
| `article6_sonic_transit_drilling_nn.py` | Prediction of Sonic Wave Transit Times From Drilling Parameters While Horizontal Drilling in Carbonate Rocks Using Neural Networks | Gowida, Elkatatny | 10.30632/PJV61N5-2020a6 |
| `article7_multiphysics_rock_classification.py` | Integrated Multiphysics Workflow for Automatic Rock Classification and Formation Evaluation Using Multiscale Image Analysis and Conventional Well Logs | Gonzalez, Kanyan, Heidari, Lopez | 10.30632/PJV61N5-2020a7 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2020_10.pdf`)
> has a text layer, so the article titles, authors, page ranges, DOIs, equation
> numbers, variable definitions, and many numeric constants were read directly
> from the paper bodies. The PDF-to-text conversion dropped most typeset formula
> *glyphs* (keeping the equation numbers and surrounding prose), so the numbered
> formulas in the modules are **faithful standard-form reconstructions** built
> from the preserved variable definitions and constants, using the standard
> textbook expressions each paper cites (Oliver-Pharr, Archie, symmetric
> Bruggeman, GLCM/Haralick, silhouette, Fjær dynamic moduli).

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions. A few practical notes:

- **Article 1 (Esatyana et al.)**: the Oliver-Pharr hardness H = Pmax/Ac and
  indentation modulus M = (√π/2)·S/(α·√Ac) (Eqs. 1a-1b, Berkovich α = 1.03),
  the specimen Young's modulus Es = M·(1−ν²) (Eq. 2), the ideal Berkovich area
  Ac = 24.5·hc², and the Johnson plastic-zone radius (Eq. 4) that sets the
  lower bound on indent spacing. Reproduces the ~20 GPa modulus basis and the
  < 6% Poisson sensitivity.

- **Article 2 (Tian et al.)** *(classification, no equations)*: the standard
  BET linearization for monolayer volume / surface area, the IUPAC five-type
  classifier, the micro/meso/macro pore-size classes, a sorting class from the
  pore-size-distribution spread, and the paper's new three-parameter
  (shape × size × sorting) scheme giving 27 curve types.

- **Article 3 (Sen et al.)**: the rolling coefficient of variation of bulk
  density (Eq. 5) and caliper rugosity (Eq. 1) as features, with the TICC
  good-hole/bad-hole clustering (Eqs. 2-4) represented by a Gaussian k-means
  proxy plus temporal (median-filter) smoothing - recovers a planted cave-in
  zone with > 80% recall and < 10% false flags.

- **Article 4 (Ramamoorthy et al.)**: Archie (Eqs. 1a-1c), the formation factor
  and effective cementation exponent (Eq. 6), and a symmetric-Bruggeman
  homogenization (Eqs. 4-5) showing the effective m varies with vug fraction
  (separate vugs elevate m above 2, the classic vuggy-carbonate effect) and that
  the vuggy resistivity index rises with an effective saturation exponent well
  below the Archie value of 2 (near unity).

- **Article 5 (Mukherjee et al.)**: the Young-Laplace capillary pressure and
  Washburn pore-throat radius, the wettability pore-type fractions (water-wet /
  oil-wet / mixed) from spontaneous-imbibition volumes (Eqs. 1-3), and the
  water-blockage trapped-water saturation with the ~1,500 psi threshold to
  restore oil-phase continuity within the 7,000 psi step-pressurization.

- **Article 6 (Gowida & Elkatatny)**: a compact single-hidden-layer tanh ANN
  predicting sonic transit time from six drilling parameters (WOB, RPM, ROP,
  torque, SPP, GPM), scored by the correlation coefficient R and AAPE, then the
  dynamic Poisson's ratio and Young's modulus from Vp, Vs, ρ (Eqs. 1-2). The
  numpy net reaches R ≈ 0.99 / AAPE ≈ 1.3% on synthetic data, comparable to the
  paper's reported R ≈ 0.94 / AAPE ≈ 1-1.9%.

- **Article 7 (Gonzalez et al.)**: the mean gray level (Eq. 1), GLCM contrast
  and energy (Eqs. 2-3), the experimental variogram for window selection
  (Eq. 5), the silhouette coefficient (Eq. 6), and k-means rock classification
  with the permeability cost function (Eq. 7) whose convergence picks the
  optimum number of classes (here matching the three formations).

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
