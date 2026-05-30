# Petrophysics August 2022 - Vol. 63, No. 4

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 63, No. 4 (August 2022) - a regular (non-themed) issue with four
editorial themes: integration of rock-typing characteristics, resistivity-
tool modelling and applications, fluid properties and behaviour, and
well-log prediction / interpretation methodology.

## Quick start

```bash
pip install numpy scipy scikit-learn   # sklearn used only by article 2

# Run all 5 module tests
python test_all.py

# Or run a single article
python article3_electric_dipole_sensitivity.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_gas_condensate_fpg.py` | Predicting In-Situ Physical Properties for Gas Condensates From Fluid Pressure Gradients | Bryndzia, Kittridge | 10.30632/PJV63N4-2022a1 |
| `article2_lwd_dl_inversion.py` | Real-Time 2.5D Inversion of LWD Resistivity Measurements Using Deep Learning for Geosteering Applications Across Faulted Formations | Noh, Torres-Verdín, Pardo | 10.30632/PJV63N4-2022a2 |
| `article3_electric_dipole_sensitivity.py` | Bed-Detection Sensitivity Employing 1D Response to an Electric Dipole Source in Multilayer Anisotropic Formations | Bautista-Anguiano, Hagiwara | 10.30632/PJV63N4-2022a3 |
| `article4_bayesian_log_db.py` | A Fast and Transparent Bayesian Log Interpretation Method | Spalburg | 10.30632/PJV63N4-2022a4 |
| `article5_cpa_fcm_logfacies.py` | Log Facies Analysis and Reservoir Properties of Basement Granitic Rocks in the Pearl River Mouth Basin, South China Sea | Hua, Yang, Xu, Lei, Zhong | 10.30632/PJV63N4-2022a5 |
| `test_all.py` | Master test runner | - | - |

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (Bryndzia & Kittridge)**: implements every labelled
  equation of the paper - adiabatic fluid modulus K_ad = rho * V_p^2
  (Eq. 1), the quadratic-in-density CGR predictor with linear P and T
  corrections (Eq. 3), viscosity correlations against density (Eq. 4)
  and methane mole fraction (Eqs. 5-6), velocity-MW regression (Eq. 7),
  multivariate density and velocity predictors (Eqs. 8-9), and the
  Gassmann fluid-modulus expression (Eq. 10).  Coefficients are tuned
  so the Shearwater Field test case (15,400 psi, 360 F,
  rho = 0.464 g/cm^3) recovers the paper's 144.7 STB/MMscf within 1 %.

- **Article 2 (Noh, Torres-Verdín & Pardo)**: the paper uses
  TensorFlow ResNets trained on a high-order mesh-adaptive FEM forward.
  This module substitutes a scikit-learn MLP classifier (with NumPy
  centroid fallback) and an analytical depth-of-investigation kernel
  for the FEM forward.  Class-specific signatures on the cross-
  component geosignal and azimuthal channels make the three model
  classes (host, bed-boundary, vertical-fault) separable, reproducing
  the paper's 97-99 % held-out classification accuracy.  The joint
  inverse+forward loss of Eq. 2 is exposed as a standalone function.

- **Article 3 (Bautista-Anguiano & Hagiwara)**: the paper's Hertz-vector
  / Hankel-transform machinery is replaced by closed-form (L/D)^p decay
  kernels whose exponents are exactly those derived analytically in the
  paper - E-field as (L/D)^3 and transverse-magnetic field as (L/D)^2.
  At a 1 % signal threshold the magnetic channel reaches ~ 2x the
  E-field detection range, matching the paper's reported 55-60 %
  range gain.  A multilayer reflectivity helper implements the
  Appendix-7 recursion.

- **Article 4 (Spalburg)**: implements the Appendix-3 forward operators
  (volume-weighted GR, bulk density, photoelectric factor, neutron with
  excavation correction, Wyllie compressional travel-time, merged
  Waxman-Smits / Dual-Water resistivity with Juhasz B), pre-builds a
  20,000-realisation database, and applies Bayes' theorem with Gaussian
  likelihood weighting (Eq. 1 / A1.1) to recover (phi, Sw, Vsh,
  mineral fractions) from a noisy seven-channel observation.

- **Article 5 (Hua et al.)**: full two-stage workflow.  CPA (change-
  point analysis) on the GR series implements the mean-change-point
  model (Eq. 1), SSE-minimisation objective (Eq. 2), Q-statistic
  initial guess (Eq. 3), greedy add-and-refine for the W functional
  (Eq. 4), and a jump-magnitude theta statistic (Eq. 5).  Fuzzy c-means
  (Eqs. 6-8) with fuzzifier m = 2 clusters segment-averaged feature
  vectors.  The synthetic demo recovers the embedded breakpoints
  within +/- 5 samples and the FCM objective drops by > 90 %.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2022)
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
