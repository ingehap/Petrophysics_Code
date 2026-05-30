# Petrophysics April 2022 - Vol. 63, No. 2

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 63, No. 2 (April 2022) - a regular issue containing one "Best of the
2021 Symposium" paper followed by six regular submissions covering through-
tubing casing-deformation imaging, chalk permeability modelling,
pyrite-aware water saturation, time-lapse micro-CT of filter cakes,
methane solubility in oil-based mud, gas-hydrate rock physics, and
digital-core wellbore stability.

## Quick start

```bash
pip install numpy

# Run all 7 module tests
python test_all.py

# Or run a single article
python article2_kozeny_permeability_chalk.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_dec_tool_bayesian_gpr.py` | Through-Tubing Casing Deformation and Tubing Eccentricity Image Tool (Best of 2021 Symposium) | Yang, Qin, Olson, Rourke | 10.30632/PJV63N2-2022a1 |
| `article2_kozeny_permeability_chalk.py` | Permeability Modeling in Clay-Rich Carbonate Reservoir | Storebo, Meireles, Fabricius | 10.30632/PJV63N2-2022a2 |
| `article3_pyrite_saturation_hs_bounds.py` | Effect of Pyrite in Water Saturation Evaluation of Clay-Rich Carbonate | Storebo, Hjuler, Meireles, Fabricius | 10.30632/PJV63N2-2022a3 |
| `article4_microct_filtercake.py` | In-Situ Visualization and Characterization of Filter-Cake Deposition Using Time-Lapse Micro-CT Imaging | Schroeder, Torres-Verdín | 10.30632/PJV63N2-2022a4 |
| `article5_methane_solubility_obm.py` | Experimental Investigation on the Effect of Methane Solubility in Oil-Based Mud Under Downhole Conditions | Song, Sukari, Wang, Jiang, Cai, Xu, Huang | 10.30632/PJV63N2-2022a5 |
| `article6_gas_hydrate_rock_physics.py` | Rock Physics Modeling of Gas Hydrate Reservoirs Through Integrated Core and Well-Log Data in NGHP-02 Area, India | Kumar, Mishra, Chatterjee, Tiwari, Avadhani | 10.30632/PJV63N2-2022a6 |
| `article7_digital_core_wellbore_stability.py` | Application of Digital Core Technology in Wellbore Stability Research | Zhou, Ye, Zhu, Cheng, Song, Wang, Cai | 10.30632/PJV63N2-2022a7 |
| `test_all.py` | Master test runner | - | - |

> **Note on coverage.** Articles 1-3 are implemented directly against
> the equations in the paper bodies.  Articles 4-7 were only represented
> by the editor's letter and table of contents in the source-PDF extract
> used to build this folder; their modules are **methodology proxies**
> that demonstrate the algorithmic concept described in the editor's
> narrative.  When the full paper bodies become available, the proxy
> modules can be replaced in place.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (Yang et al., Best of 2021 Symposium)**: linearised
  magnetostatic transfer function (Eq. 1), casing/tubing flux-density
  ratio (Eqs. 2-3), eccentricity ratio (Eq. 4), deformation factor
  (Eq. 5), and a Gaussian-Process-Regression Bayesian inversion with a
  Matern-5/2 covariance kernel (Eqs. 7-9).  Cyclic angle handling uses
  the (cos, sin) Cartesian projection trick so theta wrap-around does
  not break the GPR; recovers (Ecc, theta, Def, gamma) to within
  ~0.005, ~0.5 deg, ~0.005, ~5 deg on a 24-Hall-probe synthetic
  measurement.

- **Article 2 (Storebo et al.)**: Kozeny permeability k = c * phi^3 /
  S_phi^2 (Eq. 1) with Mortensen shielding factor (Eq. 2), ternary
  density-based porosity (Eqs. 3-4), and four pore-space SSA
  estimators - mineralogy (Eq. 5), spectral GR (Eq. 6),
  Sw + pseudo-water-film thickness (Eqs. 7-8), and NMR T2 of the water
  peak (Eqs. 9-10).  Flow-zone-indicator FZI machinery (Eqs. 11-13)
  is exposed as standalone helpers.

- **Article 3 (Storebo et al.)**: Archie sigma_t = sigma_w * phi^m *
  Sw^n (Eq. 1) extended with Wiener bounds (Eqs. 2-3), Hashin-Shtrikman
  bounds for an isotropic two-component medium (Eqs. 4-5), Archie with
  extra conductivity (Eq. 6), Waxman-Smits and Clavier dual-water
  formulations (Eqs. 7-13), and weighted HS mixing for pyrite (Eqs.
  48-52).  Default constants reproduce the paper's Boje-2C numbers:
  mineral-bound water 82.9 S/m at 91 C, pyrite mineral conductivity
  1,500 S/m, weighting constant w = 0.03.

- **Article 4 (Schroeder & Torres-Verdín)** *(proxy)*: Dewan-Chenevert
  / Outmans sqrt(t) mudcake-growth model, mudcake porosity evolution
  under compaction stress, Kozeny-Carman permeability with evolving
  porosity, and a synthetic 2-D CT slice with detected mudcake
  thickness.

- **Article 5 (Song et al.)** *(proxy)*: Henry's-law /
  Krichevsky-Kasarnovsky methane solubility form plus a multivariate
  linear regression for ln(x_CH4) against (P, T, base-oil fraction,
  mud viscosity).  Synthetic dataset recovers planted coefficients
  within 2 %.

- **Article 6 (Kumar et al.)** *(proxy)*: VRH mineral mixing, Jason
  grain-supported and cementing-model hydrate end-members, Gassmann
  fluid substitution, Vp/Vs from K_sat, G, rho_b, and a Vp/Vs
  classifier that discriminates hydrate-bearing sand / sand /
  calcite / shale lithologies.

- **Article 7 (Zhou et al.)** *(proxy)*: 3-D voxel sand-pack as the
  digital-core analogue, VRH solid moduli, Krief porosity softening,
  Plumb-Allen UCS predictor, exponential water-immersion weakening,
  and a Kirsch + Mohr-Coulomb critical-mud-weight check for vertical-
  well stability.

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
