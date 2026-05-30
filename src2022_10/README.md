# Petrophysics October 2022 - Vol. 63, No. 5

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 63, No. 5 (October 2022) - a regular (non-themed) issue spanning rock
mechanics, capillary pressure modelling, tight-rock permeability methodology,
in-situ CT visualisation of mud-filtrate invasion, and acid-gas cement
degradation.

## Quick start

```bash
pip install numpy scipy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article3_stress_dependent_permeability.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_nanoindentation.py` | A Guide to Nanoindentation | Sondergeld, Rai | 10.30632/PJV63N5-2022a1 |
| `article2_shale_capillary_pressure.py` | Empirical Relation for Capillary Pressure in Shale | Alipour K., Kasha, Sakhaee-Pour, Sadooni, Al-Kuwari | 10.30632/PJV63N5-2022a2 |
| `article3_stress_dependent_permeability.py` | An Efficient Laboratory Method to Measure Stress-Dependent Tight Rock Permeability With the Steady-State Flow Method | Zhang, Liu, Duncan | 10.30632/PJV63N5-2022a3 |
| `article4_mud_filtrate_invasion_ct.py` | Mud-Filtrate Invasion in Laminated and Spatially Heterogeneous Rocks: High-Resolution In-Situ Visualization and Analysis Using Time-Lapse X-Ray Microcomputed Tomography (Micro-CT) | Schroeder, Torres-Verdín | 10.30632/PJV63N5-2022a4 |
| `article5_cement_acid_gas_corrosion.py` | Corrosion Behavior and Mechanism Analysis of Oilwell Cement Under CO2 and H2S Conditions | Zhou, Zeng, Sun, Zhou, Lei, Wan, Luo, Wu, Zhang, Xiao | 10.30632/PJV63N5-2022a5 |
| `test_all.py` | Master test runner | - | - |

> **Note on article 6.** The issue table of contents also lists a sixth
> paper, *"Coring Method for Dolomite Rocks With Well-Developed Joint
> Fissures Based on Permeability Reinforcement"* (Gao, Ma, Kong, Wang,
> Lang, Fan, Zhu, Zhu, p. 652 onward, presumed DOI suffix
> `10.30632/PJV63N5-2022a6`).  The text body of this article was not
> present in the source-PDF extract used to build this folder, so no
> module is included for it.  Drop a faithful implementation in as
> `article6_dolomite_coring_method.py` once the full text is available
> and register it in `test_all.py`.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (Sondergeld & Rai)**: implements the full Oliver-Pharr
  framework: hardness H = P_max / A (Eq. 1); unloading stiffness S = dP/dh
  (Eq. 2); ideal Berkovich tip-area A = 24.5 * h_c^2 (Eq. 3); composite
  compliance to E_s with diamond E_i = 1141 GPa, nu_i = 0.07 (Eq. 4);
  Gupta et al. (2018) shear-modulus estimator G = 95.3 * slope - 0.35 GPa
  (Eq. 5); log-creep fit h(t) - h0 = b * log10(t/t0) (Eqs. 9-10); and the
  mixed-mode fracture toughness K_c = alpha * sqrt(E/H) * P_max / c^(3/2)
  (Eq. 11).  The 100-indent demo reproduces the paper's Woodford-shale
  array statistic (E_s mean ~ 30 +/- 3 GPa).

- **Article 2 (Alipour et al.)**: fits Brooks-Corey (Eq. 3), van Genuchten
  (Eq. 5), and the proposed three-parameter form Pc = pe + alpha1 *
  ((1 - Sw*) / Sw*)^alpha2 (Eq. 6) to a synthetic MICP dataset, scoring
  with R^2 and MSE (Eq. 7).  The proposed model recovers a non-zero
  entry pressure pe that Brooks-Corey cannot.

- **Article 3 (Zhang, Liu & Duncan)**: closed-form three-measurement
  inversion for (k0, alpha, beta) from steady-state mass-flow runs at
  large pressure gradients.  Pair 1 (same pu, pd; two confining stresses)
  gives alpha (Eq. 15); Pair 2 (same sigma_c, two pp_mean values) gives
  alpha*beta after correcting for the integrated (pu^2 - pd^2) ratio
  (Eqs. 16-18); k0 follows from any single run.  Reproduces the paper's
  carbonate-source-rock plug exactly: alpha ~ 4.7e-4 /psi, beta ~ 0.83,
  k0 ~ 100 nD.

- **Article 4 (Schroeder & Torres-Verdin)**: pure-analytical analogue of
  the time-lapse micro-CT analysis - capillary and Bond numbers, Brooks-
  Corey relative permeabilities, Leverett J-function, fractional flow,
  Welge-tangent Buckley-Leverett front saturation, and the Dewan-
  Chenevert sqrt(t) mudcake-controlled invasion-front position.  Default
  parameters reproduce the paper's Leopard-sandstone N_ca ~ 2e-5 / 7e-7
  spurt-vs-late transition.

- **Article 5 (Zhou et al.)**: implements the labelled gas-Darcy
  permeability formula k = (2 Q P0 mu L) / (A (P1^2 - P2^2)) (Eq. 1)
  and a phenomenological exponential-in-time permeability evolution
  k(t) = k_init * exp(B * t) fitted to the paper's three measurements
  (~200x rise from day 7 to day 30).  A cylindrical reaction-front
  geometry x_f(t) = K * sqrt(t) drives the tensile-strength loss
  trajectory toward the paper's reported ~ 9.8 MPa at day 30.

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
