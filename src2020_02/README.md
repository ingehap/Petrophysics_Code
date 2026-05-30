# Petrophysics February 2020 - Vol. 61, No. 1

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 61, No. 1 (February 2020) - a regular issue opening with an invited
tutorial on borehole-nuclear Monte Carlo modeling, followed by five articles
spanning a numerical comparison of Russian and Western resistivity logs, the
response of an array-induction tool in anisotropic formations, a physics-driven
deep-learning network for nonlinear inverse problems, Bayesian geosteering with
Sequential Monte Carlo, and a "boomerang" workflow for porosity and net/gross in
shaly gas reservoirs.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article1_montecarlo_nuclear_fsf_tutorial.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_montecarlo_nuclear_fsf_tutorial.py` | *Tutorial:* Simulation of Borehole Nuclear Measurements - A Practical Tutorial Guide for Implementation of Monte Carlo Methods and Approximations Based on Flux Sensitivity Functions | Luycx, Bennis, Torres-Verdín, Preeg | 10.30632/PJV61N1-2020T1 |
| `article2_russian_western_resistivity.py` | Comparison of the Russian and Western Resistivity Logs in Typical Western Siberian Reservoir Environments: A Numerical Study | Epov, Sukhorukova, Nechaev, Petrov, Rabinovich, Weston, Tyurin, Wang, Abubakar, Claverie | 10.30632/PJV61N1-2020a1 |
| `article3_hdil_array_induction_anisotropic.py` | Response Characteristics of an Array Induction Tool (HDIL) in Heterogeneous Anisotropic Formations | Liu, Zhang, Zhang, Xu, Kang, Xiao | 10.30632/PJV61N1-2020a2 |
| `article4_physics_deeplearning_inversion.py` | A Physics-Driven Deep-Learning Network for Solving Nonlinear Inverse Problems | Jin, Shen, Wu, Chen, Huang | 10.30632/PJV61N1-2020a3 |
| `article5_bayesian_geosteering_smc.py` | Bayesian Geosteering Using Sequential Monte Carlo Methods | Akkam Veettil, Clark | 10.30632/PJV61N1-2020a4 |
| `article6_boomerang_porosity_netgross.py` | Untangle Shale and Gas Effects to Estimate Porosity and Net/Gross Ratio Using a Boomerang Workflow - A Case Study in Shoreface Reservoirs in Brunei | Xu, Sharif | 10.30632/PJV61N1-2020a5 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2020_02.pdf`)
> has a text layer, so the article titles, authors, page ranges, and DOIs were
> read directly from the contents page and paper bodies. The machine text
> extraction captured the full bodies of the **Tutorial** and **Article 1**, was
> **partial for Article 2** (through page 78 of 72-85), and contained **Articles
> 3-5 only as table-of-contents entries**. Articles 4-6 (and the focusing detail
> of Article 3) are therefore implemented as **methodology proxies** of the
> standard, well-established methods their titles describe; and, as with the
> other issues, the typeset formula glyphs were dropped in extraction, so the
> numbered formulas are faithful standard-form reconstructions built from the
> preserved variable definitions.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions. A few practical notes:

- **Article 1 (Luycx et al.)** *(tutorial)*: the F4 track-length flux estimator
  φ = Σ(W·T)/V (Eqs. 3, 18-19) with a Monte-Carlo demonstration, the detector
  reaction rate N = c∫φ(E)σ(E)dE (Eq. 4), the importance/adjoint function
  Imp = Score/Weight (Eq. 17), the flux sensitivity function (background flux ×
  importance, normalized to unity, Eqs. 20-22), and the first-order perturbed
  response N = N_b + ∫FSF·Δσ (Eqs. 23-24).

- **Article 2 (Epov et al.)**: the galvanic apparent resistivity ρ_a = k·U/I,
  the laminated parallel/series (Rh/Rv) resistivities and anisotropy coefficient
  λ = √(ρv/ρh), and the EM skin depth δ = 503·√(ρ/f) controlling the
  induction depth of investigation versus frequency (VEMKZ 0.875-14 MHz vs IK
  50-100 kHz).

- **Article 3 (Liu et al.)** *(extract ended at p.78):* the anisotropy
  coefficient λ = √(Rh/Rv), the constrained least-squares focusing weights
  (Σw = 1, solved via the KKT system), the focused apparent resistivity
  ρ_a = 1/Re(Σw·σ_a), and the anisotropic apparent resistivity vs relative dip
  (negligible at 0°, significant at ≥60°).

- **Article 4 (Jin et al.)** *(methodology proxy):* the regularized nonlinear-
  inversion framework - forward operator d = G·m, the Tikhonov objective, the
  closed-form ridge inversion and an equivalent gradient-descent ("training")
  solver, showing regularization stabilizes the ill-posed noisy problem.

- **Article 5 (Akkam Veettil & Clark)** *(methodology proxy):* a Sequential
  Monte Carlo / particle filter for the distance to a geological boundary -
  particle propagation, Gaussian-likelihood Bayesian reweighting, effective
  sample size, systematic resampling, and the posterior-mean estimate tracking a
  moving boundary within the measurement noise.

- **Article 6 (Xu & Sharif)** *(methodology proxy):* density porosity, the
  gas-corrected total porosity √((φN²+φD²)/2), the shale-corrected effective
  porosity, shale volume from gamma ray, and the net/gross ratio from porosity
  and shale-volume cutoffs - the shale/gas "boomerang" crossplot untangling.

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
