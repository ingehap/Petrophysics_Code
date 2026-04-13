# Petrophysics Vol. 64, No. 4 (August 2023) — Python implementations

One Python module per article from the August 2023 issue of *Petrophysics:
The SPWLA Journal of Formation Evaluation and Reservoir Description*.

| File | Article | DOI |
|---|---|---|
| `article1_nuclear_logging.py` | Fitz, D.E. — *Evolution of Casedhole Nuclear Surveillance Logging Through Time*, pp. 473–501 | 10.30632/PJV64N4-2023a1 |
| `article2_invasion_simulation.py` | Merletti, G. et al. — *Assessment of True Formation Resistivity and Water Saturation in Deeply Invaded Tight-Gas Sandstones…*, pp. 502–517 | 10.30632/PJV64N4-2023a2 |
| `article3_mineralogical_inversion.py` | Jácomo, M.H. et al. — *Mineralogical Modeling and Petrophysical Properties of the Barra Velha Formation, Santos Basin, Brazil*, pp. 518–543 | 10.30632/PJV64N4-2023a3 |
| `article4_obm_imager_inversion.py` | Chen, Y.-H. et al. — *Fracture Imaging and Response Characterization of the High-Definition Oil-Based Mud Borehole Imagers Through Modeling and Inversion*, pp. 544–554 | 10.30632/PJV64N4-2023a4 |
| `article5_iterative_resistivity.py` | Merletti, G. et al. — *New Iterative Resistivity Modeling Workflow Reduces Uncertainty in the Assessment of Water Saturation in Deeply Invaded Reservoirs*, pp. 555–567 | 10.30632/PJV64N4-2023a5 |
| `article6_well_log_qc.py` | Jin, Y. et al. — *Python Dash for Well Data Validation, Visualization, and Processing*, pp. 568–573 | 10.30632/PJV64N4-2023a6 |

## Rules satisfied
* **(a)** Every module is runnable as a standalone script: `python article<i>_*.py` runs that module's `test_all()`.
* **(b)** Every module exposes a `test_all(verbose=True)` function that exercises every public function on synthetic data with assertions.

## Running everything
```bash
python run_all_tests.py        # runs all six test_all() suites
```

## Dependencies
* `numpy` (required, all modules)
* `scipy` (optional — `article3_mineralogical_inversion.py` uses
  `scipy.optimize.nnls` if available; otherwise it falls back to
  unconstrained least-squares + clipping).

## What each module implements

**Article 1 (Fitz)** — Pulsed-Neutron Capture (PNC) volumetric mixing law
(Eq. 4), time-lapse Sw monitoring (Eq. 5), salinity → Σ_w conversion,
Larionov gamma-ray clay-volume estimators.

**Article 2 (Merletti et al. 1)** — Sw_in regression (Eq. 1),
Land/Jerauld trapped-gas (Eq. 2), Brooks-Corey relative permeabilities
(Eqs. 3–4), Brooks-Corey capillary pressure (Eq. 5), Dewan & Chenevert
mudcake permeability/porosity (Eqs. 6–7), Chin mudcake-thickness ODE
(Eq. 8), Archie's law (Eq. 9), and a radial Sw / salinity / Rt(r) profile
generator that mirrors the figures in the paper.

**Article 3 (Jácomo et al.)** — Volumetric photoelectric factor
U = PEF·ρ_b (Eq. 1), Larionov GR clay volumes (Eqs. 2–3), NMR clay
volume (Eq. 4), hybrid clay volume (Eqs. 6–7), the linear
multicomponent inversion ML = A·V (Eq. 8) solved with NNLS under a
unit-sum constraint, and the weighted RMS error metric (Eq. 9).

**Article 4 (Chen et al.)** — Series-circuit two-frequency button-impedance
forward model (mud layer + formation), damped Gauss-Newton inversion for
(R_fmt, ε_fmt at F2, sensor standoff), the mud-angle helper, and the
fracture-equivalent-standoff trend that explains the conductive-fracture
behaviour in resistive formations.

**Article 5 (Merletti et al. 2)** — Sliding-window bed-boundary detector,
P5/P50/P95 OBM-equivalent Sw–φ envelope (Eq. 1), Archie-derived Rt
envelope used as a soft prior, simplified array-laterolog forward model
with depth-of-investigation weights, single-layer Bayesian/MCMC inversion
of (Rt, Rxo) constrained by the Rt envelope, and the outer iterative
loop that refines L_xo via grid search.

**Article 6 (Jin et al.)** — `ValidationConfig` dataclass plus the
4-rule check (missing/redundant/units/values), summary-table builder,
log-difference (Eq. 1), Pearson correlation (Eq. 2), and the
depth-shift cross-correlation that powers the Plotly-Dash repeatability
panel shown in Fig. 5.
