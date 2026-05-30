# Petrophysics August 2021 - Vol. 62, No. 4

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 62, No. 4 (August 2021) - a regular issue opening with an invited
tutorial on thinly bedded formations, followed by seven papers spanning
deep-Q-learning depth matching, NMR fluid substitution, borehole-sonic
dispersion analysis, a machine-learning synthetic-sonic contest, an oil-based-
mud resistivity imager, an acoustic volcanic-rock saturation model, and the
capillary-pressure / resistivity-index relationship in tight sandstones.

## Quick start

```bash
pip install numpy

# Run all 8 module tests
python test_all.py

# Or run a single article
python article4_sonic_dispersion_dpsm.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_thinly_bedded_petrophysics.py` | Tutorial: Petrophysics of Thinly Bedded Formations | Aldred | 10.30632/PJV62N4-2021t1 |
| `article2_depth_matching_deep_q.py` | Multiple Well-Log Depth Matching Using Deep Q-Learning | Bittar, Wang, Wu, Chen | 10.30632/PJV62N4-2021a1 |
| `article3_nmr_fluid_substitution.py` | NMR Fluid Substitution - A New Method of Reconstructing T2 Distributions Under Primary Drainage and Imbibition Conditions | Li, Kesserwan, Jin, Ma | 10.30632/PJV62N4-2021a2 |
| `article4_sonic_dispersion_dpsm.py` | Borehole Sonic Data Dispersion Analysis With a Modified Differential-Phase Semblance Method | Wang, Coates, Zhao | 10.30632/PJV62N4-2021a3 |
| `article5_synthetic_sonic_ml_contest.py` | Synthetic Sonic Log Generation With Machine Learning: A Contest Summary From Five Methods | Yu, Xu, Misra, Li, Ashby, et al. | 10.30632/PJV62N4-2021a4 |
| `article6_obm_resistivity_imager.py` | Quantitative Demonstration of a High-Fidelity Oil-Based Mud Resistivity Imager Using a Controlled Experiment | Guner, Fouda, Ewe, Torres, Barrett | 10.30632/PJV62N4-2021a5 |
| `article7_volcanic_saturation_gassmann.py` | Experimental Study on the Saturation Model of Volcanic Rock Based on Fluid Distribution | Pan, Zhou, Guo, Si, Lin | 10.30632/PJV62N4-2021a6 |
| `article8_capillary_resistivity_index.py` | Experimental Study on the Relationship Between Capillary Pressure and Resistivity Index in Tight Sandstone Rocks | Xiao, Yang, Li, Yang, Bernabe, Zhao, Li, Ren | 10.30632/PJV62N4-2021a7 |
| `test_all.py` | Master test runner | - | - |

> **Note on coverage.** All eight modules are implemented directly against the
> methods described in the paper bodies.  Throughout this issue the *typeset
> equations were stored as images* and did not survive text extraction (only
> the equation numbers remained), so the numbered formulas here are faithful
> **standard-form reconstructions** of the methods the prose describes - not
> byte-perfect transcriptions.  Several papers (2, 4, 5) use deep neural
> networks or proprietary inversions; those are represented by a compact,
> numpy-only implementation of the same underlying method (tabular Q-learning,
> phase-semblance beamforming, linear-regression baseline).  Article 1's
> parallel/series equations were reconstructed and verified against the
> paper's own worked numbers.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (Aldred, tutorial)**: parallel (horizontal) and series
  (vertical) resistor models for laminated sand-shale (Eqs. 2-3), the
  anisotropy coefficient lambda = sqrt(Rv/Rh), the Moran-Gianzero apparent-
  resistivity-vs-dip relation (Eq. 1), sand-resistivity inversion from Rh vs
  Rv (showing the series route is far more robust to shale-volume error), and
  the Thomeer capillary-pressure curve (Eq. 4).  Reproduces the paper's worked
  numbers (Rh -> 1.82, Rv = 5.5, Rss = 10 ohm-m).

- **Article 2 (Bittar et al.)**: the depth-matching MDP and the Q-learning
  Bellman update with epsilon-greedy exploration.  The paper's CNN Rainbow-DQN
  is replaced by a compact tabular Q-learner that solves the same shift-action
  MDP and reliably navigates to the match point and triggers the stop action.

- **Article 3 (Li et al.)**: NMR surface-relaxation physics (1/T2 = 1/T2bulk +
  rho*S/V), T2 -> pore-radius mapping, BVI/BVM split at the 33-ms cutoff, and
  the porosity-conserving fluid-substitution that re-amplifies the movable-
  water peak by 1/Sw_eff to reconstruct the Sw=1 distribution.

- **Article 4 (Wang et al.)**: the differential-phase frequency-slowness
  semblance estimator (phase back-propagation across the receiver array) and
  group delay from a phase spectrum, validated by recovering a known slowness
  from a synthetic nondispersive array.

- **Article 5 (Yu et al.)**: the contest scoring (pooled DTC+DTS RMSE, per-log
  RMSE, R^2), z-score / min-max normalization, log-resistivity transform, and a
  numpy linear-regression baseline.  The five contest models (zone-ANN, BiLSTM,
  Random Forest, stacked-ensemble + LightGBM, Elman RNN) are summarized rather
  than retrained.

- **Article 6 (Guner et al.)**: the capacitively-coupled imager physics -
  parallel-RC element values (Eq. 1), complex button impedance (Eq. 2),
  apparent impedivity with the low-resistivity limit Re(xi) ~ rho (Eqs. 3-4),
  the capacitive oil-mud term, and the DC-conductivity / dielectric-loss
  decoupling (Eq. 6).  Reproduces dielectric rollover and ~ -90 deg mud phase.

- **Article 7 (Pan et al.)** *(acoustic, not electrical)*: the Gassmann
  equation (Eq. 1), Wood-Lindsay (Reuss) / Domenico (Voigt) / Brie fluid moduli
  (Eqs. 2-4), the White patchy modulus (Eq. 5), and the Gassmann-Brie-Patchy
  blend (Eq. 6), with P-wave velocity from (K, mu, rho).  Confirms the patchy
  curve is the upper velocity bound and the uniform curve the lower bound, with
  the two converging at Sw = 0 and Sw = 1.

- **Article 8 (Xiao et al.)**: the Archie resistivity index and formation
  factor (Eqs. 1-2), the Waxman-Smits clay-corrected index (Eq. 5), the
  Li & Williams power-law (Eq. 9) and Szabo linear (Eq. 6) Pc-I relations, the
  Toledo fractal capillary-pressure model (Eq. 18), the Washburn pore-throat
  radius, and the reported beta(k) and b(k) permeability regressions.

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
