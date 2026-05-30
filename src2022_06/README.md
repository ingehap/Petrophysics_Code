# Petrophysics June 2022 - Vol. 63, No. 3

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 63, No. 3 (June 2022) - the **NMR Special Interest Group (SIG)
Special Issue** edited by Philip Singer.  Eleven peer-reviewed papers
organised into three sub-themes: *Machine Learning and Data Processing*
(articles 1-3), *Log Analysis and Tools* (articles 4-7), and *Core
Analysis* (articles 8-11).

## Quick start

```bash
pip install numpy scipy

# Run all 11 module tests
python test_all.py

# Or run a single article
python article1_nmf_clustering_t1t2.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_nmf_clustering_t1t2.py` | Integrated Reservoir Characterization Using Unsupervised Learning on NMR T1-T2 Logs | Jiang, Bonnie, Correa, Krueger, Kelly, Wasson | 10.30632/PJV63N3-2022a1 |
| `article2_fuzzy_genetic_nmr.py` | Unlocking the Full Potential of NMR Using Machine Learning - A Gas Field With an Oil Problem | Cuddy | 10.30632/PJV63N3-2022a2 |
| `article3_nmr_processing_toolbox.py` | NMR Logging Data Processing | Shao, Balliet | 10.30632/PJV63N3-2022a3 |
| `article4_bssica_dt2_invasion.py` | NMR-Supported Near-Wellbore Data Analysis ... Barite-Enriched Water-Based Mud | Romero Rojas, Tagarieva, Panchal, AlTurki, Qubian | 10.30632/PJV63N3-2022a4 |
| `article5_nppm_pore_size_perm.py` | NMR T1-T2 Logging in Unconventional Reservoirs: Pore-Size, Permeability, RQ | Ijasan, Macquaker, Luycx, Alzobaidi, Oyewole, Rudnicki | 10.30632/PJV63N3-2022a5 |
| `article6_ddtw_mudgas_integration.py` | Formation Evaluation Using NMR, Mud Gas, and Triple-Combo Data | Thern, Kotwicki, Ritzmann, Petersen, Mohnke | 10.30632/PJV63N3-2022a6 |
| `article7_slimhole_lwd_factor.py` | Learnings From a New Slimhole LWD NMR Technology | Hursan, Silva, Van Steene, Muna | 10.30632/PJV63N3-2022a7 |
| `article8_highfield_al_nmr.py` | Accurate Rock Mineral Characterization With NMR | Wang, Sun, Yang, Seltzer, Wigand | 10.30632/PJV63N3-2022a8 |
| `article9_t2_imbibition_wettability.py` | NMR-Based Wettability Index for Unconventional Rocks | Dick, Veselinovic, Bonnie, Kelly | 10.30632/PJV63N3-2022a9 |
| `article10_pcr_nmr_micp_perm.py` | Estimating the Permeability of Rocks by Principal Component Regressions of NMR and MICP Data | Rios, Azeredo, Moss, Pritchard, Domingues | 10.30632/PJV63N3-2022a10 |
| `article11_core_nmr_review.py` | Review of Recent Developments in NMR Core Analysis | Dick, Veselinovic, Green | 10.30632/PJV63N3-2022a11 |
| `test_all.py` | Master test runner | - | - |

> **Note on coverage.** Articles 1-6 are implemented directly against
> the equations in the paper bodies.  Articles 7-11 were only
> represented by the Guest Editor's column and table of contents in the
> source-PDF extract used to build this folder; their modules are
> **methodology proxies** that demonstrate the same algorithmic concept
> described in the editor's narrative (factor analysis, 27Al peak
> identification, time-lapse T2 wettability, PCA + PCR permeability,
> VST / SPRITE / slice-selective). When the full paper bodies become
> available, the proxy modules can be replaced in place.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (Jiang et al.)**: implements Lee-Seung multiplicative-
  update NMF for V ~ W @ H, average-link agglomerative clustering on
  the end-member spectra, and two simple fluid-typing rules (T1/T2
  threshold for HC, T2 cutoff for mobility).  The wettability index
  (Eq. 8) is the normalised T1/T2 ratio of the oil cluster.

- **Article 2 (Cuddy)**: uses a real-valued GA (tournament selection,
  blended crossover, Gaussian mutation) to evolve polynomial mappings
  from log features to predicted T2; triangular-membership fuzzy
  classifier and distance-weighted kNN regressor benchmark.  The
  DTE diffusion-decay formula 1/T2_eff = 1/T2 + (gamma*G*TE)^2 * D/12
  is exposed as a standalone helper.

- **Article 3 (Shao & Balliet)**: implements the CPMG forward kernel
  (Eq. 1), exact NNLS Tikhonov inversion via the augmented system
  [K; lambda I] x = [E; 0] (the standard Tikhonov formulation),
  Timur-Coates (Eq. 52) and SDR (Eq. 56) permeability predictors,
  log-mean T2, and a log-linear ML permeability model (Eq. 62).
  T2 inversion is benchmarked on the BVI-vs-FFV partition (the
  physically meaningful summary) rather than full L2 recovery,
  consistent with the known ill-conditioning of multi-exponential
  fitting.

- **Article 4 (Romero Rojas et al.)**: parallel relaxation rates
  (Eqs. 1-2), porosity undercall (Eq. 3), Timur-Coates and
  permeability-ratio-index KRI (Eqs. 4-5), and a FastICA implementation
  with symmetric orthogonalisation for the Blind Source Separation
  step (Eqs. 6-11).  The synthetic-mixture test recovers three sources
  with correlation > 0.95.

- **Article 5 (Ijasan et al.)**: multiphase relaxation rates (Eqs. 1-3),
  greedy peeling fit of 2-D log-normal Gaussian components on the
  T1-T2 map (NPPM), and Kozeny-Carman permeability with a Herron-
  style mineralogy factor (Eqs. 4-6).

- **Article 6 (Thern et al.)**: NMR polarisation function (Eq. 1),
  variable matrix density, the DDTW closed-form solution that solves
  the linear-in-(phi, phi*Sg) system from the density and apparent-NMR
  porosities, and a C1-C5 mud-gas hydrogen-index formula.  The DDTW
  solution exactly recovers the planted (phi, Sg).

- **Articles 7-11**: methodology proxies (see the note above) that
  implement the concept described by the Guest Editor.  Each is
  clearly flagged in the module docstring with a notice that the
  paper body was not in the available PDF extract.

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
