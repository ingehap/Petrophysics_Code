# Merging Petrophysics_Code into one common Python library

**A detailed migration plan for extracting a shared `petrolib` package from the 75 issue
directories, so every article script imports common code instead of re-implementing it.**

This plan is based on a full inventory of the repository (all 75 `srcYYYY_MM` directories,
580 article modules, ~106,500 lines of Python), a cross-cutting duplication analysis of 15
petrophysics domains that mapped **190 duplicated function families** (~1,550 separate
implementations of shared concepts), and an evaluation of three competing migration
architectures scored independently for migration safety, end-state quality, and
single-maintainer execution realism.

> **Status — July 2026: executed to completion.** All eleven domain trains are merged:
> `flow_transport`, `nmr`, `porosity_lithology`, `acoustic_geomech`, `geochem_fluids`,
> `em_dielectric`, `nuclear`, `inversion_numerics`, the depth/imaging group
> (`depth_matching`, `depth_correction`, `borehole_image`, `wellbore_geometry`),
> `integrity_drilling`, and `data_qc_io` — plus the earlier foundation modules
> (`constants`, `units`, `ml_stats`, `saturation_resistivity`, `capillary_pressure`,
> `relperm_wettability`, `testing`). Every adoption PR was verified byte-identical on
> stdout against the pre-change article, alongside the four regression gates below.
> Scope deltas vs. the section-7 sketches are documented in the PR bodies; the largest:
> the planned `data_qc_io` `units.py`/`metrics.py`/`align.py`/`pay.py` files were
> dropped because `petrolib.units`, `petrolib.ml_stats`, `petrolib.depth_matching`,
> and `petrolib.porosity_lithology` already covered them (`detect_bed_boundaries`
> lives in `data_qc_io.filt`). Corpus forms that are not bit-reproducible from the
> canonical API (e.g. `+1e-10`/`+1e-12` guard variants, scipy-kernel smoothers,
> legacy `RandomState` generators) deliberately remain local to their articles.

---

## 1. Summary of the recommendation

Create a **`petrolib/` package at the repository root** (deliberately *not* `src/` layout)
containing one module per petrophysics domain, extracted from the best existing article
implementations. **No issue directory ever moves, no article file is renamed, no
`test_all()` is ever rewritten.** Inside article files, each duplicated function keeps its
name and exact historical signature but its *body* becomes a two-line delegation to the
canonical `petrolib` function — the article's local dialect mapped once, in review, onto a
keyword-only canonical API.

The migration is a **strangler-fig sequence of small PR trains** (one train per domain,
ordered by duplication weight, piloted on pure-math code first), each gated by four
regression checks that make it impossible for a swap to silently change numerics:

1. every issue directory's own test runner stays green,
2. a golden-stdout diff over all 75 directories,
3. a shadow equivalence check (`old body == new call` to `rtol=1e-12`) before any local
   body is deleted,
4. `petrolib`'s own unit + property tests, seeded from the articles' assert constants.

The invariant at every merged commit: **`python articleN_x.py` and `python test_all.py`
keep working from inside any issue directory, on a bare clone, with no install step.**
The plan can pause indefinitely after any PR with zero debt — un-migrated articles simply
still carry local copies.

Rough size: **~75–90 PRs**, each under ~400 diff lines, over 6 phases. Every phase leaves
the repository fully working.

---

## 2. Where the repository stands today

| Fact | Value |
|---|---|
| Issue directories | 75 (`src2014_02` … `src2026_06`, one per bimonthly SPWLA *Petrophysics* issue) |
| Article modules | 580 (`.py` files excluding test runners and `__init__.py`) |
| Total Python | 660 files, ~106,500 lines |
| Self-test functions | 501 article modules define a module-level `test_all()` (513 files counting runners); the seven package-era dirs test centrally in their runner |
| Packaging / CI | none — no `pyproject.toml`, no `setup.py`, no CI workflows |
| Dependencies | numpy everywhere; scipy in 21 dirs; scikit-learn in 7; torch in 3 (`src2023_12`, `src2024_04`, `src2025_12`); xgboost + scikit-image in 1 (`src2023_04`) |

Three generations of conventions coexist:

- **70 flat directories** (2014–2024): scripts named `articleN_<topic>.py`, no
  `__init__.py`, a `test_all.py` runner that imports modules by bare name via
  `importlib` — which only works with the directory itself on `sys.path` (i.e. run from
  inside the directory).
- **5 package directories** (`src2024_10`, `src2025_12`, `src2026_02`, `src2026_04`,
  `src2026_06`): real packages with `__init__.py` and short snake_case module names.
  three of them import themselves under a public package name that does not exist in a
  bare checkout — `src2025_12` as `petrophysics_v66n6`, `src2026_02` as
  `petrophysics_2026` (whose test file also carried a dead
  `sys.path.insert(0, "/home/claude")`), and `src2026_04` as
  `petrophysics_spwla_2026` — so their test suites only ran in the original author's
  environment until the Phase 0 alias repairs.
- **6 nonstandard test runners**: `run_all_tests.py` (`src2023_08`, `src2023_10`,
  `src2024_02`, `src2025_06`), `run_all.py` (`src2023_12`), `test_all_modules.py`
  (`src2024_12`). Everything else uses `test_all.py`.

Every article module follows the same internal shape (citation docstring → equation
functions → `test_all()` → `if __name__ == "__main__":`), which is exactly what makes a
library extraction tractable: the public surface is plain functions with numpy inputs.

## 3. The duplication evidence

Because each article module is deliberately self-contained, generic physics is
re-implemented over and over. The cross-repo analysis found **190 families** of duplicated
functions. The largest:

| Concept | Files with an implementation | Domain |
|---|---:|---|
| Straight-line least-squares wrappers (`np.polyfit` deg-1 in transformed space) | ~45 | inversion_numerics |
| Archie water saturation (inverse + forward forms) | ~39 | saturation_resistivity |
| Regression/agreement metrics (RMSE, R², Pearson r, MAE) | ~30 | data_qc_io / ml_stats |
| Brownstein–Tarr NMR surface relaxation (forward + inverse) | ~27 | nmr |
| Corey / Brooks-Corey relative permeability curves | ~25 | flow_transport / relperm |
| Young–Laplace Pc ↔ Washburn pore-throat radius | ~24 | capillary_pressure |
| Synthetic well-log suite generators | ~20 | data_qc_io |
| Pressure/depth unit conversions (psi/bar/MPa/Pa, ft/m) | ~19 | data_qc_io |
| Timur–Coates / SDR / Timur NMR permeability | ~19 | nmr / flow_transport |
| Density porosity φ=(ρma−ρb)/(ρma−ρfl) | ~18 | porosity_lithology |
| Archie formation factor F=a/φ^m | ~18 | saturation_resistivity |
| Shale volume from gamma ray (linear/Larionov/Clavier/Steiber) | ~17 | porosity_lithology |
| Elastic moduli ↔ velocity conversions | ~14 | acoustic_geomech |
| CPMG/T2 forward models and inversion kernels | ~14 | nmr |
| Feature scaling (z-score / min-max) | ~14 | ml_stats |
| Capillary number / Bond number | ~13 | relperm / flow_transport |
| Brooks-Corey Pc(Sw) / Sw(Pc) curves | ~13 | capillary_pressure |
| Waxman-Smits conductivity + Sw solver | ~12 | saturation_resistivity |
| Tikhonov/ridge regularized linear inversion | ~12 | inversion_numerics |
| Leverett J-function | ~11 | capillary_pressure |

Duplication is not cosmetic: module-level constants are redefined per file (`EPS0`,
`GAMMA_H` — with a real 4th-digit disagreement between `2.675e8` and `2π·42.58e6` rad/s/T),
NumPy 2.0 compat shims (`np.trapz` vs `np.trapezoid`) are copy-pasted, and the *same
function is sometimes duplicated within a single directory* (`crim_water_saturation` twice
in `src2014_12`; `capillary_number` with **three different argument orders**, two of them
inside `src2014_08`).

Crucially, the analysis also found that many look-alikes are **not** duplicates — see the
hazard catalog in §9. That catalog is the most valuable input to the migration: it lists
where a naive "merge by name" would silently change published numbers.

## 4. Goals, constraints, non-goals

**Goals**

1. One importable library (`petrolib`) that owns every genuinely shared equation, constant,
   and utility, with a coherent, documented, numpy-vectorized API.
2. Every article script imports shared code instead of re-implementing it.
3. New issue directories (`src2026_08` onward) import `petrolib` from day one.
4. Zero silent numeric change to any reproduced article result.

**Constraints (these drove the architecture choice)**

- *Educational repo*: article files are the learning material. They must remain readable,
  individually runnable, and true to the papers' notation and units.
- *Bare-clone runnability*: today `git clone` + `cd src2019_06` + `python test_all.py`
  works with nothing but numpy installed. That property must survive.
- *Single maintainer*: every PR must be honestly reviewable by one person; the migration
  must be able to stall at any point and leave a healthy repo.
- *The 501 module-level `test_all()` functions (plus the runner-level suites of the seven
  package-era dirs) are the regression oracle*. They are never rewritten,
  loosened, or deleted — they gate their own migration.

**Non-goals (for this migration)**

- Publishing to PyPI (the packaging supports it later; dist name `spwla-petrolib` since
  `petrolib` may be taken — import name stays `petrolib`).
- Rewriting articles into thin demos, deleting per-dir runners, or restructuring the 281 KB
  README. A "demo-ification" end state is kept as an optional horizon (§12), not a phase.
- Reformatting or linting the 660 legacy files (pure churn; pollutes blame).

## 5. Target architecture

### 5.1 Repository layout after migration

```
Petrophysics_Code/
├── pyproject.toml              # name=petrolib; packages=["petrolib", ...]; alias petrophysics_2026 -> src2026_02
├── README.md                   # existing index stays; gains a short "common library" section
├── CONVENTIONS.md              # API rules: units, keyword-only args, clip=, form=, provenance
├── petrolib/                   # THE library — flat at repo root, numpy-only at import time
│   ├── __init__.py             # __version__, lazy (PEP 562) submodule loading
│   ├── constants.py            # GAMMA_H, EPS0, MU0, PA_PER_PSI, M_PER_FT, ... (single source of truth)
│   ├── units.py                # convert(x, "psi", "MPa"); temperature, depth, slowness adapters
│   ├── _compat.py              # numpy 1.x/2.x shims (trapezoid), deprecation helper
│   ├── testing.py              # assert_matches_original(), golden-capture utilities
│   ├── saturation_resistivity.py   # archie_sw, formation_factor, waxman_smits, dual_water, ...
│   ├── porosity_lithology.py       # density_porosity, vshale_from_gr(method=), thomas_stieber, TOC
│   ├── capillary_pressure.py       # young_laplace, washburn, leverett_j, brooks_corey_pc, thomeer(log_base=)
│   ├── relperm_wettability.py      # corey_kr, normalized_saturation(snr=), LET, Land, Amott
│   ├── flow_transport.py           # darcy, klinkenberg_apparent/_corrected, kozeny_carman, RQI/FZI
│   ├── nmr.py                      # brownstein_tarr, timur_coates(form=), sdr, t2_logmean, CPMG kernels
│   ├── acoustic_geomech.py         # moduli<->velocity (strict SI), gassmann, VRH/Brie, Thomsen, stress
│   ├── em_dielectric.py            # eps* (-j convention), CRIM, cole_cole, skin_depth, anisotropy
│   ├── nuclear.py                  # sigma mixing + sw_from_sigma(clip=), beer_lambert, spectral GR (coeffs=)
│   ├── geochem_fluids/             # brine.py (arps_correct(unit=)), asphaltene.py, mudgas.py,
│   │                               #   adsorption.py, pvt.py — the one domain big enough to warrant a subpackage
│   ├── inversion_numerics/         # linear.py (penalty applied exactly once), costs.py, nonlinear.py, sampling.py
│   ├── ml_stats.py                 # rmse/r2_score/pearson_r (R vs R² split), scaling, kmeans, tiny NNs
│   ├── depth_imaging.py            # xcorr_shift(edge="trim"), dtw, sinusoid dip picks, otsu, min-curvature
│   ├── integrity_drilling.py       # bond_index(method=, input_kind=), impedance classify, casing tools
│   └── data_qc_io.py               # outliers, despiking, sentinel/NaN handling, synthetic log/image/T2 generators
├── src2014_02/ … src2026_06/   # ALL 75 dirs stay exactly where they are; filenames unchanged;
│   └── test_all.py             #   runners untouched; duplicated bodies become 2-line delegations
├── tests/                      # pytest suite for petrolib
│   ├── petrolib/               # per-domain unit tests + hazard-trap tests + property tests
│   └── test_articles.py        # subprocess harness: runs each dir's runner with cwd=dir, 75 params
└── tools/
    ├── run_all_issues.py       # drives all 75 runners (knows the 6 nonstandard runner names)
    ├── golden_capture.py       # normalized stdout snapshots per directory
    ├── golden_diff.py
    └── shadow_check.py         # old-body vs petrolib-call equivalence to rtol=1e-12
```

Why **flat root layout, not `src/`**: with `petrolib/` at the root, the package is
importable whenever the repo root is on `sys.path` — which a 5-line bootstrap header (§6)
guarantees for a bare clone with no install. A `src/` layout would make every migrated
script die with `ModuleNotFoundError` until `pip install -e .`, a hard regression for an
educational repo. `pip install -e .` remains available as optional sugar.

Module granularity: **one module per domain** (~15 modules), splitting into a subpackage
only where the domain is genuinely large and heterogeneous (`geochem_fluids`,
`inversion_numerics`). Resist premature deep nesting — `petrolib.nmr.timur_coates` is
easier to discover than `petrolib.nmr.permeability.transforms.timur_coates`.

### 5.2 API conventions (to be codified in CONVENTIONS.md)

These rules come straight from the failure modes the duplication analysis found:

1. **Keyword-only physics parameters.** Data arrays may be positional; everything else
   after `*`. The repo contains at least four positional orders for `archie_sw`, two for
   `sigma_water_saturation` *within one directory*, three for `capillary_number`, and
   swapped `(rho_b, rho_ma)` vs `(rho_ma, rho_b)` — all silent because every argument is a
   float. Keyword-only kills the entire trap class.
2. **Distinct math never merges under one name.** Where the repo has two algebras under
   one name, the canonical API forces an explicit switch: `timur_coates(..., form=)`,
   `klinkenberg_apparent` vs `klinkenberg_corrected`, `bond_index(method=, input_kind=)`,
   `brooks_corey_sw(..., exponent_convention=)` or reciprocal conversion at the call site,
   `thomeer(..., log_base=)`, `anisotropy_coefficient` (√(Rv/Rh)) vs `anisotropy_ratio`.
3. **No silent clipping.** Library defaults to unclipped physics with an explicit
   `clip=` parameter; facades pass each article's historical clip. (Clipping biases SCAL
   fits and error propagation; the repo mixes none / [0,1] / [0,0.6] / [1e-6,1].)
4. **SI-first with explicit unit adapters.** The moduli/velocity code alone exists in
   three unit conventions; impedance in three scalings; stress in psi/ft, MPa, and Pa.
   Core functions are unit-neutral or strict SI; unit-suffixed wrappers and `unit=` kwargs
   (e.g. `arps_correct(..., unit="degF")`) live at the edges. Constants like 0.433 psi/ft
   move to `constants.py`/`units.py`.
5. **No baked-in field-study defaults.** `Rw=0.04`, Bakken Schmoker coefficients,
   `k_mineral=5.0 GPa` (opal!), lithology-specific cutoff defaults — locale calibrations
   are required arguments or documented presets, never invisible defaults.
6. **Vectorized, guarded, typed.** numpy-broadcastable; zero-division guards standardized
   (return NaN or raise — never a hidden `+1e-12` bias); type hints; numpydoc docstrings.
7. **Provenance is mandatory.** Every function's docstring carries a `Sources:` line
   naming the article files it consolidates (e.g. `sources: src2016_08/article3,
   src2019_10/article1`). This preserves the repo's citation culture and doubles as the
   reverse index when a question arises about a variant.
8. **numpy-only at import time.** `import petrolib` must succeed with numpy alone; scipy
   users get lazy in-function imports with a clear error; sklearn/torch/xgboost/skimage
   stay in article land (they are article-specific pipelines, not shared physics — see
   §5.3).

### 5.3 What goes into the library — and what stays in articles

**Goes in:** the 190 duplicated families; constants; unit conversions; synthetic-data
generators (well logs, T2 spectra, images) used by many demos; the numpy-only ML/stats
helpers (metrics, scaling, k-means, tiny NN trainers) that appear in dozens of files.

**Stays in the article file:** paper-specific constants and data tables (reported
saturation exponents, field results, Yen–Mullins catalogs where used once), narrative
"data-anchor" modules (some review articles are dicts of reported values, not physics),
one-off workflows, sklearn/torch model pipelines, and any variant whose physics genuinely
appears once. When in doubt, leave it local — a function can always be promoted later;
demoting one back out of a published API is worse.

The rule of thumb from the inventory: **~450 of the 580 article modules contain at least
one family hit** and will gain facade delegations; the rest are untouched.

## 6. How every existing script keeps working

The compatibility invariant, enforced by CI at every commit:

> From a bare clone with only numpy (plus that directory's existing extras), both
> `python articleN_x.py` and `python test_all.py` succeed from inside every issue
> directory — no install step, at every point of the migration.

Mechanics:

1. **Directories never move; filenames never change.** The cwd-relative `importlib`
   pattern in the 70 flat runners keeps working untouched.
2. **Bootstrap header** — the only edit a converted script needs at its top:

   ```python
   try:
       import petrolib
   except ImportError:                      # bare clone, not installed
       import sys, pathlib
       sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
       import petrolib
   ```

   `parents[1]` of `src20YY_MM/article.py` is the repo root, so this works from any cwd.
   If a directory is vendored out of the repo, it degrades to a *loud* `ImportError`,
   never a wrong number.
3. **Facades preserve the public surface.** Each article keeps e.g.
   `def corey_krw(sw, swir, sor, krw_max, nw):` with its historical positional signature,
   clip behavior, and units — the body becomes
   `return petrolib.relperm_wettability.corey_krw(sw, swir=swir, sor=sor, krw_end=krw_max, nw=nw)`.
   `from article3_x import corey_krw` and copy-paste readers see no change; `test_all()`
   exercises `petrolib` transitively through the article-era signature. The facades are
   **permanent** — they are the article's local dialect and its documentation.
4. **`pip install -e .` is optional.** `pyproject.toml` packages `petrolib` and maps
   `package-dir {"petrophysics_2026": "src2026_02"}` so the existing
   `from petrophysics_2026.udar_methods import ...` works when installed; the bootstrap
   covers the uninstalled case.
5. **Nothing else changes:** no `__init__.py` added to flat dirs, no console scripts, no
   namespace packages, no README path churn.

## 7. Migration phases

Ordering principles (from the judged proposals): safety net before anything moves; a
pure-math pilot before physics; extraction interleaved with adoption so each API is
validated against real call sites immediately; batching by **duplication family** (never
by publication year) so a reviewer holds one physics context per PR.

### Phase 0 — Freeze behavior (2–3 PRs, no script edits)

- `tools/run_all_issues.py`: subprocess-runs every directory's runner with `cwd=dir`;
  a `RUNNERS` map covers the six nonstandard names (`run_all_tests.py` ×4, `run_all.py`,
  `test_all_modules.py`). Asserts a per-dir pass verdict; asserts **exactly 501 module-level `test_all()`
  functions are discovered** repo-wide so coverage loss is loud.
- `tools/golden_capture.py` / `golden_diff.py`: record normalized stdout (timings
  stripped) per directory into `tools/golden/`. This freezes even *printed-but-unasserted*
  values, which the `test_all` tolerances (typically 1e-6..1e-3) would let drift.
- `tests/test_articles.py`: the same harness as parametrized pytest.
- The **only** pre-existing repairs allowed now are the ones needed to get a green
  baseline, each behavior-preserving and clearly separated. Phase 0 execution found two
  kinds: (a) the three self-renaming package dirs (`src2025_12`, `src2026_02`,
  `src2026_04`) import themselves under public names that don't resolve in a bare
  checkout — repaired with a `__file__`-relative `sys.modules` alias in each `test_all.py`
  (replacing `src2026_02`'s dead `sys.path.insert(0, "/home/claude")`); (b) three modules
  called `np.trapz` directly, which NumPy 2.x removed — repaired with the
  `_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))` shim already used
  elsewhere in the repo.

### Phase 1 — Packaging skeleton (1–2 PRs, nothing imports petrolib yet)

- `pyproject.toml` (§10), `petrolib/{__init__,constants,units,_compat,testing}.py`,
  `CONVENTIONS.md`.
- CI (GitHub Actions): lint + petrolib unit tests on every PR; the full 75-dir article
  suite + golden diff in **both** modes (bare checkout and `pip install -e .`), running
  nightly, on a `run-articles` label, and automatically when a diff touches `src20*`.
- ruff scoped to `petrolib/`, `tests/`, `tools/` only (`extend-exclude = ["src20*"]`).

### Phase 2 — Pilot train: pure math first (1 train ≈ 4–6 PRs)

`ml_stats` + the metrics/scaling half of `data_qc_io` (~56 files): regression metrics,
z-score/min-max scaling, k-means, linear-fit wrappers. Chosen because the math is
trivially checkable, the R-vs-R² and intercept-position traps are representative, and it
touches many files — it proves the whole workflow (API PR → era-chunked adoption PRs →
shadow check → golden diff) before any physics is at stake. **Do not proceed to Phase 3
until this train is merged and the process felt sustainable.**

Train pattern (used by every domain from here on):

- **PR-A**: add `petrolib/<domain>.py` + unit tests + property tests. Touches zero
  article files — the repo cannot break.
- **PR-B1..Bn**: convert article call sites in era chunks of ~10–15 files, <400 diff
  lines each. Per file: add the bootstrap header, replace duplicated bodies with
  delegations, leave article-unique code alone. Gates: shadow check green, the dir's own
  runner green, golden diff clean.
- **Hazard call sites are excluded from batch PRs** and get their own one-file PRs with a
  before/after numeric note (§9).

### Phase 3 — Domain trains, ordered by duplication weight and coupling (~13 trains, ~50–65 PRs)

1. `saturation_resistivity` — the 39-file Archie family, formation factor, RI, Waxman-Smits, dual-water.
2. `capillary_pressure` — Young–Laplace/Washburn (24), Brooks-Corey Pc, Leverett J, Thomeer.
3. `relperm_wettability` **together with** the Corey/Se/fractional-flow families of
   `flow_transport` — they share normalized saturation; resolve the Se-denominator
   convention (`snr=` explicit) exactly once.
4. rest of `flow_transport` — Darcy, Klinkenberg, Kozeny-Carman, RQI/FZI, Winland/Swanson,
   diffusion; coordinate the NMR-permeability transforms (Timur-Coates/SDR) with `nmr`.
5. `porosity_lithology` — density/N-D porosity, Vsh methods, Thomas-Stieber, TOC/kerogen.
6. `nmr` — Brownstein–Tarr, CPMG kernels, T2 stats and cutoffs, T1/T2 typing, surface-relaxivity pore size.
7. `inversion_numerics` — polyfit wrappers, Tikhonov (penalty applied exactly once —
   callers' λ values must be explicitly converted), NNLS/simplex machinery, misfits, MCMC helpers.
8. `acoustic_geomech` — moduli/velocity (strict SI), Gassmann (keyword-only!), VRH/Brie,
   Thomsen, impedance, stress/pore-pressure.
9. `geochem_fluids` — brine R(T,salinity) with `unit=`, FHZ/Yen–Mullins, mud-gas ratios
   (percent-vs-fraction!), isotherms, PVT.
10. `em_dielectric` — eps*/CRIM (−j convention), Cole-Cole vs HN kept separate, skin depth,
    EM anisotropy.
11. `nuclear` — Σ mixing + `sw_from_sigma(clip=)`, Beer–Lambert, spectral GR with `coeffs=`
    tuple, electron density.
12. `integrity_drilling` — bond index (`method=`, `input_kind=`), impedance-based annulus
    classification (coordinate impedance with `acoustic_geomech`), casing tools, leak rates.
13. `depth_imaging` + rest of `data_qc_io` — xcorr shift (edge handling fixed to trim, sign
    convention documented), DTW, dip picking, Otsu, smoothing/outliers/sentinels, synthetic
    generators.

### Phase 4 — Hazard audits (~8–10 one-file or few-file PRs)

One PR per trap class from §9, each with a call-site table and before/after numbers.
This is also where the **known live bugs** are fixed as *documented behavior changes*
(golden files regenerated in the same PR, CHANGELOG entry):
`src2023_08`'s Arps correction applying the °F constant to °C, `src2025_04`'s Eaton
psi/ft-gradient × meters mismatch, and any others the shadow checks surface.

### Phase 5 — The five package dirs + steady state (2–3 PRs)

- Adopt `petrolib` inside `src2024_10`–`src2026_06` (they can import it directly — they
  are packages; keep the bootstrap for bare-clone parity).
- New-issue template (`docs/new_issue_template/`) so `src2026_08+` is package-style and
  imports `petrolib` from day one — the library stops the duplication growth, which is the
  whole point.
- Tag `petrolib` 1.0.0 when Phase 4 closes; before that, 0.x with a minor bump per train.

### Phase 6 (optional horizon) — see §12.

## 8. Regression safety: the four gates

| Gate | What it catches | When it runs |
|---|---|---|
| Per-dir runners (all 75, subprocess, `cwd=dir`) | Broken imports, changed asserts — the articles' own tolerances are the final arbiter and are **never loosened**; a failing assert means the facade's defaults are wrong, not the test | every adoption PR + nightly |
| Golden stdout diff (normalized) | Drift in printed-but-unasserted values that 1e-3 asserts would miss | every adoption PR + nightly |
| Shadow equivalence (`tools/shadow_check.py`) | Any numeric difference between the old local body and the new delegation: asserts equality at `rtol=1e-12, atol=0` on the article's own synthetic inputs plus random draws in physical ranges; the old body is kept in the PR's test file until merge; documented fallback `rtol=1e-9` only for float-reassociation from vectorization; where algebra intentionally differs, the facade selects the variant reproducing the old number — still 1e-12 | inside each adoption PR |
| petrolib unit + property tests | Library-level correctness independent of facades: golden values seeded from article assert constants; round-trip identities (`archie_sw∘archie_rt`, `washburn∘young_laplace`, `psi_to_bar∘bar_to_psi`); hazard-trap fixtures that pin **both** the canonical output *and* the facade output (so facade drift is caught too), e.g. `klinkenberg_apparent(k,b,p) != klinkenberg_corrected(k,b,p)` | every PR (fast lane) |

CI matrix: Python 3.10–3.13 × {bare checkout, editable install} × numpy {1.26, 2.x}.
Fast PR lane (~2 min: ruff + petrolib tests); full article lane (nightly, on `run-articles`
label, and on any `src20*` diff). Heavy-dep dirs (torch/sklearn/xgboost/skimage) are
skipped unless the `articles` extra is installed — isolated to the full lane with CPU wheels.

## 9. Hazard catalog — where naive merging would silently change numbers

The single most important artifact from the duplication analysis. Every entry below is a
place where two implementations look interchangeable and are not. Each gets keyword-only
canonical parameters, an explicit disambiguating kwarg, and (where call sites exist) a
dedicated Phase-4 audit PR.

**Same name, different math**

| Trap | Detail | Resolution |
|---|---|---|
| `klinkenberg` | computes *apparent* k in 9 files, but the **inverse** (liquid-k correction, pressure in bar) in `src2021_12` | split `klinkenberg_apparent` / `klinkenberg_corrected` |
| Timur–Coates | two incompatible algebras: `(φ/C)⁴(FFI/BVI)²` vs `C·φ^m(FFI/BVI)^n`; plus φ in p.u. vs fraction×1e6 | mandatory `form=`, documented units |
| `kozeny_carman` | four equations (surface-area, capillary-bundle, ratio-update, +tortuosity), outputs in µm², mD or m² | explicit branches + `out_unit` documented |
| Brooks-Corey Sw | `(Pe/Pc)^λ` vs `(Pe/Pc)^(1/N)` — reciprocal exponents under one name | one convention + per-call-site conversion |
| Thomeer | `log10` vs natural log — G differs by ×2.303 | `log_base=` |
| Bond index (CBL) | linear-amplitude, log-amplitude, and attenuation forms (the last *increases* where the others decrease) | `method=`, `input_kind=` |
| `anisotropy_*` | √(Rv/Rh) vs Rv/Rh (no sqrt) vs √(Rh/Rv) | distinct names: `_coefficient` vs `_ratio` |
| Eaton | ratio direction flips between resistivity (obs/normal) and sonic (normal/obs) forms, and between absolute-pressure and gradient variants | separate named forms |
| `elastic_stretch` | integrated distributed load vs point load T·L/EA; correction *added* in some files, *subtracted* in one | two functions; sign documented |
| Mud-gas wetness/balance/character | wetness in % vs fraction (100× cutoff error); two "balance" numerators; three "character" definitions | explicit definitions + `as_percent=` |
| `compute_snr` | known noise_std vs residual-estimated noise | two names |
| `counting_precision` | absolute σρ vs relative 1/√N with composed N | two names |
| `electron_density` | ρe→ρb in three files, ρb→ρe (reversed) in `src2025_06` | forward/inverse named pair |
| `slowing_down_length` | empirical (φ,E) fit vs harmonic mineral mix | two names |
| `net_to_gross` | denominator = whole interval vs reservoir-flag thickness | `gross=` parameter |
| Cole-Cole vs Havriliak–Negami | `(jωτ)^(1−α)` vs `(jωτ)^α` — α is not one parameter | separate functions, never a shared `alpha` |
| `leverett_j_function` in `src2024_12` | an empirical J(Sw) correlation, *not* the normalization | different name in library |
| `recovery_factor` in `src2025_12` | produced-volume ratio, not the saturation form | different name |

**Convention and unit conflicts**

| Trap | Detail |
|---|---|
| Complex permittivity sign | `src2018_06` builds eps* with **+j**·σ/(ωε₀); everything else −j. Standardize −j; flip those call sites explicitly |
| Arps brine-resistivity constants | 6.77 (°F), 21.5 (°C), and 7.0 coexist; `src2023_08` applies 7.0 to °C against a 75 °F reference — live bug. `unit=` mandatory |
| GR spectral coefficients | 16/8/4 hardcoded in four modules vs 16.32/8.09/3.93 in `src2020_12`; arg order flips (th,u,k) vs (k,u,th) | 
| Moduli/velocity/impedance units | unit-neutral vs GPa+g/cc (1e9/1e3 baked in) vs GPa+kg/m³; impedance "MRayl" from three scalings; thresholds (0.5/2.6/3.0 MRayl) assume one of them |
| Normalized saturation Se | denominators (1−Swr−Sor) vs (1−Swr) vs (Swgt−Swirr) — `snr=` explicit, no default |
| Se/kr/Pc clipping | none vs [0,1] vs [1e-6,1] vs [0,0.6] — `clip=` explicit everywhere |
| T2 units | ms nearly everywhere, seconds in `src2024_04`/`src2017_06` (cutoffs 33e-3 s); ρ surface relaxivity in m/s vs µm/s vs µm/ms |
| Pore-size geometry factor | r=2ρT2 vs 3ρT2 vs d=6ρT2 vs generic shape α — `geometry_factor=` |
| GAMMA_H | 2.675e8 vs 2π·42.58e6 rad/s/T (4th-digit) — one CODATA value in `constants.py` |
| diffusion_length | √(Dt) vs √(2Dt) — `geometry_factor=` |
| Regularization λ | λ²I vs λI vs λL'L vs MAP form — canonical solver applies the penalty exactly once; every migrated caller's λ explicitly converted |
| Misfit normalizations | plain L2 vs Σ((d−s)/d)² (explodes at d→0) vs energy-normalized vs 1/|d| complex weights — MCMC step sizes/LM damping are tuned per normalization; re-verify convergence when switching |
| `fit_linear` vs `ols_fit` | intercept at `beta[0]` vs `coef[-1]` — migrate fit+predict pairs together |
| R vs R² | `pearson`/`correlation`/`correlation_coefficient` return R; `r_squared`/`r2_score` return R²; `src2022_10`'s `r2(y_pred, y_obs)` argument order is reversed |
| Washburn/MICP sign | mercury θ=140° makes cosθ negative: some variants abs(), some signed, one −4cosθ — standardize |cos θ|, document |
| Winland/Pittman | all copies take φ as *fraction* and ×100 internally — percent callers would double-scale |
| xcorr edge handling | 5+ variants use `np.roll` (wraps ends, corrupts lags — one file even warns about it); canonical default = trim/interp; shift sign convention documented |
| Swanson constants | c=399 vs 339 — verify against Swanson (1981) and expose as parameters |
| Zero-division guards | `+1e-12` (biases results) vs conditional vs none (NaN/inf) — standardized policy per CONVENTIONS.md |

**Process rules that follow:** never migrate positionally (facades map positions→keywords
once, in review); never merge two variants because their names match; every flagged call
site gets its own one-file PR with a before/after numeric note; wrapper selects whichever
variant reproduces the article's published number.

## 10. Tooling

`pyproject.toml` sketch:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "spwla-petrolib"          # import name: petrolib ("petrolib" may be taken on PyPI)
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
scipy = ["scipy>=1.10"]
ml = ["scikit-learn>=1.2", "xgboost>=1.7"]
torch = ["torch"]
image = ["scikit-image>=0.21"]
articles = ["spwla-petrolib[scipy,ml,torch,image]"]   # everything the 75 dirs need
dev = ["pytest", "pytest-xdist", "ruff", "mypy"]

[tool.hatch.build.targets.wheel]
packages = ["petrolib"]
# plus package-dir mapping so `petrophysics_2026` (src2026_02) resolves when installed

[tool.ruff]
line-length = 100
extend-exclude = ["src20*"]      # never reformat the 660 legacy files

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.mypy]
strict = true
files = ["petrolib"]
```

- **Lint/format/type-check `petrolib/`, `tests/`, `tools/` only.** Legacy scripts are
  exempt forever; facade edits match each file's local style.
- **Versioning:** SemVer 0.x during Phases 2–4 (minor bump per merged train), 1.0.0 when
  the hazard audits close. `CHANGELOG.md` records every deliberate behavior change
  (each with its golden-file regeneration).
- **Docs:** generated API reference (pdoc or mkdocs+mkdocstrings) from the numpydoc
  docstrings — the mandatory `Sources:` lines double as a provenance index. Off the
  critical path; README gains only a short install/import section.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Silent numeric drift while swapping duplicates (the dominant risk) | The four gates (§8); keyword-only APIs; the hazard catalog (§9) with per-call-site audit PRs; never migrating positionally; goldens freezing printed values |
| Facade drift (petrolib edited later, facade default forgotten) | hazard-trap tests pin facade outputs, not just canonical outputs |
| API designed wrong before real usage (library built in a vacuum) | trains interleave extraction and adoption — each API is validated against its real call sites in the same train, starting with the pilot |
| Maintainer stall midway | strangler-fig: after any merged PR the repo is fully working; un-migrated articles simply keep local copies; no phase depends on a future phase to be healthy |
| Bootstrap header noise in educational files | 5 lines, uniform, at the very top; a documented idiom in CONVENTIONS.md; the price of zero-install runnability |
| A vendored/copy-pasted article dir breaking outside the repo | degrades to a loud `ImportError` (never a wrong number); CI's bare-checkout leg guards the in-repo case |
| Heavy deps creeping into the core | numpy-only import policy + lazy scipy; sklearn/torch/xgboost stay article-side; extras-gated CI |
| Known live bugs (Arps °C, Eaton psi/ft×m, dead sys.path) | each fixed in its own flagged PR with golden regeneration + CHANGELOG — never mixed into a refactor PR |
| Losing per-article provenance when ~1,500 local bodies collapse into ~350 canonical functions | git history + mandatory `Sources:` docstring lines + permanent facades keeping the article-side names |
| PyPI name clash | dist `spwla-petrolib`, import `petrolib`; publishing is optional anyway |

## 12. Optional horizon (explicitly not part of this migration)

Once the library is stable and adopted, a future round *could* move toward the cleaner
end state that the architecture evaluation's runner-up favored: articles as ~30-line
cite-import-run demos, a docs site with a per-issue gallery replacing the 281 KB README
index, and pytest-native collection replacing per-dir runners. That direction trades away
inline educational implementations and bare-clone ergonomics, so it should be a separate,
deliberate decision after `petrolib` 1.0 — the plan above neither requires nor precludes
it.

## 13. Immediate next steps

1. Merge this plan document (this PR).
2. Phase 0, PR 1: `tools/run_all_issues.py` + `tests/test_articles.py` (the 75-dir
   harness with the `RUNNERS` map) — confirm all 75 directories pass today.
3. Phase 0, PR 2: golden capture + diff; commit `tools/golden/`.
4. Phase 1: `pyproject.toml` + `petrolib` skeleton + `CONVENTIONS.md` + CI.
5. Pilot train on `ml_stats` (+ metrics/scaling): PR-A then two adoption chunks; hold a
   retrospective against the four gates before green-lighting the 13 physics trains.

---

## Appendix A — Draft canonical API sketches per domain

Starting points for each domain train's PR-A, produced by the cross-repo duplication
analysis. Signatures are indicative, not final: each extraction PR refines its module
against the real call sites and the conventions in §5.2 (keyword-only physics parameters
apply even where a sketch shows plain positional args). Layout references like
`petrolib/<domain>.py` follow §5.1.

### flow_transport

petrolib/flow_transport.py (numpy-vectorized, keyword args; SI unless suffixed; k_md=mD, phi=fraction)

# relperm & displacement
corey_krw(sw, swir=0.15, sor=0.20, krw_end=0.3, nw=2.0)  # Swn=(sw-swir)/(1-swir-sor), clipped, denom-guarded
corey_kro(sw, swir=0.15, sor=0.20, kro_end=1.0, no=2.0)
corey_krg(sg, sgc=0.0, slr=0.15, krg_end=1.0, ng=2.0)
let_kr(s_norm, L, E, T, kr_end=1.0)
kr_from_darcy(q, mu, length, k_abs, area, dp)  # kr=q*mu*L/(k*A*dP)
fit_corey(sw, kr, swir, sor, phase='w') -> (kr_end, n)
fractional_flow(kr_inj, kr_disp, mu_inj, mu_disp)  # zero-mobility guarded
fractional_flow_corey(sw, mu_w, mu_o, **corey)
welge_shock(sw, fw, swi) -> (swf, fwf, sw_avg)
recovery_factor(soi, sor)

# single-phase Darcy & gas flow
darcy_permeability(q, mu, length, area, dp); darcy_rate(k, area, dp, mu, length); darcy_pressure_drop(q, mu, k, area, length, kr=1.0)
darcy_gas_permeability(q_ref, mu, length, area, p_up, p_down, p_ref)  # p^2 form
klinkenberg_apparent(k_inf, b, p_mean, c2=0.0)  # k_inf*(1+b/P+c2/P^2)
klinkenberg_corrected(k_app, b, p_mean)  # explicit inverse
fit_klinkenberg(p_mean, k_app) -> (k_inf, b)
mean_free_path(pressure, T, mu=None, molar_mass=None, d_collision=None)  # viscosity- or kinetic-theory branch
knudsen_number(mfp, pore_diameter); flow_regime(kn) -> str
stress_permeability(k0, gamma, ncs, ncs0=0.0); fit_stress_permeability(ncs, k)
net_confining_stress(total_stress, pore_pressure, biot=1.0)

# poro-perm transforms (return k_md)
timur_coates(phi, ffi, bvi, c=10.0, m=4.0, n=2.0, form='coates')  # 'coates': ((phi/c)^m)(ffi/bvi)^n | 'prefactor': c*phi^m*(ffi/bvi)^n
timur(phi, swirr, a=4800.0, b=4.4, c=2.0)
sdr(phi, t2lm_ms, a=4.0, m=4.0, n=2.0)
kozeny_carman(phi, specific_surface=None, r_pore=None, tau=1.0, c=5.0)  # m^2; surface-area or capillary-bundle branch
kozeny_carman_ratio(k0, phi, phi0, grain_term=True)
winland_r35(k_md, phi); winland_permeability(r35_um, phi)
swanson_permeability(sb_pc_apex, c=399.0, d=1.691); micp_apex(shg, pc)
poroperm_powerlaw(phi, a, b); fit_poroperm(phi, k_md)  # k=10^(a+b*log10 phi)
lucia_permeability(phi, rfn)

# rock typing & upscaling
rqi(k_md, phi); phi_z(phi); fzi(k_md, phi); k_from_fzi(phi, fzi_val); classify_hfu(fzi, n_units=4)
permeability_average(k, method='geometric', weights=None); wiener_bounds(k)

# diffusion & dimensionless groups
diffusion_length(D, t, geometry_factor=1.0)  # sqrt(f*D*t)
diffusion_time(L, D, geometry_factor=1.0)
stokes_einstein(T_K, mu, radius)
fick_flux(D, dc, dx); erfc_profile(c0, x, D, t); early_time_uptake(D, t, half_length)
millington_quirk(D0, phi, sw=1.0); pore_velocity(u_darcy, phi, sw=1.0)
advect_disperse_1d(c_in, length, n_cells, t_total, v, D, k_rxn=0.0) -> (x, c)
capillary_number(mu, v, sigma); bond_number(delta_rho, length_sq, sigma, g=9.81)

### porosity_lithology

petrolib/porosity_lithology.py — numpy-broadcastable, v/v fractions, g/cc; clipping opt-in.

# shale/clay volume
gamma_ray_index(gr, gr_clean, gr_shale, clip=(0,1))
vshale_from_gr(gr, gr_clean, gr_shale, method="linear")  # "linear"|"larionov_tertiary"|"larionov_older"|"clavier"|"steiber"
vshale_neutron_density(phi_n, phi_d, phi_n_sh, phi_d_sh)  # Vsh from N-D separation
combine_clay_indicators(*vcl, how="mean")

# porosity from logs
density_porosity(rho_b, rho_ma=2.65, rho_fl=1.0, clip=None)
neutron_density_porosity(phi_n, phi_d, method="rms")  # "rms" gas-corrected | "mean"
effective_porosity(phi_t, vsh, phi_sh, clip=(0, None))

# porosity from core / digital rock
porosity_from_volumes(v_bulk, v_grain)  # (BV-GV)/BV
boyle_grain_volume(v_cell, v_expansion, p1, p2)
fluid_summation_porosity(bv_oil, bv_water, bv_gas=0.0)
porosity_from_voxel_count(pore_voxels, total_voxels)
ct_porosity(mu, mu_grain, mu_fluid); ct_saturation(mu, mu_dry, mu_sat)

# mixing laws
matrix_density_from_volumes(v, rho)  # arithmetic sum(v_i*rho_i)
matrix_density_from_masses(w, rho)   # harmonic 1/sum(w_i/rho_i), incl. kerogen term
fluid_density(saturations, rhos)     # sum(S_i*rho_i)
bulk_density(phi, rho_ma, rho_fl=1.0, v_k=0.0, rho_k=1.30)  # (1-phi-vk)rho_ma + vk*rho_k + phi*rho_fl
volume_to_weight_fractions(v, rho); weight_to_volume_fractions(w, rho)
log_response(volumes, endpoints)     # linear tool-response mix sum(V_j*R_j)

# Thomas-Stieber
thomas_stieber_phit(v_lam, phi_sand, phi_sh, v_disp=0.0)   # forward trend
thomas_stieber_vlam(phi_t, phi_sand, phi_sh)               # inverse; FNTG = 1 - Vlam
thomas_stieber_sand_porosity(phi_t, v_lam, phi_sh)         # (phi_t - Vlam*phi_sh)/(1 - Vlam)

# organic matter / TOC
kerogen_mass_fraction(toc_wtpct, k=1.2)                    # OM = k*TOC (carbon frac 1/k)
kerogen_volume_from_toc(toc_wtpct, rho_ref, rho_k=1.30, carbon_frac=0.80)  # rho_ref = rho_b or rho_ma per source model
toc_schmoker(rho_b, a=154.497, b=57.261)                   # wt%
toc_passey_dlogr(rt, dt, rt_base, dt_base, lom=10.0, k_dt=0.02)

# mineral solving
multimineral_solve(measured, endpoints, sigma=None, closure=True, closure_weight=1e3, nonneg=False, method="lstsq")  # "lstsq"|"nnls"|"simplex"

# volumetrics & cutoffs
bulk_volume_water(phi, sw)                                 # phi*Sw
hydrocarbon_pore_volume(phi, sw)                           # phi*(1-Sw)
pay_flag(phi, vsh=None, sw=None, phi_cut=0.08, vsh_cut=0.40, sw_cut=0.60)
interval_thickness(depth, flag)                            # sum |dz| where flag
net_to_gross(depth, net_flag, gross_flag=None)             # gross = whole interval unless gross_flag given

### ml_stats

petrolib/ml_stats.py (pure numpy, vectorized)

METRICS
- rmse(y_true, y_pred) -> float
- mae(y_true, y_pred) -> float
- aape(y_true, y_pred, *, pct=True, eps=0.0) -> float  # mean|err/y_true|
- r2_score(y_true, y_pred, *, eps=1e-12) -> float
- pearson(x, y, *, eps=1e-12) -> float  # 0.0 on zero variance
- spearman(x, y) -> float  # tie-averaged ranks
- confusion_matrix(y_true, y_pred, *, classes=None) -> ndarray
- classification_report(y_true, y_pred, *, positive=1) -> dict  # acc/prec/rec/f1

REGRESSION
- linfit(x, y) -> (slope, intercept)
- ols_fit(X, y, *, intercept=True) -> beta  # INTERCEPT FIRST; lstsq
- ols_predict(X, beta, *, intercept=True)
- residual_std(x, y, *, ddof=2) -> float
- powerlaw_fit(x, y, *, log_binned=False) -> (a, b)  # log10 y = a + b log10 x; exponent = -b (+1 if log_binned)
- deming_slope(x, y, *, delta=1.0) -> float  # orthogonal/TLS
- moments_slope(x, y, sigma_x2) -> float  # attenuation-corrected EIV

SCALING
- zscore(X, *, axis=0, eps=1e-12) -> (Xs, mean, std)
- minmax(X, *, axis=0, lo=0.0, hi=1.0, eps=1e-12) -> (Xs, xmin, xmax)
- scale_apply(X, params) / scale_invert(Xs, params)

DECOMPOSITION / INTERPOLATION
- pca_fit(X, n_components=None) -> PCAModel(scores, components, mean, explained_var)  # SVD
- pca_transform(X, model) -> scores
- rbf_fit(X, y, *, kernel="gaussian", sigma=1.0, ridge=0.0) -> RBFModel  # ridge=0 = exact interp
- rbf_predict(model, X)

CLUSTERING
- kmeans(X, k, *, init="farthest", weights=None, iters=100, seed=0) -> (labels, centers)
- agglomerative(X, k, *, linkage="average", standardize=True) -> labels
- silhouette_score(X, labels) -> float

SHALLOW ML
- mlp_fit(X, y, *, hidden=20, iters=5000, lr=0.05, x_scale="zscore", y_clip=None, seed=0) -> MLPModel  # 1-hidden tanh, batch GD; scaler stored in model
- mlp_predict(model, X)  # de-standardizes, applies y_clip
- logistic_fit(X, y, *, iters=3000, lr=0.1) -> model
- logistic_predict(model, X, *, proba=False, threshold=0.5)
- softmax_fit(X, y, n_classes, *, iters=3000, lr=0.2, seed=0); softmax_predict(model, X)
- knn_predict(X_train, y_train, X_query, *, k=5, task="regress", weights="uniform", standardize=True)
- pnn_classify(X_train, y_train, X_query, *, sigma) -> labels

MODEL SELECTION
- train_test_split(X, y, *, test_frac=0.25, seed=0)
- kfold_indices(n, *, k=5, seed=0) -> [(train_idx, test_idx)]
- loo_score(score_fn, X, y) -> float
- grid_search_1d(grid, score_fn) -> (best, score)

BAYES / SAMPLING
- metropolis(log_post, x0, *, n=4000, step=0.1, proposal="gauss", burn=0, seed=0)  # also "lognorm"
- systematic_resample(weights, *, seed=0) -> indices
- effective_sample_size(weights) -> float
- gaussian_reweight(weights, residuals, sigma)

TESTS
- welch_t(mean1, std1, n1, mean2, std2, n2) -> float
- chi_square(table) -> (chi2, dof)

### saturation_resistivity

Module: petrolib/saturation_resistivity.py  (numpy-vectorized; resistivity ohm-m, conductivity S/m, CEC meq/g, Qv meq/cm3; keyword-only after the data args)

# Archie core
formation_factor(phi, *, a=1.0, m=2.0, b=0.0) -> F           # F = b + a/phi**m; b=0 is Archie
r0(rw, phi, *, a=1.0, m=2.0) -> R0                           # R0 = F*Rw
cementation_exponent(F, phi, *, a=1.0) -> m                  # point inversion log(F/a)/log(1/phi)
archie_sw(rt, rw, phi, *, a=1.0, m=2.0, n=2.0, clip=True)    # Sw=(a*Rw/(phi^m*Rt))^(1/n); also Sxo via (rxo, rmf)
archie_rt(sw, rw, phi, *, a=1.0, m=2.0, n=2.0)               # forward Rt
archie_conductivity(sw, cw, phi, *, a=1.0, m=2.0, n=2.0)     # sigma_t = cw*phi^m*sw^n/a
resistivity_index(rt, r0)                                     # I = Rt/R0
ri_from_sw(sw, *, n=2.0, b=1.0)                               # I = b*Sw^-n
sw_from_ri(ri, *, n=2.0, b=1.0, clip=True)                    # inverse
apparent_water_resistivity(rt, phi, *, a=1.0, m=2.0)          # Rwa
bulk_volume_water(phi, sw); sw_from_bvw(bvw, phi, *, clip=True)
fit_archie_m(phi, F, *, a=None) -> (m, a)                     # log-log regression; a fixed if given
fit_archie_n(sw, ri) -> n
pickett_fit(bvw, rt) -> (m, rw)                               # joint fit on the R0 line

# Shaly sand
qv_from_cec(cec, phi_t, *, rho_grain=2.65); cec_from_qv(qv, phi_t, *, rho_grain=2.65)
juhasz_qv(vcl_dry, cec_clay, phi_t, *, rho_clay=2.78)
b_counterion(cw, *, temp_c=25.0) -> B                         # Waxman-Thomas B(Cw,T)
waxman_smits_ct(sw, cw, phi_t, qv, *, b=None, m=2.0, n=2.0, a=1.0, temp_c=25.0)
waxman_smits_sw(rt, rw, phi_t, qv, *, b=None, m=2.0, n=2.0, a=1.0, temp_c=25.0, tol=1e-9)  # vectorized bisection
dual_water_ct(sw, cw, cwb, swb, phi_t, *, m=2.0, n=2.0)       # bound-water form; adapters swb_from_qv(qv, *, alpha=1.0, vq=0.28), cwb_from_shale(csh, phit_sh)
dual_water_sw(ct, cw, cwb, swb, phi_t, *, m=2.0, n=2.0, tol=1e-9)
simandoux_sw(rt, rw, phi, vsh, rsh, *, a=1.0, m=2.0, n=2.0)   # numeric solve; quadratic fast path at n=2
indonesia_sw(rt, rw, phi, vcl, rcl, *, a=1.0, m=2.0, n=2.0)
modified_simandoux_ct(sw, cw, csh, vsh, phi, *, m=2.0)

# Laminated / anisotropy (fractions: scalar pair or N-layer arrays)
laminated_rh(fractions, resistivities)                        # 1/Rh = sum(v_i/R_i)
laminated_rv(fractions, resistivities)                        # Rv = sum(v_i*R_i)
solve_laminated(rh, rv, rshale_h, *, rshale_v=None) -> (vshale, rsand)
anisotropy_coefficient(rv, rh) -> sqrt(Rv/Rh); anisotropy_ratio(rv, rh) -> Rv/Rh

# Other saturation sources
sigma_sw(sigma_log, phi, *, sigma_ma, sigma_w, sigma_hc, clip=True)   # pulsed-neutron balance
surface_conductivity(cec, phi, F, *, rho_grain=2.65, mobility=None, temp_c=25.0)  # Stern term
total_conductivity_surface(cw, F, sigma_s)                     # sigma = cw/F + sigma_s

### geochem_fluids

petrolib/geochem_fluids/ — numpy-vectorized, keyword-explicit, SI-first.

brine.py
  rw75_from_salinity(nacl_ppm)               # Bateman-Konen R75=0.0123+3647.5/C^0.955
  salinity_from_rw75(rw75_ohmm)              # exact inverse
  arps_correct(r1, t1, t2, *, unit="F")      # R2=R1*(T1+c)/(T2+c); c=6.77 F / 21.5 C
  rw_from_salinity(nacl_ppm, temp, *, unit="F"); salinity_from_rw(rw, temp, *, unit="F")
  sigma_w_from_salinity(nacl_ppm, temp_c=24.0)   # capture cross-section, c.u.
  nacl_meq_per_liter(nacl_ppm)
  brine_density_bw92(nacl_ppm, temp_c, press_mpa=0.1)

asphaltene.py
  YEN_MULLINS_DIAMETERS_M = {"molecule":1.5e-9,"nanoaggregate":2e-9,"cluster":5e-9}
  molar_volume_from_diameter(d_m); diameter_from_molar_volume(va)
  fhz_ratio(dz_m, va_m3mol, delta_rho, temp_k, *, entropy=0.0, dsol2_pa=0.0)  # dz>0 down; 3-term
  fhz_profile(depth_m, od_ref, depth_ref, va, delta_rho, temp_k)
  fhz_invert_molar_volume(od1, z1, od2, z2, delta_rho, temp_k)
  nearest_yen_mullins(d_m, rtol=0.25)

mudgas.py
  normalize_composition(comps, axis=-1)      # zero-safe sum-to-one closure
  wetness_ratio(c1..c5, *, percent=True)
  balance_ratio(c1..c5, *, convention="haworth")  # or "light_heavy"
  character_ratio(c3, c4, c5)                # (C4+C5)/C3
  pixler_ratios(c1..c5) -> dict; bernard_ratio(c1, c2, c3)
  apply_eec(comps, alphas)                   # extraction-efficiency correction
  classify_fluid_gor(gor_sm3, thresholds=(180,360,640,5000,15000))
  classify_fluid_wetness(wh_pct, bh)

adsorption.py
  langmuir(p, v_l, p_l, *, rho_b=None)       # rho_b folds the Gc form
  gibbs_excess(gc, rho_free, rho_ads)
  bet_isotherm(x_rel, vm, c)
  bet_fit(x_rel, v_ads, *, cross_nm2=0.162) -> (vm, c, ssa_m2_g)
  free_gas(phi, sw, bg); gas_in_place(area_m2, h_m, phi, sg, bg)

pvt.py
  pseudo_reduced(p, t, ppc, tpc)
  z_beggs_brill(ppr, tpr); z_peng_robinson(p_pa, t_k, tc_k, pc_pa, omega, phase="vapor")
  gas_density(p_pa, t_k, *, m_kg_mol=0.01604, z=1.0); pressure_from_gas_density(rho, t_k, ...)
  mixture_mw(y, mw); gas_gravity(mw_kg_mol)
  wilson_k(p_pa, t_k, pc_pa, tc_k, omega); rachford_rice(z, k, tol=1e-12)

gradients.py
  fit_pressure_gradient(depth_m, p) -> (dpdz, p0)
  density_from_gradient(dpdz, *, p_unit="Pa") -> kg/m3   # no clipping
  fluid_contact(depth_a, p_a, depth_b, p_b) -> contact_m

solubility.py
  henry_constant_co2(t_k); setschenow_factor(m_nacl, t_k, *, ks25=0.11)
  co2_solubility_brine(p_mpa, t_k, m_nacl=0.0, *, m_ch4=0.0) -> mol/kg
  henry_solubility_ln(p_mpa, t_k, a, b, dh_j_mol)  # KK gas-in-oil

contamination.py
  mix_linear(p_v, p_f, eta); contamination_fraction(p, p_v, p_f)
  cleanup_powerlaw(v, eta0, v_star, *, exponent=5/12)
  volume_to_target(eta0, v_star, eta_t, *, exponent=5/12)

core_geochem.py
  dean_stark(v_water, v_hc, *, v_bulk=None, v_pore=None) -> (phi, sw, s_hc)
  oxide_closure(oxides) -> (closed, factor)
  osi(s1_mg_g, toc_wt_pct)

### acoustic_geomech

petrolib/acoustic_geomech.py — numpy-vectorized, SI (Pa, kg/m3, m/s) unless noted

# elastic conversions
moduli_from_velocity(vp, vs, rho) -> (K, G)  # G=rho*Vs^2, K=rho*Vp^2-4G/3
velocity_from_moduli(k, g, rho) -> (Vp, Vs)
stiffness_from_velocity(rho, v)  # C=rho*V^2 (M=rho*Vp^2)
lame_from_velocity(vp, vs, rho) -> (lam, mu)
velocity_from_slowness(dt, dt_unit="us/ft") -> m/s
youngs_poisson_dynamic(vp, vs, rho) -> (E, nu)
youngs_from_kg(k, g); poisson_from_kg(k, g)

# mixing & fluid substitution
voigt(f, m); reuss(f, m); voigt_reuss_hill(f, m)  # axis kw
wood_fluid_modulus(saturations, moduli)  # Reuss over fluids
brie_fluid_modulus(sw, k_liquid, k_gas, e=3.0)
gassmann_ksat(*, k_dry, k_mineral, k_fluid, phi)
gassmann_kdry(*, k_sat, k_mineral, k_fluid, phi)

# impedance / interface / attenuation
acoustic_impedance(rho, v, out="rayl")  # or "mrayl"
reflection_coefficient(z1, z2)  # (Z2-Z1)/(Z2+Z1)
transmission_energy(z1, z2)  # 1-R^2
attenuation_db(a1, a2)  # 20*log10(A1/A2)
attenuation_coefficient(a0, ax, x)  # alpha in A=A0*exp(-alpha*x)
snell_angle(v1, v2, degrees=True)

# borehole waveforms
semblance(traces, dt_s, offsets_m, slowness_s_m, t0_s, window_s) -> float
stc_pick(traces, dt_s, offsets_m, slowness_grid, t0_grid, window_s) -> (s, t0, coh)
alford_rotation(xx, xy, yx, yy, theta) -> (fast, slow, cross)
fast_shear_azimuth(xx, xy, yx, yy, n_theta=361)
shear_wave_splitting(v_fast, v_slow)  # (Vf-Vs)/Vf

# anisotropy
thomsen_epsilon(c11, c33); thomsen_gamma(c66, c44); thomsen_delta(c13, c33, c44)
annie_c11(c33, c44, c66); annie_c13(c11, c66); mannie_c11(c33, c44, c66, kp=1.0); mannie_c13(c11, c66, k=1.0); mannie2_c66(c11, c33, c44, k)
c13_from_45deg(c11, c33, c44, rho, v45)
vti_engineering_moduli(c11, c33, c44, c66, c13, c12=None) -> dict(Ev, Eh, nu_v, nu_h, Gvh, Ghh)  # c12 default c11-2*c66
engineering_moduli(C) -> dict  # generic 6x6 via S=inv(C)
compliance(C); is_positive_definite(C)

# geomechanics (units="si"|"field" [psi, ft, g/cm3])
biot_coefficient(k_dry, k_mineral)
effective_stress(sigma_total, pp, biot=1.0)
overburden_stress(depth, rho_bulk, water_depth=0.0, rho_water=1030.0, units="si")  # cumulative rho*g*dz
hydrostatic_pressure(depth, gradient=None, rho=1000.0, units="si")
eaton_pore_pressure(s_v, p_normal, ratio, exponent=3.0)  # Pp=Sv-(Sv-Pn)*ratio^n; caller forms ratio
min_horizontal_stress(sv, pp, nu, biot=1.0, E=0.0, eps_h=0.0, eps_H=0.0, tectonic=0.0)  # Thiercelin-Plumb superset
breakdown_pressure(sh, sH, pp, tensile_strength=0.0)  # 3sh-sH-Pp+T0
reopening_pressure(sh, sH, pp); shmax_from_reopening(pr, sh, pp)
kirsch_hoop_stress(sH, sh, pw, theta_deg)
brittleness_rickman(E, nu, e_range=(10.0, 80.0), nu_range=(0.15, 0.40), percent=False)
vs_from_vp(vp_km_s, method="castagna_ls")  # pickett_ls|pickett_dol|carroll|brocher

### nmr

Module petrolib/nmr.py (constants GAMMA_H etc. in petrolib/constants.py). All numpy-vectorized, scalar-in scalar-out. Units: T1/T2 in ms, rho in um/s, S/V in 1/um, D in um^2/ms, G in T/m, TE in ms, k in mD, phi as fraction.

# relaxation physics
relaxation_rate(t2_bulk_ms=np.inf, rho_um_s=0.0, s_over_v_per_um=0.0, D=0.0, G=0.0, TE=0.0, gamma=GAMMA_H) -> 1/ms  # Brownstein-Tarr: 1/T2B + rho*(S/V) + D*(gamma*G*TE)^2/12
t2_apparent(**same) -> ms  # 1/relaxation_rate
diffusion_relaxation_rate(D, G, TE, gamma=GAMMA_H) -> 1/ms  # (gamma*G*TE)^2*D/12
combine_relaxation_times(*times_ms) -> ms  # parallel rates 1/T = sum(1/Ti)
surface_to_volume(t2_ms, rho_um_s, t2_bulk_ms=np.inf) -> 1/um
pore_radius_from_t2(t2_ms, rho_um_s, shape_factor=3.0, t2_bulk_ms=np.inf) -> um  # shape_factor: 3 sphere radius, 2 cylinder, 6 sphere diameter
surface_relaxivity_from_pore(t2_ms, radius_um, shape_factor=3.0) -> um/s
larmor_frequency(b0_T, gamma=GAMMA_H) -> Hz

# distribution statistics / partitions
t2_logmean(t2_ms, amplitude) -> float  # exp(sum(A*lnT2)/sum(A)); nan on zero mass
total_porosity(amplitude) -> float
t2_partition(t2_ms, amplitude, cutoffs_ms=(33.0,), fractions=False) -> tuple  # n+1 band sums; generalizes CBW(3)/BVI-FFI(33/100)/clay-cap-free 3-way
bvi_ffi(t2_ms, amplitude, cutoff_ms=33.0) -> (bvi, ffi)  # convenience wrapper

# forward models & inversion
cpmg_kernel(t_ms, t2_grid_ms) -> (nt, nT2)  # exp(-t/T2)
multiexp_decay(t_ms, amplitudes, t2_ms, noise=0.0, rng=None) -> array  # covers mono-exp with scalars
t1_saturation_recovery(t_ms, m0, t1_ms); t1_inversion_recovery(t_ms, m0, t1_ms)
t1t2_kernel(t_echo_ms, t_wait_ms, t1_grid_ms, t2_grid_ms, mode='saturation'|'inversion') -> Kronecker 2D kernel
invert_t2(t_ms, signal, t2_grid_ms, alpha=0.1, method='tikhonov-nnls'|'l1', auto_alpha=False) -> amplitudes  # [K; alpha*I] NNLS; BRD auto-alpha
invert_t1t2(signal, t1_obs_ms, t2_obs_ms, t1_grid_ms, t2_grid_ms, alpha=1e-2) -> map
fit_t1(t_ms, signal, model='saturation') -> (m0, t1_ms)

# permeability transforms
timur_coates(phi, ffi, bvi, C=10.0, m=4.0, n=2.0, form='classic') -> mD  # 'classic': (phi/C)^m*(FFI/BVI)^n; 'prefactor': C*phi^m*(FFI/BVI)^n
sdr(phi, t2lm_ms, a=4.0, m=4.0, n=2.0, rho_um_s=None) -> mD  # rho given -> KSDR a*phi^m*(rho*T2lm)^n

# fluid typing / HI
t1_t2_ratio(t1, t2) -> array
classify_t1t2(t1t2, cutoff=2.0) -> bool array (True=hydrocarbon)
partition_t1t2_map(t1t2, amplitudes, cutoff=2.0) -> (v_hc, v_water)
nmr_saturation(v_fluid, v_total) -> float
hydrogen_index(rho_g_cc, n_protons, mol_weight) -> float  # vs water 2/18.02
porosity_hi_correction(phi_apparent, hi) -> float  # phi_true = phi_app/HI

# relaxation theory / diffusion
bpp_spectral_density(omega, tau_c); bpp_t1_t2(omega0, tau_c, dipolar_constant=1.0) -> (T1, T2)  # full 3/10, 3/20 prefactors
mitra_short_time(d0, t, s_over_v, normalized=True) -> D(t)/D0 or D(t)
tortuosity(d0, d_inf) -> float

### inversion_numerics

petrolib/inversion_numerics/:

linear.py
- tikhonov_solve(A, b, lam=1.0, *, reg_op=None, x_ref=None, sigma=None, nonneg=False): argmin ||W(Ax-b)||^2 + lam*||L(x-x_ref)||^2; L=I default; lam applied once (never lam^2).
- map_estimate(G, d, noise_var, prior_strength, L=None, m_prior=None) -> (x, posterior_cov): Bayesian MAP.
- ista_l1(A, b, eta=0.1, nonneg=True, max_iter=500): sparse L1 inversion.
- nnls_solve(A, b, reg=0.0): NNLS on augmented system.
- project_simplex(v, total=1.0): simplex projection.
- unmix(measured, response_matrix, *, sigma=None, closure=None, nonneg='nnls', normalize=False): mineral/spectral unmixing; closure {None,'row','simplex'}; nonneg {None,'clip','nnls','simplex'}.
- svd_solve(A, b, rank=None); condition_number(A).
- convolution_matrix(kernel, n); deconvolve(d, G, rank=None).
- difference_operator(n, order=1).

costs.py
- misfit(sim, obs, *, weights=None, kind='l2', log_space=False): kind {'l2','rms','rel_data','rel_norm','chi2'}.
- reg_lambda_multiplicative(misfit, alpha, beta, lam_max=inf): Habashy-Abubakar schedule.
- reg_lambda_brd(A, b, chi2_target, bracket=(1e-6, 1e3)): discrepancy bisection.

nonlinear.py
- fd_jacobian(forward, m, eps=1e-6, scheme='central', relative=True); fd_gradient(f, x, **kw).
- levenberg_marquardt(forward, data, m0, *, bounds=None, log_params=False, lam0=1e-3, lam_up=5.0, lam_dn=0.5, damping='diag', max_iter=50, tol=1e-8) -> InvResult.
- occam(forward, data, m0, noise_level, *, reg_order=2, lam0=100.0, max_iter=50): smoothest model to target misfit.
- grid_search(forward, data, grids, misfit='log_l2').
- multistart(solver, bounds, n_starts=50, seed=0, aggregate='misfit_weighted').
- feasible_set_sampling(forward, data, m_center, bounds, noise_level, n_samples=2000, step_frac=0.15, seed=1) -> dict(P5, P50, P95, models).

stochastic.py
- gaussian_loglik(obs, pred, sigma, weights=None, log_space=False); uniform_logprior(x, bounds); soft_envelope_logprior(x, lo, hi).
- metropolis(log_post, x0, step, n_samples=4000, *, log_space=False, burn_in=0, seed=0) -> Chain(samples, acceptance).
- mala(log_post, x0, step, n_samples, seed=0): FD-gradient drift.
- enkf_update(ens, obs, obs_cov, obs_op, seed=0).
- lm_enrml(prior_mean, prior_cov, obs, obs_cov, forward, n_ens=80, n_iter=12, lam0=1.0, lam_up=4.0, lam_dn=0.4, seed=0).

optimize.py
- pso(objective, bounds, n_particles=30, n_iter=200, omega=0.7, inertia_decay=None, c1=1.5, c2=1.5, seed=0) -> (x_best, f_best, history).
- gradient_descent(f, x0, lr=None, backtracking=False, max_iter=4000).

fitting.py
- fit_line(x, y, *, xform=None, yform=None) -> LineFit(slope, intercept, r2).
- fit_powerlaw_decay(x, y, exponent=None): linearized if exponent given, else curve_fit.
- fit_exponential_approach(t, y, three_point=False) -> (asymptote, tau).
- fit_cosine(az, y) -> (mean, amplitude, phase).

pde.py
- effective_conductivity_2d(sigma_map, n_iter=5000, tol=1e-7).
- diffusion_step_1d(u, alpha, dt, dx, source=None, bc='neumann'); cfl_number(alpha, dt, dx).

### relperm_wettability

petrolib/relperm_wettability.py — numpy-vectorized; saturations as fractions, theta in degrees, sigma in N/m, mu in Pa*s.

normalized_saturation(s, sr, snr=0.0, *, clip=(0.0,1.0)) -> Se  # Se=(S-Sr)/(1-Sr-Snr); snr=0 gives the (1-Swr) convention; clip=None to disable
corey_kr(sw, *, swr, sor=0.0, krw_max=1.0, kro_max=1.0, nw=2.0, no=2.0) -> (krw, kro)  # standard two-phase Corey pair on Se; replaces all corey_kr/corey_relperm/krw+kro variants
corey_kr_gas(sg, *, sgc, swc=0.0, sorg=0.0, krg_max=1.0, ng=2.0) -> krg  # gas-phase Corey on Sg*=(Sg-Sgc)/(1-Swc-Sgc-Sorg)
brooks_corey_burdine_kr(sw, *, swr, lam, snwr=0.0) -> (krw, krnw)  # Burdine: krw=Se^((2+3L)/L), krnw=(1-Se)^2*(1-Se^((2+L)/L))
let_kr(sw, *, swr, L, E, T, kr_max=1.0, phase="wetting") -> kr  # Lomeland LET correlation; call per phase
fit_corey(sw, krw_obs, krnw_obs, *, swr, sor=0.0) -> dict  # least-squares Corey endpoints/exponents (scipy)
stone_i(...), stone_ii(...), baker_kro(...)  # three-phase kro (single source: src2014_08)
phase_mobility(kr, mu) -> lam
fractional_flow(kr_w, kr_nw, *, mu_w, mu_nw) -> fw  # Buckley-Leverett, zero-safe
fractional_flow_curve(*, swr, sor, krw_max, kro_max, nw, no, mu_w, mu_nw, n=2000) -> (sw, fw)
welge_shock(sw, fw, swc) -> (swf, fwf, sw_avg)  # tangent construction: front saturation, fw at front, mean Sw behind
endpoint_mobility_ratio(krw_max, kro_max, mu_w, mu_o) -> M
ss_darcy_kr(fw, q, dp, *, k, L, A, mu_w, mu_nw) -> (krw, krnw)  # steady-state kr from per-phase Darcy law
capillary_number(*, mu, v, sigma) -> Nca  # keyword-only: positional order varies repo-wide
bond_number(*, drho, k, sigma, g=9.81) -> Nb
trapping_number(nca, nb) -> Nt
capillary_desaturation(n, *, sor_max, sor_min, n_crit, exponent=1.0) -> Sor  # sor_min+(sor_max-sor_min)/(1+(n/n_crit)**exponent)
land_c(s_i_max, s_r_max) -> C  # C = 1/Sr_max - 1/Si_max
land_trapped(s_i, *, C=None, s_r_max=None) -> Sr  # Sr=Si/(1+C*Si); s_r_max form assumes Si_max=1
amott_indices(vw_spont, vw_forced, vo_spont, vo_forced) -> (Iw, Io, Iah)  # accepts volumes or delta-saturations
usbm_index(area_drainage, area_imbibition) -> W  # log10(A1/A2)
nmr_wettability_index(w_signal, o_signal) -> Iw  # (w-o)/(w+o); works for amplitudes, relaxation rates, or T2 areas
young_contact_angle(sigma_so, sigma_sw, sigma_wo) -> theta_deg  # Young's law, arccos clipped
wenzel_angle(theta_young_deg, roughness) -> theta_app_deg
young_laplace_pc(sigma, r, theta_deg=0.0, *, absolute=False) -> Pc  # 2*sigma*cos(theta)/r; absolute=True for MICP-style |cos|
classify_wettability_angle(theta_deg, *, cuts=(75.0, 105.0)) -> str
classify_wettability_index(i, *, scheme="3class") -> str  # "3class" (+-0.3) or "5class" (+-0.3/+-0.1)
displacement_efficiency(soi, sor) -> Ed  # (Soi-Sor)/Soi

### integrity_drilling

petrolib/integrity_drilling.py — numpy-vectorized, SI in/out unless noted, keyword defaults = most common observed.

# Cased-hole acoustics / cement
acoustic_impedance(rho, v, *, rho_unit="kg/m3") -> MRayl  # Z=rho*v; accepts "g/cc"
reflection_coefficient(z1, z2)  # (Z2-Z1)/(Z2+Z1)
transmission_energy(z1, z2)  # 1-R**2
attenuation_db(a_near, a_far, spacing_m=None)  # 20*log10(A1/A2); dB/m if spacing given
attenuation_coefficient(a0, ax, x_m)  # ln(a0/ax)/x, 1/m
bond_index(measured, free_pipe, well_bonded, *, method="linear", input_kind="amplitude", clip=(0.0, 1.0))  # method in {"linear","log"}; input_kind="attenuation" flips orientation
bond_index_combined(bi_a, bi_b, *, weights=(0.6, 0.4), corrector=None)
classify_annulus(z_mrayl, *, gas_max=0.5, liquid_max=2.6, cement_min=3.0)  # -> 'gas'/'liquid'/'transition'/'cement'
cement_quality_score(z_mrayl, *, cement_type="Portland", thresholds=None)  # continuous 0-1
classify_cement_from_cbl(relative_amp, *, good_max=0.15, medium_max=0.30)  # 'Good'/'Medium'/'Poor'

# Casing condition
casing_resonance_frequency(thickness_m, *, v=5900.0, n=1, correction=1.0)  # f=corr*n*v/(2d)
casing_thickness_from_resonance(freq_hz, *, v=5900.0, n=1, correction=1.0) -> m
metal_loss_pct(measured, nominal)  # clipped 0-100
casing_condition(loss_pct, *, bands=(10.0, 25.0, 42.5))  # 'good'/'fair'/'poor'/'critical'
remaining_life_years(thickness_mm, min_acceptable_mm, rate_mm_per_yr)
corrosion_front_depth(t, K)  # K*sqrt(t)

# Microannulus leaks
microannulus_omega(r_casing_m, aperture_m)  # annular-gap moment, m^4
leak_rate_liquid(aperture_m, r_casing_m, dp_pa, length_m, mu_pa_s, *, rho=1000.0, inclination_deg=90.0)  # gravity-corrected Hagen-Poiseuille, m3/s
leak_rate_gas(aperture_m, r_casing_m, p_in_pa, p_out_pa, length_m, mu_pa_s)  # isothermal (P1^2-P2^2)
cubic_law_conductivity(aperture_m, *, rho=1000.0, mu=1e-3)  # parallel-plate crack, m/s

# Mud gas
haworth_ratios(c1, c2, c3, c4, c5, *, percent=True)  # -> (Wh, Bh, Ch); Bh=(C1+C2)/(C3+C4+C5)
pixler_ratios(c1, c2, c3, c4, c5)  # dict {C1/C2..C1/C5, bernard}
classify_fluid_haworth(wh, bh, ch=None, *, n_classes=4)
normalize_gas(total_gas, rop, flow, bit_diameter, *, mud_weight=None, units="metric", reference=None)  # gas per drilled-rock volume; units="field" -> SCF/ton

# Pressures / drilling window
hydrostatic_pressure(tvd_m, *, rho=1030.0) -> Pa  # + psi/bar wrappers
overburden_pressure(tvd_m, rho_bulk, *, water_depth_m=0.0, rho_sw=1030.0) -> Pa
eaton_pore_pressure(overburden, hydrostatic, observed, normal, *, exponent=3.0, log_type="sonic")  # ratio direction set by log_type ("sonic"|"resistivity")
bowers_pore_pressure(velocity, overburden, *, A=10.0, B=0.7, unloading=False, U=3.0, v0=5000.0)
drilling_window_margin(pore, frac); within_drilling_window(ecd, pore, frac) -> bool

# Drilling fluid
mudcake_thickness(t_s, *, k_mc_m2, dp_pa, mu_pa_s, solids_ratio, model="dewan")  # {"sqrt_k","dewan","chin_ode"} -> m

### capillary_pressure

petrolib/capillary_pressure.py — numpy-vectorized, SI core (Pa, N/m, m, kg/m3; theta in degrees); unit-suffixed wrappers for oilfield conventions.

# Young-Laplace / Washburn
young_laplace_pc(r_m, sigma=0.480, theta_deg=140.0) -> Pa        # 2*sigma*|cos(theta)|/r
washburn_radius(pc_pa, sigma=0.480, theta_deg=140.0) -> m        # inverse; washburn_diameter = 2*r
washburn_radius_um_from_psi(pc_psi, sigma_dyne_cm=480.0, theta_deg=140.0) -> um   # r ~ 107.6/Pc[psia]

# Leverett J and rescaling
leverett_j(pc, k, phi, sigma, theta_deg=0.0) -> J                # Pc*sqrt(k/phi)/(sigma*|cos|)
pc_from_j(j, k, phi, sigma, theta_deg=0.0) -> Pc
pc_rescale_system(pc, sigma_from, theta_from, sigma_to, theta_to) -> Pc   # |sigma*cos| ratio, lab->res
pc_rescale_stress(pc, k_ref, phi_ref, k_new, phi_new) -> Pc      # J-invariant sqrt((k_ref/phi_ref)/(k_new/phi_new))

# Saturation normalization + curve models
normalized_sw(sw, swirr=0.0, sor=0.0, clip=(1e-6, 1.0)) -> Swn   # (Sw-Swirr)/(1-Swirr-Sor)
brooks_corey_pc(sw, pe, lam, swirr=0.0) -> Pc                    # Pe*Swn^(-1/lam)
brooks_corey_sw(pc, pe, lam, swirr=0.0) -> Sw                    # Swirr+(1-Swirr)*(Pe/Pc)^lam; Sw=1 below Pe
drainage_pc_threshold(swn, pc_th, a, b) -> Pc                    # Pcth + a*Swn^-b
thomeer_shg(pc, bv, g, pd, log_base=10.0) -> Shg                 # Bv*exp(-G/log(Pc/Pd)); 0 below Pd
thomeer_sw(pc, pe, g, swirr) -> Sw
van_genuchten_pc(sw, alpha, m, n, swirr) -> Pc

# Saturation-height
pc_from_height(h_m, rho_w, rho_hc, g=9.80665) -> Pa
height_from_pc(pc_pa, rho_w, rho_hc, g=9.80665) -> m             # serves entry_height / max seal column
buoyancy_pc_psi(h_ft, sg_w, sg_hc, grad_psi_ft=0.433) -> psi
sw_vs_height_power(h, a, b, swirr=0.0) -> Sw                     # Swirr + a*h^-b

# Laboratory methods
centrifuge_pc(delta_rho, r_inner, r_outer, omega=None, rpm=None) -> Pa   # HB; r_inner may be array -> Pc(r) profile
centrifuge_critical_speed(p_threshold, delta_rho, r_inner, r_outer) -> rad/s

# MICP rock typing
winland_r35(k_md, phi_frac) -> um
winland_k(r35_um, phi_frac) -> mD
r35_from_micp(pc_psi, shg_frac) -> um                            # interpolate curve at Shg=0.35
threshold_pressure(shg_frac, pc, cut=0.05) -> float
swanson_permeability(sb_pc_max, c=399.0, d=1.691) -> mD
swanson_k_thomeer(g, bv, pd) -> mD                               # 3.8068*G^-1.3334*(Bv/Pd)^2 (+ inverse thomeer_g_from_k)

# Imbibition dynamics
lucas_washburn_length(t_s, r_m, sigma, theta_deg, mu_pas) -> m   # clipped 0 when cos<0
capillary_rise_height(r_m, sigma, theta_deg, delta_rho, g=9.80665) -> m

### data_qc_io

petrolib/data_qc_io/ (numpy-vectorized; y_true first; rng: int|Generator|None)

units.py
  PA_PER_PSI=6894.757293; PSI_PER_BAR=14.5037738; M_PER_FT=0.3048
  convert(x, from_unit, to_unit)  # 'psi','bar','MPa','Pa','ft','m'

clean.py
  sentinels_to_nan(x, sentinels=(-999.0,-999.25,-9999.0), atol=1e-4)
  impute_gaps(x, index=None)  # linear interp over NaN runs
  outlier_mask(x, method='zscore', threshold=3.0, k=1.5)  # 'zscore'|'iqr'|'mad' -> bool
  despike(x, threshold=3.0, side='both', replace='interp')  # MAD*1.4826
  closure_residual(*fracs, target=1.0)  # e.g. Sw+So+Sg-1
  renormalize(fracs, axis=-1)  # force sum-to-1 (oxide/composition closure)
  relative_discrepancy(a, b)  # 2|a-b|/(a+b)

scale.py
  zscore(x, axis=None, eps=1e-12, return_params=False)
  minmax_scale(x, out_range=(0.,1.), in_range=None, axis=None, eps=1e-12)
  normalize_to_reference(x, ref_lo, ref_hi, pct=(5,95))  # Shier GR normalization
  match_moments(x, target_mean, target_std)  # affine mean/std matching

filt.py
  smooth(x, window=5, kind='boxcar', sigma=None, mode='nearest')  # boxcar|gaussian|median
  moving_stat(x, window, stat='mean', center=True)  # mean|std|mad|cv; sliding_window_view
  tool_response(x, dz, fwhm)  # Gaussian vertical-response convolution (upscaling)
  median_filter2d(img, size=3)
  window_features(X, window=11)  # edge-padded depth-window stack (n,d)->(n,d*window)

align.py
  xcorr_shift(ref, other, max_shift=50, standardize=True) -> (lag, peak_r)
  dtw(ref, other, band=None, derivative=False) -> (distance, path)
  shift_curve(x, lag, fill=nan)
  detect_bed_boundaries(curve, window=11, threshold=None) -> indices

metrics.py
  rmse(y_true, y_pred); mae(...); mape(...); r2(...); pearson_r(x, y)

pay.py
  pay_flags(phi=None, sw=None, vsh=None, phi_cut=0.08, sw_cut=0.6, vsh_cut=0.4, extra=None)
  flagged_thickness(depth, flag)  # sum |gradient(depth)| where flag
  net_to_gross(depth, net_flag, gross_flag=None)  # None -> whole interval

signal.py
  snr_db(signal, noise=None, noise_std=None)  # 10*log10(Psig/Pnoise)
  stack_gain(n)  # sqrt(n)
  block_stack(x, n, axis=-1)  # block-mean stacking
  add_gaussian_noise(x, sigma_fraction, rng=None)  # sigma = frac*mean|x|

synth.py
  blocky_log(n, n_beds=8, level_range=(20,150), noise=2.0, rng=None) -> (curve, bounds)
  log_suite(n=600, curves=('GR','RD','RHOB','DT','NPHI'), n_facies=4, rng=None) -> (dict, facies)
  shifted_pair(n, shift, rng=None)  # for alignment tests
  gaussian_mixture_spectrum(axis, centres, amps, widths, noise=0.0, log_axis=True)  # T2/T1T2
  sphere_pack_volume(shape, r_range, target_phi, rng=None); disk_image(size, n_features, rng=None)

io.py
  class WellboreData: add_channel(name, data, units, axes); to_json()/from_json()  # adopt src2023_12/bradley_wellbore_format.py

### em_dielectric

petrolib/em_dielectric.py — numpy-vectorized, freq_hz in Hz, engineering -j loss convention, keyword-only physics params.
Constants: EPS0=8.8541878128e-12, MU0=4e-7*pi

complex_permittivity(eps_real, *, sigma=0.0, freq_hz=None, eps_imag=0.0) -> complex  # eps* = e' - j(e'' + sigma/(w*eps0))
imag_permittivity_from_sigma(sigma, freq_hz)   # e'' = sigma/(2*pi*f*eps0)
sigma_from_imag_permittivity(eps_imag, freq_hz)  # inverse
loss_tangent(eps_star)                          # -Im/Re of eps*
impedivity(eps_r, sigma, freq_hz)               # 1/(j*w*eps0*eps_r + sigma)
water_permittivity(rw, freq_hz, *, eps_real=55.0, eps_dl=15.0) -> complex  # brine eps* from Rw at GHz
debye(freq_hz, *, eps_inf, eps_s, tau)
cole_cole(freq_hz, *, eps_inf, eps_s, tau, alpha=0.0)   # (jwt)^(1-alpha); alpha=0 -> Debye
havriliak_negami(freq_hz, *, eps_inf, eps_s, tau, alpha=1.0, beta=1.0)  # HN convention; alpha=beta=1 -> Debye
cole_cole_resistivity(freq_hz, *, rho0, chargeability, tau, c)  # Pelton IP form
mix_power_law(fractions, eps_components, *, alpha=0.5)  # Lichtenecker-Rother; alpha=0.5=CRIM; complex-safe N-component
crim(phi, sw=1.0, *, eps_w=78.0, eps_hc=2.2, eps_matrix=5.0, alpha=0.5)  # 3-component rock forward
sw_from_permittivity(eps_meas, phi, *, eps_w, eps_hc, eps_matrix, alpha=0.5, clip=True)
bvw_from_permittivity(eps_meas, phi, *, eps_w, eps_hc, eps_matrix, alpha=0.5)  # phi*Sw, salinity-robust
water_filled_porosity(eps_meas, *, eps_matrix, eps_w, alpha=0.5, clip=True)  # 2-component simplified CRIM
maxwell_garnett(eps_host, eps_incl, f_incl, *, depol=1/3)  # complex-safe; works for conductivities
bruggeman_symmetric(fractions, eps_components)  # N-phase EMA, complex-capable root solve
hanai_bruggeman(eps_host, eps_incl, f_incl)     # asymmetric EMA, robust Newton (no abs-damping)
depolarization_spheroid(aspect) -> (Lx, Ly, Lz) # Osborn closed forms; 1 -> spheres
skin_depth(rho, freq_hz, *, mu_r=1.0)           # sqrt(2*rho/(w*mu))
induction_number(spacing_m, rho, freq_hz)       # L/delta; near/far-field discriminator
complex_wavenumber(freq_hz, sigma, eps_r, *, mu_r=1.0) -> (k_r, k_i)  # exact lossy medium
sigma_eps_from_wavenumber(k_r, k_i, freq_hz, *, mu_r=1.0) -> (sigma, eps_r)  # closed-form inverse
phase_shift_deg(rho, freq_hz, spacing_m)        # induction-limit L/delta
attenuation_db(rho, freq_hz, spacing_m)         # 8.686*L/delta
resistivity_from_phase(phase_deg, freq_hz, spacing_m)  # induction-limit inverse
attenuation_phase_from_voltages(v_near, v_far) -> (atten_db, phase_deg)  # 20log10|ratio|, angle
anisotropy_coefficient(rh, rv)                  # lambda = sqrt(Rv/Rh) (canonical)
apparent_resistivity_dip(rh, rv, dip_deg)       # rh*sqrt(cos^2 + (rv/rh)*sin^2)
doll_radial_geometric_factor(r, spacing_m)      # G = r^2/(r^2+(L/2)^2)
apparent_conductivity_two_zone(sigma_xo, sigma_t, r_invasion, spacing_m)

### depth_imaging

petrolib/depth_matching.py
- dtw(ref, target, *, band=None, cost="sqdiff", root=False) -> DtwResult(distance, path, D): banded DTW, path always backtracked; cost in {"sqdiff","absdiff"}; root=True gives sqrt-of-sum variants.
- rddtw(ref, target, *, tau=4.0, lam=0.5, band=None) -> DtwResult: regularized derivative DTW (EWRF penalty).
- cow(ref, target, *, n_segments=10, slack=8) -> (aligned, warp): correlation-optimised warping.
- warp_to_reference(target, path, n_ref, *, reduce="mean") -> ndarray: collapse DTW path to aligned curve.
- path_depth_shifts(path, depth_ref, depth_target) -> (depths, shifts): per-depth correction from path.
- xcorr_shift(ref, target, *, max_lag=50, edge="trim", return_curve=False) -> ShiftResult(lag, corr, curve): integer-lag bulk shift by max Pearson corr; edge in {"trim","wrap"}.
- xcorr_shift_depth(depth_a, a, depth_b, b, *, max_shift=2.0, step=0.125, use_abs_corr=False) -> (shift, corr): physical-unit core-to-log homing via interpolation; b may be (n,k) multi-curve.
- local_shifts(ref, target, *, window=60, step=30, max_lag=20) -> ndarray: windowed non-wrapping lag profile.
- apply_integer_shift(log, lag, *, fill=nan) -> ndarray; apply_depth_shift(values, depth, shift) -> ndarray (continuous, interp).

petrolib/depth_correction.py
- elastic_stretch(force, length, area, *, E=2.0e11): point-load F*L/(E*A).
- distributed_stretch(length, weight_per_length, ea, *, end_load=0.0, buoyancy=1.0): hanging string (F*L + 0.5*w*L^2)/EA.
- thermal_elongation(length, dT, *, alpha=1.2e-5): alpha*L*dT.
- cable_tension(depth, total_depth, tool_weight, cable_weight_per_length): tension profile.
- corrected_depth(measured, *, stretch=0.0, thermal=0.0, convention="tally"): "tally" adds, "payout" subtracts.

petrolib/borehole_image.py
- bed_sinusoid(azimuth_deg, z0, radius, dip_deg, dip_azimuth_deg): depth trace of planar bed.
- fit_sinusoid(azimuth_deg, z, *, mask=None) -> (z0, amplitude, phase_deg): lstsq on [1, cos, sin].
- dip_from_amplitude(amplitude, radius, *, sample_spacing=1.0) -> deg: arctan(A*spacing/r).
- apparent_dip(true_dip_deg, section_azimuth_deg) -> deg: tan(app)=tan(true)*cos(beta).
- fit_plane(points_ENZ) -> (dip_deg, dip_azimuth_deg): SVD plane fit.
- otsu_threshold(image, *, bins=256) -> float on native data range.
- class_fractions(image, thresholds); phase_saturation(phase, pore); porosity_from_mask(pore): accept bool masks or counts.
- glcm(image, *, levels=16, offset=(0,1), symmetric=True) -> P; glcm_features(P) -> {contrast, energy, correlation}.
- sobel_gradient(image) -> (gx, gy, magnitude).

petrolib/wellbore_geometry.py
- dogleg_angle(inc1, azi1, inc2, azi2) -> rad; minimum_curvature_step(md1, inc1, azi1, md2, inc2, azi2) -> (dTVD, dN, dE).
- survey_to_path(md, inc, azi) -> (n,3) TVD/N/E, vectorized.
- md_to_tvd(md, inc_deg, *, method="min_curvature") -> ndarray; method "tangential" reproduces legacy sum(dMD*cos(inc)).

### nuclear

petrolib/nuclear.py — numpy-vectorized, keyword params; Sigma in c.u. (1 c.u.=1e-3/cm), K wt%, U/Th ppm, time us.

# Capture cross section (PNC)
sigma_forward(phi, sw, *, sigma_ma=10, sigma_w=55, sigma_hc=21, vsh=0, sigma_sh=27) - volumetric mixing law
sigma_forward_3phase(phi, so, sg, sw, *, sigma_oil=22, sigma_gas=8, sigma_w=80, sigma_ma=10)
sw_from_sigma(sigma_t, phi, *, sigma_ma, sigma_w, sigma_hc, vsh=0, clip=(0,1)) - invert for Sw
delta_sw_timelapse(sigma_base, sigma_mon, phi, *, sigma_w_base, sigma_w_mon, sigma_hc, sw_base)
sigma_sensitivity(phi, sigma_w, sigma_hc) - dSigma/dSw
sigma_w_from_salinity(ppm_nacl, *, temperature_c=75, model='fitz2023')
number_density(rho_g_cc, wfrac, atomic_mass) - rho*NA*w/A
macroscopic_sigma(number_densities, micro_barns, *, units='cu')
sigma_from_tau(tau_us); tau_from_sigma(sigma_cu) - 4550/x
pnc_decay(t_us, n0, sigma_cu, *, background=0) - N0*exp(-Sigma*v*t)+bg
sigma_from_decay_fit(t_us, counts, *, fit_window=None) - log-linear fit

# Attenuation
beer_lambert(i0, mu, x); mu_from_intensity(i0, i, x) - ln(I0/I)/x
attenuation_map(i, i_ref) - -ln(I/I_ref), zero-guarded
gamma_count(rho, *, n0=1e6, mu_mass=0.06, spacing_cm=30); density_from_count(...) - inverse

# Density logging
dual_detector_density(near, far, *, a, b) - a + b*ln(near/far)
spine_ribs(rho_ls, rho_ss, *, rib_coeffs=(0,1,0)) -> (rho_b, drho)
electron_density_index(z, a, rho_m); electron_density_mixture(z, a, mass_frac, rho_m)
rhob_from_rhoe(rho_e) - 1.0704*x-0.1883; rhoe_from_rhob(rho_b) - exact inverse

# Gamma ray
gr_api(k_pct, u_ppm, th_ppm, *, coeff=(16,8,4)) - SGR
cgr_api(k_pct, th_ppm, *, coeff=(16,4)) - uranium-free CGR

# Neutron / hydrogen index
hydrogen_index_fluid(fluid='water', *, rho_gas=0.2, hi_oil=0.9) -> HI (not HI*phi)
hydrogen_index_chemical(rho, n_protons, mol_weight)
hi_mix(phi, *, hi_fluid=1, hi_matrix=0); phi_from_hi(hi_log, *, hi_matrix, hi_fluid=1)
phi_hi_correction(phi_apparent, hi) - phi/HI (neutron & NMR)
collision_parameter(A); average_lethargy_gain(A); moderating_power(A, sigma_s)
slowing_down_length_empirical(phi, e_mev, *, ls0=20, e_ref=4.5) + phi_from_ls inverse
transport_length_mix(vol_fracs, lengths) - harmonic Ls/Lm mix
phi_n_from_lm(lm_star, *, lithology='limestone')
compensated_neutron_porosity(near, far, *, a=0, b=-30, c=45) - a+b*lnR+c*lnR^2, p.u.

# C/O and spectroscopy
co_ratio(c_yield, o_yield); so_from_co(cor, cor_water, cor_oil) - endpoint interp, clipped
co_forward_3phase(phi, so, sg, sw, *, c_oil=0.85, c_gas=0.75, c_mat=0.1, o_w=1, o_mat=0.55)
yields_to_weights(fy2w, s, y); weights_to_yields(fy2w, s, w); toc_from_yield(y_toc, calib)

# Counting stats / decay
counting_precision(counts) - relative 1/sqrt(N); counting_sigma(value, counts) - absolute
radioactive_decay(n0, t, half_life); decay_constant(half_life); activity(n, half_life)


---

## Appendix B — Per-directory inventory

Module counts exclude test runners and `__init__.py`. "(pkg)" marks the five directories
that are already Python packages; a bracketed name marks a nonstandard test runner.
Dominant domains are the top module-tag categories from the inventory.

| Directory | Issue | Modules | LOC | Deps beyond numpy | Dominant domains |
|---|---|---:|---:|---|---|
| `src2014_02` | Vol. 55 No. 1 (Feb 2014) | 5 | 1767 | — | saturation_resistivity, capillary_pressure, relperm_wettability |
| `src2014_04` | Vol. 55 No. 2 (Apr 2014) - Special Issue on Deepwater | 6 | 1684 | — | geochem_fluids, integrity_drilling, other |
| `src2014_06` | Vol. 55 No. 3 (Jun 2014) | 5 | 856 | — | porosity_lithology, nmr, acoustic_geomech |
| `src2014_08` | Vol. 55 No. 4 (Aug 2014) | 6 | 965 | — | relperm_wettability, flow_transport, capillary_pressure |
| `src2014_10` | Vol. 55 No. 5 (Oct 2014) | 6 | 1013 | — | saturation_resistivity, inversion_numerics, geochem_fluids |
| `src2014_12` | Vol. 55 No. 6 (Dec 2014) | 6 | 1147 | — | em_dielectric, saturation_resistivity, inversion_numerics |
| `src2015_02` | Vol. 56 No. 1 (Feb 2015) | 5 | 663 | — | flow_transport, porosity_lithology, relperm_wettability |
| `src2015_04` | Vol. 56 No. 2 (Apr 2015) | 5 | 649 | — | flow_transport, porosity_lithology, data_qc_io |
| `src2015_06` | Vol. 56 No. 3 (Jun 2015) | 4 | 568 | — | geochem_fluids, nmr, porosity_lithology |
| `src2015_08` | Vol. 56 No. 4 (Aug 2015) | 5 | 713 | — | geochem_fluids, nuclear, nmr |
| `src2015_10` | Vol. 56 No. 5 (Oct 2015) | 6 | 852 | — | porosity_lithology, acoustic_geomech, flow_transport |
| `src2015_12` | Vol. 56 No. 6 (Dec 2015) | 4 | 528 | — | saturation_resistivity, nmr, flow_transport |
| `src2016_02` | Vol. 57 No. 1 (Feb 2016) | 5 | 801 | — | flow_transport, relperm_wettability, saturation_resistivity |
| `src2016_04` | Vol. 57 No. 2 (Apr 2016) | 5 | 828 | — | porosity_lithology, geochem_fluids, ml_stats |
| `src2016_06` | Vol. 57 No. 3 (Jun 2016) | 5 | 726 | — | flow_transport, capillary_pressure, saturation_resistivity |
| `src2016_08` | Vol. 57 No. 4 (Aug 2016) | 5 | 776 | — | nmr, nuclear, flow_transport |
| `src2016_10` | Vol. 57 No. 5 (Oct 2016) | 5 | 846 | — | saturation_resistivity, em_dielectric, porosity_lithology |
| `src2016_12` | Vol. 57 No. 6 (Dec 2016) | 6 | 926 | — | acoustic_geomech, depth_imaging, nmr |
| `src2017_02` | Vol. 58 No. 1 (Feb 2017) | 5 | 524 | — | relperm_wettability, flow_transport, capillary_pressure |
| `src2017_04` | Vol. 58 No. 2 (Apr 2017) | 5 | 587 | — | nmr, geochem_fluids, ml_stats |
| `src2017_06` | Vol. 58 No. 3 (Jun 2017) | 6 | 684 | — | porosity_lithology, nmr, capillary_pressure |
| `src2017_08` | Vol. 58 No. 4 (Aug 2017) | 6 | 676 | — | nmr, porosity_lithology, relperm_wettability |
| `src2017_10` | Vol. 58 No. 5 (Oct 2017) | 5 | 588 | — | saturation_resistivity, porosity_lithology, relperm_wettability |
| `src2017_12` | Vol. 58 No. 6 (Dec 2017) | 6 | 693 | — | porosity_lithology, saturation_resistivity, data_qc_io |
| `src2018_02` | Vol. 59 No. 1 (Feb 2018) | 10 | 1005 | — | flow_transport, saturation_resistivity, porosity_lithology |
| `src2018_04` | Vol. 59 No. 2 (Apr 2018) | 10 | 1035 | — | porosity_lithology, saturation_resistivity, em_dielectric |
| `src2018_06` | Vol. 59 No. 3 (Jun 2018) | 9 | 1045 | — | saturation_resistivity, em_dielectric, porosity_lithology |
| `src2018_08` | Vol. 59 No. 4 (Aug 2018) - Special Issue on Flow Diagnostics | 9 | 1148 | — | flow_transport, acoustic_geomech, integrity_drilling |
| `src2018_10` | Vol. 59 No. 5 (Oct 2018) - "Best of 2018 SPWLA Symposium" issue | 11 | 1136 | — | porosity_lithology, flow_transport, ml_stats |
| `src2018_12` | Vol. 59 No. 6 (Dec 2018) — Special Issue: Data-Driven Analytics in Logging and Petrophysics | 11 | 1206 | — | ml_stats, acoustic_geomech, depth_imaging |
| `src2019_02` | Vol. 60 No. 1 (Feb 2019) | 11 | 1177 | — | depth_imaging, integrity_drilling, porosity_lithology |
| `src2019_04` | Vol. 60 No. 2 (Apr 2019) | 12 | 1301 | — | relperm_wettability, flow_transport, inversion_numerics |
| `src2019_06` | Vol. 60 No. 3 (Jun 2019) | 9 | 1050 | scipy | porosity_lithology, inversion_numerics, geochem_fluids |
| `src2019_08` | Vol. 60 No. 4 (Aug 2019) | 6 | 743 | — | relperm_wettability, ml_stats, geochem_fluids |
| `src2019_10` | Vol. 60 No. 5 (Oct 2019) | 10 | 1185 | — | porosity_lithology, ml_stats, flow_transport |
| `src2019_12` | Vol. 60 No. 6 (Dec 2019) | 10 | 1260 | — | acoustic_geomech, inversion_numerics, saturation_resistivity |
| `src2020_02` | Vol. 61 No. 1 (Feb 2020) | 6 | 766 | — | inversion_numerics, saturation_resistivity, em_dielectric |
| `src2020_04` | Vol. 61 No. 2 (Apr 2020) | 9 | 1007 | scipy | flow_transport, relperm_wettability, capillary_pressure |
| `src2020_06` | Vol. 61 No. 3 (Jun 2020) | 5 | 679 | — | nuclear, relperm_wettability, ml_stats |
| `src2020_08` | Vol. 61 No. 4 (Aug 2020) | 4 | 689 | — | acoustic_geomech, integrity_drilling, saturation_resistivity |
| `src2020_10` | Vol. 61 No. 5 (Oct 2020) | 7 | 1106 | — | ml_stats, acoustic_geomech, integrity_drilling |
| `src2020_12` | Vol. 61 No. 6 (Dec 2020) | 7 | 909 | — | nuclear, saturation_resistivity, geochem_fluids |
| `src2021_02` | Vol. 62 No. 1 (Feb 2021) | 9 | 1364 | scipy | geochem_fluids, flow_transport, acoustic_geomech |
| `src2021_04` | Vol. 62 No. 2 (Apr 2021) | 5 | 716 | — | nmr, porosity_lithology, flow_transport |
| `src2021_06` | Vol. 62 No. 3 (Jun 2021) | 6 | 920 | — | other, ml_stats, acoustic_geomech |
| `src2021_08` | Vol. 62 No. 4 (Aug 2021) | 8 | 1176 | — | saturation_resistivity, acoustic_geomech, capillary_pressure |
| `src2021_10` | Vol. 62 No. 5 (Oct 2021) | 9 | 1170 | — | porosity_lithology, acoustic_geomech, other |
| `src2021_12` | Vol. 62 No. 6 (Dec 2021) | 10 | 1579 | — | ml_stats, integrity_drilling, porosity_lithology |
| `src2022_02` | Vol. 63 No. 1 (Feb 2022) | 6 | 1284 | — | geochem_fluids, flow_transport, depth_imaging |
| `src2022_04` | Vol. 63 No. 2 (Apr 2022) | 7 | 1067 | — | integrity_drilling, ml_stats, flow_transport |
| `src2022_06` | Vol. 63 No. 3 (Jun 2022) — NMR SIG Special Issue | 11 | 1615 | scipy | nmr, ml_stats, flow_transport |
| `src2022_08` | Vol. 63 No. 4 (Aug 2022) | 5 | 894 | sklearn | ml_stats, em_dielectric, porosity_lithology |
| `src2022_10` | Vol. 63 No. 5 (Oct 2022) | 5 | 864 | scipy | flow_transport, capillary_pressure, inversion_numerics |
| `src2022_12` | Vol. 63 No. 6 (Dec 2022) — Best Papers of the 2022 SPWLA Annual Symposium special issue | 7 | 1312 | — | acoustic_geomech, inversion_numerics, ml_stats |
| `src2023_02` | Vol. 64 No. 1 (Feb 2023) | 9 | 1762 | sklearn | flow_transport, porosity_lithology, depth_imaging |
| `src2023_04` | Vol. 64 No. 2 (Apr 2023) — SPWLA AI/ML Special Issue | 11 | 2120 | scipy, sklearn, skimage, xgboost | ml_stats, inversion_numerics, porosity_lithology |
| `src2023_06` | Vol. 64 No. 3 (June 2023) — Best Papers of the 2022 SCA International Symposium | 9 | 1641 | scipy | flow_transport, saturation_resistivity, data_qc_io |
| `src2023_08` [run_all_tests.py] | Vol. 64 No. 4 (Aug 2023) | 6 | 1662 | scipy | saturation_resistivity, inversion_numerics, porosity_lithology |
| `src2023_10` [run_all_tests.py] | Vol. 64 No. 5 (Oct 2023) — Energy Transition Special Issue | 11 | 2104 | — | flow_transport, data_qc_io, geochem_fluids |
| `src2023_12` [run_all.py] | Vol. 64 No. 6 (Dec 2023) | 8 | 549 | scipy, torch | saturation_resistivity, inversion_numerics, geochem_fluids |
| `src2024_02` [run_all_tests.py] | Vol. 65 No. 1 (Feb 2024) | 7 | 394 | scipy, sklearn | ml_stats, geochem_fluids, acoustic_geomech |
| `src2024_04` | Vol. 65 No. 2 (Apr 2024) | 6 | 495 | scipy, sklearn, torch | ml_stats, porosity_lithology, saturation_resistivity |
| `src2024_06` | Vol. 65 No. 3 (Jun 2024) | 8 | 1941 | scipy, sklearn | saturation_resistivity, flow_transport, porosity_lithology |
| `src2024_08` | Vol. 65 No. 4 (Aug 2024) — Special Issue on Advancements in Mud Logging | 14 | 3601 | scipy, sklearn | geochem_fluids, ml_stats, porosity_lithology |
| `src2024_10` (pkg) | Vol. 65 No. 5 (Oct 2024) | 10 | 3694 | scipy | flow_transport, data_qc_io, saturation_resistivity |
| `src2024_12` [test_all_modules.py] | Vol. 65 No. 6 (Dec 2024) | 13 | 2472 | — | ml_stats, integrity_drilling, geochem_fluids |
| `src2025_02` | Vol. 66 No. 1 (Feb 2025) | 12 | 1438 | scipy | relperm_wettability, flow_transport, capillary_pressure |
| `src2025_04` | Vol. 66 No. 2 (Apr 2025) | 9 | 3394 | scipy | porosity_lithology, acoustic_geomech, integrity_drilling |
| `src2025_06` [run_all_tests.py] | Vol. 66 No. 3 (June 2025) | 8 | 2930 | scipy | porosity_lithology, flow_transport, inversion_numerics |
| `src2025_08` | Vol. 66 No. 4 (Aug 2025) — Special Issue on Well Integrity | 11 | 3327 | scipy | integrity_drilling, ml_stats, em_dielectric |
| `src2025_10` | Vol. 66 No. 5 (Oct 2025) | 11 | 2735 | scipy | acoustic_geomech, flow_transport, porosity_lithology |
| `src2025_12` (pkg) | Vol. 66 No. 6 (Dec 2025) — Best Papers of the 2024 SCA International Symposium | 13 | 4462 | scipy, torch | relperm_wettability, flow_transport, geochem_fluids |
| `src2026_02` (pkg) | Vol. 67 No. 1 (Feb 2026) — SPWLA 66th Annual Symposium best papers | 11 | 4327 | — | porosity_lithology, geochem_fluids, inversion_numerics |
| `src2026_04` (pkg) | Vol. 67 No. 2 (Apr 2026) | 12 | 6060 | scipy | ml_stats, flow_transport, acoustic_geomech |
| `src2026_06` (pkg) | Vol. 67 No. 3 (June 2026) — "Best Petrophysics Papers From MEOS GEO 2025" | 10 | 2880 | — | em_dielectric, porosity_lithology, saturation_resistivity |
