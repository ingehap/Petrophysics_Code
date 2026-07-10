# petrolib — the common petrophysics library

`petrolib/` is the shared library extracted from the article implementations in
the `srcYYYY_MM` issue directories: one canonical, tested implementation per
petrophysics domain, replacing the ~1,550 per-article copies of ~190 duplicated
function families that the repository accumulated between Vol. 55 (2014) and
Vol. 67 (2026).

The extraction is complete: every domain planned in
[LIBRARY_MERGE_PLAN.md](../LIBRARY_MERGE_PLAN.md) is merged, and every article
function identified as a duplicate now *delegates* to petrolib while keeping
its historical name, signature, docstring, and — bit for bit — its published
numeric output.

## Design principles

- **Strangler-fig facades.** No article file was moved or renamed and no
  `test_all()` was rewritten. Inside an article, a duplicated function keeps
  its exact historical signature; only its body became a delegation to the
  canonical petrolib function.
- **Bit-exact adoption.** Every swap was verified by running the article
  before and after and byte-comparing stdout, on top of the repository's
  golden-output diff. Article forms that are *not* bit-reproducible from the
  canonical API (e.g. `+1e-10` guard variants, scipy-kernel smoothers, legacy
  `np.random.RandomState` generators) deliberately stayed local.
- **Bare-clone invariant.** `python articleN_x.py` and `python test_all.py`
  keep working from inside any issue directory with no install step: article
  files carry a small bootstrap header that puts the repository root on
  `sys.path` when petrolib is not installed.
- **numpy-only core.** `import petrolib` needs numpy alone; submodules load
  lazily (PEP 562), and the few functions that need scipy import it inside the
  function with a clear error message.
- **Corpus-grounded APIs.** Each module was written from a survey of every
  article implementation of that domain. Where the corpus disagrees on a
  convention (log10 vs ln, signed vs absolute, unit systems), the divergence
  is exposed as a keyword parameter rather than silently picking a winner.

## Installation and use

Nothing needs to be installed to run the article scripts. To use petrolib
directly:

```bash
pip install -e .          # installs petrolib (Python 3.10+, numpy only)
pip install -e ".[dev]"   # adds pytest / ruff / mypy
```

```python
import petrolib

petrolib.units.convert(7.0, "bar", "psi")            # 101.526...
petrolib.ml_stats.rmse([1.0, 2.0], [1.1, 1.9])       # 0.1
petrolib.porosity_lithology.pay_flag(
    phi=[0.05, 0.12], vsh=[0.2, 0.3], phi_cut=0.08)  # [False, True]
```

Conventions (SI-first units, keyword-only physics parameters, float64
in/float64 out, `ValueError` on unknown method names) are specified in
[CONVENTIONS.md](../CONVENTIONS.md).

## Module map

| Module | Domain |
| --- | --- |
| `petrolib.constants` | Physical and unit-conversion constants — the single source of truth |
| `petrolib.units` | Spelling-tolerant `convert()` within pressure/length/time/permeability/density/velocity/temperature families, sonic slowness adapters |
| `petrolib.ml_stats` | Error metrics, scaling, OLS/k-means/PCA and other numpy-only ML helpers |
| `petrolib.porosity_lithology` | Porosity/lithology volumetrics, shale volume, pay flags, net-to-gross |
| `petrolib.saturation_resistivity` | Archie and shaly-sand saturation/resistivity models |
| `petrolib.capillary_pressure` | Young-Laplace/Washburn, Pc curve models, Leverett scaling, MICP rock typing |
| `petrolib.relperm_wettability` | Relative permeability models and wettability indices |
| `petrolib.flow_transport` | Single-phase flow, poro-perm transforms, rock typing, diffusion |
| `petrolib.nmr` | NMR relaxation physics, T2 statistics, forward models, permeability transforms |
| `petrolib.acoustic_geomech` | Acoustic/elastic conversions, rock-physics mixing, geomechanics |
| `petrolib.geochem_fluids` (package) | Adsorption, asphaltene gradients, brine, contamination, core geochemistry, pressure gradients, mud gas, PVT, solubility |
| `petrolib.em_dielectric` | Permittivity, dispersion, and dielectric mixing laws |
| `petrolib.nuclear` | Capture cross-section, attenuation, density, GR, neutron responses |
| `petrolib.inversion_numerics` (package) | Linear/nonlinear/stochastic inversion, costs, optimization, fitting, grid PDEs |
| `petrolib.depth_matching` | DTW, cross-correlation shifts, and curve warping |
| `petrolib.depth_correction` | Cable/drill-string stretch, thermal, and tension depth corrections |
| `petrolib.borehole_image` | Bed sinusoids, dip picking, thresholding, image texture |
| `petrolib.wellbore_geometry` | Minimum-curvature surveys, dogleg, MD-to-TVD |
| `petrolib.integrity_drilling` | Cement bond, casing condition, leak rates, mud gas, pore-pressure window, mudcake |
| `petrolib.data_qc_io` (package) | Log cleaning, reference normalization, filtering, SNR/stacking, synthetic data, the Bradley wellbore JSON container |
| `petrolib.testing` | Shared test helpers |

The complete public API — every function with its one-line summary — is in
[petrolib-api.md](petrolib-api.md) (generated by
[`tools/gen_petrolib_api.py`](../tools/gen_petrolib_api.py)).

## Provenance and citations

Every function extracted from the corpus carries a `Sources:` tag naming the
`srcYYYY_MM` article file(s) it was grounded in, and every module docstring
ends with a *References* section resolving those tags to complete SPWLA
*Petrophysics* citations (title, authors, year, volume/issue/pages, DOI).
`help(petrolib.<module>)` therefore shows the full bibliography of a domain.

For a print-ready reference, the **Petrophysics Handbook**
([Petrophysics_Handbook.pdf](Petrophysics_Handbook.pdf)) documents every
public function alphabetically — purpose, input parameters, output, and full
journal sources — and is regenerated with
`python tools/gen_petrolib_handbook.py` (needs `pip install reportlab`, or
`pip install -e ".[docs]"`).

## Testing and regression gates

```bash
python -m pytest tests/petrolib        # petrolib unit tests
python tools/run_all_issues.py         # every issue directory's own suite
python tools/golden_diff.py            # printed output vs frozen baselines
ruff check && ruff format --check      # lint/format (library code only)
python -m mypy petrolib                # strict typing on the library
```

The golden diff is the load-bearing check: it re-runs all 75 issue suites and
compares their printed output against the baselines frozen in `tools/golden/`
before petrolib existed — the guarantee that the library swap changed no
published number.

## History

The migration ran as one PR train per domain — a canonical-module PR grounded
in a multi-agent corpus survey, followed by era-chunked adoption PRs — with
four regression gates on every merge. The full plan, the duplication analysis
that motivated it, and the per-domain API sketches are preserved in
[LIBRARY_MERGE_PLAN.md](../LIBRARY_MERGE_PLAN.md).
