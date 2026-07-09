# petrolib conventions

Rules for code in `petrolib/` and for migrating article code into it. They exist because
the duplication analysis behind [LIBRARY_MERGE_PLAN.md](LIBRARY_MERGE_PLAN.md) found that
most real hazards in this repository are *silent*: same-typed argument swaps, same-name
functions computing different math, and unit conventions baked into formulas. Each rule
below kills a class of those hazards (plan §5.2 and §9).

## API rules

1. **Keyword-only physics parameters.** Data arrays may be positional; every physical
   parameter, coefficient, and switch goes after `*`. The repo holds four positional
   orders for Archie saturation and three for capillary number — all floats, so a swap is
   silent. Never migrate a call site positionally: the facade maps positions to keywords
   once, in review.

   ```python
   def archie_sw(rt, rw, *, phi, a=1.0, m=2.0, n=2.0, clip=None): ...
   ```

2. **Distinct math never merges under one name.** Where the articles use one name for two
   algebras, the canonical API forces an explicit switch or splits the name:
   `timur_coates(..., form=)`, `klinkenberg_apparent` vs `klinkenberg_corrected`,
   `bond_index(method=, input_kind=)`, `thomeer(..., log_base=)`,
   `anisotropy_coefficient` (√(Rv/Rh)) vs `anisotropy_ratio` (Rv/Rh). If two
   implementations disagree and you cannot prove they are the same equation, they are
   different functions.

3. **No silent clipping.** Physics functions return unclipped values by default and take
   an explicit `clip=` parameter. Facades pass each article's historical clip. Hidden
   `np.clip` (or `+1e-12` guards that bias results) may not be added to canonical code.

4. **SI-first, explicit units at the edges.** Core functions are strict SI or
   unit-neutral; conversions happen at call sites via `petrolib.units.convert` or
   documented `unit=` kwargs. No unit factors baked into formulas; no constants like
   0.433 psi/ft hiding inside function bodies. Docstrings state the unit of every
   argument and return value.

5. **No baked-in field-study defaults.** Locale calibrations (a basin's `Rw`, Bakken
   Schmoker coefficients, a paper's mineral modulus) are required arguments or named
   presets — never invisible defaults. Generic textbook defaults (`m=2.0`, `n=2.0`) are
   fine when they are the overwhelming convention and the docstring says so.

6. **Vectorized, guarded, typed.** numpy-broadcastable inputs; division-by-zero policy is
   explicit (return NaN or raise — document which); full type hints (`mypy --strict`
   passes); numpydoc-style docstrings.

7. **Provenance is mandatory.** Every canonical function's docstring carries a
   `Sources:` line naming the article files it consolidates, e.g.
   `Sources: src2016_08/article3, src2019_10/article1`. This is the reverse index from
   library code back to the papers.

8. **numpy-only at import time.** `import petrolib` (and importing any submodule) must
   succeed with numpy alone — enforced by a test. scipy is used through lazy in-function
   imports with a clear `ImportError` message; sklearn/torch/xgboost/skimage stay in
   article land.

## Migration rules (the strangler workflow)

9. **Article files keep their public surface.** Directories never move; filenames never
   change; `test_all()` functions are never rewritten or loosened. A duplicated function
   keeps its name and exact historical signature in the article file; only its *body*
   becomes a delegation to petrolib. Facades are permanent.

10. **The bootstrap header** is the only other edit a converted script gets, at the very
    top after the docstring:

    ```python
    try:
        import petrolib
    except ImportError:  # bare clone, not installed
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        import petrolib
    ```

11. **Four gates on every adoption PR** (plan §8): the directory's own runner green;
    `python tools/golden_diff.py <dir>` clean; shadow equivalence via
    `petrolib.testing.assert_matches_original` at `rtol=1e-12` (fallback `1e-9` only for
    documented float reassociation, with the old body kept in the PR's test file until
    merge); and petrolib's own unit tests. Article assert tolerances are never loosened —
    a failing assert means the facade's defaults are wrong, not the test.

12. **Hazard call sites get their own one-file PRs** with a before/after numeric note.
    Batch PRs stay under ~400 diff lines, chunked by era, one duplication family at a
    time.

13. **Deliberate behavior changes** (bug fixes like the `src2023_08` Arps unit bug) are
    never mixed into refactor PRs: separate PR, golden regenerated in the same PR,
    CHANGELOG entry.

14. **Repository invariants are updated deliberately.** `EXPECTED_DIR_COUNT` and
    `EXPECTED_TEST_ALL_MODULES` in `tools/run_all_issues.py` are pinned; adding a new
    issue directory means updating them in the same PR, teaching the harness any new
    runner format, and capturing its golden.

## New issue directories (`src2026_08` and later)

Package-style: an `__init__.py`, short snake_case module names, a `test_all.py` that
imports modules by bare name (no self-renaming package aliases), and `import petrolib`
from day one instead of re-implementing shared physics. Add the directory to the
invariants and goldens as per rule 14.

## Tooling boundaries

ruff (lint + format) and `mypy --strict` apply to `petrolib/`, `tests/`, and `tools/`
only. The legacy article scripts are exempt — reformatting them is churn that pollutes
blame — and facade edits match each file's local style.
