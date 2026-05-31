"""
Master test runner for the Petrophysics October 2014 (Vol. 55, No. 5) issue -
the Best of the 2014 SPWLA Annual Logging Symposium: inversion-based
interpretation of LWD resistivity and nuclear measurements in high-angle and
horizontal wells, core-data quality control for elemental-spectroscopy log
interpretation, an assessment of nuclear-based alternatives to chemical-source
bulk density, kerogen/maturity/mineralogy and clay typing from DRIFTS, the
impact of petrophysical properties on near-wellbore nanoparticle distribution,
and an oil-movability quicklook from dielectric measurements at four depths.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_lwd_inversion_anisotropy.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Inversion-Based LWD Resistivity & Nuclear", "article1_lwd_inversion_anisotropy"),
    ("Article 2 - Elemental Spectroscopy Core Data QC",       "article2_elemental_spectroscopy_qc"),
    ("Article 3 - Nuclear Alternatives to Cs-137 Density",    "article3_nuclear_density_alternatives"),
    ("Article 4 - DRIFTS Kerogen, Mineralogy & Clay Typing",  "article4_drifts_kerogen_mineralogy"),
    ("Article 5 - Near-Wellbore Nanoparticle Transport",      "article5_nanoparticle_transport"),
    ("Article 6 - Dielectric Oil-Movability Quicklook",       "article6_dielectric_oil_movability"),
]


def run_all():
    results = {}
    for title, mod_name in MODULES:
        print()
        print("#" * 70)
        print(f"# {title}")
        print("#" * 70)
        try:
            t0 = time.time()
            mod = importlib.import_module(mod_name)
            r = mod.test_all()
            dt = time.time() - t0
            results[mod_name] = {"status": "PASS", "time": dt, "result": r}
            print(f"  -> {dt:.2f}s")
        except Exception as e:
            traceback.print_exc()
            results[mod_name] = {"status": "FAIL", "error": str(e)}
    print()
    print("=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        status = r["status"]
        extra = f"  ({r['time']:.2f}s)" if status == "PASS" else f"  ({r.get('error', '')})"
        print(f"  {status:5s}  {name}{extra}")
    n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
    print(f"\n  {n_pass}/{len(MODULES)} modules passed")
    return results


if __name__ == "__main__":
    run_all()
