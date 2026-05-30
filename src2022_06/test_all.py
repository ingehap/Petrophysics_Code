"""
Master test runner for the Petrophysics June 2022 (Vol. 63, No. 3) issue.

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article1_nmf_clustering_t1t2.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1  - NMF + Hierarchical Clustering on T1-T2",  "article1_nmf_clustering_t1t2"),
    ("Article 2  - Fuzzy + GA + kNN NMR-WD",                 "article2_fuzzy_genetic_nmr"),
    ("Article 3  - NMR Logging Data Processing",             "article3_nmr_processing_toolbox"),
    ("Article 4  - BSS-ICA + D-T2 Invasion",                 "article4_bssica_dt2_invasion"),
    ("Article 5  - NPPM Pore-Size + Kozeny-Carman",          "article5_nppm_pore_size_perm"),
    ("Article 6  - DDTW NMR + Mud-Gas Integration",          "article6_ddtw_mudgas_integration"),
    ("Article 7  - Slimhole LWD NMR + Factor Analysis",      "article7_slimhole_lwd_factor"),
    ("Article 8  - 27Al MAS NMR Mineral Identification",     "article8_highfield_al_nmr"),
    ("Article 9  - T2-Imbibition Wettability Index",         "article9_t2_imbibition_wettability"),
    ("Article 10 - PCR Permeability from NMR + MICP",        "article10_pcr_nmr_micp_perm"),
    ("Article 11 - NMR Core Analysis Review (VST / SPRITE)", "article11_core_nmr_review"),
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
