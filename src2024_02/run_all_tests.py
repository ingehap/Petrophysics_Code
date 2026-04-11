"""Run test_all() for every article module in this package.

Petrophysics, Vol. 65, No. 1, February 2024 (SPWLA)
https://www.spwla.org/SPWLA/Publications/Journals/Recent_Petrophysics_Journals.aspx
"""
import importlib
import sys

MODULES = [
    "article1_waxman_smits_dual_water",
    "article2_contamination_transient",
    "article3_co2_storage",
    "article4_least_squares",
    "article5_granite_thermal",
    "article6_ml_contest",
    "article7_dtw_rockmech",
]


def test_all():
    failed = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            mod.test_all()
        except Exception as e:
            failed.append((name, repr(e)))
            print(f"  !! {name} FAILED: {e}")
    print("\n==", len(MODULES) - len(failed), "/", len(MODULES), "modules passed ==")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    test_all()
