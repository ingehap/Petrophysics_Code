"""
test_all.py — run test_all() for every article module in this package.

Each module corresponds to one article in
Petrophysics, Vol. 65, No. 2 (April 2024):

  grader_digital_rock          - Grader et al., pp. 149-157
  liu_ionic_capacitor          - Liu et al.,    pp. 158-172
  zhang_nmr_core               - Zhang et al.,  pp. 173-193
  xiong_productivity_factors   - Xiong et al.,  pp. 194-214
  lee_mwd_triple_combo         - Lee et al.,    pp. 215-232
  chen_sem_pore_segmentation   - Chen et al.,   pp. 233-245
"""
import importlib

MODULES = [
    "grader_digital_rock",
    "liu_ionic_capacitor",
    "zhang_nmr_core",
    "xiong_productivity_factors",
    "lee_mwd_triple_combo",
    "chen_sem_pore_segmentation",
]


def test_all():
    for name in MODULES:
        m = importlib.import_module(name)
        m.test_all()
    print("\nAll %d article modules passed." % len(MODULES))


if __name__ == "__main__":
    test_all()
