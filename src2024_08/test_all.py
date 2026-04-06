#!/usr/bin/env python3
"""
Petrophysics August 2024 – Complete Test Suite
================================================
Runs test_all() from each of the 14 article modules, implementing ideas from:

  SPWLA Petrophysics, Vol. 65, No. 4 (August 2024)
  Special Issue on Advancements in Mud Logging

Articles and corresponding modules:
  1.  Arief & Yang (pp.433-454)    → gor_prediction_ml.py
  2.  Yang et al. (pp.455-469)     → shale_fluid_prediction.py
  3.  Kopal et al. (pp.470-483)    → realtime_fluid_id.py
  4.  Yang et al. (pp.484-495)     → standard_mudgas_typing.py
  5.  Cely et al. (pp.496-506)     → ml_fluid_typing.py
  6.  Bravo et al. (pp.507-518)    → heavy_oil_viscosity.py
  7.  Ungar et al. (pp.519-531)    → prospect_fluid_estimation.py
  8.  Caldas et al. (pp.532-547)   → pvt_gor_snorre.py
  9.  Cheng et al. (pp.548-564)    → membrane_gas_logging.py
  10. Donovan (pp.565-584)         → mudgas_response.py
  11. Qubaisi et al. (pp.585-592)  → alkene_hydrogen_dbm.py
  12. Yang et al. (pp.593-603)     → gpc_uv_cuttings.py
  13. Banks et al. (pp.604-623)    → magnetic_permeability.py
  14. Yamada et al. (pp.624-648)   → lithobia_cuttings.py

Usage:
    python test_all.py              # Run all tests
    python test_all.py --module 3   # Run only module 3

References:
    All articles published in Petrophysics, Vol. 65, No. 4, August 2024.
    DOI prefix: 10.30632/PJV65N4-2024a{N}
"""

import sys
import time
import importlib
import traceback

# Module definitions: (name, description, article reference)
MODULES = [
    ("gor_prediction_ml",
     "ML Approach to Predict GOR from AMG",
     "Arief & Yang, pp.433-454"),
    ("shale_fluid_prediction",
     "AMG Fluid Prediction in Shale Reservoirs",
     "Yang et al., pp.455-469"),
    ("realtime_fluid_id",
     "Real-Time Fluid ID from AMG + Petrophysical Logs",
     "Kopal et al., pp.470-483"),
    ("standard_mudgas_typing",
     "Standard Mud Gas Fluid Typing",
     "Yang et al., pp.484-495"),
    ("ml_fluid_typing",
     "ML-Based Fluid Typing from Standard Mud Gas",
     "Cely et al., pp.496-506"),
    ("heavy_oil_viscosity",
     "Heavy Oil Viscosity from Standard Mud Gas",
     "Bravo et al., pp.507-518"),
    ("prospect_fluid_estimation",
     "Prospect Fluid Estimation Using Mud Gas",
     "Ungar et al., pp.519-531"),
    ("pvt_gor_snorre",
     "PVT Comparison & GOR Prediction (Snorre Field)",
     "Caldas et al., pp.532-547"),
    ("membrane_gas_logging",
     "Membrane Degasser + IR Spectroscopy",
     "Cheng et al., pp.548-564"),
    ("mudgas_response",
     "Mud Gas Response Variation & Quantification",
     "Donovan, pp.565-584"),
    ("alkene_hydrogen_dbm",
     "Drill-Bit Metamorphism via Alkene & H2",
     "Qubaisi et al., pp.585-592"),
    ("gpc_uv_cuttings",
     "GPC-UV Fluid Analysis from Drill Cuttings",
     "Yang et al., pp.593-603"),
    ("magnetic_permeability",
     "Magnetic Susceptibility → Permeability",
     "Banks et al., pp.604-623"),
    ("lithobia_cuttings",
     "LiOBIA: Object-Based Cuttings Image Analysis",
     "Yamada et al., pp.624-648"),
]


def run_all_tests(selected_module: int = None):
    """Run test_all() from every module (or a single selected module)."""
    print("╔" + "═" * 70 + "╗")
    print("║  PETROPHYSICS Vol. 65, No. 4 (August 2024) – Full Test Suite     ║")
    print("║  Special Issue: Advancements in Mud Logging                      ║")
    print("╚" + "═" * 70 + "╝")
    print()

    results = []
    total_start = time.time()

    modules_to_run = MODULES
    if selected_module is not None:
        idx = selected_module - 1
        if 0 <= idx < len(MODULES):
            modules_to_run = [MODULES[idx]]
        else:
            print(f"  ERROR: Module {selected_module} not found (1-{len(MODULES)})")
            return False

    for i, (mod_name, description, reference) in enumerate(modules_to_run, 1):
        mod_num = MODULES.index((mod_name, description, reference)) + 1
        print(f"\n{'━' * 72}")
        print(f"  Module {mod_num}/14: {description}")
        print(f"  Reference: {reference}")
        print(f"{'━' * 72}")

        start = time.time()
        try:
            module = importlib.import_module(mod_name)
            success = module.test_all()
            elapsed = time.time() - start
            results.append((mod_num, mod_name, True, elapsed, ""))
        except Exception as e:
            elapsed = time.time() - start
            error_msg = traceback.format_exc()
            results.append((mod_num, mod_name, False, elapsed, str(e)))
            print(f"\n  [FAIL] {mod_name}: {e}")
            print(error_msg)

    # Summary
    total_elapsed = time.time() - total_start
    passed = sum(1 for r in results if r[2])
    failed = len(results) - passed

    print(f"\n\n{'╔' + '═' * 70 + '╗'}")
    print(f"{'║'} {'TEST SUMMARY':^68} {'║'}")
    print(f"{'╠' + '═' * 70 + '╣'}")
    print(f"{'║'} {'#':>3}  {'Module':<35} {'Status':>8}  {'Time':>8} {'║'}")
    print(f"{'║'} {'─'*3}  {'─'*35} {'─'*8}  {'─'*8} {'║'}")
    for num, name, success, elapsed, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{'║'} {num:>3}  {name:<35} {status:>8}  {elapsed:>7.2f}s {'║'}")
    print(f"{'╠' + '═' * 70 + '╣'}")
    print(f"{'║'} Passed: {passed}/{len(results)} | "
          f"Failed: {failed}/{len(results)} | "
          f"Total time: {total_elapsed:.1f}s{' ' * 16}{'║'}")
    print(f"{'╚' + '═' * 70 + '╝'}")

    return failed == 0


def test_all():
    """Entry point: run all module tests."""
    return run_all_tests()


if __name__ == "__main__":
    selected = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--module" and len(sys.argv) > 2:
            selected = int(sys.argv[2])
        elif sys.argv[1] == "--list":
            print("Available modules:")
            for i, (name, desc, ref) in enumerate(MODULES, 1):
                print(f"  {i:2}. {desc} ({ref})")
            sys.exit(0)

    success = run_all_tests(selected)
    sys.exit(0 if success else 1)
