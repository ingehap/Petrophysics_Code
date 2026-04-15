"""
test_all.py
============
Run the ``test_all()`` synthetic-data test of every article module in
the Petrophysics Vol. 64, No. 3 (June 2023) issue collection.

Each article module is a self-contained Python file that can also be
run individually as a script:

    python article1_hdt.py
    python article2_wiri.py
    ...

Running this orchestrator imports every module and calls its
``test_all()`` function in turn.  A summary is printed at the end.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback


MODULES = [
    ("article1_hdt", "Hybrid Drainage Technique - Fernandes et al."),
    ("article2_wiri", "WiRI vs UFPCRI vs PP - Danielczick et al."),
    ("article3_overburden_frf_ri", "FRF / RI at overburden - Nourani et al."),
    ("article4_gas_trapping", "Gas trapping in sandstone - Gao et al."),
    ("article5_shale_t1t2star", "Shale T1-T2* MR - Zamiri et al."),
    ("article6_ultrasonic_reflection",
     "Angle-dependent ultrasonic reflection - Olszowska et al."),
    ("article7_dielectric_nmr",
     "NMR-mapped dielectric dispersion - Funk et al."),
    ("article8_thz_porosity", "THz microporosity imaging - Eichmann et al."),
    ("article9_xray_invasion",
     "X-ray invasion / mudcake - Aerens et al."),
]


def main() -> int:
    print("=" * 70)
    print("Petrophysics Vol. 64 No. 3 (June 2023) - run all module tests")
    print("=" * 70)

    results = []
    for mod_name, description in MODULES:
        print()
        print(f">>> {mod_name}: {description}")
        print("-" * 70)
        t0 = time.time()
        try:
            mod = importlib.import_module(mod_name)
            mod.test_all()
            dt = time.time() - t0
            results.append((mod_name, True, dt, None))
        except Exception as exc:               # noqa: BLE001
            dt = time.time() - t0
            traceback.print_exc()
            results.append((mod_name, False, dt, str(exc)))

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'module':<40s} {'status':<8s} {'time (s)':>8s}")
    print("-" * 70)
    n_ok = 0
    for name, ok, dt, _ in results:
        status = "OK" if ok else "FAIL"
        if ok:
            n_ok += 1
        print(f"{name:<40s} {status:<8s} {dt:>8.3f}")
    print("-" * 70)
    print(f"{n_ok}/{len(results)} modules passed.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
