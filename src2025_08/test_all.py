#!/usr/bin/env python3
"""
Master Test Suite — Petrophysics August 2025 Special Issue
==========================================================
Runs test_all() for every module implementing ideas from each article
in Petrophysics, vol. 66, no. 4, August 2025 (Special Issue on
Well Integrity).

References
----------
1. Kolay et al., "From Archives to Abandonment: Applying Generative AI
   to Optimize Plug and Abandon Processes in Old Oil Wells," pp. 545–554.
2. Bazaid et al., "Pioneering Well Logging: The Role of Fiber Optics
   in Modern Monitoring for Well Integrity Diagnosis," pp. 555–565.
3. Fouda et al., "First-Ever Seven-Pipe Corrosion Evaluation for
   Comprehensive Assessment of Pipe Integrity in Complex Well
   Completions," pp. 566–577.
4. Jawed et al., "Application of Advanced Sectorial Electromagnetic
   Scanning Addressing Well Integrity Challenges," pp. 578–593.
5. Bazaid et al., "Innovative Approach to Enhance Evaluation of Well
   Integrity in Unconventional Completions With FBE Coating,"
   pp. 594–615.
6. Alatigue et al., "Revolutionizing Complex Casing Integrity Analysis
   in the Middle East Using High-Resolution Acoustic Imaging,"
   pp. 616–630.
7. Jawed et al., "Pulsed Eddy Current Logging Technology for Through-
   Tubing Casing Break Detection up to the Third Tubular," pp. 631–646.
8. Wang et al., "Automated Anomaly Detection of Multimetallic Tubulars
   in Well Integrity Logs Using Signal Mode Decomposition and
   Physics-Informed Decision Making," pp. 647–661.
9. Manh et al., "Through-Tubing Casing Deformation Inspection Based on
   Data-Driven Koopman Modeling and Ensemble Kalman Filter,"
   pp. 662–676.
10. Zeghlache et al., "Challenges and Solutions for Advanced Through-
    Tubing Cement Evaluation," pp. 677–688.
11. Sun et al., "Formation Slowness Estimation Behind Casing via a
    Time-Variant Wave Separation Method," pp. 689–700.

Usage
-----
    python test_all.py
"""

import sys
import time
import traceback

MODULES = [
    ("Article  1 — GenAI P&A Extraction",
     "pa_genai_extraction"),
    ("Article  2 — Fiber-Optic DTS/DAS Sensing",
     "fiber_optics_sensing"),
    ("Article  3 — Seven-Pipe EM Corrosion Evaluation",
     "seven_pipe_em_corrosion"),
    ("Article  4 — Sectorial EM Scanning",
     "sectorial_em_scanning"),
    ("Article  5 — FBE Cement Bond Evaluation",
     "fbe_cement_evaluation"),
    ("Article  6 — High-Res Acoustic Imaging",
     "acoustic_imaging"),
    ("Article  7 — Pulsed Eddy Current Break Detection",
     "pulsed_eddy_current"),
    ("Article  8 — Anomaly Detection (HMVMD + Bayes)",
     "anomaly_detection_vmd"),
    ("Article  9 — Koopman + EnKF Deformation",
     "koopman_enkf_deformation"),
    ("Article 10 — SNHR / EMI Cement Evaluation",
     "cement_snhr_emi"),
    ("Article 11 — Time-Variant Wave Separation",
     "wave_separation_slowness"),
]


def main():
    print("=" * 70)
    print("  Petrophysics August 2025 — Full Test Suite")
    print("  (Special Issue on Well Integrity)")
    print("=" * 70)

    passed, failed, errors = 0, 0, []
    t0 = time.time()

    for title, module_name in MODULES:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"  Module: {module_name}")
        print(f"{'─' * 60}")
        try:
            mod = __import__(module_name)
            mod.test_all()
            passed += 1
        except Exception:
            failed += 1
            tb = traceback.format_exc()
            errors.append((title, tb))
            print(f"  [FAIL] {module_name}\n{tb}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed} passed, {failed} failed  "
          f"({elapsed:.1f} s)")
    print(f"{'=' * 70}")

    if errors:
        print("\nFailures:")
        for title, tb in errors:
            print(f"  • {title}")
        sys.exit(1)
    else:
        print("\n  All 11 modules passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
