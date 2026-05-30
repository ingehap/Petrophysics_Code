"""
Master test runner for the Petrophysics October 2021 (Vol. 62, No. 5) issue -
the special issue on "Applications of 3D Printing and Synthetic Rocks in
Petrophysics, Rock Physics, and Rock Mechanics".

Runs the test_all() function of every article module and prints a summary.
Run from this directory:

    python test_all.py

Or run a single article:

    python article5_fractal_digital_rock.py
"""

import importlib
import time
import traceback


MODULES = [
    ("Article 1 - Binder Saturation -> Porosity",            "article1_binder_saturation_porosity"),
    ("Article 2 - Image Processing for Petrophysics",        "article2_image_processing_petrophysics"),
    ("Article 3 - Original-Size Carbonate Pore Replication", "article3_carbonate_pore_replication"),
    ("Article 4 - 3D-Printed Mudrock Micromodels",           "article4_3dprint_mudrock_micromodel"),
    ("Article 5 - Fractal Characterization of Digital Rocks","article5_fractal_digital_rock"),
    ("Article 6 - Pore-Volume Compressibility of Sand",      "article6_pore_volume_compressibility"),
    ("Article 7 - Fluids in Anisotropic 3D-Printed Rock",    "article7_3dprint_anisotropic_elastic"),
    ("Article 8 - Joint Roughness & Shear (3D-printed)",     "article8_joint_roughness_shear"),
    ("Article 9 - Perforation Fracture Morphology (proxy)",  "article9_perforation_fracture_morphology"),
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
