"""
Article 4: Maintaining and Reconstructing In-Situ Saturations: A Comparison
           Between Whole Core, Sidewall Core, and Pressurized Sidewall Core in
           the Permian Basin
Blount, McMullen, Durand, Driskill (2019)
DOI: 10.30632/PJV60N1Y2019a3

Core-derived saturations depend on how much fluid is lost between the reservoir
and the laboratory.  Dean-Stark extraction measures retained water and oil
volumes; conventional whole and sidewall cores lose light hydrocarbon and gas
during retrieval, biasing the saturation, whereas pressurized sidewall cores
retain the fluids.  A mass balance reconstructs the in-situ saturation from the
measured volumes and a fluid-loss correction.

Implements:

  - Dean-Stark water / oil saturation from extracted volumes
  - Fluid-loss correction by core type (whole / sidewall / pressurized)
  - Mass-balance reconstruction of in-situ saturation
  - Saturation closure check (Sw + So + Sg = 1)

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the Dean-Stark / fluid-loss saturation reconstruction the paper compares.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Dean-Stark --------------

def dean_stark_saturation(v_water, v_oil, pore_volume):
    """Water and oil saturation from Dean-Stark extracted volumes."""
    _, sw, so = petrolib.geochem_fluids.core_geochem.dean_stark(v_water, v_oil, v_pore=pore_volume)
    return float(sw), float(so)


def fluid_loss_factor(core_type):
    """Fraction of the original light/gas fluid retained by core type.

    Pressurized sidewall retains nearly all; whole and conventional sidewall
    cores lose progressively more during depressurization/retrieval.
    """
    return {"pressurized": 0.98, "whole": 0.85, "sidewall": 0.70}[core_type]


def reconstruct_insitu(so_measured, sg_measured, retention):
    """Reconstruct in-situ oil/gas saturation by dividing out the retention factor."""
    so_true = so_measured / retention
    sg_true = sg_measured / retention
    return float(so_true), float(sg_true)


def closure(sw, so, sg):
    """Saturation closure residual  Sw + So + Sg - 1 (should be ~0)."""
    return sw + so + sg - 1.0


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: In-Situ Saturations - Core Type Comparison")
    print("=" * 60)

    # Dean-Stark saturations from extracted volumes
    sw, so = dean_stark_saturation(v_water=2.0, v_oil=3.0, pore_volume=10.0)
    print(f"  Dean-Stark Sw / So     = {sw:.2f} / {so:.2f}")
    assert abs(sw - 0.20) < 1e-9 and abs(so - 0.30) < 1e-9

    # Pressurized core retains the most fluid; sidewall the least
    assert fluid_loss_factor("pressurized") > fluid_loss_factor("whole") \
        > fluid_loss_factor("sidewall")

    # Reconstruction recovers a higher (true) in-situ hydrocarbon saturation than
    # the loss-biased measured value, most strongly for the lossy sidewall core
    so_meas, sg_meas = 0.30, 0.05
    so_ws, sg_ws = reconstruct_insitu(so_meas, sg_meas, fluid_loss_factor("sidewall"))
    so_pr, sg_pr = reconstruct_insitu(so_meas, sg_meas, fluid_loss_factor("pressurized"))
    print(f"  So reconstructed sidewall/pressurized = {so_ws:.3f} / {so_pr:.3f}")
    assert so_ws > so_meas and so_ws > so_pr      # bigger correction for lossy core

    # Closure: a consistent saturation set sums to one
    assert abs(closure(0.30, 0.55, 0.15)) < 1e-9
    assert abs(closure(0.30, 0.50, 0.15)) > 0      # inconsistent set flagged
    print("  PASS")
    return {"sw": sw, "so_sidewall_recon": so_ws, "so_pressurized_recon": so_pr}


if __name__ == "__main__":
    test_all()
