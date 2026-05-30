"""
Article 6: Pore Volume Compressibility of Unconsolidated Sand Reservoirs:
           Insights Gained Using Laboratory-Created Sand Pack Analogs
Hathon, Myers, Arya (2021)
DOI: 10.30632/PJV62N5-2021a6

Production-timescale uniaxial pore-volume compressibility of unconsolidated
sands, measured on laboratory sand-pack analogs while varying one textural or
mineralogic parameter at a time.  Reports how grain size, sorting, angularity,
feldspar and ductile-grain content shift the compressibility-vs-depletion-
stress curve.

Implements:

  - Uniaxial compaction coefficient  Cm = (1/L)(dL/d sigma_a)     (Eq. 1)
  - Pore-volume compressibility       Cp = Cm / phi               (Eq. 2)
  - Trask (1930) sorting coefficient  So = sqrt(GS25 / GS75)      (Eq. 3)
  - A peaked Cm-vs-effective-stress demonstrator (Regions A/B/C)

Note: the paper's three equations were image-rendered and are reconstructed
here in standard form (Eq. 3 is the standard Trask sorting coefficient).  The
paper publishes no forward compressibility model, so the peaked curve is a
synthetic interpolant through the reported peak-Cm / peak-stress anchors.
Compressibility in microsips (1e-6 / psi); stress in psi.
"""

import numpy as np

MICROSIP = 1e-6        # 1 microsip = 1e-6 / psi


# ---------------------------------------------- Eq. 1: compaction coef. -

def compaction_coefficient(length, axial_stress):
    """Uniaxial compaction coefficient  Cm = (1/L)(dL/d sigma_a)  (Eq. 1).

    Numerical derivative from sampled (length, stress) arrays; returns
    per-step Cm at the segment midpoints (positive for compaction).
    """
    L = np.asarray(length, float)
    s = np.asarray(axial_stress, float)
    dL = -np.diff(L)                 # shortening is positive compaction
    Lmid = 0.5 * (L[1:] + L[:-1])
    ds = np.diff(s)
    return (1.0 / Lmid) * (dL / ds)


# ---------------------------------------------- Eq. 2: Cp ---------------

def pore_volume_compressibility(Cm, phi):
    """Pore-volume compressibility  Cp = Cm / phi  (Eq. 2)."""
    return np.asarray(Cm, float) / phi


# ---------------------------------------------- Eq. 3: Trask sorting ----

def trask_sorting(gs25, gs75):
    """Trask (1930) sorting coefficient  So = sqrt(GS25 / GS75)  (Eq. 3).

    GS25 and GS75 are the 25th- and 75th-percentile grain sizes (GS25 coarser),
    so So >= 1; 1.0 = very well sorted, larger = more poorly sorted.
    """
    return np.sqrt(gs25 / gs75)


# ---------------------------------------------- demonstrator curve ------

def compressibility_curve(stress, peak_cm, peak_stress, width=2500.0):
    """Peaked Cm(stress) demonstrator (microsips) through a reported anchor.

    A log-normal-like bump in effective depletion stress reproducing the
    paper's Region A (rise), B (peak), C (decline) shape.
    """
    s = np.asarray(stress, float)
    return peak_cm * np.exp(-((np.log(s) - np.log(peak_stress)) ** 2) /
                            (2.0 * (width / peak_stress) ** 2))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Pore-Volume Compressibility of Sand Packs")
    print("=" * 60)

    # Cm from a length-vs-axial-stress record
    stress = np.array([1000.0, 1500.0, 2000.0, 2500.0])
    length = np.array([2.0000, 1.9890, 1.9800, 1.9730])     # inches, shortening
    Cm = compaction_coefficient(length, stress)
    print(f"  Cm (per step)          = {(Cm/MICROSIP).round(1)} microsips")
    assert np.all(Cm > 0)

    # Eq. 2 check: Cm = 22 microsips at phi = 0.31 -> Cp ~ 71 microsips
    Cp = pore_volume_compressibility(22 * MICROSIP, 0.31)
    print(f"  Cp (Cm=22, phi=0.31)   = {Cp/MICROSIP:.1f} microsips")
    assert abs(Cp / MICROSIP - 71.0) < 0.6

    # Trask sorting: well-sorted ~ 1, poorly sorted larger
    so_well = trask_sorting(0.30, 0.30)
    so_poor = trask_sorting(0.45, 0.13)
    print(f"  Trask So well / poor   = {so_well:.2f} / {so_poor:.2f}")
    assert abs(so_well - 1.0) < 1e-9
    assert 1.5 < so_poor < 2.0

    # Demonstrator curve peaks near the reported peak stress
    s = np.logspace(np.log10(500), np.log10(10000), 60)
    cm = compressibility_curve(s, peak_cm=22.0, peak_stress=2000.0)
    s_peak = s[int(np.argmax(cm))]
    print(f"  curve peak near        = {s_peak:.0f} psi (expect ~2000)")
    assert abs(s_peak - 2000.0) < 600.0
    print("  PASS")
    return {"Cp_microsips": Cp / MICROSIP, "So_poor": so_poor,
            "peak_stress": s_peak}


if __name__ == "__main__":
    test_all()
