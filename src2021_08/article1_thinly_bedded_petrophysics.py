"""
Article 1: Tutorial - Petrophysics of Thinly Bedded Formations
Aldred (2021)
DOI: 10.30632/PJV62N4-2021t1

A tutorial on evaluating thinly bedded (laminated) reservoirs, where
conventional directionless resistivity overestimates water saturation
because conductive shale laminae short-circuit the deep measurement and hide
resistive hydrocarbon-bearing sand laminae (low-resistivity / low-contrast
pay).  Horizontal/vertical resistivity (Rh, Rv) feed parallel/series resistor
models and an anisotropy analysis.

Implements:

  - Parallel (horizontal) conductivity  1/Rh = sum(v_i / R_i)     (Eq. 2)
  - Series (vertical) resistivity        Rv = sum(v_i * R_i)       (Eq. 3)
  - Anisotropy coefficient  lambda = sqrt(Rv / Rh)
  - Moran-Gianzero apparent resistivity vs relative dip            (Eq. 1)
  - Sand resistivity inverted from Rh (parallel) and from Rv (series)
  - Thomeer capillary-pressure mercury-injection curve             (Eq. 4)

Note: the journal's typeset equations are image-rendered and were not in the
machine-readable text; the parallel/series forms here are reconstructed and
verified against the paper's own worked numbers (Vshl=0.5, Rh_sh=1, Rh=1.8 ->
Rss=10; Rv=5.5 -> Rss=10).  Resistivities in ohm-m, dips in degrees.
"""

import numpy as np


# ---------------------------------------------- Eqs. 2-3: laminated R ---

def horizontal_resistivity(fractions, resistivities):
    """Parallel resistor model: 1/Rh = sum(v_i / R_i)  (Eq. 2)."""
    v = np.asarray(fractions, float)
    R = np.asarray(resistivities, float)
    return 1.0 / np.sum(v / R)


def vertical_resistivity(fractions, resistivities):
    """Series resistor model: Rv = sum(v_i * R_i)  (Eq. 3)."""
    v = np.asarray(fractions, float)
    R = np.asarray(resistivities, float)
    return float(np.sum(v * R))


def anisotropy_coefficient(Rv, Rh):
    """Coefficient of anisotropy  lambda = sqrt(Rv / Rh)  (>= 1)."""
    return np.sqrt(Rv / Rh)


# ---------------------------------------------- Eq. 1: Moran-Gianzero ---

def moran_gianzero_apparent(Rh, Rv, dip_deg):
    """Apparent resistivity vs relative dip (Eq. 1).

    Ra = Rh * sqrt(cos^2(theta) + lambda^2 * sin^2(theta)), lambda=sqrt(Rv/Rh).
    Ra = Rh at theta=0; Ra = sqrt(Rh*Rv) (geometric mean) at theta=90 deg.
    """
    lam = anisotropy_coefficient(Rv, Rh)
    t = np.radians(dip_deg)
    return Rh * np.sqrt(np.cos(t) ** 2 + lam ** 2 * np.sin(t) ** 2)


# ---------------------------------------------- sand resistivity --------

def sand_resistivity_from_parallel(Rh, Vshl, Rh_sh):
    """Invert the parallel model for sand resistivity (sensitive to Vshl).

    Rss = (1 - Vshl) / (1/Rh - Vshl/Rh_sh).
    """
    return (1.0 - Vshl) / (1.0 / Rh - Vshl / Rh_sh)


def sand_resistivity_from_series(Rv, Vshl, Rv_sh):
    """Invert the series model for sand resistivity (robust to Vshl).

    Rss = (Rv - Vshl*Rv_sh) / (1 - Vshl).
    """
    return (Rv - Vshl * Rv_sh) / (1.0 - Vshl)


# ---------------------------------------------- Eq. 4: Thomeer ----------

def thomeer_bulk_volume(Pc, Binf, G, Pd):
    """Thomeer mercury-injection bulk volume  Bv = Binf*exp(-G/ln(Pc/Pd)) (Eq. 4)."""
    Pc = np.asarray(Pc, float)
    return np.where(Pc > Pd, Binf * np.exp(-G / np.log(Pc / Pd)), 0.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Tutorial - Petrophysics of Thinly Bedded Formations")
    print("=" * 60)

    # Worked example: 50% laminar shale (1 ohm-m), 50% sand (10 ohm-m)
    Vshl, Rh_sh, Rss_true = 0.50, 1.0, 10.0
    Rh = horizontal_resistivity([Vshl, 1 - Vshl], [Rh_sh, Rss_true])
    Rv = vertical_resistivity([Vshl, 1 - Vshl], [Rh_sh, Rss_true])
    print(f"  Rh (parallel)          = {Rh:.3f} ohm-m  (paper rounds to 1.8)")
    print(f"  Rv (series)            = {Rv:.3f} ohm-m  (expect 5.5)")
    assert abs(Rh - 1.8182) < 1e-3 and abs(Rv - 5.5) < 1e-9

    # Anisotropy
    lam = anisotropy_coefficient(Rv, Rh)
    print(f"  anisotropy lambda      = {lam:.3f}")
    assert abs(lam - np.sqrt(Rv / Rh)) < 1e-12 and lam > 1.0
    assert Rv >= Rh                      # always true for laminated media

    # Moran-Gianzero: Rh at 0 deg, geometric mean at 90 deg, monotonic
    assert abs(moran_gianzero_apparent(Rh, Rv, 0) - Rh) < 1e-9
    ra90 = moran_gianzero_apparent(Rh, Rv, 90)
    print(f"  Ra at 90 deg dip       = {ra90:.3f}  (= geometric mean)")
    assert abs(ra90 - np.sqrt(Rh * Rv)) < 1e-9

    # Sand-resistivity inversion: series route is far more stable to Vshl error
    rss_par = sand_resistivity_from_parallel(Rh, Vshl, Rh_sh)
    rss_ser = sand_resistivity_from_series(Rv, Vshl, Rh_sh)
    print(f"  Rss from parallel/series = {rss_par:.2f} / {rss_ser:.2f}")
    assert abs(rss_par - 10.0) < 1e-6 and abs(rss_ser - 10.0) < 1e-6
    # +5% Vshl error: parallel explodes, series stays ~10
    par_err = sand_resistivity_from_parallel(Rh, 0.55, Rh_sh)
    ser_err = sand_resistivity_from_series(Rv, 0.55, Rh_sh)
    print(f"  Rss at Vshl=0.55       = {par_err:.0f} (par) / {ser_err:.1f} (ser)")
    assert par_err > 1000.0 and abs(ser_err - 11.0) < 0.1

    # Thomeer curve rises with capillary pressure above the entry pressure
    bv = thomeer_bulk_volume([5.0, 50.0, 500.0], Binf=20.0, G=0.3, Pd=2.0)
    assert bv[0] == bv[0] and np.all(np.diff(bv) > 0)
    print("  PASS")
    return {"Rh": Rh, "Rv": Rv, "lambda": float(lam), "Rss_series": rss_ser}


if __name__ == "__main__":
    test_all()
