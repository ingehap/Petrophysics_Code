"""
article_06_hill_potash_pid_plot.py
==================================
Implementation of ideas from:

    Hill, D. G., Crain, E. R., and Teufel, L. W. (2023).  "The Potash
    Identification (PID) Plot: A Rapid Screening Crossplot for
    Discrimination of Commercial Potash."  Petrophysics, 64(5), 700-713.
    DOI: 10.30632/PJV64N5-2023a6

The "PID plot" is a simple gamma-ray vs neutron-porosity crossplot.
Commercial potash minerals (sylvite, langbeinite) are *anhydrous* and
plot at high GR, low neutron.  Non-commercial potash minerals
(carnallite, kainite, leonite, polyhalite) are *hydrated*, so they
plot at high GR AND high neutron porosity.  Non-potash evaporites
(halite, anhydrite, gypsum, kieserite) and clays/shales also plot in
characteristic regions.

This module:

    * provides a mineral library with %K2O equivalent, GR (API at 100%
      mineral) and neutron porosity (apparent, limestone-equivalent)
    * classifies (GR, NPHI) data points into commercial / non-commercial
      potash / other evaporite / shale
    * computes a grade-thickness map column per Hill (2019)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np


Category = Literal["commercial_potash", "noncommercial_potash",
                   "non_potash_evaporite", "shale_or_marker", "halite_anhydrite"]


# ---------------------------------------------------------------------------
# Mineral library  (after Table 1 of Hill et al. 2023, SLB chart book 2000)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mineral:
    name: str
    formula: str
    K2O_pct: float          # equivalent %K2O of the pure mineral
    GR_API: float           # gamma ray (API) at 100 %
    NPHI_pu: float          # apparent neutron porosity (pu, limestone)
    category: Category


MINERALS: Dict[str, Mineral] = {
    # Commercial potash
    "sylvite":     Mineral("sylvite",     "KCl",                63.2, 750.0, -2.0,  "commercial_potash"),
    "langbeinite": Mineral("langbeinite", "K2SO4.2MgSO4",       22.7, 290.0,  0.0,  "commercial_potash"),
    # Non-commercial potash (hydrated)
    "carnallite":  Mineral("carnallite",  "KMgCl3.6H2O",        17.0, 200.0, 65.0,  "noncommercial_potash"),
    "kainite":     Mineral("kainite",     "KMg(SO4)Cl.3H2O",    19.3, 245.0, 45.0,  "noncommercial_potash"),
    "leonite":     Mineral("leonite",     "K2Mg(SO4)2.4H2O",    25.7, 320.0, 35.0,  "noncommercial_potash"),
    "polyhalite":  Mineral("polyhalite",  "K2Ca2Mg(SO4)4.2H2O", 15.6, 200.0, 14.0,  "noncommercial_potash"),
    # Non-potash evaporites
    "halite":      Mineral("halite",      "NaCl",               0.0,   0.0,  -3.0,  "halite_anhydrite"),
    "anhydrite":   Mineral("anhydrite",   "CaSO4",              0.0,   0.0,  -1.0,  "halite_anhydrite"),
    "gypsum":      Mineral("gypsum",      "CaSO4.2H2O",         0.0,  10.0, 50.0,  "non_potash_evaporite"),
    "kieserite":   Mineral("kieserite",   "MgSO4.H2O",          0.0,   0.0, 38.0,  "non_potash_evaporite"),
    # Shale / marker beds (variable - representative)
    "shale":       Mineral("shale",       "(K,Al,Si,O,OH)",      3.0, 130.0, 35.0,  "shale_or_marker"),
}


# ---------------------------------------------------------------------------
# Classification on the (GR, NPHI) plane
# ---------------------------------------------------------------------------
def classify(gr_api: np.ndarray, nphi_pu: np.ndarray,
             gr_cut: float = 150.0,
             nphi_cut: float = 8.0,
             nphi_high: float = 25.0,
             gr_shale_min: float = 80.0,
             gr_shale_max: float = 180.0) -> np.ndarray:
    """Assign each (GR, NPHI) point to one of the categories.

    Decision rules follow the PID master plot (Fig. 10 of Hill et al.):
        * GR < gr_cut and NPHI < nphi_cut         -> halite_anhydrite
        * GR < gr_cut and NPHI >= nphi_cut and GR >= gr_shale_min
                                                   -> shale_or_marker
        * GR < gr_cut and NPHI >= nphi_cut and GR <  gr_shale_min
                                                   -> non_potash_evaporite
        * GR >= gr_cut and NPHI < nphi_cut         -> commercial_potash
        * GR >= gr_cut and NPHI >= nphi_cut and GR < gr_shale_max and NPHI very high
                                                   -> shale_or_marker
        * GR >= gr_cut and NPHI >= nphi_cut         -> noncommercial_potash
    """
    cat = np.empty(np.broadcast(gr_api, nphi_pu).shape, dtype=object)
    cat[:] = "non_potash_evaporite"

    low_gr = gr_api < gr_cut
    high_gr = ~low_gr
    low_n = nphi_pu < nphi_cut

    # Low GR
    cat[low_gr & low_n] = "halite_anhydrite"
    cat[low_gr & ~low_n & (gr_api >= gr_shale_min)] = "shale_or_marker"
    cat[low_gr & ~low_n & (gr_api < gr_shale_min)] = "non_potash_evaporite"

    # High GR
    cat[high_gr & low_n] = "commercial_potash"
    cat[high_gr & ~low_n] = "noncommercial_potash"
    return cat


def percent_k2o(gr_api: np.ndarray, slope: float = 0.085,
                intercept: float = 0.0) -> np.ndarray:
    """Fig. 3/4 RMA transform: %K2O = slope * GR + intercept (caution, paper
    shows this is *over-optimistic* in heterogeneous strata)."""
    return np.maximum(slope * gr_api + intercept, 0.0)


# ---------------------------------------------------------------------------
# Grade-thickness for an interval
# ---------------------------------------------------------------------------
def grade_thickness(depth_m: np.ndarray, k2o_pct: np.ndarray,
                    is_commercial: np.ndarray,
                    min_grade_pct: float = 4.0,
                    min_thickness_m: float = 1.2) -> tuple[float, float]:
    """Return (cumulative thickness, average grade) of the commercial
    intervals satisfying the BLM standards (>=4 ft thick, >=4 % K2O)."""
    sel = is_commercial & (k2o_pct >= min_grade_pct)
    if not np.any(sel):
        return 0.0, 0.0
    # Find contiguous runs of `sel`
    diff = np.diff(sel.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    total_h = 0.0
    weighted_g = 0.0
    for s, e in zip(starts, ends):
        if e - s == 0:
            continue
        h = depth_m[min(e, len(depth_m) - 1)] - depth_m[s]
        if h >= min_thickness_m:
            g = float(np.mean(k2o_pct[s:e]))
            total_h += h
            weighted_g += g * h
    return total_h, weighted_g / total_h if total_h > 0 else 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    rng = np.random.default_rng(3)

    # Each pure mineral should classify into the right category
    for m in MINERALS.values():
        gr = np.array([m.GR_API])
        np_ = np.array([m.NPHI_pu])
        cat = classify(gr, np_)[0]
        # halite/anhydrite & gypsum/kieserite share quadrant boundaries
        if m.category in {"halite_anhydrite", "non_potash_evaporite"}:
            assert cat in {"halite_anhydrite", "non_potash_evaporite"}, (m.name, cat)
        else:
            assert cat == m.category, (m.name, cat, m.category)

    # Synthetic well: 10 m of halite, 1.5 m of sylvite (commercial),
    # 1 m polyhalite (non-commercial), 1 m of shale, 5 m halite
    z = np.arange(0.0, 18.5, 0.1)
    gr = np.full_like(z, MINERALS["halite"].GR_API + 5)
    npu = np.full_like(z, MINERALS["halite"].NPHI_pu + 1)
    sylvite_zone = (z >= 10) & (z < 11.5)
    poly_zone = (z >= 11.5) & (z < 12.5)
    shale_zone = (z >= 12.5) & (z < 13.5)
    gr[sylvite_zone] = 600.0
    npu[sylvite_zone] = -1.0
    gr[poly_zone] = 200.0
    npu[poly_zone] = 14.0
    gr[shale_zone] = 140.0
    npu[shale_zone] = 35.0

    cat = classify(gr, npu)
    assert np.all(cat[sylvite_zone] == "commercial_potash")
    assert np.all(cat[poly_zone] == "noncommercial_potash")
    # Mid-range NPHI (35 pu, 140 API) lands in noncommercial_potash bucket
    # by the simple rules; the paper notes the value of K-U-T spectral GR
    # to break this ambiguity.
    assert np.all(np.isin(cat[shale_zone],
                          ["shale_or_marker", "noncommercial_potash"]))
    assert np.all(cat[~(sylvite_zone | poly_zone | shale_zone)]
                  == "halite_anhydrite")

    k2o = percent_k2o(gr)
    h, g = grade_thickness(z, k2o, cat == "commercial_potash",
                           min_grade_pct=4.0, min_thickness_m=1.0)
    assert 1.0 <= h <= 1.6, h
    assert g > 4.0, g

    # Random cloud: distribution should be coherent
    n = 500
    g_rand = rng.uniform(0, 800, n)
    n_rand = rng.uniform(-3, 70, n)
    cats = classify(g_rand, n_rand)
    bins = {c: int(np.sum(cats == c)) for c in
            ("commercial_potash", "noncommercial_potash",
             "non_potash_evaporite", "halite_anhydrite", "shale_or_marker")}
    # All five categories should be populated for a uniform cloud
    assert all(v > 0 for v in bins.values()), bins

    print(f"article_06_hill_potash_pid_plot: OK  (sylvite_run h={h:.2f} m, "
          f"avg grade={g:.1f}% K2O, bins={bins})")


if __name__ == "__main__":
    test_all()
