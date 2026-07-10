"""
Constructing Depositional Models of the Cretaceous Reservoirs in
Southeastern Iraq: Integration of Borehole Images, Petrophysical Logs,
and Core Data

Reference:
    Sultan, G., El Araby, A., Selmy, A. G., Shabeeb, A. M., Basso, S. M.,
    Al-Jubouri, M., Alward, W., Nie, X., Aboud, M., Akbi, H., Kun, Z.,
    Omer, I. S., Mahmoud, B. A., and Kadhim, A. A. (2026). Constructing
    Depositional Models of the Cretaceous Reservoirs in Southeastern Iraq.
    Petrophysics, 67(3), 525-542.
    DOI: 10.30632/PJV67N3-2026a4  (orig. SPE-227079-MS)

This is a qualitative sedimentological / sequence-stratigraphic study with no
mathematical equations. This module provides faithful, reproducible
implementations of the workflow building blocks the authors describe:

  - Dunham (1962) carbonate texture classification (mudstone -> grainstone).
  - A high-resolution synthetic-resistivity (SRES) proxy from borehole-image
    conductivity, used to differentiate Dunham textural classes.
  - Core-calibrated electrofacies propagation (nearest-centroid in
    log-response space) across uncored intervals.
  - Sequence-stratigraphic surface flagging (unconformity / karst surface as
    a chronostratigraphic marker; transgressive/regressive trend from a
    grain-size proxy).
  - Reservoir-quality ranking of environments of deposition (EOD).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------------------------------------
# 1. Dunham (1962) carbonate texture classification
# ---------------------------------------------------------------------------

DUNHAM_ORDER = ["mudstone", "wackestone", "packstone", "grainstone"]


def dunham_class(grain_fraction: float, grain_supported: bool,
                 has_mud: bool = True) -> str:
    """
    Simplified Dunham depositional-texture classifier.

    Parameters
    ----------
    grain_fraction : fraction of allochems / grains (0-1).
    grain_supported : True if the fabric is grain-supported.
    has_mud         : True if carbonate mud (micrite) is present.

    Returns
    -------
    'mudstone' | 'wackestone' | 'packstone' | 'grainstone'
    """
    if not grain_supported:
        # Mud-supported.
        return "wackestone" if grain_fraction >= 0.10 else "mudstone"
    # Grain-supported.
    return "packstone" if has_mud else "grainstone"


def dunham_energy_rank(dunham: str) -> int:
    """Depositional-energy / reservoir-quality rank (higher = better)."""
    return DUNHAM_ORDER.index(dunham) if dunham in DUNHAM_ORDER else -1


# ---------------------------------------------------------------------------
# 2. Synthetic high-resolution resistivity (SRES) from image conductivity
# ---------------------------------------------------------------------------

def synthetic_resistivity(image_conductivity: np.ndarray,
                          rt_shallow: float, rt_deep: float) -> np.ndarray:
    """
    Build a high-resolution synthetic-resistivity curve by histogram-matching
    the (dimensionless) borehole-image conductivity to the dynamic range of a
    low-resolution resistivity log.

    Parameters
    ----------
    image_conductivity : per-depth image conductivity (arbitrary units).
    rt_shallow, rt_deep : resistivity scale endpoints (ohm.m), low->high.

    Returns
    -------
    SRES (ohm.m), inversely scaled (high conductivity -> low resistivity).
    """
    c = np.asarray(image_conductivity, dtype=float)
    cmin, cmax = c.min(), c.max()
    if cmax == cmin:
        return np.full_like(c, 0.5 * (rt_shallow + rt_deep))
    norm = (c - cmin) / (cmax - cmin)               # 0..1, high = conductive
    return rt_deep - norm * (rt_deep - rt_shallow)  # conductive -> low Rt


# ---------------------------------------------------------------------------
# 3. Core-calibrated electrofacies propagation
# ---------------------------------------------------------------------------

@dataclass
class Electrofacies:
    """A facies centroid in normalised log-response space."""
    name: str
    centroid: np.ndarray  # vector of normalised log values

    def distance(self, x: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(x, float) - self.centroid))


def train_electrofacies(core_logs: Dict[str, np.ndarray],
                        labels: Sequence[str]) -> List[Electrofacies]:
    """
    Compute facies centroids from cored intervals.

    Parameters
    ----------
    core_logs : dict log_name -> array of values at cored depths.
    labels    : facies label per cored depth.

    Returns
    -------
    List of Electrofacies centroids (one per unique label).
    """
    names = list(core_logs.keys())
    X = np.column_stack([np.asarray(core_logs[n], float) for n in names])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd
    labels = np.asarray(labels)
    centroids = []
    for lab in sorted(set(labels)):
        centroids.append(Electrofacies(lab, Xn[labels == lab].mean(0)))
    # stash normalisation for reuse
    for c in centroids:
        c.centroid = c.centroid  # noqa (kept explicit)
    train_electrofacies._mu = mu  # type: ignore[attr-defined]
    train_electrofacies._sd = sd  # type: ignore[attr-defined]
    train_electrofacies._names = names  # type: ignore[attr-defined]
    return centroids


def classify_electrofacies(centroids: List[Electrofacies],
                           logs: Dict[str, np.ndarray]) -> List[str]:
    """Propagate facies to uncored depths by nearest centroid."""
    names = train_electrofacies._names  # type: ignore[attr-defined]
    mu = train_electrofacies._mu        # type: ignore[attr-defined]
    sd = train_electrofacies._sd        # type: ignore[attr-defined]
    X = np.column_stack([np.asarray(logs[n], float) for n in names])
    Xn = (X - mu) / sd
    out = []
    for row in Xn:
        out.append(min(centroids, key=lambda c: c.distance(row)).name)
    return out


# ---------------------------------------------------------------------------
# 4. Sequence-stratigraphic surface flagging
# ---------------------------------------------------------------------------

def flag_unconformity(porosity: np.ndarray, gr: np.ndarray,
                      poro_jump: float = 0.05, gr_drop: float = 20.0
                      ) -> np.ndarray:
    """
    Flag candidate karst / unconformity (subaerial-exposure) surfaces where a
    sharp porosity increase coincides with a gamma-ray drop (clean,
    karstified carbonate above the surface).

    Returns a boolean array, True at flagged depths.
    """
    poro = np.asarray(porosity, float)
    gr = np.asarray(gr, float)
    dporo = np.diff(poro, prepend=poro[0])
    dgr = np.diff(gr, prepend=gr[0])
    return (dporo > poro_jump) & (dgr < -gr_drop)


def trend_regressive(grain_size_proxy: np.ndarray) -> str:
    """
    Classify a vertical succession as regressive (coarsening / shoaling-up,
    e.g. mudstone -> grainstone) or transgressive (fining-up) from the slope
    of a grain-size proxy versus depth index.
    """
    y = np.asarray(grain_size_proxy, float)
    slope = petrolib.inversion_numerics.fitting.fit_line(np.arange(len(y)), y).slope
    return "regressive" if slope > 0 else "transgressive"


# ---------------------------------------------------------------------------
# 5. Reservoir-quality ranking of environments of deposition
# ---------------------------------------------------------------------------

EOD_QUALITY = {
    "inner-ramp grainstone": 5,
    "rudist buildup": 5,
    "tidal bar/channel": 4,
    "barrier-island sand": 4,
    "deltaic lobe": 3,
    "packstone shoal": 3,
    "lagoonal mudstone": 1,
    "offshore mudstone": 1,
}


def rank_eod(eod: str) -> int:
    """Reservoir-quality rank (1 worst .. 5 best) for an EOD label."""
    return EOD_QUALITY.get(eod, 0)


# ---------------------------------------------------------------------------
# 6. Convenience: full workflow example
# ---------------------------------------------------------------------------

def example_workflow():
    """Run a complete example and print key results."""
    print("=" * 64)
    print("Cretaceous Depositional Model (image + log + core)")
    print("Ref: Sultan et al., Petrophysics 67(3) 2026")
    print("=" * 64)

    print("\nDunham classification:")
    for gf, gs, mud in [(0.05, False, True), (0.3, False, True),
                        (0.6, True, True), (0.7, True, False)]:
        d = dunham_class(gf, gs, mud)
        print(f"  grains={gf:.2f} gs={gs!s:<5} mud={mud!s:<5} -> "
              f"{d:<10} (rank {dunham_energy_rank(d)})")

    cond = np.array([0.2, 0.8, 0.9, 0.3, 0.25, 0.85])
    sres = synthetic_resistivity(cond, rt_shallow=2.0, rt_deep=200.0)
    print(f"\nSynthetic resistivity (SRES): "
          f"{np.array2string(sres, precision=1)}")

    # Electrofacies training + propagation.
    core = {"GR": [25, 80, 30], "RHOB": [2.68, 2.45, 2.70], "NPHI": [0.05, 0.25, 0.04]}
    labels = ["grainstone", "mudstone", "grainstone"]
    cents = train_electrofacies(core, labels)
    pred = classify_electrofacies(
        cents, {"GR": [28, 78], "RHOB": [2.69, 2.46], "NPHI": [0.06, 0.24]})
    print(f"\nPropagated electrofacies: {pred}")

    poro = np.array([0.08, 0.09, 0.20, 0.18, 0.10])
    gr = np.array([60, 58, 25, 30, 55])
    flags = flag_unconformity(poro, gr)
    print(f"Unconformity flags: {flags.tolist()}")
    print(f"Vertical trend: {trend_regressive([1, 2, 2, 3, 4])}")

    print("\nEOD reservoir-quality ranking:")
    for eod in ["rudist buildup", "deltaic lobe", "lagoonal mudstone"]:
        print(f"  {eod:<22s} rank {rank_eod(eod)}")

    return cents


if __name__ == "__main__":
    example_workflow()
