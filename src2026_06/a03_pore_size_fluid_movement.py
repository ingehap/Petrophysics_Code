"""
Effect of Pore-Size Distribution on Fluid Movement in Complex Reservoirs

Reference:
    Manuaba, I. B. G. H., Najrani, H., Cavalleri, C., Moge, A., and
    Chapura, M. (2026). Effect of Pore-Size Distribution on Fluid Movement
    in Complex Reservoirs. Petrophysics, 67(3), 509-524.
    DOI: 10.30632/PJV67N3-2026a3  (orig. SPE-227259-MS)

This is a workflow / multiphysics-integration paper. It applies the standard
published forms of several models; this module implements them faithfully:

  - Archie saturation with the dielectric textural parameter MN used as a
    dynamic exponent (m = n = MN) and a zoned/calibrated salinity.
  - Gaussian decomposition of the NMR T2 distribution into macro / meso /
    micro pore modes (porosity partitioning, after Marzouk et al. 1995).
  - NMR SDR permeability (carbonate-adapted, using the T2 logarithmic mean).
  - Timur-Coates permeability from NMR free-fluid / bound-fluid volumes.
  - A simple multiphysics volumetric forward model + misfit, where the
    invertible parameters are Vw, Vhc, salinity, MN (Fig. 4 of the paper).

Field context: macroporosity + "Type 1" micropore throats host/flow movable
hydrocarbons, whereas micrite-dominated layers (Types 2/3) trap immovable
heavy oil; downhole-calibrated salinity + dielectric MN refine deep
saturation and reveal thin movable-oil layers that fixed-salinity Archie
would miss.
"""

import math
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
# 1. Archie with dielectric textural exponent MN
# ---------------------------------------------------------------------------

def archie_sw(Rw: float, Rt: float, phi: float,
              MN: float, a: float = 1.0) -> float:
    """
    Archie water saturation with m = n = MN (the dielectric textural
    parameter doubles as cementation and saturation exponent):

        Sw = ( a * Rw / (phi**MN * Rt) ) ** (1/MN)

    Parameters
    ----------
    Rw  : formation-water resistivity, ohm.m
    Rt  : true (deep) resistivity, ohm.m
    phi : total porosity, fraction
    MN  : dielectric textural parameter (= m = n)
    a   : tortuosity factor (default 1.0)

    Returns
    -------
    Sw clipped to [0, 1].
    """
    if phi <= 0.0 or Rt <= 0.0:
        return float("nan")
    # HAZARD (LIBRARY_MERGE_PLAN.md section 9): argument order (Rw, Rt) — the
    # canonical order is (rt, rw); the textural exponent MN doubles as m and n.
    return float(petrolib.saturation_resistivity.archie_sw(Rt, Rw, phi=phi, a=a, m=MN, n=MN, clip=(0.0, 1.0)))


def rw_from_salinity(salinity_ppm: float, temp_c: float = 75.0) -> float:
    """
    Approximate NaCl-brine resistivity from salinity and temperature
    (Arps relation), used to convert downhole-calibrated salinity to Rw.

        Rw75 = (0.0123 + 3647.5 / Cppm**0.955)         at 75 deg F (ref)
        Rw   = Rw_ref * (T_ref + 6.77) / (T + 6.77)    Arps temperature law

    Parameters
    ----------
    salinity_ppm : NaCl-equivalent salinity, ppm
    temp_c       : temperature, deg C
    """
    Cppm = max(salinity_ppm, 1.0)
    rw75f = 0.0123 + 3647.5 / Cppm ** 0.955  # ohm.m at 75 deg F (~23.9 C)
    t_f = temp_c * 9.0 / 5.0 + 32.0
    return rw75f * (75.0 + 6.77) / (t_f + 6.77)


# ---------------------------------------------------------------------------
# 2. Gaussian decomposition of the NMR T2 distribution
# ---------------------------------------------------------------------------

@dataclass
class GaussianMode:
    """One Gaussian pore-body mode in log10(T2) space."""
    amplitude: float   # peak height
    center: float      # log10(T2) center
    width: float       # standard deviation in log10 units


def gaussian_t2_model(log_t2: np.ndarray,
                      modes: Sequence[GaussianMode]) -> np.ndarray:
    """Sum of Gaussian pore-body modes evaluated on a log10(T2) axis."""
    log_t2 = np.asarray(log_t2, dtype=float)
    out = np.zeros_like(log_t2)
    for m in modes:
        out += m.amplitude * np.exp(-0.5 * ((log_t2 - m.center) / m.width) ** 2)
    return out


def partition_porosity(t2_ms: np.ndarray, amplitude: np.ndarray,
                       t2_cutoffs: Tuple[float, float] = (3.0, 33.0)
                       ) -> Dict[str, float]:
    """
    Partition the porosity (area under the T2 amplitude) into micro / meso /
    macro fractions using two T2 cutoffs.

    Parameters
    ----------
    t2_ms      : T2 axis, ms.
    amplitude  : incremental porosity amplitude at each T2.
    t2_cutoffs : (micro|meso, meso|macro) cutoffs in ms.  Default carbonate
                 cutoffs (3 ms, 33 ms).

    Returns
    -------
    dict with 'micro', 'meso', 'macro' porosity fractions summing to 1.0.
    """
    t2 = np.asarray(t2_ms, float)
    amp = np.asarray(amplitude, float)
    total = amp.sum()
    if total <= 0.0:
        return {"micro": 0.0, "meso": 0.0, "macro": 0.0}
    c1, c2 = t2_cutoffs
    micro = amp[t2 < c1].sum()
    meso = amp[(t2 >= c1) & (t2 < c2)].sum()
    macro = amp[t2 >= c2].sum()
    return {"micro": micro / total, "meso": meso / total, "macro": macro / total}


def t2_logmean(t2_ms: np.ndarray, amplitude: np.ndarray) -> float:
    """Logarithmic mean of the T2 distribution (T2LM), ms."""
    t2 = np.asarray(t2_ms, float)
    amp = np.asarray(amplitude, float)
    w = amp.sum()
    if w <= 0.0:
        return float("nan")
    return float(np.exp(np.sum(amp * np.log(t2)) / w))


# ---------------------------------------------------------------------------
# 3. NMR permeability transforms
# ---------------------------------------------------------------------------

def sdr_permeability(phi: float, t2lm_ms: float,
                     a: float = 4.0, m: float = 4.0, n: float = 2.0) -> float:
    """
    Schlumberger-Doll-Research (SDR) NMR permeability, ms-based:

        k = a * phi**m * T2LM**n

    The T2 logarithmic mean is used as a proxy for average pore size.
    """
    return a * phi ** m * t2lm_ms ** n


def timur_coates_permeability(phi: float, ffi: float, bvi: float,
                              C: float = 10.0) -> float:
    """
    Timur-Coates NMR permeability:

        k = ( phi/C )**4 * ( FFI / BVI )**2

    Parameters
    ----------
    phi : total porosity, fraction
    ffi : free-fluid (movable) porosity fraction
    bvi : bound-fluid (irreducible) porosity fraction
    C   : calibration constant (default 10)
    """
    if bvi <= 0.0:
        return float("inf")
    return (phi / C) ** 4 * (ffi / bvi) ** 2


# ---------------------------------------------------------------------------
# 4. Simplified multiphysics volumetric forward model + misfit (Fig. 4)
# ---------------------------------------------------------------------------

@dataclass
class FormationModel:
    """Invertible volumetric model of one depth station."""
    Vw: float          # water volume (fraction)
    Vhc: float         # hydrocarbon volume (fraction)
    salinity_ppm: float
    MN: float          # textural parameter

    @property
    def porosity(self) -> float:
        """Total porosity = Vw + Vhc (sum of inverted fluid volumes)."""
        return self.Vw + self.Vhc

    @property
    def sw(self) -> float:
        phi = self.porosity
        return self.Vw / phi if phi > 0 else float("nan")


def forward_resistivity(model: FormationModel, Rt_measured: float,
                        temp_c: float = 75.0) -> float:
    """
    Reconstruct deep resistivity from the volumetric model via Archie and
    return its misfit (relative) against the measured Rt.
    """
    Rw = rw_from_salinity(model.salinity_ppm, temp_c)
    phi = model.porosity
    if phi <= 0 or model.sw <= 0:
        return float("inf")
    Rt_model = Rw / (phi ** model.MN * model.sw ** model.MN)
    return abs(Rt_model - Rt_measured) / Rt_measured


# ---------------------------------------------------------------------------
# 5. Convenience: full workflow example
# ---------------------------------------------------------------------------

def example_workflow():
    """Run a complete example and print key results."""
    print("=" * 64)
    print("Pore-Size Distribution & Fluid Movement (multiphysics)")
    print("Ref: Manuaba et al., Petrophysics 67(3) 2026")
    print("=" * 64)

    # Build a synthetic NMR T2 distribution (macro + Type-1 micro + micrite).
    t2 = np.logspace(-1, 3.3, 64)  # 0.1 ms - 2000 ms
    log_t2 = np.log10(t2)
    modes = [
        GaussianMode(1.0, math.log10(200.0), 0.30),  # macro
        GaussianMode(0.7, math.log10(15.0), 0.30),   # Type-1 micro (movable)
        GaussianMode(0.9, math.log10(1.0), 0.25),    # micrite (immovable)
    ]
    amp = gaussian_t2_model(log_t2, modes)

    part = partition_porosity(t2, amp)
    t2lm = t2_logmean(t2, amp)
    print(f"\nPorosity partition: macro={part['macro']:.0%}  "
          f"meso={part['meso']:.0%}  micro={part['micro']:.0%}")
    print(f"T2 log-mean: {t2lm:.1f} ms")

    phi = 0.22
    bvi = phi * part["micro"]
    ffi = phi * (part["macro"] + part["meso"])
    k_sdr = sdr_permeability(phi, t2lm)
    k_tc = timur_coates_permeability(phi, ffi, bvi)
    print(f"\nPermeability:  SDR = {k_sdr:.1f} md   Timur-Coates = {k_tc:.1f} md")

    # Archie with calibrated salinity vs fixed salinity.
    Rw = rw_from_salinity(185_000)  # Field Example 1: 185 kppm
    sw_dyn = archie_sw(Rw, Rt=8.0, phi=phi, MN=2.1)
    sw_fix = archie_sw(rw_from_salinity(100_000), Rt=8.0, phi=phi, MN=2.0)
    print(f"\nArchie Sw (calibrated 185 kppm, MN=2.1): {sw_dyn:.2f}")
    print(f"Archie Sw (fixed 100 kppm,   m=n=2):     {sw_fix:.2f}")

    # Multiphysics forward misfit.
    model = FormationModel(Vw=0.154, Vhc=0.066, salinity_ppm=185_000, MN=2.1)
    mf = forward_resistivity(model, Rt_measured=8.0)
    print(f"\nForward model: phi={model.porosity:.3f}  Sw={model.sw:.2f}  "
          f"misfit={mf:.3f}")

    return part


if __name__ == "__main__":
    example_workflow()
