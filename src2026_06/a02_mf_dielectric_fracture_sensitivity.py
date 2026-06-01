"""
Experimental Study of Sensitivity of Multifrequency (MF) Dielectric
Measurements to Hydraulic Fractures in Sandstone and Carbonate Formations

Reference:
    Al-Qouzi, W., Hassan, A., Attia, H., El-Husseiny, A., and Mahmoud, M.
    (2026). Experimental Study of Sensitivity of Multifrequency (MF)
    Dielectric Measurements to Hydraulic Fractures in Sandstone and Carbonate
    Formations. Petrophysics, 67(3), 482-508.
    DOI: 10.30632/PJV67N3-2026a2  (orig. SPE-227827-MS)

Implements:
  - Debye / Cole-Cole Maxwell-Wagner effective complex permittivity (Eq. 1):
        e*(w) = e_inf + d_eps / (1 + (j w tau)^(1-alpha))
    with the pure-Debye case alpha = 0:
        e*(w) = e_inf + d_eps / (1 + j w tau)
  - Effective conductivity from the imaginary permittivity.
  - "Area between curves" sensitivity metric used by the authors to quantify
    the before/after-fracture permittivity contrast.
  - Percentage permittivity / conductivity increase after fracturing.
  - A qualitative orientation-sensitivity rule (vertical > horizontal for
    carbonates).

Setup context (verbatim from the paper):
  - Indiana limestone (97 % calcite, phi ~18 %, k ~12 md) and Bandera
    sandstone (~70 % quartz, phi ~19.5 %, k ~27 md), plugs 1.5 in x 3 in.
  - Frequency sweep 1 MHz - 3 GHz (40 pre-fracture / 80 post-fracture pts)
    on a Keysight E5071C VNA with an open-ended coaxial probe.
  - Saturated with 3 wt% KCl brine at 1900 psi confining for 24 h.
  - Reported: carbonate permittivity rise avg ~+22 units (45 -> >60);
    conductivity rise ~12-34 %; vertical fractures more sensitive than
    horizontal for carbonates.
"""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

EPS0 = 8.8541878128e-12  # vacuum permittivity, F/m


# ---------------------------------------------------------------------------
# 1. Maxwell-Wagner / Cole-Cole effective complex permittivity (Eq. 1)
# ---------------------------------------------------------------------------

def cole_cole_permittivity(freq_hz, eps_inf: float, d_eps: float,
                           tau: float, alpha: float = 0.0):
    """
    Effective complex permittivity of the fractured heterogeneous system.

    Cole-Cole form (Eq. 1, generalised):

        e*(w) = e_inf + d_eps / (1 + (j w tau)^(1-alpha))

    For alpha = 0 this reduces to the single-relaxation Debye model:

        e*(w) = e_inf + d_eps / (1 + j w tau)

    Parameters
    ----------
    freq_hz : float or array-like
        Frequency, Hz.
    eps_inf : float
        High-frequency permittivity e_inf (negligible interfacial polarisation).
    d_eps : float
        Polarisation strength  d_eps = e_static - e_inf.
    tau : float
        Relaxation time, s.
    alpha : float
        Cole-Cole distribution parameter in [0, 1).  0 = pure Debye.

    Returns
    -------
    complex ndarray (or scalar) of e*(w).
    """
    w = 2.0 * math.pi * np.asarray(freq_hz, dtype=float)
    jwt = 1j * w * tau
    return eps_inf + d_eps / (1.0 + jwt ** (1.0 - alpha))


def effective_conductivity(freq_hz, eps_imag) -> np.ndarray:
    """
    Effective conductivity from the imaginary permittivity:

        sigma = w * eps0 * e''  = 2*pi*f * eps0 * e''

    Returns conductivity in S/m.
    """
    w = 2.0 * math.pi * np.asarray(freq_hz, dtype=float)
    return w * EPS0 * np.asarray(eps_imag, dtype=float)


def relaxation_regime(freq_hz: float, tau: float) -> str:
    """
    Classify the polarisation regime relative to the relaxation time:

      - 'interfacial'  if  w*tau << 1  (low frequency, charge accumulates at
                        fracture-matrix interfaces -> elevated permittivity)
      - 'bulk'         if  w*tau >> 1  (high frequency, response governed by
                        bulk water content + intrinsic matrix)
      - 'transition'   otherwise.
    """
    wt = 2.0 * math.pi * freq_hz * tau
    if wt < 0.1:
        return "interfacial"
    if wt > 10.0:
        return "bulk"
    return "transition"


# ---------------------------------------------------------------------------
# 2. Before / after-fracture sensitivity metrics
# ---------------------------------------------------------------------------

def area_between_curves(freq_hz: Sequence[float],
                        eps_before: Sequence[float],
                        eps_after: Sequence[float]) -> float:
    """
    The authors quantify fracture sensitivity by the area between the
    pre- and post-fracture permittivity spectra (integrated over log-freq).

    Reported example values: IND1 = 49.4, IND2 = 63.7.

    Parameters
    ----------
    freq_hz    : frequency points (Hz), ascending.
    eps_before : permittivity before fracturing at each frequency.
    eps_after  : permittivity after fracturing at each frequency.

    Returns
    -------
    Area between the two curves in log10(frequency) space.
    """
    x = np.log10(np.asarray(freq_hz, dtype=float))
    diff = np.abs(np.asarray(eps_after, float) - np.asarray(eps_before, float))
    # np.trapezoid (NumPy >= 2.0) with fallback to np.trapz (NumPy < 2.0).
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapezoid(diff, x))


def percent_increase(before: float, after: float) -> float:
    """Percentage increase of a property after fracturing."""
    if before == 0.0:
        return float("nan")
    return 100.0 * (after - before) / before


@dataclass
class FractureCase:
    """One measurement set (rock x orientation x distance x condition)."""
    rock: str                  # 'carbonate' or 'sandstone'
    orientation: str           # 'vertical' or 'horizontal'
    distance: str              # 'near' or 'far'
    eps_before: float          # low-frequency permittivity before fracture
    eps_after: float           # low-frequency permittivity after fracture
    cond_before: float         # conductivity before, S/m
    cond_after: float          # conductivity after, S/m

    @property
    def eps_gain(self) -> float:
        return self.eps_after - self.eps_before

    @property
    def cond_pct(self) -> float:
        return percent_increase(self.cond_before, self.cond_after)


def orientation_more_sensitive(rock: str) -> str:
    """
    Reported orientation sensitivity: for carbonates the vertical fracture
    (longer conductive path, direct probe contact) gives the larger dielectric
    response; for sandstone the orientation effect is ambiguous (clay masking).
    """
    return "vertical" if rock.lower().startswith("carb") else "ambiguous"


# ---------------------------------------------------------------------------
# 3. Convenience: full workflow example
# ---------------------------------------------------------------------------

def example_workflow():
    """Run a complete example and print key results."""
    print("=" * 64)
    print("MF Dielectric Sensitivity to Hydraulic Fractures")
    print("Ref: Al-Qouzi et al., Petrophysics 67(3) 2026")
    print("=" * 64)

    freqs = np.logspace(6, 9.5, 80)          # 1 MHz - ~3 GHz

    # Pre-fracture: weak interfacial polarisation.
    before = cole_cole_permittivity(freqs, eps_inf=6.0, d_eps=39.0, tau=2e-8)
    # Post-fracture: enhanced d_eps (more interfacial area / connectivity).
    after = cole_cole_permittivity(freqs, eps_inf=6.0, d_eps=64.0, tau=4e-8)

    area = area_between_curves(freqs, before.real, after.real)
    print(f"\nArea between permittivity curves: {area:.1f}")
    print(f"  (paper reports IND1 = 49.4, IND2 = 63.7)")

    lo = 0  # 1 MHz index
    print(f"\nLow-frequency permittivity  before = {before.real[lo]:.1f}, "
          f"after = {after.real[lo]:.1f}  (+{after.real[lo]-before.real[lo]:.1f})")
    print(f"  regime @ 12 MHz: {relaxation_regime(12e6, tau=4e-8)}")
    print(f"  regime @  2 GHz: {relaxation_regime(2e9,  tau=4e-8)}")

    # Conductivity from imaginary permittivity.
    sig_after = effective_conductivity(freqs, -after.imag)
    sig_before = effective_conductivity(freqs, -before.imag)
    pct = percent_increase(sig_before[-1], sig_after[-1])
    print(f"\nConductivity increase @ high freq: {pct:.0f} %  "
          f"(paper: 12-34 %)")

    cases = [
        FractureCase("carbonate", "vertical", "near", 45.0, 70.0, 0.30, 0.40),
        FractureCase("carbonate", "horizontal", "near", 49.0, 68.0, 0.28, 0.34),
        FractureCase("sandstone", "vertical", "near", 35.0, 50.0, 0.20, 0.224),
    ]
    print("\nMeasurement sets:")
    for c in cases:
        print(f"  {c.rock:<10s}{c.orientation:<11s} "
              f"d(eps)={c.eps_gain:+5.1f}  d(sigma)={c.cond_pct:+5.0f}%")
    print(f"\nMost sensitive orientation (carbonate): "
          f"{orientation_more_sensitive('carbonate')}")

    return cases


if __name__ == "__main__":
    example_workflow()
