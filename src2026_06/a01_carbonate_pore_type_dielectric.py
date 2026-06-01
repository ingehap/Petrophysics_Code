"""
Unlocking the Effect of Pore Types on Dielectric Permittivity in Carbonates

Reference:
    AlZoukani, A. M., Al-Hamad, M., and Abdallah, W. (2026). Unlocking the
    Effect of Pore Types on Dielectric Permittivity in Carbonates.
    Petrophysics, 67(3), 470-481.
    DOI: 10.30632/PJV67N3-2026a1
    (Best Petrophysics Papers from MEOS GEO 2025; orig. SPE-227798-MS)

Implements:
  - Complex dielectric permittivity definition (Eq. 1):  e* = e' - i e''
  - Porosity-normalised permittivity index  (e' / phi) at 12 MHz, used to
    isolate the pore-geometry effect from porosity.
  - Digital Image Analysis (DIA) pore-geometry descriptors:
        PoA      - perimeter-over-area of pore space
        DOMSize  - dominant pore size (upper bound of pores making up 50%
                   of the porosity)
        AR       - aspect ratio (max/min axis of the bounding ellipse)
  - A simple rule-based pore-type classifier (moldic / interparticle /
    intercrystalline) following the trends reported in the paper.

The paper deliberately applies NO dielectric-dispersion model; the analysis
is performed directly on measured permittivity normalised by porosity and
cross-plotted against the DIA descriptors.

Measurement context (verbatim from the paper):
  - 17 carbonate outcrop samples, porosity 9.2-45.0 % (avg ~24 %),
    permeability 3.9-1468 md.
  - Permittivity measured 12 MHz - 1 GHz with an Agilent E50771C network
    analyzer; samples saturated with 30-kppm NaCl brine.
  - Reported ranges: PoA 30-290, DOMSize 21-493, AR 0.31-0.62; normalised
    permittivity at 12 MHz between 1.5 and 3.5.
"""

import cmath
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1. Complex dielectric permittivity (Eq. 1)
# ---------------------------------------------------------------------------

def complex_permittivity(eps_real: float, eps_imag: float) -> complex:
    """
    Complex dielectric permittivity (Eq. 1):

        e* = e' - i e''

    Parameters
    ----------
    eps_real : float
        Real part e' (relative permittivity / dielectric constant).
    eps_imag : float
        Imaginary part e'' (related to the conductivity of the sample),
        a non-negative loss term.

    Returns
    -------
    complex
        e* = e' - i*e''  (engineering sign convention used in the paper).
    """
    return complex(eps_real, -abs(eps_imag))


def loss_tangent(eps_real: float, eps_imag: float) -> float:
    """Dielectric loss tangent  tan(delta) = e'' / e'."""
    return abs(eps_imag) / eps_real


def imag_from_conductivity(sigma: float, freq_hz: float) -> float:
    """
    Imaginary permittivity contribution from ionic conductivity:

        e'' = sigma / (2*pi*f*eps0)

    Parameters
    ----------
    sigma   : conductivity, S/m
    freq_hz : frequency, Hz

    Returns
    -------
    e'' (dimensionless, relative).
    """
    eps0 = 8.8541878128e-12  # F/m
    return sigma / (2.0 * cmath.pi * freq_hz * eps0)


# ---------------------------------------------------------------------------
# 2. Porosity-normalised permittivity index
# ---------------------------------------------------------------------------

def normalized_permittivity(eps_real: float, phi: float) -> float:
    """
    Porosity-independent permittivity index used by the authors to remove
    the first-order porosity control and isolate the pore-geometry effect:

        eps_norm = e' / phi

    Parameters
    ----------
    eps_real : real permittivity (typically at 12 MHz)
    phi      : porosity (fraction)

    Returns
    -------
    Normalised permittivity (dimensionless). Reported range 1.5-3.5.
    """
    if phi <= 0.0:
        raise ValueError("porosity must be positive")
    return eps_real / phi


# ---------------------------------------------------------------------------
# 3. Digital Image Analysis (DIA) pore-geometry descriptors
# ---------------------------------------------------------------------------

def perimeter_over_area(perimeters: Sequence[float],
                        areas: Sequence[float]) -> float:
    """
    PoA = (total perimeter enclosing pore spaces) / (total pore-space area)

    Larger PoA indicates a more intricate / complex pore network.
    """
    tot_p = float(sum(perimeters))
    tot_a = float(sum(areas))
    if tot_a <= 0.0:
        raise ValueError("total pore area must be positive")
    return tot_p / tot_a


def dominant_pore_size(pore_sizes: Sequence[float],
                       pore_areas: Sequence[float]) -> float:
    """
    DOMSize - the upper boundary of the pore sizes that, cumulatively,
    account for 50 % of the porosity (pore area) on a thin section.

    Parameters
    ----------
    pore_sizes : characteristic size of each pore (e.g. equivalent diameter)
    pore_areas : area of each pore (weights for the porosity fraction)

    Returns
    -------
    The size threshold below which 50 % of the pore area resides.
    """
    pairs = sorted(zip(pore_sizes, pore_areas), key=lambda t: t[0])
    total = sum(a for _, a in pairs)
    if total <= 0.0:
        raise ValueError("total pore area must be positive")
    cum = 0.0
    for size, area in pairs:
        cum += area
        if cum >= 0.5 * total:
            return float(size)
    return float(pairs[-1][0])


def aspect_ratio(major_axis: float, minor_axis: float) -> float:
    """
    AR = ratio of the minimum to maximum axis of the ellipse bounding a
    pore.  Reported range 0.31-0.62 (1.0 = perfectly round).
    """
    if major_axis <= 0.0:
        raise ValueError("major axis must be positive")
    return min(major_axis, minor_axis) / max(major_axis, minor_axis)


def mean_aspect_ratio(majors: Sequence[float],
                      minors: Sequence[float]) -> float:
    """Area-unweighted mean aspect ratio over all segmented pores."""
    ars = [aspect_ratio(a, b) for a, b in zip(majors, minors)]
    return sum(ars) / len(ars)


# ---------------------------------------------------------------------------
# 4. Pore-type classification from the DIA + dielectric trends
# ---------------------------------------------------------------------------

@dataclass
class CarbonateSample:
    """A single carbonate thin-section / plug observation."""
    name: str
    phi: float                 # porosity, fraction
    perm_md: float             # permeability, md
    eps_real_12mhz: float      # real permittivity at 12 MHz (brine-saturated)
    poa: float                 # perimeter-over-area
    domsize: float             # dominant pore size
    ar: float                  # aspect ratio

    @property
    def eps_norm(self) -> float:
        return normalized_permittivity(self.eps_real_12mhz, self.phi)


def classify_pore_type(sample: CarbonateSample,
                       poa_hi: float = 150.0,
                       domsize_hi: float = 200.0) -> str:
    """
    Rule-based pore-type label following the reported trends:

      - Moldic         : high normalised permittivity, low PoA, low AR,
                         high DOMSize (simple, isolated network).
      - Intercrystalline: high PoA and high AR, small DOMSize
                         (most complex network).
      - Interparticle  : everything else (lowest normalised permittivity,
                         intermediate DIA descriptors).

    Thresholds are tunable; defaults sit roughly mid-range of the reported
    PoA (30-290) and DOMSize (21-493) windows.
    """
    if sample.poa < poa_hi and sample.domsize >= domsize_hi:
        return "moldic"
    if sample.poa >= poa_hi and sample.domsize < domsize_hi:
        return "intercrystalline"
    return "interparticle"


# ---------------------------------------------------------------------------
# 5. Convenience: full workflow example
# ---------------------------------------------------------------------------

def example_workflow():
    """Run a complete example and print key results."""
    print("=" * 64)
    print("Carbonate Pore Type vs Dielectric Permittivity")
    print("Ref: AlZoukani, Al-Hamad & Abdallah, Petrophysics 67(3) 2026")
    print("=" * 64)

    # Eq. 1 demonstration with conductivity-derived loss term.
    epp = imag_from_conductivity(sigma=0.5, freq_hz=12e6)
    es = complex_permittivity(eps_real=18.0, eps_imag=epp)
    print(f"\nComplex permittivity (e' = 18, sigma = 0.5 S/m @ 12 MHz):")
    print(f"  e* = {es:.3f}   tan(delta) = {loss_tangent(18.0, epp):.4f}")

    # Synthetic suite spanning the three dominant pore types.
    samples = [
        CarbonateSample("moldic-1",         0.30, 1468, 13.5, 45.0, 320.0, 0.34),
        CarbonateSample("intercrystalline", 0.18,  3.9, 18.0, 250.0,  35.0, 0.58),
        CarbonateSample("interparticle",    0.24,   85, 14.0, 120.0, 110.0, 0.45),
    ]

    print("\nDIA + dielectric classification:")
    print(f"  {'sample':<18s}{'phi':>5s}{'eps_n':>8s}"
          f"{'PoA':>7s}{'DOM':>7s}{'AR':>6s}  type")
    for s in samples:
        ptype = classify_pore_type(s)
        print(f"  {s.name:<18s}{s.phi:>5.2f}{s.eps_norm:>8.2f}"
              f"{s.poa:>7.0f}{s.domsize:>7.0f}{s.ar:>6.2f}  {ptype}")

    # DIA descriptors computed from raw segmented-pore data.
    poa = perimeter_over_area([12.0, 8.0, 30.0], [4.0, 3.0, 9.0])
    dom = dominant_pore_size([20, 50, 120, 300], [1.0, 2.0, 3.0, 4.0])
    ar = mean_aspect_ratio([10.0, 8.0], [4.0, 5.0])
    print(f"\nComputed-from-raw DIA:  PoA = {poa:.2f}  "
          f"DOMSize = {dom:.0f}  mean AR = {ar:.2f}")

    return samples


if __name__ == "__main__":
    example_workflow()
