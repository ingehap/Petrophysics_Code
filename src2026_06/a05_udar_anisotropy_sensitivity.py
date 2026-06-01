"""
Anisotropy Sensitivity in Ultradeep Azimuthal Resistivity Technologies

Reference:
    Bower, M., Xie, H., Wang, G. L., Leveque, S., and Dolan, J. (2026).
    Anisotropy Sensitivity in Ultradeep Azimuthal Resistivity Technologies.
    Petrophysics, 67(3), 544-559.
    DOI: 10.30632/PJV67N3-2026a5

The paper studies how the sensitivity of a tilted-antenna ultradeep azimuthal
resistivity (UDAR) tool to electromagnetic formation anisotropy varies with:
    - transmitter-receiver spacing,
    - firing (operating) frequency,
    - formation resistivity,
    - near-field vs far-field detection regimes.

Key reported findings, encoded here:
    - Longer spacing AND higher firing frequency maximise near-field
      anisotropy sensitivity.
    - Far-field anisotropy sensitivity is generally low.
    - The dominant controlling factor is the resistivity of the medium the
      tool currently sits in.

This module implements the standard transversely-isotropic (TI) layered-earth
building blocks used to reach those conclusions:
    - Skin depth and induction number.
    - Apparent anisotropy ratio (lambda) and the TI horizontal/vertical
      resistivity pair.
    - A coaxial/coplanar tilted-coil response proxy whose anisotropy
      sensitivity can be probed against spacing/frequency/resistivity.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

MU0 = 4.0e-7 * math.pi  # vacuum permeability, H/m


# ---------------------------------------------------------------------------
# 1. Fundamental EM scales
# ---------------------------------------------------------------------------

def skin_depth(resistivity: float, freq_hz: float) -> float:
    """
    EM skin depth in a homogeneous medium:

        delta = sqrt(2 * rho / (w * mu0)) = sqrt(rho / (pi * f * mu0))

    Parameters
    ----------
    resistivity : ohm.m
    freq_hz     : Hz

    Returns
    -------
    Skin depth, m.
    """
    return math.sqrt(resistivity / (math.pi * freq_hz * MU0))


def induction_number(spacing: float, resistivity: float, freq_hz: float
                     ) -> float:
    """
    Dimensionless induction number L/delta (spacing over skin depth).
    Near-field regime when L/delta << 1, far-field when L/delta >> 1.
    """
    return spacing / skin_depth(resistivity, freq_hz)


def field_regime(spacing: float, resistivity: float, freq_hz: float) -> str:
    """Return 'near-field' or 'far-field' from the induction number."""
    return "near-field" if induction_number(spacing, resistivity, freq_hz) < 1.0 \
        else "far-field"


# ---------------------------------------------------------------------------
# 2. TI anisotropy description
# ---------------------------------------------------------------------------

def anisotropy_ratio(rh: float, rv: float) -> float:
    """Anisotropy ratio lambda = sqrt(Rv / Rh) (>= 1 for Rv >= Rh)."""
    return math.sqrt(rv / rh)


@dataclass
class TILayer:
    """A transversely-isotropic layer."""
    rh: float          # horizontal resistivity, ohm.m
    rv: float          # vertical resistivity, ohm.m

    @property
    def lam(self) -> float:
        return anisotropy_ratio(self.rh, self.rv)


# ---------------------------------------------------------------------------
# 3. Tilted-coil response proxy and anisotropy sensitivity
# ---------------------------------------------------------------------------

def tilted_coil_response(layer: TILayer, spacing: float, freq_hz: float,
                         tilt_deg: float = 45.0) -> complex:
    """
    Simplified analytic proxy for the mutual-coupling voltage of a tilted-coil
    induction tool in a homogeneous TI whole-space.

    The coaxial (Hzz, sees Rh) and coplanar (Hxx, sees the anisotropic
    combination) couplings are blended by the antenna tilt; each follows the
    standard magnetic-dipole low-induction-number form

        V ~ (i k^2 / L) * exp(i k L),   k = (1 - i) / delta

    Parameters
    ----------
    layer    : TI layer (Rh, Rv).
    spacing  : transmitter-receiver spacing L, m.
    freq_hz  : firing frequency, Hz.
    tilt_deg : antenna tilt angle, degrees.

    Returns
    -------
    Complex response proxy (arbitrary units).
    """
    w = 2.0 * math.pi * freq_hz
    t = math.radians(tilt_deg)

    def coupling(rho_eff: float) -> complex:
        delta = skin_depth(rho_eff, freq_hz)
        k = (1.0 - 1.0j) / delta
        return 1j * k * k / spacing * np.exp(1j * k * spacing)

    # Coaxial sees Rh; coplanar sees an effective rho that depends on lambda.
    v_zz = coupling(layer.rh)
    v_xx = coupling(layer.rh * layer.lam)
    return (math.cos(t) ** 2) * v_zz + (math.sin(t) ** 2) * v_xx


def anisotropy_sensitivity(rh: float, lam: float, spacing: float,
                           freq_hz: float, tilt_deg: float = 45.0,
                           d_lam: float = 0.05) -> float:
    """
    Anisotropy sensitivity = |dV / d(lambda)| evaluated by finite difference,
    normalised by the isotropic response magnitude.

    Larger values indicate a configuration more capable of resolving
    anisotropy.  Used to demonstrate the spacing / frequency / resistivity
    trends reported in the paper.
    """
    base = TILayer(rh, rh * lam ** 2)
    pert = TILayer(rh, rh * (lam + d_lam) ** 2)
    v0 = tilted_coil_response(base, spacing, freq_hz, tilt_deg)
    v1 = tilted_coil_response(pert, spacing, freq_hz, tilt_deg)
    ref = abs(tilted_coil_response(TILayer(rh, rh), spacing, freq_hz, tilt_deg))
    if ref == 0.0:
        return float("nan")
    return abs(v1 - v0) / d_lam / ref


def absolute_anisotropy_sensitivity(rh: float, lam: float, spacing: float,
                                    freq_hz: float, tilt_deg: float = 45.0,
                                    d_lam: float = 0.05) -> float:
    """
    Absolute (un-normalised) anisotropy sensitivity |dV/d(lambda)|.

    Unlike :func:`anisotropy_sensitivity`, this is NOT divided by the isotropic
    response magnitude, so it retains the strong dependence of the raw
    measurable signal level on the surrounding formation resistivity - which
    is what governs how resolvable anisotropy actually is in practice.
    """
    base = TILayer(rh, rh * lam ** 2)
    pert = TILayer(rh, rh * (lam + d_lam) ** 2)
    v0 = tilted_coil_response(base, spacing, freq_hz, tilt_deg)
    v1 = tilted_coil_response(pert, spacing, freq_hz, tilt_deg)
    return abs(v1 - v0) / d_lam


def dominant_factor(rh_values, spacing_values, freq_values) -> str:
    """
    Identify which swept variable produces the largest relative variation in
    the (absolute) anisotropy sensitivity, illustrating the paper's conclusion
    that the medium resistivity is the dominant control: the raw response
    level - and hence the achievable anisotropy resolution - varies far more
    with formation resistivity than with spacing or firing frequency.
    """
    lam = 1.5

    def spread(values, fn):
        s = [fn(v) for v in values]
        s = [x for x in s if math.isfinite(x) and x > 0]
        return (max(s) / min(s)) if s and min(s) > 0 else 0.0

    rh0, sp0, f0 = rh_values[0], spacing_values[0], freq_values[0]
    by_rho = spread(rh_values,
                    lambda r: absolute_anisotropy_sensitivity(r, lam, sp0, f0))
    by_sp = spread(spacing_values,
                   lambda s: absolute_anisotropy_sensitivity(rh0, lam, s, f0))
    by_f = spread(freq_values,
                  lambda f: absolute_anisotropy_sensitivity(rh0, lam, sp0, f))
    best = max([("resistivity", by_rho), ("spacing", by_sp),
                ("frequency", by_f)], key=lambda t: t[1])
    return best[0]


# ---------------------------------------------------------------------------
# 4. Convenience: full workflow example
# ---------------------------------------------------------------------------

def example_workflow():
    """Run a complete example and print key results."""
    print("=" * 64)
    print("UDAR Anisotropy Sensitivity")
    print("Ref: Bower, Xie, Wang, Leveque & Dolan, Petrophysics 67(3) 2026")
    print("=" * 64)

    rh = 10.0
    print(f"\nField regime (Rh={rh} ohm.m):")
    for L in (5.0, 20.0, 80.0):
        for f in (2e3, 4e4):
            reg = field_regime(L, rh, f)
            ld = induction_number(L, rh, f)
            print(f"  L={L:>5.0f} m  f={f:>7.0f} Hz  L/delta={ld:5.2f}  {reg}")

    print("\nAnisotropy sensitivity (lambda=1.5):")
    print("  longer spacing & higher frequency -> larger sensitivity")
    for L in (5.0, 40.0):
        for f in (2e3, 4e4):
            s = anisotropy_sensitivity(rh, 1.5, L, f)
            print(f"  L={L:>5.0f} m  f={f:>7.0f} Hz  sensitivity={s:.3e}")

    df = dominant_factor(rh_values=[2.0, 10.0, 50.0, 200.0],
                         spacing_values=[5.0, 20.0, 40.0],
                         freq_values=[2e3, 1e4, 4e4])
    print(f"\nDominant controlling factor: {df}")

    return rh


if __name__ == "__main__":
    example_workflow()
