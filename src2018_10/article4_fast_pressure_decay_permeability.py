"""
Article 4: Fast Pressure-Decay Core Permeability Measurement for Tight Rocks
Gan, Griffin, Dacy, Xie, Lee (2018)
DOI: 10.30632/PJV59N5-2018a3

Permeability of tight rocks (microdarcy to nanodarcy) is measured by the
pressure-decay (pulse-decay) method: a pressure pulse across the core relaxes
exponentially as gas flows through it, with a time constant inversely
proportional to permeability.  A fast variant shortens the test by fitting the
early-time decay.

Implements:

  - Pulse-decay pressure relaxation  dP(t) = dP0*exp(-t/tau)
  - Decay time constant  1/tau = (k*A/(mu*L*beta))*(1/Vu + 1/Vd)
  - Permeability from the fitted decay rate
  - Klinkenberg gas-slippage note (apparent vs liquid permeability)

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the pulse-decay permeability relations the paper applies.  SI units.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- pulse decay -------------

def decay_rate(k, A, L, mu, beta, Vu, Vd):
    """Pulse-decay rate  1/tau = (k*A/(mu*L*beta))*(1/Vu + 1/Vd)  (1/s).

    k = permeability (m^2), A = area, L = length, mu = gas viscosity,
    beta = gas compressibility, Vu/Vd = upstream/downstream reservoir volumes.
    """
    return (k * A / (mu * L * beta)) * (1.0 / Vu + 1.0 / Vd)


def pressure_decay(dp0, rate, t):
    """Differential pressure decay  dP(t) = dP0*exp(-rate*t)."""
    return dp0 * np.exp(-rate * np.asarray(t, float))


def fit_permeability(t, dp, A, L, mu, beta, Vu, Vd):
    """Recover permeability from a measured pressure-decay curve.

    Fits ln(dP) = ln(dP0) - rate*t, then inverts the decay-rate relation for k.
    """
    t = np.asarray(t, float); dp = np.asarray(dp, float)
    slope, _ = np.polyfit(t, np.log(dp), 1)
    rate = -slope
    geom = (A / (mu * L * beta)) * (1.0 / Vu + 1.0 / Vd)
    return rate / geom


def klinkenberg(k_l, b, p_mean):
    """Apparent gas permeability  k_app = k_l*(1 + b/Pm)."""
    return petrolib.flow_transport.klinkenberg_apparent(k_l, b=b, p_mean=p_mean)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Fast Pressure-Decay Core Permeability")
    print("=" * 60)

    # Test conditions for a tight core
    A = np.pi * (0.0254 / 2) ** 2          # 1-in. diameter core
    L = 0.0254                              # 1-in. length
    mu = 1.8e-5                             # N2 viscosity (Pa.s)
    beta = 1e-5                             # 1/Pa
    Vu = Vd = 5e-6                          # m^3 reservoir volumes

    # Higher permeability -> faster decay
    k_true = 1e-18                          # ~1 microdarcy (1 mD ~ 1e-15 m^2)
    rate = decay_rate(k_true, A, L, mu, beta, Vu, Vd)
    rate_hi = decay_rate(1e-17, A, L, mu, beta, Vu, Vd)
    print(f"  decay rate 1uD / 10uD  = {rate:.4f} / {rate_hi:.4f} 1/s")
    assert rate_hi > rate

    # Pressure decays exponentially; fit recovers the planted permeability
    t = np.linspace(0, 600, 60)
    dp = pressure_decay(5e4, rate, t)
    k_fit = fit_permeability(t, dp, A, L, mu, beta, Vu, Vd)
    print(f"  fitted k               = {k_fit:.3e} m^2 (true {k_true:.0e})")
    assert abs(k_fit - k_true) / k_true < 1e-6

    # Klinkenberg: apparent gas permeability exceeds the liquid value at low Pm
    assert klinkenberg(k_true, 5e5, 1e6) > k_true
    assert klinkenberg(k_true, 5e5, 1e6) > klinkenberg(k_true, 5e5, 5e6)
    print("  PASS")
    return {"decay_rate": float(rate), "k_fit": float(k_fit)}


if __name__ == "__main__":
    test_all()
