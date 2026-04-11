"""
Khan et al. (2023), Petrophysics 64(6): 954-969.
Nonlinear creep-damage modeling of solution-mined salt caverns used for
H2/CO2 storage. Goal: predict time-dependent strain and cavern closure
during cyclic operational pressure.

Implements a Norton-power-law creep with a scalar damage variable D
following a Kachanov-type evolution.
"""
import numpy as np


def norton_creep_rate(sigma, A=1e-12, n=4.5):
    """Steady-state strain rate (1/day) given differential stress (MPa)."""
    return A * sigma ** n


def damage_rate(sigma, D, B=1e-14, r=5.0, k=0.7):
    """Kachanov damage evolution."""
    return B * (sigma / (1 - D + 1e-12)) ** r * (1 - D) ** (-k)


def simulate_cavern(p_cavern, p_litho, days, dt=0.5, A=1e-12, n=4.5):
    """Time-march creep strain + damage. Returns t, eps, D."""
    t = np.arange(0, days, dt)
    sigma = np.maximum(p_litho - p_cavern, 1e-3)  # diff stress, MPa
    if np.isscalar(sigma): sigma = np.full_like(t, sigma)
    eps = np.zeros_like(t); D = np.zeros_like(t)
    for i in range(1, len(t)):
        s_eff = sigma[i] / (1 - D[i - 1] + 1e-12)
        eps[i] = eps[i - 1] + norton_creep_rate(s_eff, A, n) * dt
        D[i] = min(0.99, D[i - 1] + damage_rate(sigma[i], D[i - 1]) * dt)
    closure = 1 - np.exp(-eps)  # fractional volume reduction
    return t, eps, D, closure


def test_all():
    days = 365
    t, eps, D, closure = simulate_cavern(p_cavern=10.0, p_litho=25.0, days=days)
    print("Khan et al. salt-cavern creep model:")
    print(f"  final strain    : {eps[-1]:.4e}")
    print(f"  final damage    : {D[-1]:.4f}")
    print(f"  cavern closure  : {closure[-1]*100:.2f} %")
    assert eps[-1] > 0 and D[-1] > 0 and 0 < closure[-1] < 1
    print("  PASS")


if __name__ == "__main__":
    test_all()
