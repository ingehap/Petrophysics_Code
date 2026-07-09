"""
Article 7: Use of Data Analytics to Optimize Hydraulic Fracture Locations Along
           Borehole
Gupta, Rai, Devegowda, Sondergeld (2018)
DOI: 10.30632/PJV59N6Y2018a6

Hydraulic-fracture stages should be placed where the rock is brittle (easier to
fracture) and the minimum horizontal stress is low (fractures initiate and grow
more readily).  A composite completion-quality score from a brittleness index
and the stress profile ranks candidate perforation clusters, and an optimizer
selects the best locations subject to a minimum-spacing constraint.

Implements:

  - Rickman brittleness index from dynamic E and nu
  - Minimum-horizontal-stress profile (poroelastic)
  - Completion-quality score (high brittleness, low stress)
  - Optimal stage placement with a minimum-spacing constraint

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the brittleness / stress / placement-optimization analytics the paper applies.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- brittleness / stress ----

def brittleness_index(E, nu, E_min=10.0, E_max=80.0, nu_min=0.15, nu_max=0.40):
    """Rickman brittleness index (0-1): high E and low nu -> brittle.

        BI = 0.5*[(E - E_min)/(E_max - E_min) + (nu_max - nu)/(nu_max - nu_min)]
    """
    return np.clip(
        petrolib.acoustic_geomech.brittleness_rickman(
            E, nu, e_range=(E_min, E_max), nu_range=(nu_min, nu_max)),
        0.0, 1.0)


def min_horizontal_stress(overburden, p_pore, nu, biot=1.0):
    """Poroelastic minimum horizontal stress.

        sigma_h = (nu/(1-nu))*(S_v - alpha*Pp) + alpha*Pp
    """
    return petrolib.acoustic_geomech.min_horizontal_stress(overburden, p_pore, nu, biot=biot)


def completion_quality(bi, stress):
    """Completion-quality score: high brittleness, low (normalized) stress."""
    s = np.asarray(stress, float)
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return np.asarray(bi, float) * (1.0 - s_norm)


def optimize_stages(score, n_stages, min_spacing):
    """Greedily select stage indices by descending score with a min spacing."""
    order = np.argsort(score)[::-1]
    chosen = []
    for i in order:
        if all(abs(i - j) >= min_spacing for j in chosen):
            chosen.append(int(i))
        if len(chosen) == n_stages:
            break
    return sorted(chosen)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Optimizing Hydraulic Fracture Locations")
    print("=" * 60)

    # Brittleness: stiff, low-Poisson rock is more brittle
    assert brittleness_index(70.0, 0.18) > brittleness_index(20.0, 0.35)

    # Minimum horizontal stress rises with Poisson's ratio (more ductile)
    assert min_horizontal_stress(5000.0, 2000.0, 0.35) > \
        min_horizontal_stress(5000.0, 2000.0, 0.20)

    # Build a synthetic lateral profile and rank completion quality
    rng = np.random.default_rng(6)
    n = 50
    E = np.full(n, 30.0); E[[10, 25, 40]] = 70.0      # three brittle sweet spots
    nu = np.full(n, 0.30); nu[[10, 25, 40]] = 0.18
    Sv, Pp = 6000.0, 2500.0
    bi = brittleness_index(E, nu)
    stress = min_horizontal_stress(Sv, Pp, nu)
    cq = completion_quality(bi, stress)
    print(f"  CQ at sweet spots / mean = {cq[[10,25,40]].mean():.2f} / {cq.mean():.2f}")
    assert cq[[10, 25, 40]].mean() > cq.mean()

    # Optimizer places stages at the brittle sweet spots, honoring spacing
    stages = optimize_stages(cq, n_stages=3, min_spacing=5)
    print(f"  selected stages        = {stages}")
    assert set(stages) == {10, 25, 40}
    # spacing respected
    assert all(stages[i + 1] - stages[i] >= 5 for i in range(len(stages) - 1))
    print("  PASS")
    return {"stages": stages, "cq_sweet": float(cq[[10,25,40]].mean())}


if __name__ == "__main__":
    test_all()
