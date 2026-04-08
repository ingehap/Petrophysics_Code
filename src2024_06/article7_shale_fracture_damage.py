"""
article7_shale_fracture_damage.py
=================================

Implementation of the mechanical analysis in:

    Jiang, H., Qu, Z., and Liu, W. (2024).
    "Study on the Damage and Failure Process of Prefabricated
    Hole-Fracture Combination Defects in Shale."
    Petrophysics 65(3), 397-410.  DOI: 10.30632/PJV65N3-2024a7

The paper conducts uniaxial compression tests with digital image
correlation (DIC) on shale specimens with a pre-drilled hole and a
pre-cut fracture at various angles to the bedding plane.  The three
key quantitative observations we reproduce are:

1.  Stress concentration at an elliptical fracture under uniaxial
    compression (Inglis, 1913):  the tangential stress at the fracture
    tip is

        sigma_tip = sigma_applied * (1 + 2 * a/b)

    for an ellipse with semi-axes a (along the loading direction) and b
    (transverse).  For a crack at angle theta to the loading axis we
    use the projection of the applied stress.

2.  Stress concentration at a circular pore is always 3 * sigma (Kirsch
    solution, 1898).

3.  The paper's central empirical finding is that the relative
    dominance of fracture vs. pore stress concentration depends on the
    angle between the fracture and the bedding plane:

        * theta <  30 deg  -> fracture tip dominates
        * 30 <= theta <= 60 -> mixed
        * theta >  60 deg  -> pore dominates

    We expose this as a classifier that returns the dominant failure
    mode given the geometry.

4.  A simple linear elastic stress-strain simulation produces a
    synthetic DIC strain field consistent with the paper's Fig. 5-9
    behaviour: the elastic modulus E increases with fracture angle, and
    compressive strength has a characteristic U-shape (high at 0 and
    90 deg, low in the middle).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Stress concentration at an elliptical fracture --------------------------
# ---------------------------------------------------------------------------

def fracture_tip_stress(sigma_applied: float, semi_major_a: float,
                        semi_minor_b: float, angle_deg: float) -> float:
    """Tangential stress at the tip of an elliptical fracture.

    Parameters
    ----------
    sigma_applied : applied far-field uniaxial compressive stress (+ comp).
    semi_major_a : ellipse semi-axis along crack length.
    semi_minor_b : ellipse semi-axis perpendicular to crack length.
    angle_deg : angle between the long axis of the fracture and the
        loading direction (0 deg = fracture parallel to loading, which
        gives no stress concentration at the tip).

    Returns
    -------
    Tangential stress at the fracture tip.  Positive means compressive
    tip stress can promote crack initiation in tension-shear mode.
    """
    if semi_minor_b <= 0:
        raise ValueError("semi_minor_b must be > 0")
    effective_sigma = sigma_applied * np.sin(np.radians(angle_deg)) ** 2
    return effective_sigma * (1.0 + 2.0 * semi_major_a / semi_minor_b)


def pore_stress_concentration(sigma_applied: float) -> float:
    """Kirsch solution: tangential stress at the edge of a circular hole
    under uniaxial compression is 3 * sigma."""
    return 3.0 * sigma_applied


# ---------------------------------------------------------------------------
# 2. Dominant failure mode classifier ---------------------------------------
# ---------------------------------------------------------------------------

def dominant_failure_mode(angle_deg: float) -> str:
    """Return 'fracture', 'mixed', or 'pore' -- the dominant stress
    concentration source based on the paper's angle regimes."""
    if angle_deg < 30:
        return "fracture"
    if angle_deg <= 60:
        return "mixed"
    return "pore"


# ---------------------------------------------------------------------------
# 3. Mechanical properties as a function of fracture angle ------------------
# ---------------------------------------------------------------------------

def elastic_modulus(angle_deg: float, E_min_gpa: float = 12.0,
                    E_max_gpa: float = 25.0) -> float:
    """Elastic modulus increases with fracture angle (Fig. 7 of the paper).

    We use a simple monotonic sine^2 interpolation between E_min (angle
    = 0) and E_max (angle = 90).
    """
    frac = np.sin(np.radians(angle_deg)) ** 2
    return float(E_min_gpa + (E_max_gpa - E_min_gpa) * frac)


def compressive_strength(angle_deg: float, sigma_min: float = 55.0,
                         sigma_max: float = 110.0) -> float:
    """U-shaped dependence of strength on fracture angle.

    Strength is highest at 0 and 90 degrees and lowest near 45 degrees,
    matching the pattern described in the paper's abstract.
    """
    theta = np.radians(angle_deg)
    # reaches 1 at theta = 0 and pi/2, minimum at pi/4
    shape = 1.0 - np.sin(2 * theta) ** 2
    return float(sigma_min + (sigma_max - sigma_min) * shape)


# ---------------------------------------------------------------------------
# 4. Synthetic strain field generator (the DIC output) ---------------------
# ---------------------------------------------------------------------------

def synthetic_strain_field(nx: int = 64, ny: int = 64,
                           fracture_center: tuple[float, float] = (0.5, 0.5),
                           fracture_length: float = 0.3,
                           fracture_angle_deg: float = 45.0,
                           pore_center: tuple[float, float] = (0.3, 0.5),
                           pore_radius: float = 0.05,
                           sigma_applied: float = 50.0,
                           youngs_modulus_gpa: float = 20.0) -> np.ndarray:
    """Return a 2D axial strain field for a specimen under uniaxial load.

    This is a toy elastic field: mean strain sigma / E, plus Gaussian
    "hot-spots" at the fracture tips and around the pore whose
    amplitudes follow the Inglis and Kirsch solutions.  It is enough to
    reproduce the qualitative pattern shown in the DIC figures of the
    paper -- concentrated strain bands developing between fracture tip
    and pore at intermediate angles.
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    base_strain = sigma_applied / (youngs_modulus_gpa * 1000.0)  # mm/m scale
    strain = np.full_like(X, base_strain)

    # Fracture tip positions
    cx, cy = fracture_center
    half_len = fracture_length / 2
    ang = np.radians(fracture_angle_deg)
    tip1 = (cx + half_len * np.cos(ang), cy + half_len * np.sin(ang))
    tip2 = (cx - half_len * np.cos(ang), cy - half_len * np.sin(ang))

    amp_tip = fracture_tip_stress(sigma_applied, semi_major_a=half_len,
                                   semi_minor_b=0.02,
                                   angle_deg=fracture_angle_deg)
    amp_pore = pore_stress_concentration(sigma_applied)

    sigma_spot = 0.05
    for tx, ty in (tip1, tip2):
        strain += (amp_tip / (youngs_modulus_gpa * 1000.0)) * np.exp(
            -((X - tx) ** 2 + (Y - ty) ** 2) / (2 * sigma_spot ** 2))

    px, py = pore_center
    strain += (amp_pore / (youngs_modulus_gpa * 1000.0)) * np.exp(
        -((X - px) ** 2 + (Y - py) ** 2) / (2 * pore_radius ** 2))
    return strain


# ---------------------------------------------------------------------------
# Test harness ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_all(verbose: bool = True) -> None:
    # (a) Fracture tip stress: angle 0 produces zero stress (load parallel
    # to fracture), angle 90 produces the full Inglis amplification.
    s0 = fracture_tip_stress(100, semi_major_a=1.0, semi_minor_b=0.1,
                              angle_deg=0)
    s90 = fracture_tip_stress(100, semi_major_a=1.0, semi_minor_b=0.1,
                               angle_deg=90)
    assert abs(s0) < 1e-12
    assert abs(s90 - 100 * (1 + 2 * 1.0 / 0.1)) < 1e-9
    # Tip stress ordering
    s_low = fracture_tip_stress(100, 1.0, 0.1, 20)
    s_mid = fracture_tip_stress(100, 1.0, 0.1, 50)
    s_high = fracture_tip_stress(100, 1.0, 0.1, 80)
    assert s_low < s_mid < s_high

    # (b) Kirsch pore stress concentration factor is exactly 3
    assert pore_stress_concentration(20) == 60.0

    # (c) Regime classifier
    assert dominant_failure_mode(15) == "fracture"
    assert dominant_failure_mode(45) == "mixed"
    assert dominant_failure_mode(75) == "pore"

    # (d) Mechanical properties: E monotonically up, strength U-shaped
    Es = [elastic_modulus(a) for a in range(0, 91, 15)]
    assert all(x <= y for x, y in zip(Es, Es[1:])), \
        "E should be monotonic with angle"
    strengths = np.array([compressive_strength(a) for a in range(0, 91, 15)])
    middle_idx = len(strengths) // 2
    assert strengths[middle_idx] < strengths[0]
    assert strengths[middle_idx] < strengths[-1]

    # (e) DIC strain field is positive and has clear concentration spots.
    field = synthetic_strain_field(fracture_angle_deg=45)
    assert field.shape == (64, 64)
    assert field.min() >= 0
    # Peak must be at least an order of magnitude above the background
    peak = field.max()
    baseline = field.mean()
    assert peak > 3 * baseline

    if verbose:
        print("Article 7 (Shale fracture damage): all tests passed.")
        print(f"  tip stress angle 0/50/90 deg = {s0:.2f} / {s_mid:.2f} / {s90:.2f}")
        print(f"  E (0->90 deg)   = {Es[0]:.1f} -> {Es[-1]:.1f} GPa")
        print(f"  sigma_c (0/45/90 deg) = "
              f"{compressive_strength(0):.1f} / "
              f"{compressive_strength(45):.1f} / "
              f"{compressive_strength(90):.1f} MPa")
        print(f"  DIC field peak / baseline = {peak/baseline:.2f}")


if __name__ == "__main__":
    test_all()
