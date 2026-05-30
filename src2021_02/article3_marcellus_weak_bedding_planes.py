"""
Article 3: Conclusive Proof of Weak Bedding Planes in the Marcellus Shale and
           Proposed Mitigation Strategies
Kowan, Schanken, Jacobi (2021)
DOI: 10.30632/PJV62N1-2021a2

Borehole-image and core evidence shows the Marcellus fails preferentially along
its bedding.  The mechanics are captured by Jaeger's single-plane-of-weakness
theory: a rock containing a weak plane oriented at angle beta to the maximum
principal stress fails either by sliding along that plane (over a band of
orientations) or, outside that band, by intact Mohr-Coulomb shear.  The result
is the classic U-shaped strength-vs-bedding-angle curve with a minimum that
locates the weakest orientation -- the basis for the mitigation strategy of
orienting wells / raising mud weight to avoid bedding-parallel shear.

Implements:

  - Plane-of-weakness differential strength (Jaeger)
      (s1 - s3) = 2(c_w + mu_w s3) / [(1 - mu_w cot(beta)) sin(2 beta)]
  - Intact Mohr-Coulomb differential strength
  - Combined (minimum of the two) strength vs bedding angle
  - Weakest-orientation angle  beta_min = 45 deg + phi_w/2
  - Mud-weight floor to suppress bedding-plane slip

Note: this issue's source PDF has no usable text layer, so the formulas are
faithful standard-form reconstructions of the single-weak-plane mechanics the
paper's argument rests on.  Stresses in MPa, angles in degrees.
"""

import numpy as np


# ---------------------------------------------- weakness-plane ----------

def weakness_strength(beta_deg, sigma3, c_w, mu_w):
    """Differential stress to slide on a weak plane at angle beta to sigma1.

        (s1 - s3) = 2(c_w + mu_w*s3) / [(1 - mu_w*cot(beta)) * sin(2*beta)]
    Valid only where the denominator is positive (the slip band); elsewhere the
    plane cannot slip and np.inf is returned (intact rock governs).
    """
    beta = np.radians(np.asarray(beta_deg, float))
    denom = (1.0 - mu_w / np.tan(beta)) * np.sin(2.0 * beta)
    out = np.full_like(beta, np.inf)
    ok = denom > 1e-9
    out[ok] = 2.0 * (c_w + mu_w * sigma3) / denom[ok]
    return out


def intact_strength(sigma3, c0, mu_i):
    """Intact Mohr-Coulomb differential strength (orientation independent).

        (s1 - s3) = 2(c0 + mu_i*s3) * (sqrt(mu_i^2+1) + mu_i)
    Returns a scalar differential stress for the intact rock matrix.
    """
    return 2.0 * (c0 + mu_i * sigma3) * (np.sqrt(mu_i ** 2 + 1.0) + mu_i)


def combined_strength(beta_deg, sigma3, c_w, mu_w, c0, mu_i):
    """Operative strength = min(weak-plane slip, intact failure) at each beta."""
    weak = weakness_strength(beta_deg, sigma3, c_w, mu_w)
    intact = intact_strength(sigma3, c0, mu_i)
    return np.minimum(weak, intact)


def weakest_angle(mu_w):
    """Bedding angle of minimum strength  beta_min = 45 deg + phi_w/2."""
    phi_w = np.degrees(np.arctan(mu_w))
    return 45.0 + phi_w / 2.0


# ---------------------------------------------- mitigation --------------

def min_mud_weight(sigma3_gradient, depth_m, c_w, mu_w, beta_deg,
                   sigma1_minus_sigma3):
    """Minimum equivalent mud density (kg/m^3) so the well stays below the
    bedding-slip strength at orientation beta.

    Increasing mud weight raises the near-wellbore confining stress s3; this
    returns the s3 (and the equivalent mud density at the given depth) at which
    the weak-plane strength just exceeds the applied differential stress.
    """
    beta = np.radians(beta_deg)
    denom = (1.0 - mu_w / np.tan(beta)) * np.sin(2.0 * beta)
    if denom <= 0:
        return 0.0                       # plane cannot slip at this beta
    # solve weakness_strength(beta, s3) = applied  ->  for s3
    s3_req = (sigma1_minus_sigma3 * denom / 2.0 - c_w) / mu_w
    s3_req = max(s3_req, 0.0)
    # equivalent mud density: s3 = rho g depth  (MPa = 1e6 Pa)
    return s3_req * 1e6 / (9.81 * depth_m)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Weak Bedding Planes in the Marcellus Shale")
    print("=" * 60)

    sigma3 = 20.0          # MPa confining
    c_w, mu_w = 5.0, 0.5   # weak-plane cohesion / friction
    c0, mu_i = 20.0, 0.8   # intact cohesion / friction

    beta = np.arange(1.0, 90.0, 1.0)
    strength = combined_strength(beta, sigma3, c_w, mu_w, c0, mu_i)

    # The minimum strength occurs near beta_min = 45 + phi_w/2
    b_min_pred = weakest_angle(mu_w)
    b_min_obs = beta[np.argmin(strength)]
    print(f"  weakest angle  pred / obs = {b_min_pred:.1f} / {b_min_obs:.1f} deg")
    assert abs(b_min_obs - b_min_pred) < 3.0

    # Strength is reduced by the weak plane only inside a band; at beta -> 0 or
    # 90 deg the plane cannot slip and intact strength governs.
    intact = intact_strength(sigma3, c0, mu_i)
    print(f"  intact strength        = {intact:.1f} MPa")
    print(f"  min (bedding) strength = {strength.min():.1f} MPa @ {b_min_obs:.0f} deg")
    assert strength.min() < intact                       # weak plane lowers it
    assert abs(strength[0] - intact) < 1e-6              # beta~0 -> intact
    assert abs(strength[-1] - intact) < 1e-6            # beta~90 -> intact

    # Mitigation: raising mud weight (s3) raises the bedding-slip strength
    s_lo = weakness_strength(b_min_obs, 10.0, c_w, mu_w)
    s_hi = weakness_strength(b_min_obs, 30.0, c_w, mu_w)
    print(f"  bedding strength s3=10 / 30 = {float(s_lo):.1f} / {float(s_hi):.1f} MPa")
    assert s_hi > s_lo

    mw = min_mud_weight(0.0226, 2500.0, c_w, mu_w, b_min_obs, 25.0)
    print(f"  required mud density   = {mw:.0f} kg/m^3")
    assert mw > 0
    print("  PASS")
    return {"beta_min": b_min_obs, "min_strength": float(strength.min()),
            "intact": intact}


if __name__ == "__main__":
    test_all()
