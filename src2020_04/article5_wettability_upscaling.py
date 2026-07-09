"""
Article 5: Workflow for Upscaling Wettability From the Nanoscale to Core Scale
Rucker, Bartels, Bultreys, Boone, Singh, Garfi, Scanziani, Spurin,
Yesufu-Rufai, Krevor, Blunt, Wilson, Mahani, Cnudde, Luckham, Georgiadis,
Berg (2020)
DOI: 10.30632/PJV61N2-2020a5

An integrated multiscale workflow combines atomic-force-microscopy surface
roughness (nanoscale), micro-CT pore-scale imaging, and topological drainage
simulation to upscale wettability (contact-angle distribution) from the
nanometer to the centimeter scale.  Drainage follows the Young-Laplace
threshold (higher capillary pressure invades smaller pores), and surface
roughness modifies the apparent contact angle through the Wenzel relation.

Implements:

  - Young-Laplace drainage threshold radius  r = 2*sigma*cos(theta)/Pc
  - Wenzel roughness-corrected contact angle  cos(theta_app) = r_w*cos(theta)
  - Volume-weighted contact-angle upscaling to the core scale

Note: this issue's PDF has a text layer; the paper invokes the Young-Laplace
equation (qualitatively) and Wenzel/Cassie roughness concepts but writes no
numbered closed-form equations, so this module implements those standard
relations.  Paper anchors: Ketton carbonate (99.9% calcite), water-wet
reference contact angle 30 deg.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Young-Laplace -----------

def drainage_threshold_radius(pc, sigma, theta_deg):
    """Largest pore radius still water-filled at capillary pressure Pc.

        r = 2*sigma*cos(theta)/Pc
    Higher Pc invades smaller pores (drainage).
    """
    return petrolib.capillary_pressure.washburn_radius(
        pc, sigma=sigma, theta_deg=theta_deg, absolute=False)


def capillary_pressure(sigma, theta_deg, r):
    """Young-Laplace capillary entry pressure  Pc = 2*sigma*cos(theta)/r."""
    return petrolib.capillary_pressure.young_laplace_pc(
        r, sigma=sigma, theta_deg=theta_deg, absolute=False)


# ---------------------------------------------- Wenzel roughness --------

def wenzel_contact_angle(theta_young_deg, roughness):
    """Apparent contact angle from surface roughness  cos(theta_app)=r*cos(theta).

    roughness r_w >= 1 is the actual/projected area ratio; it amplifies the
    intrinsic wettability (a water-wet surface becomes more water-wet).
    Returns the apparent contact angle in degrees (clipped to [0, 180]).
    """
    c = roughness * np.cos(np.radians(theta_young_deg))
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


# ---------------------------------------------- upscaling ---------------

def upscale_contact_angle(thetas_deg, volumes):
    """Pore-volume-weighted mean contact angle (nanoscale -> core scale)."""
    thetas = np.asarray(thetas_deg, float)
    v = np.asarray(volumes, float)
    return float(np.sum(thetas * v) / np.sum(v))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Upscaling Wettability Nanoscale -> Core")
    print("=" * 60)

    # Young-Laplace: higher capillary pressure invades smaller pores
    sigma, theta = 0.03, 30.0
    r_lo = drainage_threshold_radius(1e4, sigma, theta)
    r_hi = drainage_threshold_radius(1e5, sigma, theta)
    print(f"  threshold r @1e4/1e5 Pa = {r_lo*1e6:.2f} / {r_hi*1e6:.2f} um")
    assert r_hi < r_lo
    # round-trip with the entry pressure
    assert abs(capillary_pressure(sigma, theta, r_lo) - 1e4) < 1e-3

    # Wenzel: roughness makes a water-wet surface more water-wet (lower angle)
    th_app = wenzel_contact_angle(30.0, roughness=1.5)
    print(f"  Wenzel angle (30 deg, r=1.5) = {th_app:.1f} deg")
    assert th_app < 30.0
    # an oil-wet surface (>90) becomes more oil-wet under roughness
    assert wenzel_contact_angle(120.0, 1.5) > 120.0

    # Upscaling: a volume-weighted mix of water-wet and oil-wet pores
    theta_core = upscale_contact_angle([30.0, 120.0], [0.7, 0.3])
    print(f"  upscaled contact angle = {theta_core:.1f} deg")
    assert 30.0 < theta_core < 120.0 and abs(theta_core - 57.0) < 1e-6
    print("  PASS")
    return {"r_hi_um": float(r_hi * 1e6), "wenzel_30": float(th_app),
            "theta_core": theta_core}


if __name__ == "__main__":
    test_all()
