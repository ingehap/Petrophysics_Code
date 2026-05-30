"""
Article 6: Maximizing Net Pay in Penta-Lateral Well With Advanced Proactive
           Geosteering and 2D Structural Analysis Using Azimuthal Resistivity
           Measurements
Antonov, Kushnir, Martakov, Pazos, Small, Tropin, Maraj, Itter, Nelson,
Rabinovich (2021)
DOI: 10.30632/PJV62N3-2021a5

A case study using azimuthal/extra-deep azimuthal resistivity LWD to proactively
geosteer a five-branch (penta-lateral) well and maximize net pay, with a post-
well 2D structural inversion reconstructing bed and fault geometry from distance-
to-boundary (D2B) picks across the wells.

Implements:

  - Measured depth -> true vertical depth  TVD = sum(dMD*cos(inc))
  - Boundary TVD from a distance-to-boundary pick
  - Apparent <-> true dip  tan(app) = tan(true)*cos(beta)
  - Structural dip from two boundary picks
  - Least-squares fault-plane fit -> dip & azimuth
  - Net-pay (reservoir contact) percentage along a lateral

Note: the paper is a descriptive case study with no equations; the relations
here are the standard borehole-geometry / plane-fitting identities the workflow
relies on (flagged).  Depths in ft, dip/azimuth in degrees, net pay in percent.
"""

import numpy as np


# ---------------------------------------------- borehole geometry -------

def md_to_tvd(dmd, inc_deg):
    """True vertical depth increment  TVD = sum(dMD * cos(inclination))."""
    return float(np.sum(np.asarray(dmd, float) * np.cos(np.radians(inc_deg))))


def boundary_tvd(tvd_well, d2b, inc_deg, side=1.0):
    """Boundary TVD from a distance-to-boundary pick.

    side = +1 for a boundary below the well, -1 for above; the measured
    distance is projected vertically by cos(inclination).
    """
    return tvd_well + side * d2b * np.cos(np.radians(inc_deg))


def apparent_dip(true_dip_deg, section_angle_deg):
    """Apparent dip in a section at angle beta from true-dip azimuth:
    tan(app) = tan(true)*cos(beta)."""
    return np.degrees(np.arctan(np.tan(np.radians(true_dip_deg)) *
                                np.cos(np.radians(section_angle_deg))))


def structural_dip(d_tvd, d_horizontal):
    """Structural dip between two boundary picks  = atan(dTVD / dHoriz)."""
    return np.degrees(np.arctan2(d_tvd, d_horizontal))


# ---------------------------------------------- fault-plane fit ---------

def _normal_to_dip_azimuth(n):
    """Convert a plane normal (E, N, Down) to (dip_deg, dip_azimuth_deg)."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    dip = np.degrees(np.arccos(abs(n[2])))
    az = np.degrees(np.arctan2(n[0], n[1])) % 360.0
    return float(dip), float(az)


def fit_fault_plane(points):
    """Least-squares plane through >=3 (E, N, TVD) points -> (dip, azimuth).

    The plane normal is the smallest-singular-value direction of the centered
    coordinates.
    """
    P = np.asarray(points, float)
    centroid = P.mean(axis=0)
    _, _, vh = np.linalg.svd(P - centroid)
    normal = vh[-1]
    return _normal_to_dip_azimuth(normal)


# ---------------------------------------------- net pay -----------------

def net_pay_percent(inside_mask, segment_lengths):
    """Reservoir-contact percentage = inside footage / total footage * 100."""
    inside = np.asarray(inside_mask, bool)
    L = np.asarray(segment_lengths, float)
    return 100.0 * L[inside].sum() / L.sum()


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Proactive Geosteering & 2D Structural Analysis")
    print("=" * 60)

    # MD -> TVD: a near-horizontal lateral barely gains TVD
    tvd = md_to_tvd([100, 100, 100.0], inc_deg=88.0)
    print(f"  TVD over 300 ft @88deg = {tvd:.2f} ft")
    assert tvd < 12.0
    # a vertical section converts MD directly to TVD
    assert abs(md_to_tvd([500.0], 0.0) - 500.0) < 1e-9

    # Boundary 7 ft below a near-horizontal well
    b = boundary_tvd(tvd_well=10000.0, d2b=7.0, inc_deg=88.0, side=1.0)
    assert b > 10000.0 and b < 10000.0 + 7.0

    # Apparent dip is never steeper than true dip
    assert apparent_dip(40.0, 60.0) < 40.0
    assert abs(apparent_dip(40.0, 0.0) - 40.0) < 1e-9

    # Fault-plane fit recovers a planted dip & azimuth (OBc: 44 deg / 23 deg)
    dip_t, az_t = 44.0, 23.0
    n = np.array([np.sin(np.radians(dip_t)) * np.sin(np.radians(az_t)),
                  np.sin(np.radians(dip_t)) * np.cos(np.radians(az_t)),
                  np.cos(np.radians(dip_t))])
    rng = np.random.default_rng(0)
    # generate points on the plane n.(p-p0)=0
    p0 = np.array([100.0, 200.0, 11626.0])
    pts = []
    for _ in range(8):
        e, nth = rng.uniform(-50, 50, 2)
        # solve for down-coordinate so the point lies on the plane
        d = p0[2] - (n[0] * (p0[0] + e - p0[0]) + n[1] * (p0[1] + nth - p0[1])) / n[2]
        pts.append([p0[0] + e, p0[1] + nth, d])
    dip_f, az_f = fit_fault_plane(pts)
    print(f"  fitted fault dip/azim  = {dip_f:.1f} / {az_f:.1f} (true {dip_t}/{az_t})")
    assert abs(dip_f - dip_t) < 1.0
    assert min(abs(az_f - az_t), abs(az_f - az_t - 180)) < 1.0

    # Net pay along a lateral
    inside = [True, True, False, True, True]
    lengths = [200, 300, 150, 250, 100.0]
    np_pct = net_pay_percent(inside, lengths)
    print(f"  net pay                = {np_pct:.1f}%")
    assert abs(np_pct - 100.0 * 850 / 1000) < 1e-9
    print("  PASS")
    return {"fault_dip": dip_f, "fault_azimuth": az_f, "net_pay_pct": np_pct}


if __name__ == "__main__":
    test_all()
