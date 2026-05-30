"""
Article 9: Wellbore Positioning While Drilling With LWD Measurements
Poedjono, Nwosu, Martin (2019)
DOI: 10.30632/PJV60N3-2019a8

Real-time wellbore position is computed from LWD survey stations (measured
depth, inclination, azimuth) using the industry-standard minimum-curvature
method, which models the path between stations as a circular arc.  The position
and its uncertainty (error ellipsoid) feed geosteering and anti-collision
decisions.

Implements:

  - Dogleg angle and ratio factor between two survey stations
  - Minimum-curvature TVD / north / east increments
  - Cumulative 3D well path from a survey
  - Simple along-hole position-uncertainty growth

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the journal metadata and these are faithful
standard-form reconstructions of the minimum-curvature survey method the paper
applies.  Angles in degrees; depths/positions in metres.
"""

import numpy as np


# ---------------------------------------------- minimum curvature -------

def dogleg(inc1, azi1, inc2, azi2):
    """Dogleg angle (rad) between two stations from inclinations/azimuths."""
    i1, a1, i2, a2 = (np.radians(x) for x in (inc1, azi1, inc2, azi2))
    cos_dl = (np.cos(i2 - i1)
              - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1)))
    return float(np.arccos(np.clip(cos_dl, -1.0, 1.0)))


def ratio_factor(dl):
    """Minimum-curvature ratio factor  RF = (2/DL)*tan(DL/2)  (1 if DL=0)."""
    if dl < 1e-9:
        return 1.0
    return (2.0 / dl) * np.tan(dl / 2.0)


def station_increment(md1, inc1, azi1, md2, inc2, azi2):
    """Minimum-curvature (dTVD, dNorth, dEast) between two survey stations."""
    dmd = md2 - md1
    dl = dogleg(inc1, azi1, inc2, azi2)
    rf = ratio_factor(dl)
    i1, a1, i2, a2 = (np.radians(x) for x in (inc1, azi1, inc2, azi2))
    half = dmd / 2.0 * rf
    dN = half * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2))
    dE = half * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2))
    dV = half * (np.cos(i1) + np.cos(i2))
    return dV, dN, dE


def compute_path(survey):
    """Cumulative (TVD, North, East) path from a list of (MD, inc, azi) stations."""
    tvd = nth = est = 0.0
    path = [(tvd, nth, est)]
    for (md1, i1, a1), (md2, i2, a2) in zip(survey[:-1], survey[1:]):
        dV, dN, dE = station_increment(md1, i1, a1, md2, i2, a2)
        tvd += dV; nth += dN; est += dE
        path.append((tvd, nth, est))
    return np.array(path)


def position_uncertainty(md, error_rate=0.0015):
    """Simple along-hole position uncertainty (radius, m): grows with MD."""
    return error_rate * md


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Wellbore Positioning While Drilling (LWD)")
    print("=" * 60)

    # A straight vertical well: all displacement is TVD, none lateral
    dV, dN, dE = station_increment(0, 0, 0, 100, 0, 0)
    print(f"  vertical: dTVD/dN/dE   = {dV:.1f} / {dN:.1f} / {dE:.1f}")
    assert abs(dV - 100.0) < 1e-9 and abs(dN) < 1e-9 and abs(dE) < 1e-9

    # A perfectly horizontal section due north: all displacement is northing
    dV, dN, dE = station_increment(0, 90, 0, 100, 90, 0)
    assert abs(dV) < 1e-9 and abs(dN - 100.0) < 1e-9 and abs(dE) < 1e-9

    # Ratio factor is 1 for a straight hole and > 1 for a dogleg
    assert abs(ratio_factor(0.0) - 1.0) < 1e-12
    assert ratio_factor(np.radians(30.0)) > 1.0

    # A build-and-turn survey: TVD increases monotonically, well steps out
    survey = [(0, 0, 0), (500, 0, 0), (800, 30, 45),
              (1100, 60, 45), (1400, 90, 45)]
    path = compute_path(survey)
    tvd, north, east = path[-1]
    horiz = np.hypot(north, east)
    print(f"  final TVD / horiz disp = {tvd:.0f} / {horiz:.0f} m")
    assert np.all(np.diff(path[:, 0]) >= -1e-9)    # TVD non-decreasing
    assert horiz > 200.0                            # significant step-out
    # the lateral heads NE (azimuth 45 deg -> north ~ east)
    assert abs(north - east) < 1.0

    # Position uncertainty grows with measured depth
    assert position_uncertainty(3000.0) > position_uncertainty(1000.0)
    print("  PASS")
    return {"final_TVD": float(tvd), "horizontal_disp": float(horiz)}


if __name__ == "__main__":
    test_all()
