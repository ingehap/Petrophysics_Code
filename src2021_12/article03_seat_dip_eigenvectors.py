"""
Article 3: Taming the Thunder Horse With Axes and Vectors
Ruehlicke, Uhrin, Veselovsky, Schlaich (2021)
DOI: 10.30632/PJV62N6-2021a3

Applies the Statistical Eigenvector Analysis Technique (SEAT) to borehole-
image dip data from the Thunder Horse Field to reconstruct depositional slope
and the architecture of mass-transport complexes (MTCs).  The key idea: the
azimuthal trend of the symmetry (eigen) axis derived from a girdle of poles-
to-bedding is robust to structural-dip uncertainty (negligible below ~40 deg
dip), so the slump-fold axis is a reliable proxy for paleoslope strike.

Implements:

  - Dip (azimuth, magnitude) -> downward pole-to-bedding unit vector  (R1)
  - Orientation / scatter matrix  T = (1/N) sum(n n^T)                (R3)
  - Eigen-decomposition -> principal axes (min-eigenvalue = axis)     (R4)
  - Woodcock (1977) shape K and strength C                           (R5)
  - Vollmer (1990) P/G/R fabric indices                              (R6)
  - Eigenvector -> trend / plunge                                    (R8)
  - Structural-dip tilt-invariance check (the paper's central claim)

Note: this is an application paper; the SEAT math lives in Ruehlicke et al.
(2019).  The relations here are standard orientation-statistics forms
(Scheidegger 1965; Woodcock 1977; Vollmer 1990) consistent with the paper's
prose, flagged as reconstructions.  Angles are in degrees; vectors use an
East-North-Down (E, N, D) frame.
"""

import numpy as np


# ---------------------------------------------- R1: pole to bedding -----

def dip_to_pole(azimuth_deg, dip_deg):
    """Downward unit normal (pole to bedding) from dip azimuth & magnitude (R1).

    Returns (E, N, D) with D positive downward.
    """
    a = np.radians(azimuth_deg)
    d = np.radians(dip_deg)
    return np.array([np.sin(d) * np.sin(a),
                     np.sin(d) * np.cos(a),
                     np.cos(d)])


# ---------------------------------------------- R3: scatter matrix ------

def orientation_matrix(poles):
    """Normalized orientation (scatter) matrix  T = (1/N) sum(n n^T)  (R3).

    poles : (N, 3) array of unit vectors.
    """
    P = np.asarray(poles, float)
    return (P.T @ P) / P.shape[0]


# ---------------------------------------------- R4: principal axes ------

def principal_axes(T):
    """Eigenvalues (descending) and matching eigenvectors of T (R4)."""
    vals, vecs = np.linalg.eigh(T)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def symmetry_axis(poles):
    """Fold / slump symmetry axis = eigenvector of the MINIMUM eigenvalue."""
    _, vecs = principal_axes(orientation_matrix(poles))
    return vecs[:, -1]


# ---------------------------------------------- R5-R6: fabric -----------

def woodcock(lams):
    """Woodcock (1977) shape K = ln(l1/l2)/ln(l2/l3) and strength C = ln(l1/l3)."""
    l1, l2, l3 = (max(v, 1e-12) for v in lams)
    K = np.log(l1 / l2) / np.log(l2 / l3)
    C = np.log(l1 / l3)
    return float(K), float(C)


def vollmer(lams):
    """Vollmer (1990) point/girdle/random indices (P + G + R = 1)."""
    l1, l2, l3 = lams
    return {"P": float(l1 - l2), "G": float(2 * (l2 - l3)), "R": float(3 * l3)}


# ---------------------------------------------- R8: trend / plunge ------

def axis_to_trend_plunge(vec):
    """Convert an (E, N, D) axis to (trend_deg, plunge_deg)."""
    e, n, d = vec
    if d < 0:                      # point to the lower hemisphere
        e, n, d = -e, -n, -d
    trend = np.degrees(np.arctan2(e, n)) % 360.0
    plunge = np.degrees(np.arcsin(np.clip(d, -1, 1)))
    return float(trend), float(plunge)


def tilt_poles(poles, tilt_deg, about="E"):
    """Apply a rigid structural tilt about a horizontal axis (E or N)."""
    t = np.radians(tilt_deg)
    c, s = np.cos(t), np.sin(t)
    if about == "E":               # rotate in the N-D plane
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:                          # about N, rotate in E-D plane
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return np.asarray(poles, float) @ R.T


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: SEAT Dip Eigenvector Analysis")
    print("=" * 60)

    # Cylindrical fold with a known E-W (trend 090) horizontal axis:
    # beds dip due north and due south at a range of magnitudes.
    measurements = [(0, 30), (0, 50), (180, 30), (180, 50),
                    (0, 10), (180, 10), (0, 70), (180, 70)]
    poles = np.array([dip_to_pole(a, d) for a, d in measurements])

    lams, _ = principal_axes(orientation_matrix(poles))
    axis = symmetry_axis(poles)
    trend, plunge = axis_to_trend_plunge(axis)
    print(f"  eigenvalues (desc)     = {np.array2string(lams, precision=3)}")
    print(f"  recovered axis trend   = {trend:.1f} deg  (expect ~90)")
    print(f"  recovered axis plunge  = {plunge:.1f} deg  (expect ~0)")
    assert min(abs(trend - 90), abs(trend - 270)) < 2.0
    assert abs(plunge) < 2.0

    # Fabric: a girdle distribution -> Woodcock K < 1, Vollmer G dominant
    K, C = woodcock(lams)
    idx = vollmer(lams)
    print(f"  Woodcock K / C         = {K:.2f} / {C:.2f}")
    print(f"  Vollmer P/G/R          = {idx['P']:.2f}/{idx['G']:.2f}/{idx['R']:.2f}")
    assert K < 1.0, "girdle distribution should give Woodcock K < 1"
    assert idx["G"] > idx["P"]
    assert abs(idx["P"] + idx["G"] + idx["R"] - 1.0) < 1e-9

    # Tilt invariance: a <40 deg structural tilt leaves the axis TREND ~ unchanged
    tilted = tilt_poles(poles, tilt_deg=25, about="N")
    trend_t, plunge_t = axis_to_trend_plunge(symmetry_axis(tilted))
    print(f"  trend after 25deg tilt = {trend_t:.1f} deg")
    assert min(abs(trend_t - 90), abs(trend_t - 270)) < 3.0
    print("  PASS")
    return {"trend": trend, "plunge": plunge, "K": K, "vollmer": idx}


if __name__ == "__main__":
    test_all()
