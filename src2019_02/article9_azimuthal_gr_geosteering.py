"""
Article 9: Modeling of Azimuthal Gamma-Ray Tools for Use in Geosteering in
           Unconventional Reservoirs
Wang, Stockhausen, Wyatt, Gulick (2019)
DOI: 10.30632/PJV60N1Y2019a8

In a horizontal well, an azimuthal gamma-ray tool measures gamma ray in
sectors around the borehole.  An approaching bed boundary is seen first on the
up- or down-facing sector, so the azimuthal contrast (up-sector minus
down-sector GR) signals proximity to the boundary and, with the bedding dip,
gives the distance to it - the core of geosteering.

Implements:

  - Azimuthal sector averaging (up / down / image)
  - Up-down azimuthal GR contrast (boundary proximity indicator)
  - Distance-to-boundary estimate from the contrast decay
  - Apparent bedding dip from the sinusoidal image response

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the azimuthal-GR geosteering relations the paper models.
"""

import numpy as np


# ---------------------------------------------- sectors -----------------

def sector_average(image_row, sector_indices):
    """Average GR over a set of azimuthal sector indices."""
    return float(np.mean(np.asarray(image_row, float)[sector_indices]))


def up_down_contrast(image_row):
    """Up-sector minus down-sector GR contrast (top half vs bottom half)."""
    img = np.asarray(image_row, float)
    n = len(img)
    up = img[np.r_[0:n // 8, 7 * n // 8:n]].mean()    # sectors around the top (0 deg)
    down = img[3 * n // 8: 5 * n // 8].mean()         # sectors around the bottom
    return float(up - down)


def distance_to_boundary(contrast, contrast_at_boundary, decay_length):
    """Distance to a boundary from the (exponentially decaying) GR contrast.

        contrast = contrast_at_boundary*exp(-d/decay_length)
        ->  d = -decay_length*ln(contrast/contrast_at_boundary)
    """
    return -decay_length * np.log(np.clip(contrast / contrast_at_boundary, 1e-9, 1.0))


def apparent_dip(image, depths, n_sectors):
    """Apparent bedding dip from the sinusoid traced by a boundary on the image.

    Fits z(phi) = z0 - r*tan(dip)*cos(phi - phi0) to the per-sector boundary
    depths; returns the dip (deg) given the tool radius r encoded as 1 here.
    """
    phi = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
    M = np.vstack([np.cos(phi), np.sin(phi), np.ones_like(phi)]).T
    c1, c2, _ = np.linalg.lstsq(M, np.asarray(depths, float), rcond=None)[0]
    amp = np.hypot(c1, c2)
    return float(np.degrees(np.arctan(amp)))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Azimuthal Gamma-Ray Geosteering")
    print("=" * 60)

    n = 16
    # Uniform formation -> no up/down contrast
    uniform = np.full(n, 60.0)
    assert abs(up_down_contrast(uniform)) < 1e-9

    # A shale above the well raises the up-sector GR -> positive contrast
    img = np.full(n, 40.0)
    img[np.r_[0:n // 8, 7 * n // 8:n]] = 110.0     # top sectors hot (shale above)
    c = up_down_contrast(img)
    print(f"  up-down GR contrast    = {c:.1f} API")
    assert c > 0

    # Distance to boundary falls as the contrast grows toward the boundary value
    d_far = distance_to_boundary(5.0, 50.0, 2.0)
    d_near = distance_to_boundary(40.0, 50.0, 2.0)
    print(f"  distance far/near      = {d_far:.2f} / {d_near:.2f} m")
    assert d_far > d_near >= 0

    # Apparent dip from a sinusoidal boundary image (plant a 20 deg dip)
    dip_true = 20.0
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    depths = 1000.0 - np.tan(np.radians(dip_true)) * np.cos(phi - 0.7)
    dip = apparent_dip(None, depths, n)
    print(f"  apparent dip           = {dip:.1f} deg  (true {dip_true})")
    assert abs(dip - dip_true) < 1.0
    print("  PASS")
    return {"contrast": c, "dist_near": float(d_near), "dip": dip}


if __name__ == "__main__":
    test_all()
