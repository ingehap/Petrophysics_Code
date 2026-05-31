"""
Article 6 (Technical Note): Normalizing Gamma-Ray Logs Acquired from a Mixture of
           Vertical and Horizontal Wells in the Haynesville Shale
Xu, Bayer, Wunderle, Bansal (2016)
Reference: Petrophysics Vol. 57, No. 6 (December 2016), pp. 638-643
DOI: none assigned (this issue predates SPWLA DOI assignment)

Gamma-ray logs from a mix of vertical and horizontal wells are normalized to a
common statistical distribution using a true-stratigraphic-projection (TSP)
workflow: normalize the vertical wells, build a synthetic GR along each
horizontal path by projecting the nearest vertical well at equal true
stratigraphic thickness, then match the raw LWD GR to that synthetic by
equalizing its mean and standard deviation.

Implements:

  - Histogram normalization of a GR log to a reference range
  - True-stratigraphic-thickness projection (equal TST -> equal GR)
  - Affine normalization to a target mean and standard deviation
  - Maximum percent shift applied by the normalization

Note: this technical note is procedural (no display equations), so the relations
below are the standard histogram / mean-sigma normalization it describes.  GR in
API.
"""

import numpy as np


# ---------------------------------------------- normalization --------------

def histogram_normalize(gr, gr_min, gr_max, ref_min, ref_max):
    """Linear histogram normalization of a GR log onto a reference [ref_min, ref_max]."""
    gr = np.asarray(gr, float)
    return ref_min + (gr - gr_min) * (ref_max - ref_min) / (gr_max - gr_min)


def tst_projection(gr_vertical, tst_vertical, tst_query):
    """Synthetic GR along a horizontal path: interpolate the vertical-well GR at
    equal true stratigraphic thickness (layer-cake: equal TST -> equal GR)."""
    return np.interp(np.asarray(tst_query, float),
                     np.asarray(tst_vertical, float), np.asarray(gr_vertical, float))


def affine_normalize(gr_raw, target_mean, target_std):
    """Affine normalization to a target mean and standard deviation

        GR_norm = (GR_raw - mean)/std * target_std + target_mean.
    """
    gr = np.asarray(gr_raw, float)
    return (gr - gr.mean()) / gr.std() * target_std + target_mean


def max_percent_shift(gr_raw, gr_norm):
    """Maximum percent shift the normalization applies to any sample."""
    gr = np.asarray(gr_raw, float)
    return float(np.max(np.abs(np.asarray(gr_norm, float) - gr) / gr) * 100.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6 (TN): GR Normalization (Haynesville)")
    print("=" * 60)

    # Histogram normalization maps the endpoints onto the reference range
    gr = np.array([20.0, 70.0, 120.0])
    n = histogram_normalize(gr, 20.0, 120.0, 30.0, 150.0)
    assert np.isclose(n[0], 30.0) and np.isclose(n[-1], 150.0)

    # TST projection interpolates the vertical-well GR at the queried TST
    grv = np.array([40.0, 90.0, 150.0])
    tst = np.array([0.0, 10.0, 20.0])
    assert np.isclose(tst_projection(grv, tst, 5.0), 65.0)

    # Affine normalization matches the target mean and standard deviation
    rng = np.random.default_rng(12)
    raw = rng.normal(80.0, 25.0, 500)
    out = affine_normalize(raw, target_mean=100.0, target_std=15.0)
    print(f"  out mean / std         = {out.mean():.2f} / {out.std():.2f}")
    assert np.isclose(out.mean(), 100.0) and np.isclose(out.std(), 15.0)

    # Maximum percent shift is reported
    shift = max_percent_shift(raw, out)
    print(f"  max percent shift      = {shift:.1f} %")
    assert shift > 0
    print("  PASS")
    return {"out_mean": float(out.mean()), "max_shift": shift}


if __name__ == "__main__":
    test_all()
