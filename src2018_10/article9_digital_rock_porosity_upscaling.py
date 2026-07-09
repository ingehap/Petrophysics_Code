"""
Article 9: Upscaling of Digital Rock Porosities by Correlation With Whole-Core
           CT-Scan Histograms
Hertel, Rydzy, Anger, Berg, Appel, de Jong (2018)
DOI: 10.30632/PJV59N5-2018a8

Digital-rock (micro-CT) porosity is measured on a tiny sub-sample that may not
be representative of the whole core.  Correlating the digital-rock porosity with
the whole-core CT-scan porosity histogram upscales the small-scale value: the
representative-elementary-volume (REV) is found where the running porosity
stabilizes, and a linear correlation maps the digital-rock value onto the
core-scale histogram.

Implements:

  - CT porosity from voxel attenuation (calibration endpoints)
  - Running-average REV convergence
  - Histogram-based core porosity (mean of the CT histogram)
  - Linear upscaling correlation (digital-rock -> whole-core)

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard CT-porosity / REV / upscaling relations the
paper's title describes.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- CT porosity -------------

def ct_porosity(mu, mu_grain, mu_fluid):
    """Porosity from CT attenuation  phi = (mu_grain - mu)/(mu_grain - mu_fluid)."""
    return petrolib.porosity_lithology.ct_porosity(mu, mu_grain, mu_fluid)


def running_porosity(phi_voxels):
    """Cumulative running-average porosity vs sample volume (REV curve)."""
    a = np.asarray(phi_voxels, float)
    return np.cumsum(a) / np.arange(1, len(a) + 1)


def rev_size(phi_voxels, tol=0.005):
    """Smallest sample count at which the running porosity is within tol of final."""
    run = running_porosity(phi_voxels)
    final = run[-1]
    for i in range(len(run)):
        if np.all(np.abs(run[i:] - final) < tol):
            return i + 1
    return len(run)


def histogram_porosity(phi_voxels):
    """Whole-core porosity = mean of the CT porosity histogram."""
    return float(np.mean(phi_voxels))


def upscale(phi_dr, slope, intercept):
    """Linear upscaling correlation  phi_core = slope*phi_dr + intercept."""
    return slope * phi_dr + intercept


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Upscaling Digital Rock Porosity (CT Histograms)")
    print("=" * 60)

    # CT porosity from attenuation endpoints
    assert abs(ct_porosity(0.7 * 1.0 + 0.3 * 0.2, 1.0, 0.2) - 0.3) < 1e-9

    # REV: running porosity stabilizes as the sample grows
    rng = np.random.default_rng(8)
    phi_vox = np.clip(0.18 + 0.05 * rng.standard_normal(4000), 0, 1)
    run = running_porosity(phi_vox)
    print(f"  running phi 50 / final = {run[49]:.3f} / {run[-1]:.3f}")
    rev = rev_size(phi_vox, tol=0.005)
    print(f"  REV size               = {rev} voxels")
    assert 1 <= rev <= len(phi_vox)
    # the running average converges (late-window spread small)
    assert np.std(run[-500:]) < 0.005

    # Histogram porosity equals the mean
    assert abs(histogram_porosity(phi_vox) - phi_vox.mean()) < 1e-12

    # Upscaling correlation maps a small-scale value onto the core scale
    phi_core = upscale(0.20, slope=0.9, intercept=0.01)
    print(f"  upscaled core porosity = {phi_core:.3f}")
    assert abs(phi_core - 0.19) < 1e-9
    print("  PASS")
    return {"REV": rev, "phi_core": float(phi_core)}


if __name__ == "__main__":
    test_all()
