"""
Article 9: Fluid Optical Database Reconstruction With Validated Mapping from
           External Oil and Gas Information Source
Chen, Jones, Dai, van Zuilekom (2018)
DOI: 10.30632/PJV59N6Y2018a8

Downhole-fluid-analysis optical sensors measure optical density (OD) across many
channels; an optical database links fluid composition to its OD spectrum.  The
database is reconstructed (and gaps filled) by a validated linear mapping from an
external oil-and-gas composition source, so OD spectra can be predicted from
composition and inverted back to composition.

Implements:

  - Beer-Lambert optical density per channel  OD = log10(I0/I) = eps*c*L
  - Composition -> OD spectrum forward mapping (channel response matrix)
  - Least-squares inversion OD -> composition (validated mapping)
  - Reconstruction error of a held-out spectrum

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard optical-density / linear-mapping relations the
paper describes.
"""

import numpy as np


# ---------------------------------------------- optics ------------------

def optical_density(I0, I):
    """Optical density  OD = log10(I0/I)."""
    return np.log10(I0 / np.asarray(I, float))


def od_spectrum(composition, response_matrix, path_length=1.0):
    """Forward map composition -> OD spectrum  OD = (A @ c)*L  (Beer-Lambert)."""
    return (np.asarray(response_matrix, float) @ np.asarray(composition, float)) * path_length


def invert_composition(od, response_matrix, path_length=1.0):
    """Least-squares inversion of an OD spectrum to fluid composition."""
    A = np.asarray(response_matrix, float) * path_length
    c, *_ = np.linalg.lstsq(A, np.asarray(od, float), rcond=None)
    return c


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Fluid Optical Database Reconstruction")
    print("=" * 60)

    # Beer-Lambert OD: more absorption (lower transmitted I) -> higher OD
    assert optical_density(1.0, 0.1) > optical_density(1.0, 0.5)
    assert abs(optical_density(1.0, 0.1) - 1.0) < 1e-9

    # Channel response matrix (10 optical channels x 4 fluid components)
    rng = np.random.default_rng(8)
    n_chan, n_comp = 10, 4
    A = np.abs(rng.normal(1.0, 0.4, (n_chan, n_comp)))

    # Forward: a planted composition gives an OD spectrum
    comp_true = np.array([0.5, 0.3, 0.15, 0.05])      # C1, C2-C5, CO2, water
    od = od_spectrum(comp_true, A)
    print(f"  OD spectrum (first 3)  = {np.array2string(od[:3], precision=3)}")
    assert np.all(od > 0)

    # Validated mapping inverts OD back to composition (overdetermined LS)
    comp_hat = invert_composition(od, A)
    print(f"  composition error      = {np.max(np.abs(comp_hat - comp_true)):.2e}")
    assert np.max(np.abs(comp_hat - comp_true)) < 1e-6

    # Reconstruction of a noisy held-out spectrum is still accurate
    od_noisy = od + rng.normal(0, 0.01, n_chan)
    comp_noisy = invert_composition(od_noisy, A)
    od_recon = od_spectrum(comp_noisy, A)
    rel = np.linalg.norm(od_recon - od) / np.linalg.norm(od)
    print(f"  spectrum reconstruction rel. error = {rel:.3f}")
    assert rel < 0.05
    print("  PASS")
    return {"comp_error": float(np.max(np.abs(comp_hat - comp_true))),
            "recon_rel_error": float(rel)}


if __name__ == "__main__":
    test_all()
