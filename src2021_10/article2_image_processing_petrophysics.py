"""
Article 2: Enhanced Learning of Fundamental Petrophysical Concepts Through
           Image Processing and 3D Printing
Alyafei, Al Musleh, Bautista, Idris, Seers (2021)
DOI: 10.30632/PJV62N5-2021a2

A petroleum-engineering education paper: transparent 3D-printed micromodels
are photographed during flooding and analysed in Fiji/ImageJ to teach
porosity, fluid saturation, wettability, grain size, and displacement
efficiency.  The quantitative core is pixel-counting on thresholded binary
images plus a few definitional petrophysics relations.

Implements:

  - Porosity from a segmented image  phi = N_pore / N_total
  - Phase saturation  S_phase = (phase pore-pixels) / (pore-pixels)
  - Irreducible / residual saturation bookkeeping
  - Displacement efficiency  E_D = (S_oi - S_or)/S_oi             (Eq. 1)
  - Equivalent grain radius  r = sqrt(A / pi)
  - Wettability decision from contact angle (water-wet if < 90 deg)

Note: only Eq. 1 is numbered in the paper, and its glyph (like all the
others) was image-rendered and not in the text; the forms here are faithful
reconstructions that reproduce the paper's worked numbers (phi = 27.01%,
S_wir = 0.332, S_or = 0.226, E_D ~ 66%).
"""

import numpy as np

WATER_WET_ANGLE = 90.0      # contact-angle threshold (deg)


# ---------------------------------------------- porosity ----------------

def porosity_from_binary(pore_mask):
    """Porosity = pore-pixel fraction of a binary image."""
    pore_mask = np.asarray(pore_mask, bool)
    return float(pore_mask.sum()) / pore_mask.size


# ---------------------------------------------- saturation --------------

def phase_saturation(phase_mask, pore_mask):
    """Saturation = phase pore-pixels / total pore-pixels."""
    phase = np.asarray(phase_mask, bool)
    pore = np.asarray(pore_mask, bool)
    occupied = np.logical_and(phase, pore).sum()
    return float(occupied) / float(pore.sum())


def irreducible_water_saturation(post_drainage_phase_fraction):
    """S_wir = 1 - (oil fraction left after drainage)."""
    return 1.0 - post_drainage_phase_fraction


# ---------------------------------------------- Eq. 1: efficiency -------

def displacement_efficiency(s_oi, s_or):
    """Displacement efficiency  E_D = (S_oi - S_or) / S_oi  (Eq. 1)."""
    return (s_oi - s_or) / s_oi


# ---------------------------------------------- grain size --------------

def equivalent_radius(area, length_per_pixel=1.0):
    """Equivalent-circle radius from a particle area  r = sqrt(A/pi)."""
    a = np.asarray(area, float) * length_per_pixel ** 2
    return np.sqrt(a / np.pi)


# ---------------------------------------------- wettability -------------

def is_water_wet(contact_angle_deg):
    """Water-wet if the contact angle is below 90 degrees."""
    return contact_angle_deg < WATER_WET_ANGLE


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Image Processing for Petrophysics Education")
    print("=" * 60)

    # Synthetic 100x100 grayscale: 27.01% of pixels are pore (intensity 0),
    # matrix is bright.  Threshold at mid-gray to recover the pore mask.
    rng = np.random.default_rng(0)
    img = np.full((100, 100), 200, dtype=int)
    n_pore = 2701
    flat = rng.permutation(img.size)[:n_pore]
    img.flat[flat] = 20                       # pore pixels are dark
    pore_mask = img < 128
    phi = porosity_from_binary(pore_mask)
    print(f"  porosity (pixel count) = {phi:.4f}  (expect 0.2701)")
    assert abs(phi - 0.2701) < 1e-9

    # Within the pore space, 66.8% is occupied by oil after drainage
    oil_mask = np.zeros_like(pore_mask)
    pore_idx = np.flatnonzero(pore_mask)
    oil_idx = pore_idx[:int(round(0.668 * pore_idx.size))]
    oil_mask.flat[oil_idx] = True
    s_oi = phase_saturation(oil_mask, pore_mask)
    print(f"  oil saturation (drain) = {s_oi:.3f}  (expect ~0.668)")
    assert abs(s_oi - 0.668) < 2e-3

    s_wir = irreducible_water_saturation(s_oi)
    print(f"  S_wir                  = {s_wir:.3f}  (expect ~0.332)")
    assert abs(s_wir - 0.332) < 2e-3

    # Displacement efficiency from drainage / imbibition saturations
    e_d = displacement_efficiency(s_oi=0.668, s_or=0.226)
    print(f"  displacement eff. E_D  = {e_d:.3f}  (expect ~0.662)")
    assert abs(e_d - 0.662) < 2e-3

    # Equivalent grain radius from a circular particle of known area
    r = equivalent_radius(np.pi * 5.0 ** 2)
    print(f"  equivalent radius      = {r:.2f}  (expect 5.0)")
    assert abs(r - 5.0) < 1e-9

    # Wettability decision
    assert is_water_wet(40.0) and not is_water_wet(120.0)
    print("  PASS")
    return {"porosity": phi, "s_wir": s_wir, "E_D": e_d}


if __name__ == "__main__":
    test_all()
