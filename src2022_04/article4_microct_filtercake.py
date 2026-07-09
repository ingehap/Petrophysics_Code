"""
Article 4: In-Situ Visualization and Characterization of Filter-Cake
Deposition Using Time-Lapse Micro-CT Imaging
Schroeder, Torres-Verdin (2022)
DOI: 10.30632/PJV63N2-2022a4

Body text was not present in the available PDF extract, so this module
is a *methodology proxy* guided by the editor's letter: high-resolution
X-ray micro-CT images mudcake deposition on borehole-wall core samples
during WBM and synthetic-OBM filtrate invasion; continuous scanning
yields time-resolved mudcake thickness, porosity, and permeability.

Implements:

  - Dewan-Chenevert / Outmans mudcake-growth model:
        h_mc(t) = sqrt(2 * k_mc * dP * t / (mu * (1 - phi_mc)))
  - Mudcake porosity evolution under increasing compaction stress:
        phi_mc(t) = phi_0 * (1 + t / tau) ^ (-c)
  - Mudcake permeability via Kozeny-Carman with the evolving porosity:
        k_mc(t) = k_0 * (phi(t)/phi_0)^3 * ((1-phi_0)/(1-phi(t)))^2
  - Synthetic 2-D micro-CT slice with a growing mudcake layer along the
    borehole wall.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Dewan-Chenevert growth -

def mudcake_thickness_m(t_s, k_mc_m2, dP_Pa, mu_Pa_s, phi_mc):
    return np.sqrt(2.0 * k_mc_m2 * dP_Pa * t_s
                   / (mu_Pa_s * max(1.0 - phi_mc, 1e-6)))


# ---------------------------------------------- porosity + permeability --

def mudcake_porosity(t_s, phi_0=0.40, tau_s=1800.0, c=0.15):
    return phi_0 * (1.0 + t_s / tau_s) ** (-c)


def kozeny_carman_mudcake(phi_t, phi_0, k_0):
    return petrolib.flow_transport.kozeny_carman_ratio(k_0, phi_t, phi_0, grain_term=True)


# ---------------------------------------------- synthetic CT slice -----

def synth_ct_slice(n=128, mudcake_thickness_px=4, seed=0):
    """2-D voxel slice with a borehole at the centre and a mudcake ring."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((n, n))
    cx, cy = n // 2, n // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_bore = n // 4
    img = np.where(r < r_bore, 0.0,           # borehole fluid
                   np.where(r < r_bore + mudcake_thickness_px, 0.6,  # mudcake
                            1.0))             # formation
    img += 0.05 * rng.standard_normal(img.shape)
    return np.clip(img, 0.0, 1.0)


def detect_mudcake_thickness(ct_slice, threshold_low=0.3, threshold_high=0.7):
    """Detect the annular thickness of pixels whose value lies in
    (threshold_low, threshold_high), interpreted as the mudcake band."""
    mask = (ct_slice > threshold_low) & (ct_slice < threshold_high)
    return float(mask.sum() / (2.0 * np.pi * (ct_slice.shape[0] // 4)))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Time-Lapse Micro-CT of Mudcake Deposition (proxy)")
    print("=" * 60)

    # Dewan-Chenevert growth at three time points
    k_mc = 0.5e-15            # m^2 (~ 500 nD)
    dP = 3.5e6                # Pa
    mu = 0.05                 # Pa.s (mud)
    for t_min in (1, 10, 60):
        h = mudcake_thickness_m(t_min * 60.0, k_mc, dP, mu, 0.40)
        print(f"  t = {t_min:3d} min   mudcake thickness = {h * 1000:6.2f} mm")

    # Mudcake porosity / permeability evolution
    print("  Time  phi_mc    k_mc (mD)")
    for t_min in (1, 5, 30, 120):
        phi_t = mudcake_porosity(t_min * 60.0)
        k_t = kozeny_carman_mudcake(phi_t, 0.40, k_mc) / 0.9869e-15
        print(f"  {t_min:4d}   {phi_t:.3f}    {k_t:8.4f}")

    # Detect mudcake on a synthetic CT slice
    img = synth_ct_slice(mudcake_thickness_px=6)
    thickness = detect_mudcake_thickness(img)
    print(f"  Detected mudcake thickness (synthetic slice) = {thickness:.2f} px")
    assert thickness > 0.5, "Detector must register a non-zero mudcake band"
    print("  PASS")
    return {"detected_thickness_px": thickness}


if __name__ == "__main__":
    test_all()
