"""
Article 1: Pore-Structure Characterization of a Complex Carbonate Reservoir in
           South Iraq Using Advanced Interpretation of NMR Logs
Saidian, Jain, Milad (2021)
DOI: 10.30632/PJV62N2-2021a1

Dual-vendor NMR logs in the Mishrif carbonate, where large vugs cause an NMR
porosity deficit (insufficient polarization) and diffusion relaxation in large
pores distorts the T2 distribution.  A single-pore forward model shows how echo
spacing and gradient control the pore-size sensitivity, a large-pore T2 cutoff
(~847 ms) is established, and a porosity correction recovers the deficit.

Implements:

  - Multi-exponential relaxation  1/T2 = 1/T2bulk + 1/T2surf + 1/T2diff  (Eq. 1)
  - Diffusion relaxation rate  1/T2diff = D*gamma^2*g^2*TE^2/12          (Eq. 2)
  - NMR porosity correction  phi_corr = phi + 0.3*Vol_largepore         (Eq. 3)
  - Surface relaxation (sphere S/V = 3/r) and single-pore T2 model
  - Timur-Coates and SDR permeability; large-pore T2-cutoff partition

Equations transcribed from the rendered article.  SI units internally
(metres, seconds, T/m); T2 reported in ms, porosity in fraction.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

GAMMA_H = 2.675e8        # rad/s/T, hydrogen gyromagnetic ratio
T2_LARGE_CUTOFF_MS = 847.0


# ---------------------------------------------- Eqs. 1-2: relaxation ----

def surface_rate(rho2_m_s, s_over_v):
    """Surface relaxation rate  1/T2surf = rho2 * (S/V)  (1/s).  rho2 in m/s."""
    return petrolib.nmr.relaxation_rate(rho=rho2_m_s, s_over_v=s_over_v)


def diffusion_rate(D, g_T_per_m, TE_s, gamma=GAMMA_H):
    """Diffusion relaxation rate  1/T2diff = D*gamma^2*g^2*TE^2/12  (Eq. 2).

    D in m^2/s, g (gradient) in T/m, TE (echo spacing) in s -> 1/s.
    """
    return petrolib.nmr.diffusion_relaxation_rate(D, G=g_T_per_m, TE=TE_s, gamma=gamma)


def t2_single_pore(radius_m, rho2_m_s, T2bulk_s, D, g_T_per_m, TE_s):
    """Combine bulk + surface (sphere S/V=3/r) + diffusion into T2 (Eq. 1).

    Returns T2 in seconds.
    """
    return petrolib.nmr.t2_apparent(
        t2_bulk=T2bulk_s, rho=rho2_m_s, s_over_v=3.0 / radius_m,
        D=D, G=g_T_per_m, TE=TE_s, gamma=GAMMA_H)


# ---------------------------------------------- Eq. 3: porosity corr ----

def porosity_correction(phi_uncorr, vol_large_pore, factor=0.3):
    """NMR porosity correction  phi_corr = phi + 0.3*Vol_largepore  (Eq. 3)."""
    return phi_uncorr + factor * vol_large_pore


def large_pore_volume(T2_ms, amplitude, cutoff_ms=T2_LARGE_CUTOFF_MS):
    """Porosity in pores with T2 above the large-pore cutoff."""
    T2 = np.asarray(T2_ms, float)
    return float(np.asarray(amplitude, float)[T2 > cutoff_ms].sum())


# ---------------------------------------------- permeability ------------

def timur_coates(phi, ffi, bvi, C=10.0):
    """Timur-Coates permeability  k = (phi/C)^4 * (FFI/BVI)^2  (mD)."""
    # This copy reports in mD scaled by 1e6 (the unit adapter stays local).
    return petrolib.nmr.timur_coates(phi, ffi, bvi, C=C) * 1e6


def sdr(phi, t2lm_ms, a=4.0, m=4.0, n=2.0):
    """SDR permeability  k = a*phi^m*T2LM^n  (mD).  T2LM in ms."""
    return petrolib.nmr.sdr(phi, t2lm_ms, a=a, m=m, n=n)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: NMR Pore-Structure of a Complex Carbonate")
    print("=" * 60)

    # Eq. 2: diffusion rate scales as TE^2 -> Vendor2/Vendor1 ratio = (900/200)^2
    D = 6.5e-9                      # m^2/s (6.5e-5 cm^2/s)
    g = 0.17                        # T/m (17 G/cm)
    r_v1 = diffusion_rate(D, g, 200e-6)
    r_v2 = diffusion_rate(D, g, 900e-6)
    print(f"  diffusion rate V2/V1   = {r_v2 / r_v1:.2f}  (expect 20.25)")
    assert abs(r_v2 / r_v1 - (900 / 200) ** 2) < 1e-6

    # Single-pore model: large-TE (Vendor 2) saturates with pore size while
    # small-TE (Vendor 1) keeps tracking -> Vendor 1 T2 larger at big pores
    rho2 = 3e-6                     # m/s (3 um/s)
    T2b = 9.85                      # s
    big = 80e-6                     # 80 um radius
    t2_v1 = t2_single_pore(big, rho2, T2b, D, g, 200e-6)
    t2_v2 = t2_single_pore(big, rho2, T2b, D, g, 900e-6)
    print(f"  T2 at 80um  V1 / V2    = {t2_v1*1e3:.0f} / {t2_v2*1e3:.0f} ms")
    assert t2_v1 > t2_v2            # Vendor 1 stays sensitive to large pores

    # Eq. 3: porosity correction recovers the large-pore deficit
    phi_c = porosity_correction(0.18, 0.05)
    print(f"  corrected porosity     = {phi_c:.3f}  (expect 0.195)")
    assert abs(phi_c - 0.195) < 1e-9

    # Large-pore volume from a bimodal T2 distribution
    T2 = np.array([1.0, 10.0, 100.0, 900.0, 2000.0])
    amp = np.array([0.02, 0.05, 0.06, 0.03, 0.02])
    vlp = large_pore_volume(T2, amp)
    print(f"  large-pore volume      = {vlp:.3f}")
    assert abs(vlp - 0.05) < 1e-9      # the 900 & 2000 ms bins

    # Permeability models positive
    assert timur_coates(0.25, 0.6, 0.4) > 0 and sdr(0.25, 100.0) > 0
    print("  PASS")
    return {"diff_ratio": r_v2 / r_v1, "phi_corr": phi_c, "vlp": vlp}


if __name__ == "__main__":
    test_all()
