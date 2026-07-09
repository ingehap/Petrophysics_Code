"""
Article 11: Review of Recent Developments in NMR Core Analysis
Dick, Veselinovic, Green (2022)
DOI: 10.30632/PJV63N3-2022a11

Body text was not present in the available PDF extract, so this module
is a *methodology proxy* guided by the Guest Editor's summary: a survey
of the last decade of NMR core-analysis advances, including pore-size
distribution, NMR-derived relative permeability and capillary pressure,
variable-spaced-tau (VST) sequences, slice-selective long-core
measurements, spatially resolved T1-T2 mapping, fast imaging including
SPRITE, high magnetic fields, and high-P/T core-holder cells.

Implements representative *concept demonstrations* for three of these
themes:

  - Variable-spaced-tau (VST) acquisition: increase tau geometrically to
    capture both fast-relaxing micropores and slow-relaxing macropores
    with low echo count.
  - Slice-selective profile: 1-D signal along the core long axis,
    obtained by applying a slice gradient and a frequency-selective RF
    pulse.
  - Fast-imaging / SPRITE-style 1-D profile: very short FID acquisition
    at each point of a phase-encoding gradient table.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ----------------------------------------------- VST tau schedule -------

def vst_schedule(n_echoes=64, tau_min_us=200.0, tau_max_us=4000.0):
    """Geometric tau schedule from tau_min to tau_max."""
    return np.geomspace(tau_min_us, tau_max_us, n_echoes)


def vst_decay_curve(T2_dist, T2_axis_ms, tau_axis_us):
    """Synthesise the multi-tau echo train from a T2 distribution."""
    # us -> s and ms -> s adapters kept local; the CPMG kernel delegates.
    tau_s = tau_axis_us * 1e-6
    K = petrolib.nmr.cpmg_kernel(tau_s, T2_axis_ms * 1e-3)
    return K @ T2_dist


# ----------------------------------------------- slice-selective profile --

def slice_selective_profile(rho_z, slice_centre_m, slice_thickness_m,
                            z_axis_m):
    """1-D NMR signal sampled inside a slab of finite thickness."""
    mask = np.abs(z_axis_m - slice_centre_m) <= 0.5 * slice_thickness_m
    return float(rho_z[mask].sum())


# ----------------------------------------------- SPRITE-style profile --

def sprite_profile(rho_z, gradient_T_m=0.10, dwell_us=10.0, n_phases=32,
                  gamma_rad_s_T=2.675e8, z_axis_m=None):
    """Phase-encoded 1-D image of rho(z) - returns the magnitude profile."""
    if z_axis_m is None:
        z_axis_m = np.linspace(-0.05, 0.05, len(rho_z))
    k_axis = (np.arange(n_phases) - n_phases // 2) \
             * gamma_rad_s_T * gradient_T_m * dwell_us * 1e-6
    # FT of rho(z) sampled at k_axis
    Z, K = np.meshgrid(z_axis_m, k_axis)
    F = np.exp(-1j * K * Z)
    S = F @ rho_z
    # Inverse FT to recover profile
    return np.abs(np.fft.fftshift(np.fft.ifft(S))) * len(rho_z)


# ----------------------------------------------- tests -----------------

def test_all():
    print("=" * 60)
    print("Article 11: NMR Core Analysis Review (concept proxies)")
    print("=" * 60)

    # VST sequence sanity - geometric tau schedule spanning two decades
    # past the dominant T2 so the slow tail is fully sampled.
    T2_axis = np.logspace(-1, 3, 64)
    T2_dist = np.exp(-((np.log10(T2_axis) - np.log10(40.0)) / 0.3) ** 2)
    T2_dist /= T2_dist.sum()
    tau = vst_schedule(n_echoes=64, tau_min_us=200, tau_max_us=400_000)
    decay = vst_decay_curve(T2_dist, T2_axis, tau)
    print(f"  VST decay  initial = {decay[0]:.3f},  final = {decay[-1]:.3f}")
    assert decay[0] > 5.0 * decay[-1], "VST decay must capture order-of-magnitude drop"

    # Slice-selective profile sanity
    z = np.linspace(-0.05, 0.05, 100)
    rho = np.exp(-((z - 0.01) / 0.02) ** 2)
    sig_in = slice_selective_profile(rho, slice_centre_m=0.01, slice_thickness_m=0.02, z_axis_m=z)
    sig_out = slice_selective_profile(rho, slice_centre_m=-0.03, slice_thickness_m=0.02, z_axis_m=z)
    print(f"  Slice-selective signal  in-slab = {sig_in:.3f},  off-slab = {sig_out:.3f}")
    assert sig_in > 2.0 * sig_out

    # SPRITE profile sanity
    img = sprite_profile(rho, n_phases=64, z_axis_m=z)
    peak_pos_idx = int(np.argmax(img))
    peak_pos = z[peak_pos_idx]
    print(f"  SPRITE image peak at z = {peak_pos:+.3f} m  (true 0.010)")
    assert abs(peak_pos - 0.01) < 0.015, "SPRITE peak should land near the source"
    print("  PASS")
    return {"vst_ratio": decay[0] / decay[-1],
            "slice_contrast": sig_in / max(sig_out, 1e-9),
            "sprite_peak_m": peak_pos}


if __name__ == "__main__":
    test_all()
