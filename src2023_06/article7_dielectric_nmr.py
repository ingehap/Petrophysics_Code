"""
article7_dielectric_nmr.py
===========================
Implementation of ideas from:

    Funk, J., Myers, M., Hathon, L.
    "NMR-Mapped Distributions of Dielectric Dispersion"
    Petrophysics, Vol. 64, No. 3 (June 2023), pp. 421-437
    DOI: 10.30632/PJV64N3-2023a7

Key ideas implemented:

  * Bloembergen-Purcell-Pound (BPP) NMR relaxation (Eqs. 1 and 2)
        1/T1  = K * [tau_c/(1 + (omega*tau_c)^2)
                   + 4*tau_c/(1 + (2*omega*tau_c)^2)]
        1/T2  = K * [3*tau_c/2 + 5*tau_c/(2*(1+(omega*tau_c)^2))
                   + tau_c/(1 + (2*omega*tau_c)^2)]
  * Debye permittivity (Eq. 5)
        eps* = eps_inf + (eps_s - eps_inf) / (1 + i*omega*tau_c)
  * Havriliak-Negami extension (Eq. 6)
        eps* = eps_inf + (eps_s - eps_inf) / (1 + (i*omega*tau)^a)^b
  * Pore Combination Model linear additive permittivity (Eq. 7)
        eps_r = eps_inf + phi_v * eps_r_vug + phi_m * eps_r_matrix
  * NMR T2 distribution -> matrix vs vug split (tau_PCM)
        sort the T2 spectrum and assign the fastest-relaxing fraction
        to the matrix until phi_m is reached, the rest to vugs.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# BPP relaxation rates
# ---------------------------------------------------------------------------
def bpp_t1(omega: float, tau_c: float | np.ndarray,
           K: float = 1.0) -> float | np.ndarray:
    """BPP longitudinal relaxation T1 (s)."""
    tau_c = np.asarray(tau_c, dtype=float)
    inv = K * (tau_c / (1.0 + (omega * tau_c) ** 2)
               + 4.0 * tau_c / (1.0 + (2.0 * omega * tau_c) ** 2))
    return 1.0 / inv


def bpp_t2(omega: float, tau_c: float | np.ndarray,
           K: float = 1.0) -> float | np.ndarray:
    """BPP transverse relaxation T2 (s)."""
    tau_c = np.asarray(tau_c, dtype=float)
    inv = K * (1.5 * tau_c
               + 2.5 * tau_c / (1.0 + (omega * tau_c) ** 2)
               + 1.0 * tau_c / (1.0 + (2.0 * omega * tau_c) ** 2))
    return 1.0 / inv


# ---------------------------------------------------------------------------
# Dielectric models
# ---------------------------------------------------------------------------
def debye_permittivity(omega: np.ndarray, eps_inf: float,
                       eps_s: float, tau: float) -> np.ndarray:
    """Complex Debye permittivity."""
    return eps_inf + (eps_s - eps_inf) / (1.0 + 1j * omega * tau)


def havriliak_negami(omega: np.ndarray, eps_inf: float, eps_s: float,
                     tau: float, alpha: float, beta: float) -> np.ndarray:
    """
    Havriliak-Negami complex permittivity (Eq. 6).
    Reduces to:
        Cole-Cole when beta=1
        Cole-Davidson when alpha=1
        Debye when alpha=beta=1
    """
    return eps_inf + (eps_s - eps_inf) \
        / (1.0 + (1j * omega * tau) ** alpha) ** beta


# ---------------------------------------------------------------------------
# Pore Combination Model: linear additive permittivity (Eq. 7)
# ---------------------------------------------------------------------------
def pcm_permittivity(omega: np.ndarray, eps_inf: float,
                     phi_matrix: float, eps_r_matrix: np.ndarray,
                     phi_vug: float, eps_r_vug: np.ndarray) -> np.ndarray:
    """
    Linear additive PCM relative permittivity:
        eps_r(omega) = eps_inf + phi_m * eps_r_matrix(omega)
                                + phi_v * eps_r_vug(omega)
    """
    return eps_inf + phi_matrix * eps_r_matrix + phi_vug * eps_r_vug


# ---------------------------------------------------------------------------
# tau_PCM:  use a NMR T2 distribution to split matrix and vug porosities
# ---------------------------------------------------------------------------
def split_t2_into_matrix_vug(t2_axis: np.ndarray,
                             t2_amplitudes: np.ndarray,
                             phi_matrix_target: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Allocate the fastest-relaxing portion of a NMR T2 distribution to
    the rock matrix until the cumulative pore-volume fraction equals
    ``phi_matrix_target``; the remaining (slowly relaxing) part is
    attributed to vugs.

    Returns
    -------
    matrix_amplitudes, vug_amplitudes  arrays of the same shape as
    ``t2_amplitudes``.
    """
    order = np.argsort(t2_axis)              # increasing T2
    amp = t2_amplitudes[order]
    total = amp.sum()
    if total <= 0:
        return np.zeros_like(t2_amplitudes), np.zeros_like(t2_amplitudes)

    target_amp = phi_matrix_target * total
    cum = np.cumsum(amp)
    matrix_part = np.zeros_like(amp)
    for i, c in enumerate(cum):
        if c <= target_amp:
            matrix_part[i] = amp[i]
        else:
            # take only the slice of this bin that fits
            previous = cum[i - 1] if i > 0 else 0
            matrix_part[i] = max(target_amp - previous, 0.0)
            break
    vug_part = amp - matrix_part

    # Re-order back to original T2 axis ordering
    inv_order = np.argsort(order)
    return matrix_part[inv_order], vug_part[inv_order]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    """Synthetic-data test for module 7 (NMR + dielectric)."""
    print("[article7] testing BPP relaxation behaviour ...")
    omega = 2 * np.pi * 2e6
    tau = np.logspace(-12, -6, 200)
    t1 = bpp_t1(omega, tau, K=1e6)
    t2 = bpp_t2(omega, tau, K=1e6)
    # In the line-narrowing limit (omega*tau_c << 1) T1 ~ T2
    short = tau < 1e-10
    rel_diff = np.max(np.abs(t1[short] - t2[short]) / t2[short])
    assert rel_diff < 0.1, "T1 ~ T2 must hold for omega*tau_c << 1"
    # In the slow-motion regime T1 > T2
    slow = tau > 1e-7
    assert np.all(t1[slow] > t2[slow])
    print(f"           T1/T2 line-narrowing rel diff = {rel_diff:.3e}")

    print("[article7] testing Debye and Havriliak-Negami permittivity ...")
    f = np.logspace(3, 9, 200)
    om = 2 * np.pi * f
    eps_d = debye_permittivity(om, 5.0, 80.0, 1e-6)
    eps_hn = havriliak_negami(om, 5.0, 80.0, 1e-6, alpha=0.9, beta=0.85)

    # Real part decreases monotonically from eps_s to eps_inf
    assert eps_d.real[0] > eps_d.real[-1]
    assert abs(eps_d.real[-1] - 5.0) < 0.5
    assert abs(eps_d.real[0] - 80.0) < 0.5
    # HN model must give reasonable values too
    assert 4 < eps_hn.real[-1] < 10

    print("[article7] testing PCM split using a synthetic T2 distribution ...")
    t2_axis = np.logspace(-4, 0, 100)
    # bimodal T2 distribution (matrix peak ~ 1ms, vug peak ~ 100ms)
    amp = np.exp(-((np.log10(t2_axis) - np.log10(1e-3)) / 0.4) ** 2) \
        + 0.6 * np.exp(-((np.log10(t2_axis) - np.log10(1e-1)) / 0.3) ** 2)
    matrix_amp, vug_amp = split_t2_into_matrix_vug(
        t2_axis, amp, phi_matrix_target=0.55)
    frac_matrix = matrix_amp.sum() / amp.sum()
    print(f"           matrix porosity fraction recovered = "
          f"{frac_matrix:.3f} (target 0.55)")
    assert abs(frac_matrix - 0.55) < 0.02

    # Ensure matrix only carries the fastest-relaxing part
    nonzero_matrix = t2_axis[matrix_amp > 1e-12]
    nonzero_vug = t2_axis[vug_amp > 1e-12]
    assert nonzero_matrix.max() <= nonzero_vug.max()
    print("[article7] OK")


if __name__ == "__main__":
    test_all()
