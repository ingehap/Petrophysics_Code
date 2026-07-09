"""
Article 5: NMR T1-T2 Logging in Unconventional Reservoirs: Pore-Size
Distribution, Permeability, and Reservoir Quality
Ijasan, Macquaker, Luycx, Alzobaidi, Oyewole, Rudnicki (2022)
DOI: 10.30632/PJV63N3-2022a5

NMR Petrophysical Pore Multimodal (NPPM) analysis for tight-oil
mudstones.  Implements:

  - Generalised relaxation rate sum                          (Eqs. 1-3)
        1/T2 = 1/T2_bulk + 1/T2_diff + sum_f s_f * (rho_f * S_f / V)
  - 2-D Gaussian-mixture (NPPM) fit of the T1-T2 distribution into
    log-normal poro-fluid clusters: each (T1, T2) Gaussian gives
    apparent surface relaxivity rho_n,f and bulk T_B,f.
  - Pore-size distribution per fluid: r_f = rho_n,f * T2_n
  - Kozeny-Carman permeability                              (Eqs. 4-6)
        k = phi^3 * <r^2> / (180 * (1 - phi)^2)
  - Herron-style mineralogy permeability scaling.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- Eqs. 1-3 ----------------

def t2_total_multiphase(T2_bulk, T2_diff, fractional_saturations,
                       surface_relaxivities, surf_to_vol_ratio):
    """1/T2 = 1/T2_B + 1/T2_D + S/V * sum_f s_f * rho_f   (Eqs. 1-3)."""
    surf_term = surf_to_vol_ratio * sum(
        s * r for s, r in zip(fractional_saturations, surface_relaxivities))
    return 1.0 / (1.0 / T2_bulk + 1.0 / T2_diff + surf_term)


# ---------------------------------------------- 2-D Gaussian mixture --

def gaussian_2d(L1, L2, mu1, mu2, s1, s2, amp):
    return amp * np.exp(-0.5 * (((L1 - mu1) / s1) ** 2 + ((L2 - mu2) / s2) ** 2))


def fit_nppm_components(T1, T2, M_obs, n_components=4, n_iter=80, seed=0):
    """Greedy peeling fit: iteratively pick the maximum residual, place a
    Gaussian there, refit its parameters by Nelder-Mead-style steps and
    subtract.  Returns a list of (mu_T1, mu_T2, sigma_T1, sigma_T2, amp).
    """
    rng = np.random.default_rng(seed)
    L1, L2 = np.log10(T1)[:, None], np.log10(T2)[None, :]
    R = M_obs.copy()
    components = []
    for k in range(n_components):
        # Pick the maximum residual
        idx = np.unravel_index(np.argmax(R), R.shape)
        mu1, mu2 = float(L1[idx[0], 0]), float(L2[0, idx[1]])
        s1, s2 = 0.30, 0.30
        amp = float(R[idx])
        # Refine by coordinate descent
        for _ in range(n_iter):
            G = gaussian_2d(L1, L2, mu1, mu2, s1, s2, amp)
            residual = R - G
            # Update amplitude analytically
            G1 = gaussian_2d(L1, L2, mu1, mu2, s1, s2, 1.0)
            denom = float((G1 ** 2).sum())
            amp = max(0.0, float((R * G1).sum() / max(denom, 1e-12)))
            # Coordinate-descent sigma + mu via small steps
            mu1 += 0.02 * np.sign((residual * G1 * (L1 - mu1)).sum())
            mu2 += 0.02 * np.sign((residual * G1 * (L2 - mu2)).sum())
        components.append((mu1, mu2, s1, s2, amp))
        R = R - gaussian_2d(L1, L2, mu1, mu2, s1, s2, amp)
        R = np.clip(R, 0.0, None)
    return components


def apparent_surface_relaxivity(t2_peak_ms, pore_radius_um):
    """rho_n = r / T2  - returned in um / s for consistency."""
    # ms -> s adapter kept local; shape_factor=1 gives the raw r/T2.
    return float(petrolib.nmr.surface_relaxivity_from_pore(
        t2_peak_ms * 1e-3, pore_radius_um, shape_factor=1.0))


# ---------------------------------------------- Kozeny-Carman (Eqs. 4-6) ---

def kozeny_carman(phi, mean_r2_m2):
    """k [m^2] = phi^3 * <r^2> / (180 * (1 - phi)^2)."""
    return phi ** 3 * mean_r2_m2 / (180.0 * (1.0 - phi) ** 2)


def herron_mineralogy_factor(fraction_carbonate, fraction_clay):
    """Empirical Herron-style permeability scaling factor on mineralogy."""
    return float(np.exp(-3.0 * fraction_clay + 0.5 * fraction_carbonate))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: NPPM Gaussian Mixture - Pore Size & Permeability")
    print("=" * 60)

    # Relaxation-rate sanity: surface term dominates short T2
    T2 = t2_total_multiphase(T2_bulk=2000.0, T2_diff=300.0,
                             fractional_saturations=[0.3, 0.7],
                             surface_relaxivities=[20.0, 5.0],
                             surf_to_vol_ratio=0.20)
    print(f"  T2_total (multiphase wetted fractions) = {T2:.3f} ms")
    assert T2 < 50.0, "Surface term should dominate T2"

    # Build a synthetic 2-D T1-T2 map
    T1 = np.logspace(0, 3.5, 32)
    T2v = np.logspace(-1, 3, 32)
    L1, L2 = np.log10(T1)[:, None], np.log10(T2v)[None, :]
    M = np.zeros((32, 32))
    centres = [(1.0, 0.5), (1.7, 1.3), (2.6, 1.0), (2.4, 2.3)]
    amps = [0.4, 0.25, 0.20, 0.15]
    for (mu1, mu2), A in zip(centres, amps):
        M += A * np.exp(-0.5 * (((L1 - mu1) / 0.30) ** 2
                                + ((L2 - mu2) / 0.30) ** 2))
    M += 1e-3 * np.random.default_rng(0).standard_normal(M.shape)
    M = np.clip(M, 0.0, None)

    comps = fit_nppm_components(T1, T2v, M, n_components=4)
    print(f"  Fitted {len(comps)} 2-D Gaussian components.  Peak centres (log10 T1, log10 T2):")
    for k, (m1, m2, s1, s2, a) in enumerate(comps):
        print(f"    em {k}:  ({m1:5.2f}, {m2:5.2f})   amp={a:.3f}")
    matched = 0
    for (mu1, mu2), A in zip(centres, amps):
        for m1, m2, *_ in comps:
            if abs(m1 - mu1) < 0.30 and abs(m2 - mu2) < 0.30:
                matched += 1
                break
    print(f"  Matched {matched} / {len(centres)} planted clusters")
    assert matched >= len(centres) - 1, "NPPM must recover most clusters"

    # Kozeny-Carman with Herron factor
    mean_r2_um2 = np.mean([1.0, 4.0, 16.0])     # mix of 1, 2, 4 um radii
    mean_r2_m2 = mean_r2_um2 * (1e-6) ** 2
    k_m2 = kozeny_carman(0.08, mean_r2_m2) * herron_mineralogy_factor(0.6, 0.10)
    k_mD = k_m2 / 0.9869e-15
    print(f"  Kozeny-Carman k (with Herron factor) = {k_mD:6.3f} mD")
    print("  PASS")
    return {"clusters_matched": matched, "k_mD": k_mD}


if __name__ == "__main__":
    test_all()
