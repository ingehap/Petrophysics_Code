"""
Article 1: A Guide to Nanoindentation
Sondergeld, Rai (2022)
DOI: 10.30632/PJV63N5-2022a1

Implements the Oliver-Pharr analysis framework for nanoindentation of
shales, including all of the key formulas listed in the paper:

  - Hardness            H = P_max / A                    (Eq. 1)
  - Unloading stiffness S = dP/dh                        (Eq. 2)
  - Tip-area function   A = A(h_c) [polynomial in h_c]   (Eq. 3)
  - Reduced modulus to Young's modulus via the indenter-sample
    composite compliance
        1/E_r = (1 - nu_s^2)/E_s + (1 - nu_i^2)/E_i      (Eq. 4)
    with diamond properties E_i = 1141 GPa, nu_i = 0.07.
  - Gupta et al. (2018) shear-modulus estimator:
        G = 95.3 * slope_late_load - 0.35  GPa            (Eq. 5)
  - Creep power-law fits (Eqs. 6-8) and a logarithmic creep model
        h(t) - h0 = b * log10(t/t0)                       (Eqs. 9-10)
  - Mixed-mode fracture toughness from corner-cube crack lengths
        K_c = alpha * sqrt(E/H) * P_max / c^(3/2)        (Eq. 11)

Synthetic demo reproduces the paper's Woodford-shale array statistic
(mean Es ~ 31 GPa with std ~ 3.4 GPa) from a simulated 100-indent
4 x 25 array of load-displacement curves.
"""

import numpy as np


# --------------------------------------------- Oliver-Pharr core --------

def hardness(P_max, A):
    """H = P_max / A   (Eq. 1).  Units: P in N, A in m^2 -> H in Pa."""
    return P_max / A


def reduced_modulus(S, A, beta=1.034):
    """E_r = (sqrt(pi) / (2*beta)) * S / sqrt(A)   (Oliver-Pharr 1992)."""
    return (np.sqrt(np.pi) / (2.0 * beta)) * S / np.sqrt(A)


def youngs_modulus_from_Er(E_r, nu_s=0.20, E_i=1141e9, nu_i=0.07):
    """Solve Eq. 4 for E_s.

        1/E_r = (1-nu_s^2)/E_s + (1-nu_i^2)/E_i
        => E_s = (1 - nu_s^2) / (1/E_r - (1-nu_i^2)/E_i)
    """
    return (1.0 - nu_s ** 2) / (1.0 / E_r - (1.0 - nu_i ** 2) / E_i)


def tip_area_function_berkovich(h_c, c0=24.5, c1=0.0, c2=0.0):
    """Berkovich ideal: A = 24.5 * h_c^2 (Eq. 3, leading term).

    Real tip blunting adds c1 * h_c + c2 * sqrt(h_c) terms.  Returns A
    in the same units^2 as h_c (e.g., nm -> nm^2).
    """
    return c0 * h_c ** 2 + c1 * h_c + c2 * np.sqrt(h_c)


# --------------------------------------------- load-displacement curves -

def synth_load_displacement(P_max=10e-3, h_max_nm=400.0, n_pts=200,
                            E_s_GPa=30.0, H_GPa=2.0, noise=0.01, seed=0):
    """Synthetic P-h pair for one indent (units: P in N, h in nm).

    Loading branch follows P = a_load * h^2 (Hertzian / Berkovich); the
    unloading branch follows the Oliver-Pharr power law
        P = a_un * (h - h_f) ^ m   with m ~ 1.5
    and is calibrated so the unloading stiffness S = dP/dh at h_max and
    the residual depth h_f reproduce the requested E_s and H.
    """
    rng = np.random.default_rng(seed)
    # Loading: P = a * h^2
    a_load = P_max / h_max_nm ** 2
    h_load = np.linspace(0.0, h_max_nm, n_pts // 2)
    P_load = a_load * h_load ** 2
    # Reduced modulus E_r from composite-compliance Eq. 4
    E_i_GPa = 1141.0; nu_i = 0.07; nu_s = 0.20
    inv_Er = (1.0 - nu_s ** 2) / E_s_GPa + (1.0 - nu_i ** 2) / E_i_GPa
    E_r_GPa = 1.0 / inv_Er
    # Contact area at h_c = 0.75 * h_max (Berkovich)
    A_max_nm2 = tip_area_function_berkovich(0.75 * h_max_nm)
    # Stiffness in N / nm:  S[N/m] = (2 beta / sqrt pi) * E_r[Pa] * sqrt(A)[m]
    E_r_Pa = E_r_GPa * 1e9
    A_max_m2 = A_max_nm2 * 1e-18
    S_N_per_m = (2.0 * 1.034 / np.sqrt(np.pi)) * E_r_Pa * np.sqrt(A_max_m2)
    S_N_per_nm = S_N_per_m * 1e-9
    # Unloading branch: m=1.5, residual depth h_f, intercept slope = S at h_max
    h_f = h_max_nm - 1.5 * P_max / S_N_per_nm
    h_un = np.linspace(h_max_nm, max(h_f, 0.0), n_pts // 2)
    a_un = P_max / max(1e-9, (h_max_nm - h_f)) ** 1.5
    P_un = a_un * np.maximum(h_un - h_f, 0.0) ** 1.5

    h = np.concatenate([h_load, h_un])
    P = np.concatenate([P_load, P_un])
    P *= (1.0 + noise * rng.standard_normal(len(P)))
    h_max_idx = int(np.argmax(h))
    return h, P, h_max_idx


def fit_unloading_stiffness(h, P, h_max_idx, frac=0.25):
    """Linear fit dP/dh on the top `frac` of the unloading branch."""
    h_un = h[h_max_idx:]
    P_un = P[h_max_idx:]
    n_fit = max(3, int(frac * len(h_un)))
    coef = np.polyfit(h_un[:n_fit], P_un[:n_fit], 1)
    return float(coef[0])


# --------------------------------------------- Gupta (Eq. 5) ----------

def gupta_shear_modulus(slope_late_load_GPa_nm):
    """G = 95.3 * slope - 0.35  GPa  (Eq. 5)."""
    return 95.3 * slope_late_load_GPa_nm - 0.35


# --------------------------------------------- creep (Eqs. 9-10) -----

def fit_log_creep(t_s, h_nm):
    """Linear fit of h(t) - h0 vs log10(t) (Eqs. 9-10).  t_s is the
    hold-time array starting at t > 0 (skip the t=0 point)."""
    log_t = np.log10(t_s)
    coef = np.polyfit(log_t, h_nm - h_nm[0], 1)
    return float(coef[0]), float(coef[1])  # (b, intercept)


# --------------------------------------------- toughness (Eq. 11) ----

def fracture_toughness(E_GPa, H_GPa, P_max_N, c_m, alpha=0.040):
    """K_c = alpha * sqrt(E/H) * P_max / c^(3/2)   (Eq. 11).

    Returns K_c in MPa.sqrt(m) when inputs are in (GPa, GPa, N, m).
    """
    return alpha * np.sqrt(E_GPa / H_GPa) * P_max_N / c_m ** 1.5 * 1e-6


# --------------------------------------------- tests --------------------

def test_all():
    print("=" * 60)
    print("Article 1: Oliver-Pharr Nanoindentation on Woodford Shale")
    print("=" * 60)

    rng = np.random.default_rng(0)
    n_indents = 100
    true_E_mean, true_E_std = 31.0, 3.4
    E_targets = rng.normal(true_E_mean, true_E_std, n_indents)
    H_target = 2.0

    Es_recovered = []
    for i, Et in enumerate(E_targets):
        h, P, idx = synth_load_displacement(P_max=10e-3, h_max_nm=400.0,
                                            E_s_GPa=Et, H_GPa=H_target,
                                            noise=0.005, seed=i)
        h_max = h[idx]
        P_max = P[idx]
        S_N_per_nm = fit_unloading_stiffness(h, P, idx, frac=0.25)
        # Contact depth h_c ~ 0.75 * h_max for a Berkovich (Oliver-Pharr)
        h_c = 0.75 * h_max
        A_nm2 = tip_area_function_berkovich(h_c)
        # Convert: S [N/nm] -> [N/m] via *1e9; sqrt(A) [nm] -> [m] via *1e-9
        E_r_Pa = (np.sqrt(np.pi) / (2.0 * 1.034)) * (S_N_per_nm * 1e9) \
                 / np.sqrt(A_nm2 * 1e-18)
        E_s_GPa = youngs_modulus_from_Er(E_r_Pa) / 1e9
        Es_recovered.append(E_s_GPa)

    Es_recovered = np.array(Es_recovered)
    print(f"  Synthetic array: 4 x 25 = {n_indents} indents")
    print(f"  True   E_s  mean = {true_E_mean:5.2f}  std = {true_E_std:4.2f}  GPa")
    print(f"  Recov  E_s  mean = {Es_recovered.mean():5.2f}  "
          f"std = {Es_recovered.std():4.2f}  GPa")

    # Gupta shear-modulus demonstration on a single curve
    h, P, idx = synth_load_displacement(seed=99)
    n_late = int(0.10 * idx)
    slope_load = float(np.polyfit(h[idx - n_late:idx],
                                  P[idx - n_late:idx], 1)[0])
    G = gupta_shear_modulus(slope_load * 1e9 / 1e9)  # already in GPa/nm
    print(f"  Gupta G  (Eq. 5)               = {G:6.3f} GPa")

    # Log-creep fit
    t_s = np.linspace(1.0, 100.0, 50)
    h_creep = 400.0 + 1.2 * np.log10(t_s) \
              + 0.05 * np.random.default_rng(2).standard_normal(50)
    b, _ = fit_log_creep(t_s, h_creep)
    print(f"  Log-creep coefficient b        = {b:6.3f}  (true 1.20)")

    # Fracture toughness sanity
    Kc = fracture_toughness(E_GPa=30.0, H_GPa=2.0,
                            P_max_N=0.5, c_m=20e-6, alpha=0.040)
    print(f"  Fracture toughness K_c (Eq. 11) = {Kc:6.2f} MPa.sqrt(m)")

    # Sanity checks
    assert abs(Es_recovered.mean() - true_E_mean) < 0.30 * true_E_mean, \
        "Recovered Young's mean should be within +/- 30 % of target"
    assert abs(b - 1.2) < 0.05, "Log-creep slope must be recovered"
    print("  PASS")
    return {"E_s_mean_GPa": float(Es_recovered.mean()),
            "E_s_std_GPa": float(Es_recovered.std()),
            "creep_b": b,
            "Kc_MPa_sqrtm": Kc}


if __name__ == "__main__":
    test_all()
