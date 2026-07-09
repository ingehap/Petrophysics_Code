"""
Article 4: Machine-Learning-Enabled Automatic Sonic Shear Processing
Liang, Lei (2021)
DOI: 10.30632/PJV62N3-2021a3

A physics-driven machine-learning method (MLADI) that replaces a slow, unstable
mode-search root-finding solver for dipole-flexural dispersion.  A neural network
trained on synthetic flexural dispersion curves (generated over a 7-parameter VTI
model using the ANNIE approximation) acts as a fast forward proxy embedded in a
least-squares inversion that labels the flexural mode and inverts for formation
shear slowness (DTS).

Implements:

  - VTI stiffness from slownesses (ANNIE approximation)            (Eqs. 1-7)
        C55 = rhob/dts^2, C33 = rhob/dtc^2, C66 = C55*(1+2*gamma),
        C13 = C33-2*C55 (Eq.3), C12 = C13 (Eq.2), C11 = 2*C66+C12,
        pr = (2*dtc^2-dts^2)/(2*dtc^2-2*dts^2)                     (Eq. 6)
  - Relative mean absolute difference RMAD                         (Eq. 9)
  - Inversion misfit O                                             (Eq. 10)
  - A surrogate flexural dispersion + DTS inversion (numpy proxy for the NN)

Equations transcribed from the rendered article.  Slowness in s/m, density in
kg/m^3, stiffnesses in Pa, frequency in kHz.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Table 1 ANNIE-model parameter bounds (units in docstring)
BOUNDS = {"bhr": (3, 9), "dtm": (180, 300), "rhom": (800, 1400),
          "dts": (60, 460), "pr": (0.1, 0.4), "rhob": (1800, 2800),
          "gamma": (0, 0.8)}
N_FREQ = 51                       # dispersion sampled at 51 frequencies
FREQS_KHZ = np.linspace(0.1, 10.1, N_FREQ)


# ---------------------------------------------- Eqs. 1-7: ANNIE VTI -----

def annie_stiffness(dts, dtc, gamma, rhob):
    """VTI stiffness matrix via the ANNIE approximation (Eqs. 1-7).

    dts, dtc are shear / compressional slownesses (s/m).  Returns the 6x6
    Voigt stiffness matrix (Pa).
    """
    C55 = rhob / dts ** 2                       # Eq. 4
    C33 = rhob / dtc ** 2                       # Eq. 5
    C66 = C55 * (1.0 + 2.0 * gamma)             # Eq. 7
    C13 = C33 - 2.0 * C55                       # Eq. 3
    C12 = C13                                   # Eq. 2 (ANNIE)
    C11 = 2.0 * C66 + C12                       # from C66 = (C11-C12)/2
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C11
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
    C[3, 3] = C[4, 4] = C55
    C[5, 5] = C66
    return C


def poisson_ratio(dts, dtc):
    """Poisson's ratio  pr = (2*dtc^2-dts^2)/(2*dtc^2-2*dts^2)  (Eq. 6)."""
    return (2 * dtc ** 2 - dts ** 2) / (2 * dtc ** 2 - 2 * dts ** 2)


def thomsen_gamma(C66, C55):
    """Thomsen gamma  = (C66 - C55)/(2*C55)  (Eq. 7)."""
    return petrolib.acoustic_geomech.thomsen_gamma(C66, C55)


# ---------------------------------------------- Eqs. 9-10: metrics ------

def rmad(y_nn, y_ms):
    """Relative mean absolute difference  sum|y_nn-y_ms|/sum y_ms  (Eq. 9)."""
    y_nn = np.asarray(y_nn, float); y_ms = np.asarray(y_ms, float)
    return float(np.sum(np.abs(y_nn - y_ms)) / np.sum(y_ms))


def misfit(y_nn, y_data):
    """Inversion misfit  O = |sum(y_nn - y_data)| / sum(y_nn)  (Eq. 10)."""
    y_nn = np.asarray(y_nn, float); y_data = np.asarray(y_data, float)
    return float(abs(np.sum(y_nn - y_data)) / np.sum(y_nn))


# ---------------------------------------------- flexural dispersion -----

def flexural_dispersion(freqs_khz, dts_us_ft, amp=0.35, fc=5.0, w=1.0):
    """Surrogate dipole-flexural dispersion curve (slowness vs frequency).

    Low-frequency asymptote -> formation shear slowness dts; rises toward a
    slower Airy region at high frequency.  Stand-in for the paper's mode-search
    / NN forward model so the module runs with numpy alone.
    """
    f = np.asarray(freqs_khz, float)
    return dts_us_ft * (1.0 + amp / (1.0 + np.exp(-(f - fc) / w)))


def invert_dts(freqs_khz, slowness):
    """Recover formation shear slowness = the low-frequency dispersion asymptote."""
    f = np.asarray(freqs_khz, float)
    order = np.argsort(f)
    return float(slowness[order][0])


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Machine-Learning Sonic Shear Processing")
    print("=" * 60)

    # ANNIE stiffness from velocities Vs=2500, Vp=4000 m/s, gamma=0.1
    Vs, Vp, rhob, gamma = 2500.0, 4000.0, 2330.0, 0.1
    dts, dtc = 1.0 / Vs, 1.0 / Vp
    C = annie_stiffness(dts, dtc, gamma, rhob)
    print(f"  C55 / C33              = {C[3,3]:.3e} / {C[2,2]:.3e} Pa")
    assert abs(C[3, 3] - rhob * Vs ** 2) < 1.0       # Eq. 4
    assert abs(C[2, 2] - rhob * Vp ** 2) < 1.0       # Eq. 5
    assert np.allclose(C, C.T)                       # symmetric
    # gamma round-trips from C66, C55 (Eq. 7)
    assert abs(thomsen_gamma(C[5, 5], C[3, 3]) - gamma) < 1e-9
    # C66 = (C11 - C12)/2
    assert abs(C[5, 5] - (C[0, 0] - C[0, 1]) / 2.0) < 1.0

    # Poisson's ratio (Eq. 6) matches the standard Vp/Vs formula
    pr = poisson_ratio(dts, dtc)
    vpvs = Vp / Vs
    pr_std = (0.5 * vpvs ** 2 - 1) / (vpvs ** 2 - 1)
    print(f"  Poisson ratio          = {pr:.4f}  (standard {pr_std:.4f})")
    assert abs(pr - pr_std) < 1e-9

    # RMAD and misfit are ~0 for identical curves
    y = flexural_dispersion(FREQS_KHZ, 260.0)
    assert rmad(y, y) < 1e-12 and misfit(y, y) < 1e-12
    # small perturbation -> small RMAD
    y2 = flexural_dispersion(FREQS_KHZ, 262.0)
    print(f"  RMAD (260 vs 262 us/ft) = {rmad(y2, y):.4f}")
    assert 0 < rmad(y2, y) < 0.05

    # Dispersion inversion recovers the formation shear slowness
    dts_true = 260.0
    disp = flexural_dispersion(FREQS_KHZ, dts_true)
    dts_hat = invert_dts(FREQS_KHZ, disp)
    print(f"  inverted DTS           = {dts_hat:.2f} us/ft  (true {dts_true})")
    assert abs(dts_hat - dts_true) < 2.0
    assert np.all(np.diff(disp) >= 0)                # slowness rises with f
    print("  PASS")
    return {"C55": C[3, 3], "poisson": pr, "dts_hat": dts_hat}


if __name__ == "__main__":
    test_all()
