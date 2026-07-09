"""
Article 6: Influence of Magnetic Susceptibility Contrast on NMR Studies -
           Experimental Analysis From Siliciclastic Reservoirs
Sarkar, Chatterjee, Lal, Kumar, Deo (2019)
DOI: 10.30632/PJV60N6-2019a6

Paramagnetic minerals create internal magnetic-field gradients that add a
diffusion term to the NMR T2 relaxation, which grows with the square of the echo
spacing TE.  Measuring T2 at several echo spacings and regressing 1/T2 against
TE^2 isolates the diffusion term and yields an (upper-limit) internal gradient
G, explaining anomalously low T2 cutoffs in magnetic-mineral-rich sandstones.

Implements:

  - Three-mechanism relaxation  1/T2 = 1/T2B + rho*(S/V) + (gamma*G*TE)^2*D/12  (Eq. 1)
  - Carr-Purcell diffusion term  1/T2D = (gamma*G*TE)^2*D/12                    (Eq. 2)
  - Internal gradient from the slope of 1/T2 vs TE^2                            (Eq. 3)
  - Length scales: structural V/S, diffusion sqrt(D*tau), dephasing sqrt(D/(gamma*G))

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are the standard NMR
internal-gradient forms anchored to those definitions.  gamma = 2.675e8 /T/s;
reported internal gradients 72-510 Gauss/cm.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

GAMMA_H = 2.675e8        # rad/s/T
GAUSS_PER_CM_TO_T_PER_M = 1e-2      # 1 Gauss/cm = 1e-2 T/m


# ---------------------------------------------- relaxation --------------

def diffusion_term(G_T_per_m, TE, D, gamma=GAMMA_H):
    """Carr-Purcell diffusion relaxation rate  (gamma*G*TE)^2*D/12  (Eq. 2)."""
    return petrolib.nmr.diffusion_relaxation_rate(D, G=G_T_per_m, TE=TE, gamma=gamma)


def t2_total(T2_bulk, rho, s_over_v, G_T_per_m, TE, D, gamma=GAMMA_H):
    """Total T2 from bulk + surface + diffusion  (Eq. 1).  Returns T2 in s."""
    return petrolib.nmr.t2_apparent(
        t2_bulk=T2_bulk, rho=rho, s_over_v=s_over_v, D=D, G=G_T_per_m, TE=TE, gamma=gamma)


# ---------------------------------------------- gradient inversion ------

def internal_gradient_from_slope(TE_array, inv_T2, D, gamma=GAMMA_H):
    """Internal gradient G from the slope of 1/T2 vs TE^2  (Eq. 3).

        1/T2 = const + (gamma^2 * G^2 * D / 12) * TE^2
    slope = gamma^2 G^2 D /12  ->  G = sqrt(slope*12/(gamma^2*D)).
    Returns G in Gauss/cm.
    """
    TE2 = np.asarray(TE_array, float) ** 2
    slope, _ = np.polyfit(TE2, np.asarray(inv_T2, float), 1)
    G_T_per_m = np.sqrt(slope * 12.0 / (gamma ** 2 * D))
    return G_T_per_m / GAUSS_PER_CM_TO_T_PER_M


# ---------------------------------------------- length scales -----------

def structural_length(volume, surface):
    """Pore structural length  L_s = V/S."""
    return volume / surface


def diffusion_length(D, tau):
    """Diffusion length  L_d = sqrt(D*tau),  tau = TE/2."""
    return petrolib.flow_transport.diffusion_length(D, tau)


def dephasing_length(D, G_T_per_m, gamma=GAMMA_H):
    """Dephasing length  L_g = (D/(gamma*G))^(1/3)."""
    return (D / (gamma * G_T_per_m)) ** (1.0 / 3.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Magnetic Susceptibility Contrast on NMR")
    print("=" * 60)

    D = 2.5e-9                                 # m^2/s
    # Diffusion term grows with TE^2: doubling TE quadruples the rate
    G = 200.0 * GAUSS_PER_CM_TO_T_PER_M        # 200 Gauss/cm -> T/m
    d1 = diffusion_term(G, 1e-3, D)
    d2 = diffusion_term(G, 2e-3, D)
    print(f"  1/T2D at TE=1/2 ms     = {d1:.3f} / {d2:.3f} 1/s")
    assert abs(d2 / d1 - 4.0) < 1e-9

    # Larger echo spacing shortens T2 in a gradient
    assert t2_total(3.0, 5e-6, 1e4, G, 2e-3, D) < t2_total(3.0, 5e-6, 1e4, G, 0.5e-3, D)

    # Recover a planted internal gradient from the 1/T2-vs-TE^2 slope
    G_true = 350.0                             # Gauss/cm
    G_si = G_true * GAUSS_PER_CM_TO_T_PER_M
    TEs = np.array([0.1, 0.5, 1.0, 2.0, 5.0]) * 1e-3
    inv_T2 = 1.0 / 3.0 + 5e-6 * 1e4 + diffusion_term(G_si, TEs, D)
    G_fit = internal_gradient_from_slope(TEs, inv_T2, D)
    print(f"  recovered gradient     = {G_fit:.1f} Gauss/cm  (true {G_true})")
    assert abs(G_fit - G_true) < 1.0
    assert 72.0 <= G_fit <= 510.0              # within the paper's reported range

    # Length scales are positive and ordered sensibly
    assert structural_length(1e-15, 1e-9) > 0
    assert diffusion_length(D, 1e-3) > 0 and dephasing_length(D, G_si) > 0
    print("  PASS")
    return {"grad_recovered": float(G_fit), "T2D_1ms": float(d1)}


if __name__ == "__main__":
    test_all()
