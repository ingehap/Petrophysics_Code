"""
Article 6: Estimation of Gas-Condensate Relative Permeability Using a Lattice
           Boltzmann Modeling Approach
Schembre-McCabe, Kamath, Fager, Crouse (2020)
DOI: 10.30632/PJV61N2-2020a6

A multiphase Lattice Boltzmann Method (LBM) computes gas/condensate relative
permeability on a 3D micro-CT pore space of a Berea sandstone across a range of
capillary numbers, for both displacement and condensate-dropout processes.  The
capillary number - the ratio of viscous to capillary forces - controls the
rate (positive coupling) effect that lifts gas relative permeability toward the
miscible limit at high flow rate / low interfacial tension.

Implements:

  - Capillary number  N_c = mu*v/sigma                           (Eq. 1)
  - Capillary-desaturation gas rel-perm vs N_c (rate effect)
  - Base (Corey) gas/condensate relative permeability vs saturation

Note: this issue's PDF text layer contained only the first page of this article
(the krg/kro-vs-Nc result tables were beyond the extract truncation), so the
capillary-number relation (Eq. 1) is implemented directly and the relative-
permeability response is the standard capillary-desaturation / Corey model the
LBM study parameterizes.  Berea micro-CT at 2.6 um/voxel.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- capillary number --------

def capillary_number(mu, v, sigma):
    """Capillary number  N_c = mu*v/sigma  (viscous / capillary forces)  (Eq. 1)."""
    return petrolib.relperm_wettability.capillary_number(mu=mu, v=v, sigma=sigma)


# ---------------------------------------------- rate effect -------------

def krg_vs_capillary_number(nc, krg_low=0.3, krg_high=1.0, nc50=1e-5, a=1.0):
    """Gas rel-perm vs capillary number (capillary-desaturation curve).

        krg(Nc) = krg_low + (krg_high - krg_low) * Nc^a/(Nc^a + Nc50^a)
    At low Nc the immiscible (capillary-dominated) value krg_low holds; at high
    Nc the rate effect lifts it toward the miscible limit krg_high.
    """
    nc = np.asarray(nc, float)
    return krg_low + (krg_high - krg_low) * nc ** a / (nc ** a + nc50 ** a)


# ---------------------------------------------- Corey base curves -------

def corey_krg(sg, sgc, sor, ng=2.0, krg_max=1.0):
    """Corey gas relative permeability  krg = krg_max*Sg*^ng."""
    # Sg* = (Sg - Sgc)/(1 - Sgc - Sor): 2-endpoint gas (swc=0, sorg=Sor).
    return petrolib.relperm_wettability.corey_krg(
        sg, sgc=sgc, swc=0.0, sorg=sor, krg_max=krg_max, ng=ng)


def corey_kro(sg, sgc, sor, no=2.0, kro_max=1.0):
    """Corey condensate (oil) relative permeability  kro = kro_max*(1-Sg*)^no."""
    # kro_max*(1-Sg*)^no on the gas-saturation normalization (Sg* = (Sg-Sgc)/
    # (1-Sgc-Sor)) is the library's (1-Se) form with the saturation = Sg.
    return petrolib.relperm_wettability.corey_kro(
        sg, swr=sgc, sor=sor, kro_max=kro_max, no=no)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Gas-Condensate Rel-Perm via Lattice Boltzmann")
    print("=" * 60)

    # Capillary number rises with velocity and falls with interfacial tension
    nc1 = capillary_number(2e-5, 1e-4, 1e-3)
    nc2 = capillary_number(2e-5, 1e-3, 1e-3)
    print(f"  Nc at v=1e-4 / 1e-3    = {nc1:.2e} / {nc2:.2e}")
    assert nc2 > nc1
    assert capillary_number(2e-5, 1e-4, 1e-2) < nc1     # higher sigma -> lower Nc

    # Rate effect: krg rises monotonically with Nc toward the miscible limit
    nc = np.logspace(-7, -2, 20)
    krg = krg_vs_capillary_number(nc)
    print(f"  krg(low Nc) / krg(high Nc) = {krg[0]:.3f} / {krg[-1]:.3f}")
    assert np.all(np.diff(krg) >= -1e-12)
    assert abs(krg[0] - 0.3) < 0.02 and krg[-1] > 0.95

    # Base Corey curves: krg rises with Sg, kro falls; they cross
    sg = np.linspace(0.1, 0.9, 17)
    krg_c = corey_krg(sg, sgc=0.05, sor=0.10)
    kro_c = corey_kro(sg, sgc=0.05, sor=0.10)
    assert np.all(np.diff(krg_c) >= -1e-12)
    assert np.all(np.diff(kro_c) <= 1e-12)
    assert krg_c[0] < kro_c[0] and krg_c[-1] > kro_c[-1]   # curves cross
    print("  PASS")
    return {"Nc": float(nc1), "krg_lowNc": float(krg[0]), "krg_hiNc": float(krg[-1])}


if __name__ == "__main__":
    test_all()
