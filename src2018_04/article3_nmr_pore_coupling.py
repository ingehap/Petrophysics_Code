"""
Article 3: Nuclear Magnetic Resonance and Pore Coupling in Clay-Coated
           Sandstones With Anomalous Porosity Preservation, Agua Grande
           Formation, Reconcavo Basin, Brazil
Jacomo, Trindade, de Oliveira, Leite, Montrazi, Andreeta, Bonagamba (2018)
DOI: 10.30632/PJV59N2-2018a2

NMR T2 relaxation of chlorite-coated sandstones is interpreted to attribute
relaxation peaks to pore families (macropores, dissolution pores, clay-coating
micropores) and to detect diffusive pore coupling.  In the fast-diffusion
regime the relaxation rate is proportional to the pore surface-to-volume ratio,
the multiexponential magnetization decay is the sum of the pore contributions,
and a field gradient adds a diffusion-relaxation term.

Implements:

  - Surface relaxation (fast diffusion)  1/T2 = rho2*(S/V)
  - Multiexponential decay  Mz(t) = sum_i M_i*exp(-t/T2_i)
  - Diffusion relaxation in a gradient  1/T2_D = D*gamma^2*G^2*TE^2/12
  - Pore surface-to-volume ratio from a measured T2

Note: this issue's PDF has a text layer; the surface-relaxation law (Eq. 1)
survived, while the decay and diffusion expressions (Eqs. 2-3) lost their
typeset glyphs and are faithful standard-form reconstructions.  SI/CGS-mixed as
in the paper: T in s, S/V in 1/m, gradient in T/m, gamma in rad/s/T.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

GAMMA_H = 2.675e8            # proton gyromagnetic ratio (rad/s/T)


# ---------------------------------------------- relaxation --------------

def surface_relaxation_t2(rho2, s_over_v):
    """Transverse surface relaxation time  T2 = 1/(rho2*(S/V))  (Eq. 1).

    rho2 = T2 surface relaxivity (m/s), S/V = surface-to-volume ratio (1/m).
    Larger pores (smaller S/V) relax more slowly (longer T2).
    """
    return petrolib.nmr.t2_apparent(rho=rho2, s_over_v=s_over_v)


def surface_to_volume(rho2, t2):
    """Invert the surface relaxation for the pore S/V  =  1/(rho2*T2)."""
    return petrolib.nmr.surface_to_volume(t2, rho=rho2)


def multiexponential_decay(t, amplitudes, t2s):
    """Total transverse magnetization decay  Mz(t) = sum_i M_i*exp(-t/T2_i) (Eq. 2)."""
    return petrolib.nmr.multiexp_decay(t, amplitudes, t2s)


def diffusion_relaxation_rate(diffusivity, gradient, te, gamma=GAMMA_H):
    """Diffusion relaxation rate in a field gradient (Eq. 3)

        1/T2_D = D*gamma^2*G^2*TE^2/12,

    D = molecular diffusion coefficient, G = field gradient, TE = interecho time.
    """
    return petrolib.nmr.diffusion_relaxation_rate(diffusivity, G=gradient, TE=te, gamma=gamma)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: NMR Pore Coupling")
    print("=" * 60)

    # Macropore (small S/V) relaxes slower than a clay-coating micropore
    t2_macro = surface_relaxation_t2(1e-5, 5e3)
    t2_micro = surface_relaxation_t2(1e-5, 5e5)
    print(f"  T2 macro / micro       = {t2_macro:.3e} / {t2_micro:.3e} s")
    assert t2_macro > t2_micro
    assert np.isclose(surface_to_volume(1e-5, t2_macro), 5e3)

    # Two-pore decay starts at the total amplitude and decays monotonically
    t = np.linspace(0, 0.5, 50)
    mz = multiexponential_decay(t, amplitudes=[0.7, 0.3], t2s=[0.2, 0.01])
    print(f"  Mz(0) / Mz(end)        = {mz[0]:.3f} / {mz[-1]:.3f}")
    assert np.isclose(mz[0], 1.0) and np.all(np.diff(mz) < 0)

    # A stronger gradient or longer echo time speeds diffusion relaxation
    r_short = diffusion_relaxation_rate(2.5e-9, 0.1, 200e-6)
    r_long = diffusion_relaxation_rate(2.5e-9, 0.1, 400e-6)
    assert r_long > r_short > 0
    print("  PASS")
    return {"T2_macro": float(t2_macro), "diff_rate": float(r_long)}


if __name__ == "__main__":
    test_all()
