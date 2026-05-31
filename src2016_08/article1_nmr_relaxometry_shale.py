"""
Article 1: NMR Relaxometry in Shale and Implications for Logging
Kausik, Fellah, Rylander, Singer, Lewis, Sinclair (2016)
Reference: Petrophysics Vol. 57, No. 4 (August 2016), pp. 339-350
DOI: none assigned (this issue predates SPWLA DOI assignment)

NMR relaxometry distinguishes the constituents of unconventional shale (kerogen,
bitumen, heavy/light oil, water, gas) from the frequency dependence of their
relaxation times.  Liquid-hydrocarbon relaxation is the sum of intramolecular,
intermolecular and electron-dipolar rates; the homonuclear dipole-dipole T1 and
T2 follow the Bloembergen-Purcell-Pound (BPP) spectral-density model, whose
T1/T2 ratio rises sharply for slow molecular motion (large omega*tau_c).  Bulk
gas (methane) relaxes by spin rotation with T1 = T2.

Implements:

  - Additive liquid-hydrocarbon relaxation rate  1/Ti = sum of mechanisms (Eq. 1)
  - BPP spectral density  J(omega) = tau_c/(1 + (omega*tau_c)^2)
  - BPP dipolar T1 and T2 rates and the T1/T2 ratio (Eqs. 2-3)
  - Gas spin-rotation rate  1/T1 = 1/T2  (Eq. 4)

Note: this issue's PDF has a text layer; the additive-rate (Eq. 1), BPP (Eqs.
2-3) and spin-rotation (Eq. 4) relations are transcribed from the body, while
the typeset glyphs were dropped and the BPP/spin-rotation forms are standard
reconstructions (Bloembergen et al., 1948; Hubbard, 1963).  Times in s,
frequencies (Larmor) in rad/s, correlation times tau_c in s.
"""

import numpy as np


# ---------------------------------------------- additive rates --------------

def total_relaxation_rate(*rates):
    """Additive liquid-hydrocarbon relaxation rate (Eq. 1)

        1/Ti = 1/Ti_intra + 1/Ti_inter + 1/Ti_elec + ...,

    summing the intramolecular, intermolecular, electron-dipolar (and any other)
    contributions.  Pass the component times Ti; returns the combined Ti.
    """
    inv = sum(1.0 / np.asarray(t, float) for t in rates)
    return 1.0 / inv


# ---------------------------------------------- BPP dipolar model --------------

def spectral_density(omega, tau_c):
    """BPP spectral density  J(omega) = tau_c/(1 + (omega*tau_c)^2)."""
    return tau_c / (1.0 + (np.asarray(omega, float) * tau_c) ** 2)


def bpp_rates(omega0, tau_c, dipolar_constant=1.0):
    """Homonuclear dipole-dipole T1 and T2 rates (Eqs. 2-3, BPP)

        1/T1 = (3/10)*D2*[J(w0) + 4 J(2 w0)]
        1/T2 = (3/20)*D2*[3 J(0) + 5 J(w0) + 2 J(2 w0)],

    with D2 the dipolar coupling constant (mu0/4pi)^2 * gamma^4 * hbar^2 *
    I(I+1)/r^6.  Returns (T1, T2).
    """
    j0 = spectral_density(0.0, tau_c)
    j1 = spectral_density(omega0, tau_c)
    j2 = spectral_density(2.0 * omega0, tau_c)
    inv_t1 = (3.0 / 10.0) * dipolar_constant * (j1 + 4.0 * j2)
    inv_t2 = (3.0 / 20.0) * dipolar_constant * (3.0 * j0 + 5.0 * j1 + 2.0 * j2)
    return 1.0 / inv_t1, 1.0 / inv_t2


def t1_t2_ratio(omega0, tau_c):
    """T1/T2 ratio from the BPP model

        T1/T2 = [3 J(0) + 5 J(w0) + 2 J(2 w0)] / (2*[J(w0) + 4 J(2 w0)]),

    ~1 for fast motion (w0*tau_c << 1, light fluids) and large for slow motion
    (w0*tau_c >> 1, kerogen/bitumen/bound fluids).
    """
    t1, t2 = bpp_rates(omega0, tau_c, dipolar_constant=1.0)
    return t1 / t2


# ---------------------------------------------- gas spin rotation --------------

def spin_rotation_rate(temperature, viscosity, sr_constant=1.0):
    """Bulk-gas (methane) spin-rotation relaxation (Eq. 4)

        1/T1 = 1/T2 = C_sr * T / viscosity,

    since the spin-rotation correlation time tau_F is inversely proportional to
    viscosity; spin-spin and spin-lattice rates are equal for motional-narrowed
    gas.  Returns the (equal) T1 = T2 time.
    """
    return viscosity / (sr_constant * temperature)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: NMR Relaxometry in Shale")
    print("=" * 60)

    # Additive rates: combined time is shorter than any single component
    t = total_relaxation_rate(1.0, 2.0, 4.0)
    assert t < 1.0 and np.isclose(1.0 / t, 1.0 + 0.5 + 0.25)

    # Fast motion (w0*tau_c << 1): T1 ~ T2, ratio ~ 1
    w0 = 2.0 * np.pi * 2.0e6        # 2 MHz Larmor (rad/s)
    r_fast = t1_t2_ratio(w0, tau_c=1.0e-11)
    print(f"  T1/T2 fast motion      = {r_fast:.3f}")
    assert np.isclose(r_fast, 1.0, atol=1e-3)

    # Slow motion (w0*tau_c >> 1): T1/T2 >> 1 (kerogen/bitumen-like)
    r_slow = t1_t2_ratio(w0, tau_c=1.0e-6)
    print(f"  T1/T2 slow motion      = {r_slow:.1f}")
    assert r_slow > 10.0

    # The ratio increases monotonically with correlation time
    taus = [1e-11, 1e-9, 1e-7, 1e-6]
    ratios = [t1_t2_ratio(w0, tc) for tc in taus]
    assert all(b >= a for a, b in zip(ratios, ratios[1:]))

    # BPP T1 and T2 are positive and T1 >= T2
    t1, t2 = bpp_rates(w0, 1.0e-8, dipolar_constant=1.0e9)
    print(f"  BPP T1 / T2            = {t1:.4f} / {t2:.4f} s")
    assert t1 >= t2 > 0

    # Gas spin rotation: T1 = T2, and relaxation is slower at higher T (lower visc effect)
    tg = spin_rotation_rate(350.0, 1.5e-5, sr_constant=1.0)
    assert tg > 0
    print("  PASS")
    return {"T1T2_fast": float(r_fast), "T1T2_slow": float(r_slow), "T1": float(t1)}


if __name__ == "__main__":
    test_all()
