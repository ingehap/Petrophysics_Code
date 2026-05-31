"""
Article 3: Magnetic Resonance Core-Plug Analysis with the Three-Magnet Array
           Unilateral Magnet
Juan C. Garcia-Naranjo, Pan Guo, Florin Marica, Guangzhi Liao, Bruce J. Balcom
           (2014)
Reference: Petrophysics Vol. 55, No. 3 (June 2014), pp. 229-239
DOI: none assigned (this issue predates SPWLA DOI assignment)

A three-magnet array unilateral (one-sided) magnet measures core plugs either in
a homogeneous "sweet spot" or in an extended constant-gradient region.  CPMG
relaxation, the gradient-enhanced effective relaxation, and a signal-ratio
porosity calibrated against a reference plug are the working relations.

Implements:

  - Larmor frequency  f = gamma*B0/(2*pi)
  - CPMG magnetization decay  M(t) = M0*exp(-t/T2eff)
  - Effective T2 with surface relaxation and gradient diffusion
  - Signal-ratio porosity against a reference plug

Note: this instrumentation paper carries no numbered display equations; the
CPMG decay, the gradient-diffusion effective relaxation and the signal-ratio
porosity are standard unilateral-NMR relations written in standard form (Hoult,
1972).  A constant gradient G = 60 G/cm and a Berea reference porosity of 21%
are reported.  Times in seconds, fields in tesla, gradient in T/m.
"""

import numpy as np

GAMMA_H = 2.675222e8  # proton gyromagnetic ratio, rad/(s*T)


# ---------------------------------------------- field & frequency --------------

def larmor_frequency(b0):
    """Proton Larmor frequency from the static field

        f = gamma*B0/(2*pi),

    with B0 in tesla; returns Hz.  (B0 ~ 0.0528 T gives ~2.25 MHz.)
    """
    return GAMMA_H * b0 / (2.0 * np.pi)


# ---------------------------------------------- CPMG relaxation --------------

def cpmg_decay(t, m0, t2eff):
    """CPMG transverse magnetization decay

        M(t) = M0*exp(-t/T2eff).
    """
    return m0 * np.exp(-np.asarray(t, float) / t2eff)


def t2_effective(t2_bulk, relaxivity, surface_to_volume, diffusion,
                 gradient, echo_spacing):
    """Effective transverse relaxation rate combining bulk, surface and
    gradient-diffusion contributions

        1/T2eff = 1/T2_bulk + rho2*(S/V) + (D*gamma^2*G^2*TE^2)/12,

    with surface relaxivity rho2, surface-to-volume S/V, diffusion D, constant
    gradient G (T/m) and echo spacing TE.
    """
    inv = (1.0 / t2_bulk + relaxivity * surface_to_volume
           + diffusion * GAMMA_H ** 2 * gradient ** 2 * echo_spacing ** 2 / 12.0)
    return 1.0 / inv


def porosity_from_signal(signal_sample, signal_ref, porosity_ref):
    """Signal-ratio porosity calibrated against a reference plug

        phi = phi_ref*(S_sample/S_ref),

    the NMR signal being proportional to the fluid (hydrogen) volume in the
    sensitive spot.
    """
    return porosity_ref * signal_sample / signal_ref


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Three-Magnet Array Unilateral MR")
    print("=" * 60)

    # Larmor frequency: the reported 2.25 MHz operating point
    f = larmor_frequency(0.05285)
    print(f"  Larmor frequency = {f/1e6:.2f} MHz")
    assert np.isclose(f / 1e6, 2.25, atol=0.05)

    # CPMG decay falls monotonically and starts at M0
    t = np.linspace(0, 1.0, 50)
    m = cpmg_decay(t, m0=100.0, t2eff=0.1)
    assert np.isclose(m[0], 100.0) and np.all(np.diff(m) < 0)

    # Gradient diffusion shortens T2eff relative to the gradient-free case
    g_field = 0.6  # 60 G/cm in T/m
    t2_grad = t2_effective(3.0, 1e-5, 1e4, 2.5e-9, gradient=g_field, echo_spacing=6e-4)
    t2_nograd = t2_effective(3.0, 1e-5, 1e4, 2.5e-9, gradient=0.0, echo_spacing=6e-4)
    print(f"  T2eff(grad)={t2_grad*1e3:.2f} ms   T2eff(no grad)={t2_nograd*1e3:.2f} ms")
    assert t2_grad < t2_nograd

    # Signal-ratio porosity against the 21% Berea reference
    phi = porosity_from_signal(signal_sample=80.0, signal_ref=100.0, porosity_ref=0.21)
    print(f"  porosity = {phi:.3f}")
    assert np.isclose(phi, 0.168)
    print("  PASS")
    return {"f_MHz": float(f / 1e6), "T2eff_grad": float(t2_grad), "phi": float(phi)}


if __name__ == "__main__":
    test_all()
