"""
Article 1: Subsurface Fluid Characterization Using Downhole and Core NMR T1-T2
           Maps Combined with Pore-Scale Imaging Techniques
Lessenger, Merkel, Medina, Ramakrishna, Chen, Balliet, Xie, Bhattad, Carnerup,
Knackstedt (2015)
Reference: Petrophysics Vol. 56, No. 4 (August 2015), pp. 313-333
DOI: none assigned (this issue predates SPWLA DOI assignment)

NMR T1-T2 (and T2-D) maps characterize fluid type, saturation and wettability in
the mixed-wet Green River sandstones.  Each fluid's relaxation is the sum of
bulk, surface and (for T2) diffusion contributions; wettability modifies the
surface term (the wetting fluid relaxes faster), and the diffusion term depends
on the molecular diffusion coefficient, tool gradient and echo spacing.  The
echo trains are inverted (with a non-negativity constraint) to the multimodal
relaxation distribution.

Implements:

  - T1 relaxation  1/T1 = 1/T1bulk + 1/T1surface  (Eq. 1)
  - Apparent T2  1/T2app = 1/T2bulk + 1/T2surface + 1/T2diffusion  (Eq. 2)
  - Intrinsic T2  1/T2int = 1/T2bulk + 1/T2surface  (Eq. 3)
  - Diffusion relaxation rate  1/T2D = (gamma*G*TE)^2*D/12
  - Multiexponential echo-decay forward model and T1/T2 fluid typing

Note: this issue's PDF has a text layer; the relaxation relations (Eqs. 1-3) and
the echo-decay model (Eq. 4) are transcribed from the body, while the typeset
glyphs were dropped and reconstructed in standard form (Coates et al., 1999).
Times in s, diffusion in m^2/s, gradient in T/m, echo spacing in s.
"""

import numpy as np

GAMMA_PROTON = 2.675e8        # rad/(s*T), proton gyromagnetic ratio


# ---------------------------------------------- relaxation components --------------

def t1_relaxation(t1_bulk, t1_surface):
    """Longitudinal relaxation  1/T1 = 1/T1bulk + 1/T1surface  (Eq. 1)."""
    return 1.0 / (1.0 / t1_bulk + 1.0 / t1_surface)


def t2_apparent(t2_bulk, t2_surface, t2_diffusion):
    """Apparent transverse relaxation (Eq. 2)

        1/T2app = 1/T2bulk + 1/T2surface + 1/T2diffusion.
    """
    return 1.0 / (1.0 / t2_bulk + 1.0 / t2_surface + 1.0 / t2_diffusion)


def t2_intrinsic(t2_bulk, t2_surface):
    """Intrinsic transverse relaxation (diffusion removed, Eq. 3)

        1/T2int = 1/T2bulk + 1/T2surface.
    """
    return 1.0 / (1.0 / t2_bulk + 1.0 / t2_surface)


def diffusion_relaxation_rate(diffusion_coeff, gradient, echo_spacing, gamma=GAMMA_PROTON):
    """Diffusion-induced T2 relaxation rate

        1/T2D = (gamma*G*TE)^2 * D / 12,

    with G the tool field gradient, TE the echo spacing and D the molecular
    diffusion coefficient (gas > water > oil, decreasing with oil viscosity).
    """
    return (gamma * gradient * echo_spacing) ** 2 * diffusion_coeff / 12.0


# ---------------------------------------------- echo decay / typing --------------

def multiexponential_decay(times, amplitudes, t2_components):
    """Forward CPMG echo-decay (Eq. 4)

        M(t) = sum_i A_i * exp(-t/T2_i),

    the multimodal sum the inversion recovers (subject to A_i >= 0, Eq. 5).
    """
    t = np.asarray(times, float)[:, None]
    a = np.asarray(amplitudes, float)[None, :]
    t2 = np.asarray(t2_components, float)[None, :]
    return np.sum(a * np.exp(-t / t2), axis=1)


def t1_t2_ratio_fluid_type(t1, t2, water_cutoff=2.0):
    """Fluid type from the T1/T2 ratio: water/brine ~1-2, oil and bound/heavy
    fluids higher (wettability and viscosity raise the ratio)."""
    return "water" if t1 / t2 < water_cutoff else "hydrocarbon/bound"


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: NMR T1-T2 Fluid Characterization")
    print("=" * 60)

    # Combined relaxation is faster (shorter time) than any single mechanism
    t1 = t1_relaxation(3.0, 0.5)
    assert t1 < 0.5 and np.isclose(1.0 / t1, 1.0 / 3.0 + 1.0 / 0.5)

    # Apparent T2 is shorter than intrinsic T2 (diffusion adds a relaxation path)
    t2d = 1.0 / diffusion_relaxation_rate(2.5e-9, gradient=0.2, echo_spacing=0.6e-3)
    t2app = t2_apparent(2.0, 0.3, t2d)
    t2int = t2_intrinsic(2.0, 0.3)
    print(f"  T2 app / intrinsic     = {t2app:.4f} / {t2int:.4f} s")
    assert t2app < t2int

    # Higher diffusion coefficient (gas) gives a faster diffusion rate than oil
    assert diffusion_relaxation_rate(1e-7, 0.2, 0.6e-3) > diffusion_relaxation_rate(1e-10, 0.2, 0.6e-3)

    # Multiexponential decay is monotonically decreasing from the total porosity
    t = np.linspace(0.0, 1.0, 50)
    m = multiexponential_decay(t, [0.1, 0.15], [0.05, 0.5])
    print(f"  echo decay M(0)/M(end) = {m[0]:.3f} / {m[-1]:.3f}")
    assert np.isclose(m[0], 0.25) and m[-1] < m[0] and np.all(np.diff(m) <= 0)

    # T1/T2 fluid typing
    assert t1_t2_ratio_fluid_type(1.0, 0.8) == "water"
    assert t1_t2_ratio_fluid_type(2.0, 0.2) == "hydrocarbon/bound"
    print("  PASS")
    return {"T2app": float(t2app), "T2int": float(t2int), "M0": float(m[0])}


if __name__ == "__main__":
    test_all()
