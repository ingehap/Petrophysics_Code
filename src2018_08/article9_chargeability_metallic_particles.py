"""
Article 9: Chargeability of Porous Rocks With or Without Metallic Particles
Revil, Tartrat, Abdulsamad, Ghorbani, Coperey (2018)
DOI: 10.30632/PJV59V4-2018a8

Induced-polarization chargeability M (Seigel's definition) measures how much a
rock polarizes under an applied current.  For a clay-rich background it is set
by the Stern-layer surface conductivity; the ratio of the polarization to
surface-conduction mobilities gives a near-universal dimensionless number R, so
in the low-salinity limit the background chargeability tends to a universal
value.  Adding metallic (e.g. pyrite) particles adds a saturation- and
temperature-independent contribution from electrodiffusion inside the grains.

Implements:

  - Chargeability from the conductivity dispersion  M = 1 - sigma0/sigma_inf
  - Stern-layer surface conductivity  sigma_S = rho_g*B*CEC/(F*phi)
  - Instantaneous background conductivity  sigma_inf = sigma_w/F + sigma_S
  - Background chargeability  Mb = Mn/sigma_inf  and the universal R = lambda/B
  - Mixture chargeability with a metallic-particle contribution

Note: this issue's PDF has a text layer, and several of this article's relations
survived extraction (Eqs. 1, 8, 10-13); the remaining display fractions were
dropped, so the mixing relations are standard-form reconstructions.  The
reported mobilities B(Na+,25C)=1.63e-8 and lambda(Na+,25C)=1.41e-9 (giving
R~0.09) are reproduced.  SI units.
"""

import numpy as np

B_NA_25C = 1.63e-8           # surface-conduction apparent mobility (m^2 s^-1 V^-1)
LAMBDA_NA_25C = 1.41e-9      # polarization apparent mobility   (m^2 s^-1 V^-1)


# ---------------------------------------------- chargeability --------------

def chargeability(sigma0, sigma_inf):
    """Chargeability from the conductivity dispersion  M = 1 - sigma0/sigma_inf.

    From sigma0 = sigma_inf*(1 - M): the DC (relaxed) conductivity sigma0 is
    always <= the instantaneous conductivity sigma_inf, and M is the fractional
    drop between them (Seigel, Eqs. 1-2).
    """
    return 1.0 - np.asarray(sigma0, float) / sigma_inf


def surface_conductivity(rho_g, cec, formation_factor, phi, mobility=B_NA_25C):
    """Stern-layer surface conductivity  sigma_S = rho_g*B*CEC/(F*phi)  (Eq. 10).

    rho_g = grain density (~2650), CEC = cation exchange capacity (C/kg),
    F = intrinsic formation factor, phi = connected porosity, B = counterion
    apparent mobility.
    """
    return rho_g * mobility * cec / (formation_factor * phi)


def instantaneous_conductivity(sigma_w, formation_factor, sigma_s):
    """Instantaneous background conductivity  sigma_inf = sigma_w/F + sigma_S (Eq. 10)."""
    return sigma_w / formation_factor + sigma_s


def background_chargeability(normalized_chargeability, sigma_inf):
    """Background chargeability  Mb = Mn/sigma_inf  (Eq. 8)."""
    return normalized_chargeability / sigma_inf


def universal_R(lmbda=LAMBDA_NA_25C, mobility=B_NA_25C):
    """Universal dimensionless number  R = lambda/B  (salinity/T-independent).

    Equals the surface-conductivity-dominated limit of the background
    chargeability (~0.08-0.09).
    """
    return lmbda / mobility


def mixture_chargeability(phi_m, m_metallic, m_background):
    """Mixture chargeability  M = phi_m*M_metallic + (1 - phi_m)*Mb  (Eqs. 3-4).

    Volume-weighted sum of the metallic-particle contribution (proportional to
    the metallic volume fraction phi_m) and the background chargeability Mb.
    """
    return phi_m * m_metallic + (1.0 - phi_m) * m_background


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Chargeability of Porous Rocks")
    print("=" * 60)

    # Chargeability is the fractional conductivity drop; sigma0 <= sigma_inf
    m = chargeability(0.0095, 0.01)
    print(f"  chargeability          = {m:.3f}")
    assert np.isclose(m, 0.05) and 0.0 <= m <= 1.0

    # Universal number R = lambda/B ~ 0.09 (low-salinity background limit)
    r = universal_R()
    print(f"  universal R = lambda/B = {r:.3f}")
    assert np.isclose(r, 0.0865, atol=0.01)

    # Surface conductivity feeds the instantaneous conductivity and Mb
    sig_s = surface_conductivity(rho_g=2650.0, cec=0.1, formation_factor=10.0, phi=0.3)
    sig_inf = instantaneous_conductivity(sigma_w=0.1, formation_factor=10.0, sigma_s=sig_s)
    print(f"  sigma_S / sigma_inf    = {sig_s:.4f} / {sig_inf:.4f} S/m")
    assert sig_inf > sig_s > 0
    mb = background_chargeability(normalized_chargeability=sig_s * r, sigma_inf=sig_inf)
    assert 0.0 < mb < 0.1

    # Adding metallic particles raises the chargeability above the background
    m_low = mixture_chargeability(phi_m=0.0, m_metallic=0.088, m_background=mb)
    m_hi = mixture_chargeability(phi_m=0.013, m_metallic=0.088, m_background=mb)
    print(f"  M (0% / 1.3% metallic) = {m_low:.3f} / {m_hi:.3f}")
    assert m_hi > m_low and np.isclose(m_low, mb)
    print("  PASS")
    return {"R": float(r), "Mb": float(mb), "M_metallic": float(m_hi)}


if __name__ == "__main__":
    test_all()
