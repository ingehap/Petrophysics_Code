"""
Article 3: An Assessment of Fundamentals of Nuclear-Based Alternatives to
           Conventional Chemical-Source Bulk-Density Measurement
Ahmed Badruzzaman (2014)
Reference: Petrophysics Vol. 55, No. 5 (October 2014), pp. 415-434
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best of the 2014 SPWLA Annual Logging Symposium.  A Monte-Carlo study assesses
chemical-source-free bulk-density measurements: a bremsstrahlung X-ray generator
(Compton density, like Cs-137) and neutron-gamma density (inelastic-gamma and
neutron-gamma).  The fundamentals are the Compton attenuation of gamma rays and
the capture-correction of the neutron-gamma response.

Implements:

  - Compton transmission  I = I0*exp(-mu_c*x)  (Eq. 1)
  - Compton attenuation coefficient  mu_c = rho_B*Av*sigma_c  (Eq. 2)
  - Neutron-gamma capture-correction fraction (Eq. 3)
  - Neutron-gamma density from the two-detector inelastic count ratio (Neuman
    et al., 1999): rho proportional to log(ratio)
  - Counting-statistics precision  sigma_rho ~ 1/sqrt(N)

Note: this issue's PDF has a text layer; Eq. 1 is transcribed, while the Eq. 2
and Eq. 3 bodies were dropped in extraction and reconstructed from the stated
variable definitions (Neuman et al., 1999).  Effective Z values (quartz 11.78,
dolomite 13.74, limestone 15.71) are from the paper.  Densities in g/cm^3,
times in seconds, lengths in cm.
"""

import numpy as np

AVOGADRO = 6.022e23  # 1/mol


# ---------------------------------------------- Compton density --------------

def compton_transmission(i0, mu_c, x):
    """Transmitted photon intensity through a slab (Eq. 1)

        I = I0*exp(-mu_c*x),

    with the Compton (mass) attenuation coefficient mu_c and path length x.
    """
    return i0 * np.exp(-mu_c * np.asarray(x, float))


def compton_attenuation(rho_b, sigma_c, av=AVOGADRO):
    """Compton attenuation coefficient (Eq. 2)

        mu_c = rho_B*Av*sigma_c,

    proportional to bulk density rho_B through the electron density; sigma_c is
    the Compton cross-section per electron (constant at a given energy).
    """
    return rho_b * av * sigma_c


def density_from_transmission(i0, i, x, sigma_c, av=AVOGADRO):
    """Invert the Compton transmission for bulk density

        rho_B = ln(I0/I)/(x*Av*sigma_c).
    """
    return np.log(i0 / i) / (x * av * sigma_c)


# ---------------------------------------------- neutron-gamma --------------

def ngamma_capture_fraction(sigma_a, velocity, t1, t2):
    """Neutron-gamma capture fraction over a post-burst window
    (Eq. 3, single-medium simplification of Neuman et al., 1999)

        f = exp(-lambda*t1) - exp(-lambda*t2),   lambda = v*Sigma_a,

    the fraction of thermal neutrons captured between t1 and t2 (the thermal-
    neutron population decays as exp(-lambda*t)), with the macroscopic absorption
    cross-section Sigma_a (1/cm) and thermal-neutron velocity v (cm/s).  This
    capture contribution must be removed to isolate the inelastic density signal.
    """
    lam = velocity * sigma_a
    return np.exp(-lam * t1) - np.exp(-lam * t2)


def neutron_gamma_density(count_ratio, slope, intercept):
    """Bulk density from the two-detector inelastic count ratio (Neuman et al.)

        rho_B = intercept + slope*log(ratio),

    a linear calibration of density against the log of the near/far inelastic
    count ratio.
    """
    return intercept + slope * np.log(np.asarray(count_ratio, float))


# ---------------------------------------------- precision --------------

def counting_precision(rho, counts):
    """Statistical (counting) precision of a count-rate-based density

        sigma_rho ~ rho/sqrt(N),

    showing the higher source intensity (larger N) of the alternatives improves
    precision for a given logging speed.
    """
    return rho / np.sqrt(np.asarray(counts, float))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Nuclear Alternatives to Cs-137 Density")
    print("=" * 60)

    # Compton transmission falls with density and path length
    sigma_c = 4.8e-25  # cm^2/electron-scale constant
    mu_water = compton_attenuation(1.0, sigma_c)
    mu_rock = compton_attenuation(2.65, sigma_c)
    i_water = compton_transmission(1e6, mu_water, 10.0)
    i_rock = compton_transmission(1e6, mu_rock, 10.0)
    print(f"  I(water)={i_water:.3e}  I(quartz)={i_rock:.3e}")
    assert i_rock < i_water  # denser rock attenuates more

    # Density inverts exactly from the transmission
    rho_rec = density_from_transmission(1e6, i_rock, 10.0, sigma_c)
    print(f"  recovered density = {rho_rec:.3f} g/cm3")
    assert np.isclose(rho_rec, 2.65)

    # Capture fraction is a fraction in (0, 1) and grows with absorption
    f10 = ngamma_capture_fraction(sigma_a=0.1, velocity=2.2e5, t1=10e-6, t2=40e-6)
    print(f"  capture fraction = {f10:.3f}")
    assert 0.0 < f10 < 1.0
    assert (ngamma_capture_fraction(0.2, 2.2e5, 10e-6, 40e-6) > f10)

    # Neutron-gamma density increases with the inelastic count ratio
    rho_lo = neutron_gamma_density(1.5, slope=1.0, intercept=2.0)
    rho_hi = neutron_gamma_density(3.0, slope=1.0, intercept=2.0)
    assert rho_hi > rho_lo

    # More counts -> better (smaller) statistical uncertainty
    assert counting_precision(2.4, 1e6) < counting_precision(2.4, 1e4)
    print("  PASS")
    return {"rho_rec": float(rho_rec), "capture_fraction": float(f10)}


if __name__ == "__main__":
    test_all()
