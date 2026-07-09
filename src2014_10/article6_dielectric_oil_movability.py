"""
Article 6: Application of an Oil-Movability Quicklook Technique Using Dielectric
           Measurements at Four Depths of Investigation
S.T. Grayson and J.L. Hemingway (2014)
Reference: Petrophysics Vol. 55, No. 5 (October 2014), pp. 461-469
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best of the 2014 SPWLA Annual Logging Symposium.  A multifrequency dielectric
pad measures apparent permittivity at four transmitter-receiver spacings (four
depths of investigation, up to ~4 in.).  A simplified CRIM model converts each
permittivity to water-filled porosity, hence a radial water/oil-saturation
profile, and the separation between the shallow and deep oil saturations gives a
moved-oil (movability) quicklook in low-salinity reservoirs.

Implements:

  - CRIM mixing permittivity and water-filled porosity from permittivity
  - Water and oil saturation  Sw = phi_water/phi_total,  So = 1 - Sw
  - Radial saturation profile from four depths of investigation
  - Moved-oil quicklook  dSo = So(deep) - So(shallow)

Note: this paper carries no numbered display equations; the simplified CRIM
relation and Sw = phi_water/phi_total are transcribed from the prose and written
in standard form (Grayson & Hemingway, 2013; Hizem et al., 2008).  Permittivities
relative, porosities and saturations as fractions.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- CRIM --------------

def crim_permittivity(sw, phi, eps_w, eps_hc, eps_matrix):
    """Complex-refractive-index (CRIM) mixing permittivity

        sqrt(eps) = (1-phi)*sqrt(eps_matrix) + phi*Sw*sqrt(eps_w)
                    + phi*(1-Sw)*sqrt(eps_hc).
    """
    return petrolib.em_dielectric.crim(
        phi, sw, eps_w=eps_w, eps_hc=eps_hc, eps_matrix=eps_matrix
    )


def water_filled_porosity(eps_apparent, eps_water, eps_matrix_star):
    """Water-filled porosity from the apparent permittivity (simplified CRIM)

        phi_water = (sqrt(eps_a) - sqrt(eps_matrix*))/(sqrt(eps_w) - sqrt(eps_matrix*)),

    where eps_matrix* is the effective permittivity of the non-water (matrix +
    hydrocarbon) system.
    """
    return petrolib.em_dielectric.water_filled_porosity(
        eps_apparent, eps_matrix=eps_matrix_star, eps_w=eps_water, clip=False
    )


# ---------------------------------------------- saturations --------------

def water_saturation(phi_water, phi_total):
    """Water saturation  Sw = phi_water/phi_total."""
    return np.asarray(phi_water, float) / phi_total


def oil_saturation(phi_water, phi_total):
    """Oil saturation  So = 1 - Sw = 1 - phi_water/phi_total."""
    return 1.0 - water_saturation(phi_water, phi_total)


def radial_saturation_profile(eps_apparent_4doi, phi_total, eps_water,
                              eps_matrix_star):
    """Radial oil-saturation profile from the four DOI apparent permittivities

    Converts each of the four spacings (shallow -> deep) to water-filled
    porosity and then to oil saturation, returning the So profile.
    """
    phi_w = water_filled_porosity(np.asarray(eps_apparent_4doi, float),
                                  eps_water, eps_matrix_star)
    return oil_saturation(phi_w, phi_total)


# ---------------------------------------------- oil movability --------------

def moved_oil(so_profile):
    """Moved-oil (movability) quicklook

        dSo = So(deep) - So(shallow),

    the deep-minus-shallow oil-saturation separation.  A positive dSo means oil
    was flushed from the shallow (invaded) zone and is therefore movable; the
    deepest reading is taken as the true oil saturation.
    """
    so = np.asarray(so_profile, float)
    return float(so[-1] - so[0])


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Dielectric Oil-Movability Quicklook")
    print("=" * 60)

    eps_w, eps_hc, eps_m = 60.0, 2.2, 5.0
    phi = 0.40  # diatomite-style high porosity

    # The full three-component CRIM permittivity rises with water saturation
    assert (crim_permittivity(0.6, phi, eps_w, eps_hc, eps_m)
            > crim_permittivity(0.2, phi, eps_w, eps_hc, eps_m))

    # Simplified (two-component) CRIM: water-filled porosity round-trips, where
    # eps_matrix* is the effective permittivity of the non-water system.
    eps_m_star = crim_permittivity(0.0, phi, eps_w, eps_hc, eps_m)  # Sw=0 system
    sw_true = 0.45
    phi_w_true = phi * sw_true
    eps_app = eps_water_to_eps(phi_w_true, phi, eps_w, eps_m_star)
    phi_w = water_filled_porosity(eps_app, eps_w, eps_m_star)
    sw = water_saturation(phi_w, phi)
    print(f"  eps_app={eps_app:.2f}  phi_water={phi_w:.3f}  Sw={sw:.3f}")
    assert np.isclose(sw, sw_true, atol=1e-6)
    assert np.isclose(oil_saturation(phi_w, phi), 1.0 - sw_true, atol=1e-6)

    # Flushed profile: shallow zone has less oil (water flushed in), deep has more
    # -> increasing So with DOI signals movable oil
    so_shallow_to_deep = [0.12, 0.30, 0.50, 0.55]
    phi_w_4 = [phi * (1 - s) for s in so_shallow_to_deep]
    eps_4 = [eps_water_to_eps(pw, phi, eps_w, eps_m_star) for pw in phi_w_4]
    so_prof = radial_saturation_profile(eps_4, phi, eps_w, eps_m_star)
    dso = moved_oil(so_prof)
    print(f"  So profile = {np.round(so_prof, 3)}  dSo = {dso:.3f}")
    assert np.allclose(so_prof, so_shallow_to_deep, atol=1e-6)
    assert dso > 0.2  # strong movability (>20 saturation units)

    # A uniform profile shows no movability
    eps_uniform = [eps_water_to_eps(phi * (1 - 0.4), phi, eps_w, eps_m_star)] * 4
    assert np.isclose(moved_oil(radial_saturation_profile(
        eps_uniform, phi, eps_w, eps_m_star)), 0.0, atol=1e-6)
    print("  PASS")
    return {"Sw": float(sw), "dSo": float(dso)}


def eps_water_to_eps(phi_water, phi_total, eps_water, eps_matrix_star):
    """Helper (test only): apparent permittivity from a water-filled porosity,
    the inverse of ``water_filled_porosity``."""
    root = np.sqrt(eps_matrix_star) + phi_water * (np.sqrt(eps_water)
                                                   - np.sqrt(eps_matrix_star))
    return root ** 2


if __name__ == "__main__":
    test_all()
