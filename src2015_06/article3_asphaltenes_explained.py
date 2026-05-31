"""
Article 3: Asphaltenes Explained for the Nonchemist
Mullins, Pomerantz, Andrews, Zuo (2015)
Reference: Petrophysics Vol. 56, No. 3 (June 2015), pp. 266-275
DOI: none assigned (this issue predates SPWLA DOI assignment)

A tutorial on asphaltenes: the Yen-Mullins model gives the nanocolloidal size
hierarchy (molecule -> nanoaggregate -> cluster), and the Flory-Huggins-Zuo
(FHZ) equation of state predicts asphaltene gradients (measured as optical
density, OD) from three contributions - gravity, solubility and entropy.  The
gravity term is the Boltzmann/barometric distribution applied to the Archimedes
buoyancy of an asphaltene particle; the solubility term explains why high-GOR
(low solubility-parameter) oils carry little asphaltene.

Implements:

  - Yen-Mullins particle hierarchy (size and aggregation number)
  - FHZ gravity (Boltzmann) optical-density ratio between two depths (Eq. 3)
  - FHZ solubility contribution to the OD ratio
  - Combined FHZ OD ratio (gravity + solubility)

Note: this is a tutorial/review; the FHZ relation (Eq. 3) is transcribed from
the body, while the typeset glyphs were dropped and reconstructed in standard
form (Mullins et al., 2012; Zuo et al., 2013).  SI units: particle volume in
m^3, density in kg/m^3, height in m, temperature in K, solubility parameters in
(MPa)^0.5.
"""

import numpy as np

KB = 1.380649e-23             # Boltzmann constant, J/K
R_GAS = 8.314                 # J/(mol*K)
G_EARTH = 9.81                # m/s^2


# ---------------------------------------------- Yen-Mullins hierarchy --------------

def yen_mullins_model():
    """Yen-Mullins asphaltene size hierarchy.

    Returns a dict of the three nanocolloidal species with approximate diameter
    (nm) and aggregation number (number of building blocks):
      molecule       ~1.5 nm
      nanoaggregate  ~2 nm   (~6 molecules)
      cluster        ~5 nm   (~8 nanoaggregates).
    """
    return {
        "molecule": {"diameter_nm": 1.5, "aggregation": 1},
        "nanoaggregate": {"diameter_nm": 2.0, "aggregation": 6},
        "cluster": {"diameter_nm": 5.0, "aggregation": 8},
    }


def particle_volume(diameter_nm):
    """Spherical asphaltene-particle volume (m^3) from a diameter in nm."""
    r = 0.5 * diameter_nm * 1e-9
    return (4.0 / 3.0) * np.pi * r ** 3


# ---------------------------------------------- FHZ --------------

def fhz_gravity_od_ratio(particle_vol, delta_rho, h_lower, h_upper, temperature):
    """FHZ gravity (Boltzmann) optical-density ratio between two depths (Eq. 3)

        OD(h_lower)/OD(h_upper) = exp[ Va*delta_rho*g*(h_upper - h_lower)/(kB*T) ],

    the Archimedes buoyancy of an asphaltene particle in the barometric
    distribution; with h_lower below h_upper this ratio exceeds 1 (asphaltenes
    settle downward).
    """
    return np.exp(particle_vol * delta_rho * G_EARTH * (h_upper - h_lower) / (KB * temperature))


def fhz_solubility_term(particle_vol, delta_oil_lower, delta_oil_upper, delta_asph, temperature):
    """FHZ solubility contribution to ln(OD ratio)

        term = (Va/(kB*T)) * [ (delta_oil_upper - delta_a)^2 - (delta_oil_lower - delta_a)^2 ],

    with the solubility parameters in (MPa)^0.5 (so the squared differences are
    converted from MPa to Pa by 1e6).  Va/(kB*T) keeps this per-particle term
    consistent with the gravity term.  It captures how a changing oil solubility
    parameter (e.g. via GOR) shifts asphaltene content; high-GOR (low delta_oil)
    oils dissolve less asphaltene.
    """
    return (particle_vol / (KB * temperature)) * 1e6 * (
        (delta_oil_upper - delta_asph) ** 2 - (delta_oil_lower - delta_asph) ** 2)


def fhz_od_ratio(particle_vol, delta_rho, h_lower, h_upper, temperature,
                 delta_oil_lower=None, delta_oil_upper=None, delta_asph=None):
    """Combined FHZ optical-density ratio (gravity + optional solubility)

        OD(h_lower)/OD(h_upper) = exp[ gravity_exponent + solubility_term ].
    """
    grav = particle_vol * delta_rho * G_EARTH * (h_upper - h_lower) / (KB * temperature)
    sol = 0.0
    if None not in (delta_oil_lower, delta_oil_upper, delta_asph):
        sol = fhz_solubility_term(particle_vol, delta_oil_lower, delta_oil_upper,
                                  delta_asph, temperature)
    return np.exp(grav + sol)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Asphaltenes Explained")
    print("=" * 60)

    # Yen-Mullins hierarchy: clusters are larger and more aggregated than molecules
    ym = yen_mullins_model()
    assert ym["cluster"]["diameter_nm"] > ym["nanoaggregate"]["diameter_nm"] > ym["molecule"]["diameter_nm"]
    assert particle_volume(5.0) > particle_volume(2.0) > particle_volume(1.5)

    # FHZ gravity: asphaltene OD increases downward; clusters give a steeper gradient
    va_nano = particle_volume(2.0)
    va_cluster = particle_volume(5.0)
    ratio_nano = fhz_gravity_od_ratio(va_nano, 60.0, h_lower=-50.0, h_upper=0.0, temperature=350.0)
    ratio_cluster = fhz_gravity_od_ratio(va_cluster, 60.0, h_lower=-50.0, h_upper=0.0, temperature=350.0)
    print(f"  OD ratio nano/cluster  = {ratio_nano:.3f} / {ratio_cluster:.3f}")
    assert ratio_cluster > ratio_nano > 1.0

    # Solubility term: a lower oil solubility parameter up-column raises the gradient
    od_grav = fhz_od_ratio(va_nano, 60.0, -50.0, 0.0, 350.0)
    od_full = fhz_od_ratio(va_nano, 60.0, -50.0, 0.0, 350.0,
                           delta_oil_lower=20.0, delta_oil_upper=19.8, delta_asph=21.0)
    print(f"  OD ratio grav/full     = {od_grav:.3f} / {od_full:.3f}")
    assert od_full != od_grav and od_full > 1.0
    print("  PASS")
    return {"ratio_cluster": float(ratio_cluster), "OD_full": float(od_full)}


if __name__ == "__main__":
    test_all()
