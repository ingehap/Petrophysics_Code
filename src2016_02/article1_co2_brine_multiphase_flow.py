"""
Article 1: The Impact of Reservoir Conditions and Rock Heterogeneity on
           CO2-Brine Multiphase Flow in Permeable Sandstone
Krevor, Reynolds, Al-Menhali, Niu (2016)
Reference: Petrophysics Vol. 57, No. 1 (February 2016), pp. 12-18
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best Papers of the 2015 SCA Symposium.  Coreflood tests across 5-20 MPa, 25-90 C
and 0-5 M NaCl show that the intrinsic CO2-brine multiphase-flow properties
(drainage capillary pressure, steady-state relative permeability, residual
trapping) are invariant with reservoir conditions: the system stays water-wet,
capillary pressure scales with interfacial tension, and the reported sensitivity
of relative permeability is really a response to capillary heterogeneity vs. the
viscous flow force.  Residual trapping follows the Land initial-residual model.

Implements:

  - Land trapping constant and initial-residual saturation (Land, 1968)
  - Leverett J-function (capillary-pressure / interfacial-tension scaling)
  - Corey two-phase relative permeability (water and CO2)
  - Capillary number (viscous-to-capillary force balance)

Note: this is an experimental SCA paper; the relations below are the standard
two-phase-flow models it relies on (Land trapping, Leverett scaling, Corey
relative permeability).  Pc and IFT in consistent units, permeability in m^2 (or
mD consistently), saturations dimensionless.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- residual trapping --------------

def land_constant(sgi_max, sgr_max):
    """Land (1968) trapping constant  C = 1/Sgr_max - 1/Sgi_max,

    from the maximum initial and residual non-wetting (CO2) saturations.
    """
    return petrolib.relperm_wettability.land_c(s_i_max=sgi_max, s_r_max=sgr_max)


def land_trapped_saturation(sgi, c):
    """Residual (trapped) saturation from the initial saturation (Land, 1968)

        Sgr = Sgi/(1 + C*Sgi),

    the initial-residual characteristic curve used to parameterize hysteresis.
    """
    return petrolib.relperm_wettability.land_trapped(sgi, C=c)


# ---------------------------------------------- capillary scaling --------------

def leverett_j(pc, sigma, k, phi):
    """Leverett J-function  J = (Pc/sigma)*sqrt(k/phi),

    the dimensionless capillary pressure; CO2-brine Pc curves at different
    conditions collapse when scaled by the interfacial tension sigma.
    """
    # This article's J omits the cos(theta) term; the library form carries it,
    # so pass theta_deg=0 (cos 0 = 1) to recover J = (Pc/sigma)*sqrt(k/phi).
    return petrolib.capillary_pressure.leverett_j(
        pc, sigma=sigma, theta_deg=0.0, k=k, phi=phi, absolute=True)


def pc_from_j(j, sigma, k, phi):
    """Capillary pressure from the J-function  Pc = J*sigma*sqrt(phi/k)."""
    return petrolib.capillary_pressure.pc_from_leverett_j(
        j, sigma=sigma, theta_deg=0.0, k=k, phi=phi, absolute=True)


# ---------------------------------------------- relative permeability --------------

def _normalized_sw(sw, swc, sor):
    return (np.asarray(sw, float) - swc) / (1.0 - swc - sor)


def corey_water_relperm(sw, swc, sor, krw_max=1.0, nw=4.0):
    """Corey wetting-phase (brine) relative permeability

        krw = krw_max * Swn^nw,   Swn = (Sw - Swc)/(1 - Swc - Sor).
    """
    # This article leaves Swn unclipped, so pass clip=None.
    return petrolib.relperm_wettability.corey_krw(
        sw, swr=swc, sor=sor, krw_max=krw_max, nw=nw, clip=None)


def corey_co2_relperm(sw, swc, sor, krn_max=1.0, nn=2.0):
    """Corey non-wetting-phase (CO2) relative permeability

        krn = krn_max * (1 - Swn)^nn.
    """
    return petrolib.relperm_wettability.corey_kro(
        sw, swr=swc, sor=sor, kro_max=krn_max, no=nn, clip=None)


def capillary_number(velocity, viscosity, sigma):
    """Capillary number  Ca = velocity*viscosity/sigma,

    the viscous-to-capillary force ratio that sets whether a core test measures
    the intrinsic or an effective (heterogeneity-influenced) relative
    permeability.
    """
    return petrolib.relperm_wettability.capillary_number(mu=viscosity, v=velocity, sigma=sigma)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: CO2-Brine Multiphase Flow")
    print("=" * 60)

    # Land trapping: Sgr = Sgr_max when Sgi = Sgi_max, and smaller for smaller Sgi
    c = land_constant(sgi_max=0.6, sgr_max=0.3)
    assert np.isclose(land_trapped_saturation(0.6, c), 0.3)
    assert land_trapped_saturation(0.3, c) < 0.3
    sgr = land_trapped_saturation(0.4, c)
    print(f"  Land C / Sgr(0.4)      = {c:.3f} / {sgr:.3f}")

    # Leverett J-function and its inverse round-trip
    pc = 5.0e3
    j = leverett_j(pc, sigma=0.03, k=2e-13, phi=0.2)
    assert np.isclose(pc_from_j(j, 0.03, 2e-13, 0.2), pc)
    # Scaling by IFT collapses curves: same J at the same saturation for two IFTs
    print(f"  Leverett J             = {j:.3f}")

    # Corey relative permeabilities: endpoints (Swn=1 at Sw=1-Sor) and monotonicity
    assert np.isclose(corey_water_relperm(1.0, 0.2, 0.0, 1.0, 4.0), 1.0)
    assert np.isclose(corey_co2_relperm(0.2, 0.2, 0.0, 1.0, 2.0), 1.0)
    assert corey_water_relperm(0.6, 0.2, 0.0) > corey_water_relperm(0.4, 0.2, 0.0)
    assert corey_co2_relperm(0.4, 0.2, 0.0) > corey_co2_relperm(0.6, 0.2, 0.0)

    # Capillary number rises with velocity (toward effective rel-perm regime)
    assert capillary_number(1e-5, 1e-3, 0.03) < capillary_number(1e-4, 1e-3, 0.03)
    print("  PASS")
    return {"Land_C": float(c), "Sgr": float(sgr), "J": float(j)}


if __name__ == "__main__":
    test_all()
