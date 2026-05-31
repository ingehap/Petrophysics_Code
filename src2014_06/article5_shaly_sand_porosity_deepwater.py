"""
Article 5: Analysis of Shale for Shaly-Sand Porosity Computation and Sedimentary
           Interpretation in Deepwater Sediments
Chunming Xu (2014)
Reference: Petrophysics Vol. 55, No. 3 (June 2014), pp. 253-259
DOI: none assigned (this issue predates SPWLA DOI assignment)

A neutron-density crossplot "shale-line" method (Thomas-Stieber / Juhasz family)
resolves the effective pore-fluid volume of a three-component system (quartz, wet
clay, pore fluid).  The wet-shale and dry-shale lines on the crossplot, and a
laminated net-to-gross, give the effective porosity and a sedimentary
interpretation of the shale distribution.

Implements:

  - Quartz-fluid and quartz-wet-clay (shale) line slopes  (Eqs. 5, 6)
  - Effective pore-fluid volume from the crossplot  (Eq. 4)
  - Wet/dry-shale neutron conversion  (Eqs. 7, 9) and dry-clay slope (Eq. 10)
  - Three-component volumetric solve (quartz, wet clay, fluid)
  - Laminated effective porosity  phi_e = Vfl/N  (Eq. 11)

Note: this issue's PDF has a text layer; the line slopes (Eqs. 5, 6, 10) and the
wet-shale mixing (Eq. 7) survived, while Eq. 4 (effective fluid volume) is the
exact algebraic solution of the linear system (Eqs. 1-3) and Eqs. 9, 11 are
reconstructed by inversion.  Quartz grain density 2.65 g/cm^3, phi_fl ~ 1.
Porosities/volumes as fractions, density in g/cm^3.
"""

import numpy as np

RHO_QUARTZ = 2.65  # g/cm^3


# ---------------------------------------------- crossplot line slopes --------------

def fluid_line_slope(rho_fl, phi_fl=1.0):
    """Quartz-fluid line slope on the neutron-density crossplot (Eq. 6)

        kfl = (2.65 - rho_fl)/phi_fl.
    """
    return (RHO_QUARTZ - rho_fl) / phi_fl


def clay_line_slope(rho_cl, phi_cl):
    """Quartz-wet-clay (shale) line slope (Eq. 5)

        kcl = (2.65 - rho_cl)/phi_cl,

    the slope of the wet-shale line (shale = quartz + wet clay, no pore fluid).
    """
    return (RHO_QUARTZ - rho_cl) / phi_cl


# ---------------------------------------------- effective fluid volume --------------

def effective_fluid_volume(phi, rho, slope_clay, rho_fl, phi_fl=1.0):
    """Effective pore-fluid volume from the crossplot (Eq. 4)

        Vfl = (kcl*phi - (2.65 - rho))/(kcl*phi_fl - (2.65 - rho_fl)),

    the exact solution of the three-component linear system (neutron, density and
    unit-sum closure).  Using the dry-clay slope kdcl gives the total pore volume.
    """
    num = slope_clay * phi - (RHO_QUARTZ - rho)
    den = slope_clay * phi_fl - (RHO_QUARTZ - rho_fl)
    return num / den


def solve_three_component(phi, rho, phi_cl, rho_cl, phi_fl=1.0, rho_fl=1.0):
    """Solve the three-component system for (Vqu, Vcl, Vfl)

        phi = phi_fl*Vfl + phi_cl*Vcl,
        rho = rho_fl*Vfl + rho_qu*Vqu + rho_cl*Vcl,
        Vqu + Vcl + Vfl = 1.

    Returns the volume fractions (quartz, wet clay, fluid).
    """
    a = np.array([[phi_fl, phi_cl, 0.0],
                  [rho_fl, rho_cl, RHO_QUARTZ],
                  [1.0, 1.0, 1.0]])
    b = np.array([phi, rho, 1.0])
    vfl, vcl, vqu = np.linalg.solve(a, b)
    return vqu, vcl, vfl


# ---------------------------------------------- dry-shale conversion --------------

def wet_shale_neutron(phi_dshale, phi_shl):
    """Wet-shale neutron porosity from the dry-shale value (Eq. 7)

        phi_shale = phi_dshale + (1 - phi_dshale)*phi_shl,

    with phi_shl the wet-shale total (core) porosity.
    """
    return phi_dshale + (1.0 - phi_dshale) * phi_shl


def dry_shale_neutron(phi_shale, phi_shl):
    """Dry-shale neutron porosity by inverting Eq. 7 (Eq. 9)

        phi_dshale = (phi_shale - phi_shl)/(1 - phi_shl).
    """
    return (phi_shale - phi_shl) / (1.0 - phi_shl)


def dry_clay_line_slope(rho_dshale, phi_dshale):
    """Quartz-dry-clay line slope (Eq. 10)

        kdcl = (2.65 - rho_dshale)/phi_dshale.
    """
    return (RHO_QUARTZ - rho_dshale) / phi_dshale


# ---------------------------------------------- net-to-gross --------------

def laminated_effective_porosity(vfl, net_to_gross):
    """Laminated effective porosity (Eq. 11)

        phi_e = Vfl/N,

    where shale laminae carry zero effective porosity and the sand carries a
    constant effective porosity; N is the net-to-gross.
    """
    return vfl / net_to_gross


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Shaly-Sand Porosity (Shale-Line Method)")
    print("=" * 60)

    # Example I (Gulf of Mexico): wet-shale slope ~0.229 for a representative
    # wet clay (phi_cl, rho_cl) reproduces the reported value
    rho_cl, phi_cl = 2.45, 0.873
    kcl = clay_line_slope(rho_cl, phi_cl)
    print(f"  wet-shale slope kcl = {kcl:.3f}")
    assert np.isclose(kcl, 0.229, atol=0.005)

    # Oil-bearing fluid line slope
    kfl = fluid_line_slope(rho_fl=0.93)
    print(f"  quartz-oil slope = {kfl:.3f}")
    assert np.isclose(kfl, 1.72, atol=0.01)

    # Three-component solve and the closed-form effective fluid volume agree
    # (a shaly-sand point above the shale line, so the fluid volume is positive)
    phi, rho = 0.30, 2.35
    vqu, vcl, vfl = solve_three_component(phi, rho, phi_cl, rho_cl,
                                          phi_fl=1.0, rho_fl=0.93)
    vfl_eq4 = effective_fluid_volume(phi, rho, kcl, rho_fl=0.93)
    print(f"  Vqu={vqu:.3f}  Vcl={vcl:.3f}  Vfl={vfl:.3f}")
    assert np.isclose(vfl, vfl_eq4) and np.isclose(vqu + vcl + vfl, 1.0)
    assert vfl > 0 and vcl > 0 and vqu > 0

    # Wet/dry-shale neutron conversion round-trips
    phi_dshale = dry_shale_neutron(0.45, phi_shl=0.10)
    assert np.isclose(wet_shale_neutron(phi_dshale, 0.10), 0.45)
    kdcl = dry_clay_line_slope(rho_dshale=2.78, phi_dshale=phi_dshale)
    print(f"  dry-clay slope kdcl = {kdcl:.3f}")

    # Laminated effective porosity exceeds the bulk fluid volume when N < 1
    phi_e = laminated_effective_porosity(vfl, net_to_gross=0.7)
    print(f"  phi_e (N=0.7) = {phi_e:.3f}")
    assert phi_e > vfl
    assert np.isclose(laminated_effective_porosity(vfl, 1.0), vfl)
    print("  PASS")
    return {"kcl": float(kcl), "Vfl": float(vfl), "phi_e": float(phi_e)}


if __name__ == "__main__":
    test_all()
