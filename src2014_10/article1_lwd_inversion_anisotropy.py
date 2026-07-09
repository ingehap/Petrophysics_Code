"""
Article 1: Inversion-Based Interpretation of Logging-While-Drilling Resistivity
           and Nuclear Measurements: Field Examples in High-Angle and Horizontal
           Wells
Olabode Ijasan, Carlos Torres-Verdin, William E. Preeg, John Rasmus,
Edward J. Stockhausen (2014)
Reference: Petrophysics Vol. 55, No. 5 (October 2014), pp. 374-391
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best of the 2014 SPWLA Annual Logging Symposium.  A layer-by-layer nonlinear
inversion jointly interprets LWD propagation-resistivity and nuclear
(density/PEF/neutron) measurements in deviated wells.  Net-sand / non-net shale
layers carry an electrical anisotropy described by the Hagiwara (1996) net-to-
gross mixing law, and Archie's equation is applied only to the net sand.

Implements:

  - Hagiwara horizontal conductivity  sigma_h = sigma_sh_h*(1-NG) + sigma_sd*NG
    (Eq. A-2)
  - Hagiwara vertical resistivity  Rv = (1-NG)*Rsh_v + NG*Rsd  (Eq. A-4)
  - Anisotropy coefficient  lambda = sqrt(Rv/Rh)
  - Net-sand Archie water saturation and a net-pay flag (N/G cutoff)
  - Quadratic data-misfit cost and a linear net/shale conductivity inversion

Note: this field-examples paper delegates the full forward model to Ijasan et
al. (2013, 2014); the anisotropy relations Eq. A-2/A-4 and lambda are transcribed
/ reconstructed in standard Hagiwara (1996) form (the Eq. A-3/A-4 bodies were
dropped in extraction).  Conductivities in S/m, resistivities in Ohm*m.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- anisotropy (Hagiwara) --------------

def horizontal_conductivity(sigma_sd, sigma_sh_h, net_to_gross):
    """Horizontal conductivity of a laminated net-sand / shale layer (Eq. A-2)

        sigma_h = sigma_sh_h*(1 - N/G) + sigma_sd*(N/G),

    the parallel (arithmetic) average weighted by the net-to-gross fraction.
    """
    ng = net_to_gross
    return sigma_sh_h * (1.0 - ng) + sigma_sd * ng


def vertical_resistivity(r_sd, r_sh_v, net_to_gross):
    """Vertical resistivity of a laminated layer (Eq. A-4)

        Rv = (1 - N/G)*Rsh_v + (N/G)*Rsd,

    the series (arithmetic resistivity) average perpendicular to bedding.
    """
    ng = net_to_gross
    return (1.0 - ng) * r_sh_v + ng * r_sd


def anisotropy_coefficient(rv, rh):
    """Electrical anisotropy coefficient

        lambda = sqrt(Rv/Rh),

    the square root of the vertical-to-horizontal resistivity ratio (>= 1 for
    laminated shaly sands; field average ~1.22).
    """
    return petrolib.em_dielectric.anisotropy_coefficient(rh, rv)


# ---------------------------------------------- net sand Archie --------------

def archie_sw_net_sand(rt, rw, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation applied to the net sand only

        Sw = (a*Rw/(phi^m*Rt))^(1/n).
    """
    return petrolib.saturation_resistivity.archie_sw(rt, rw, phi=phi, a=a, m=m, n=n)


def net_pay_flag(net_to_gross, csh, ng_cutoff=0.05, csh_cutoff=0.95):
    """Net-pay flag: a layer is non-pay where the net-to-gross falls below the
    cutoff or the shale concentration exceeds its cutoff (hydrocarbon set to 0).
    """
    ng = np.asarray(net_to_gross, float)
    csh = np.asarray(csh, float)
    return (ng >= ng_cutoff) & (csh <= csh_cutoff)


# ---------------------------------------------- inversion --------------

def misfit_cost(simulated, measured, weights=None):
    """Quadratic data-misfit cost minimized by the nonlinear inversion

        C = sum_i w_i*(s_i - d_i)^2 / sum_i w_i*d_i^2   (relative, dimensionless).
    """
    s = np.asarray(simulated, float)
    d = np.asarray(measured, float)
    w = np.ones_like(d) if weights is None else np.asarray(weights, float)
    num = petrolib.inversion_numerics.costs.misfit(s, d, weights=w, kind="l2")
    return float(num / np.sum(w * d ** 2))


def invert_sand_conductivity(sigma_h, sigma_sh_h, net_to_gross):
    """Recover the net-sand conductivity from the measured horizontal
    conductivity by inverting Eq. A-2

        sigma_sd = (sigma_h - sigma_sh_h*(1 - N/G))/(N/G).
    """
    ng = net_to_gross
    return (sigma_h - sigma_sh_h * (1.0 - ng)) / ng


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Inversion-Based LWD Resistivity & Nuclear")
    print("=" * 60)

    # Field Example II/III style values
    r_sd, r_sh_h, r_sh_v = 3.94, 0.63, 0.88
    ng = 0.6
    sigma_h = horizontal_conductivity(1.0 / r_sd, 1.0 / r_sh_h, ng)
    rv = vertical_resistivity(r_sd, r_sh_v, ng)
    lam = anisotropy_coefficient(rv, 1.0 / sigma_h)
    print(f"  sigma_h={sigma_h:.4f} S/m  Rv={rv:.3f}  lambda={lam:.3f}")
    # vertical resistivity exceeds horizontal -> anisotropy > 1
    assert lam > 1.0
    # parallel mix lies between the sand and shale conductivities
    assert 1.0 / r_sh_h > sigma_h > 1.0 / r_sd

    # Net-sand Archie and net-pay flagging
    sw = archie_sw_net_sand(rt=9.0, rw=0.05, phi=0.20)
    print(f"  net-sand Sw = {sw:.3f}")
    assert 0 < sw < 1
    flags = net_pay_flag([0.6, 0.02, 0.5], [0.4, 0.4, 0.97])
    assert list(flags) == [True, False, False]

    # Inversion recovers the sand conductivity that generated sigma_h
    sd_rec = invert_sand_conductivity(sigma_h, 1.0 / r_sh_h, ng)
    print(f"  recovered sigma_sd = {sd_rec:.4f} S/m  (true {1.0/r_sd:.4f})")
    assert np.isclose(sd_rec, 1.0 / r_sd)

    # Misfit is zero for a perfect match and positive otherwise
    assert misfit_cost([1, 2, 3], [1, 2, 3]) == 0.0
    assert misfit_cost([1.1, 2, 3], [1, 2, 3]) > 0
    print("  PASS")
    return {"lambda": float(lam), "Sw": float(sw), "sigma_sd": float(sd_rec)}


if __name__ == "__main__":
    test_all()
