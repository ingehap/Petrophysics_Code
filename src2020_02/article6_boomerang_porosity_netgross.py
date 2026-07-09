"""
Article 6: Untangle Shale and Gas Effects to Estimate Porosity and Net/Gross
           Ratio Using a Boomerang Workflow - A Case Study in Shoreface
           Reservoirs in Brunei
Xu, Sharif (2020)
DOI: 10.30632/PJV61N1-2020a5

In shaly, gas-bearing shoreface sands the neutron and density logs respond in
opposite directions to shale and to gas, so a density-neutron crossplot traces a
"boomerang" loop.  Untangling the two effects gives effective porosity and the
net/gross ratio: shale is removed with a volume-of-shale correction, and gas is
handled by combining the (suppressed) neutron porosity with the (inflated)
density porosity.

Implements:

  - Density porosity  phiD = (rho_ma - rho_b)/(rho_ma - rho_fl)
  - Gas-corrected total porosity  phi = sqrt((phiN^2 + phiD^2)/2)
  - Shale-corrected effective porosity  phie = phi - Vsh*phi_sh
  - Net/gross from porosity and shale-volume cutoffs

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard shaly-sand / gas-corrected density-neutron
relations the paper's title describes.  Porosities as fractions.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- porosity ----------------

def density_porosity(rho_b, rho_ma=2.65, rho_fl=1.0):
    """Density porosity  phiD = (rho_ma - rho_b)/(rho_ma - rho_fl)."""
    return petrolib.porosity_lithology.density_porosity(rho_b, rho_ma, rho_fl)


def gas_corrected_porosity(phi_n, phi_d):
    """Gas-corrected total porosity  phi = sqrt((phiN^2 + phiD^2)/2).

    The root-mean-square of the neutron (gas-suppressed) and density
    (gas-inflated) porosities recovers a porosity between the two.
    """
    return petrolib.porosity_lithology.neutron_density_porosity(phi_n, phi_d, method="rms")


def effective_porosity(phi_total, vsh, phi_shale):
    """Shale-corrected effective porosity  phie = phi - Vsh*phi_sh (>= 0)."""
    return petrolib.porosity_lithology.effective_porosity(phi_total, vsh, phi_shale)


# ---------------------------------------------- net/gross ---------------

def vshale_from_gr(gr, gr_clean, gr_shale):
    """Linear shale volume from gamma ray  Vsh = (GR - GR_clean)/(GR_shale - GR_clean)."""
    return petrolib.porosity_lithology.gamma_ray_index(gr, gr_clean, gr_shale)


def net_to_gross(depth, phie, vsh, phi_cut=0.08, vsh_cut=0.4):
    """Net/gross: fraction of gross thickness meeting porosity & shale cutoffs."""
    net = petrolib.porosity_lithology.pay_flag(phie, vsh, phi_cut=phi_cut, vsh_cut=vsh_cut)
    return petrolib.porosity_lithology.net_to_gross(depth, net)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Boomerang Workflow - Porosity & Net/Gross")
    print("=" * 60)

    # Density porosity: lighter rock -> higher porosity
    assert density_porosity(2.65) == 0.0
    pd = density_porosity(2.31)
    print(f"  density porosity @2.31 = {pd:.3f}")
    assert abs(pd - 0.206) < 1e-3

    # Gas correction: a gas crossover (phiD high, phiN suppressed) yields a
    # porosity between the two readings
    phi_n, phi_d = 0.12, 0.28              # gas: neutron low, density high
    phi = gas_corrected_porosity(phi_n, phi_d)
    print(f"  gas-corrected porosity = {phi:.3f}  (between {phi_n} and {phi_d})")
    assert phi_n < phi < phi_d

    # Shale correction reduces effective porosity
    phie = effective_porosity(0.22, vsh=0.3, phi_shale=0.35)
    print(f"  effective porosity     = {phie:.3f}")
    assert abs(phie - (0.22 - 0.3 * 0.35)) < 1e-9 and phie < 0.22

    # Vshale from GR and the net/gross over a synthetic shoreface interval
    depth = np.linspace(2000.0, 2019.0, 20)
    gr = np.full(20, 90.0); gr[5:15] = 25.0        # a 10-m clean sand
    vsh = vshale_from_gr(gr, gr_clean=20.0, gr_shale=120.0)
    phie_log = np.full(20, 0.04); phie_log[5:15] = 0.18
    ng = net_to_gross(depth, phie_log, vsh)
    print(f"  net/gross              = {ng:.2f}")
    assert 0.4 < ng < 0.6                          # ~half the interval is net
    print("  PASS")
    return {"phiD": float(pd), "phi_gas": float(phi), "phie": float(phie),
            "net_gross": ng}


if __name__ == "__main__":
    test_all()
