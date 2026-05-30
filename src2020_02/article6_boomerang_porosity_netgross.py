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


# ---------------------------------------------- porosity ----------------

def density_porosity(rho_b, rho_ma=2.65, rho_fl=1.0):
    """Density porosity  phiD = (rho_ma - rho_b)/(rho_ma - rho_fl)."""
    return (rho_ma - np.asarray(rho_b, float)) / (rho_ma - rho_fl)


def gas_corrected_porosity(phi_n, phi_d):
    """Gas-corrected total porosity  phi = sqrt((phiN^2 + phiD^2)/2).

    The root-mean-square of the neutron (gas-suppressed) and density
    (gas-inflated) porosities recovers a porosity between the two.
    """
    phi_n = np.asarray(phi_n, float); phi_d = np.asarray(phi_d, float)
    return np.sqrt((phi_n ** 2 + phi_d ** 2) / 2.0)


def effective_porosity(phi_total, vsh, phi_shale):
    """Shale-corrected effective porosity  phie = phi - Vsh*phi_sh (>= 0)."""
    return np.clip(np.asarray(phi_total, float)
                   - np.asarray(vsh, float) * phi_shale, 0.0, None)


# ---------------------------------------------- net/gross ---------------

def vshale_from_gr(gr, gr_clean, gr_shale):
    """Linear shale volume from gamma ray  Vsh = (GR - GR_clean)/(GR_shale - GR_clean)."""
    return np.clip((np.asarray(gr, float) - gr_clean) / (gr_shale - gr_clean),
                   0.0, 1.0)


def net_to_gross(depth, phie, vsh, phi_cut=0.08, vsh_cut=0.4):
    """Net/gross: fraction of gross thickness meeting porosity & shale cutoffs."""
    depth = np.asarray(depth, float)
    dz = np.abs(np.gradient(depth))
    net = (np.asarray(phie, float) >= phi_cut) & (np.asarray(vsh, float) <= vsh_cut)
    return float(np.sum(dz[net]) / np.sum(dz))


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
