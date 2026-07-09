"""
Article 2: Defining Net-Pay Cutoffs in Carbonates Using Advanced Petrophysical
           Methods
Skalinski, Mallan, Edwards, Sun, Toumelin, Kelly, Wushur, Sullivan (2019)
DOI: 10.30632/PJV60N1Y2019a1

Net pay in heterogeneous carbonates is defined with cutoffs that account for the
rock-fabric control on the porosity-permeability relationship.  A permeability
cutoff is mapped to a porosity cutoff through a rock-fabric (Lucia-style or
Winland R35) transform, and net pay is summed where porosity, shale-volume and
water-saturation cutoffs are simultaneously met.

Implements:

  - Winland R35 pore-throat radius  log R35 = 0.732 + 0.588 log k - 0.864 log phi
  - Lucia rock-fabric permeability transform  log k = A - B*log(phi)
  - Permeability cutoff -> porosity cutoff inversion
  - Net-pay flags and net-to-gross from integrated cutoffs

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the carbonate rock-fabric / net-pay-cutoff methods the paper applies.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- rock fabric -------------

def winland_r35(k_md, phi):
    """Winland R35 pore-throat radius  log R35 = 0.732 + 0.588 log k - 0.864 log phi.

    phi as a percentage; returns R35 in microns.
    """
    return petrolib.flow_transport.winland_r35(k_md, phi)


def lucia_permeability(phi, A=8.0, B=5.0):
    """Rock-fabric permeability transform  log10(k) = A + B*log10(phi)  (mD)."""
    # This log-log form is the library's poro-perm power law (A intercept, B slope).
    return petrolib.flow_transport.poroperm_powerlaw(phi, a=A, b=B)


def porosity_cutoff_from_perm(k_cut, A=8.0, B=5.0):
    """Invert the Lucia transform for the porosity cutoff at a permeability cutoff."""
    return 10.0 ** ((np.log10(k_cut) - A) / B)


# ---------------------------------------------- net pay -----------------

def pay_flag(phi, vsh, sw, phi_cut, vsh_cut=0.4, sw_cut=0.5):
    """Per-sample pay flag from porosity, shale-volume and Sw cutoffs."""
    return ((np.asarray(phi, float) >= phi_cut)
            & (np.asarray(vsh, float) <= vsh_cut)
            & (np.asarray(sw, float) <= sw_cut))


def net_to_gross(depth, pay):
    """Net/gross = net pay thickness / gross thickness."""
    dz = np.abs(np.gradient(np.asarray(depth, float)))
    return float(np.sum(dz[np.asarray(pay, bool)]) / np.sum(dz))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Net-Pay Cutoffs in Carbonates")
    print("=" * 60)

    # Winland R35 rises with permeability and porosity
    assert winland_r35(10.0, 0.18) > winland_r35(1.0, 0.18)
    print(f"  R35 (10 mD, 18%)       = {winland_r35(10.0, 0.18):.2f} um")

    # Lucia transform and its inverse are consistent
    k = lucia_permeability(0.15)
    phi_back = porosity_cutoff_from_perm(k)
    print(f"  Lucia k(15%) / phi back = {k:.2f} mD / {phi_back:.3f}")
    assert abs(phi_back - 0.15) < 1e-9

    # A 0.1 mD permeability cutoff maps to a porosity cutoff
    phi_cut = porosity_cutoff_from_perm(0.1)
    print(f"  porosity cutoff (0.1 mD) = {phi_cut:.3f}")
    assert 0.0 < phi_cut < 0.2

    # Net pay over a synthetic carbonate interval
    depth = np.linspace(3000.0, 3019.0, 20)
    phi = np.full(20, 0.04); phi[6:16] = 0.16
    vsh = np.full(20, 0.5); vsh[6:16] = 0.1
    sw = np.full(20, 0.8); sw[6:14] = 0.3
    pay = pay_flag(phi, vsh, sw, phi_cut)
    ng = net_to_gross(depth, pay)
    print(f"  net samples / N:G      = {int(pay.sum())} / {ng:.2f}")
    assert int(pay.sum()) == 8 and 0.0 < ng < 1.0
    print("  PASS")
    return {"phi_cut": float(phi_cut), "net_to_gross": ng}


if __name__ == "__main__":
    test_all()
