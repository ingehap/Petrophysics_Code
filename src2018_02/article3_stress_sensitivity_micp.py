"""
Article 3: Stress Sensitivity of Mercury-Injection Measurements
Guise, Grattoni, Allshorn, Fisher, Schiffer (2018)
DOI: 10.30632/petro_059_1_a2

A porosimeter that injects mercury while holding the sample under confining
stress shows that mercury-injection capillary pressure (MICP) is strongly
stress-sensitive: threshold pressures rise several-fold and pore-throat
diameters shrink under net stress.  This module implements the standard MICP
machinery the paper uses - the Washburn pore diameter from injection pressure
and the Swanson permeability from the apex of the log-log MICP curve.

Implements:

  - Washburn pore diameter  d = -4*sigma*cos(theta)/Pc
  - Swanson permeability  k = A*(SHg/Pc)_apex^B
  - Apex of the mercury-saturation / capillary-pressure curve
  - Threshold (entry) pressure detection

Note: this issue's PDF has a text layer; the Washburn relation and Swanson
constants are given in the prose, while the Swanson display equation (Eq. 1)
lost its glyph in extraction and is a faithful standard-form reconstruction.
Default Swanson constants A=339, B=1.691 (Swanson 1981).  Pc in psi, d in m.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

PSI_TO_PA = 6894.76


# ---------------------------------------------- MICP --------------

def washburn_diameter(pc_psi, sigma=0.485, theta_deg=140.0):
    """Pore diameter from mercury injection pressure  d = -4*sigma*cos(theta)/Pc.

    Mercury is non-wetting (theta ~ 140), so cos(theta) < 0 and d > 0.  sigma is
    the mercury-air interfacial tension (0.485 N/m); Pc converted from psi.
    """
    # -4*cos folds the obtuse mercury angle to a positive diameter; the library
    # expresses this as the |cos| (absolute=True) convention.  psi->Pa is kept here.
    pc_pa = np.asarray(pc_psi, float) * PSI_TO_PA
    return petrolib.capillary_pressure.washburn_diameter(
        pc_pa, sigma=sigma, theta_deg=theta_deg, absolute=True)


def micp_apex(shg_pct, pc_psi):
    """Apex of the MICP curve: max of SHg/Pc (Pittman 1992).  Returns the ratio."""
    return petrolib.flow_transport.micp_apex(shg_pct, pc_psi)


def swanson_permeability(shg_pct, pc_psi, a=339.0, b=1.691):
    """Swanson permeability from the MICP apex  k = A*(SHg/Pc)_apex^B  (Eq. 1, mD)."""
    apex, _ = micp_apex(shg_pct, pc_psi)
    return petrolib.flow_transport.swanson_permeability(apex, c=a, d=b)


def threshold_pressure(shg_pct, pc_psi, shg_cut=5.0):
    """Threshold (entry) pressure: the Pc at which mercury saturation first exceeds a cut."""
    shg = np.asarray(shg_pct, float)
    pc = np.asarray(pc_psi, float)
    idx = np.argmax(shg > shg_cut)
    return float(pc[idx])


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Stress Sensitivity of MICP")
    print("=" * 60)

    # Higher injection pressure -> smaller accessed pore diameter
    d_hi = washburn_diameter(50000.0)
    d_lo = washburn_diameter(5000.0)
    print(f"  d at 50k / 5k psi      = {d_hi * 1e9:.1f} / {d_lo * 1e9:.1f} nm")
    assert d_lo > d_hi > 0

    # Synthetic MICP curve: Swanson permeability from the apex
    pc = np.array([100, 300, 1000, 3000, 10000, 30000.0])
    shg = np.array([2, 20, 55, 75, 88, 95.0])
    k = swanson_permeability(shg, pc)
    apex, i = micp_apex(shg, pc)
    print(f"  apex SHg/Pc at idx {i}   = {apex:.4f},  Swanson k = {k:.3f} mD")
    assert k > 0 and 0 <= i < len(pc)

    # Threshold pressure is where mercury first enters (SHg > 5%)
    pth = threshold_pressure(shg, pc, shg_cut=5.0)
    print(f"  threshold pressure     = {pth:.0f} psi")
    assert pth == 300.0
    print("  PASS")
    return {"swanson_k": float(k), "threshold_psi": pth}


if __name__ == "__main__":
    test_all()
