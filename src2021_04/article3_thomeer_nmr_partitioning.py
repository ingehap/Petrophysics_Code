"""
Article 3: Free or Bound? Thomeer and NMR Porosity Partitioning in Carbonate
           Reservoirs, Alta Discovery, Southwestern Barents Sea
Gianotten, Rameil, Foyn, Kollien, Marre, Looyestijn, Zhang, Hebing (2021)
DOI: 10.30632/PJV62N2-2021a3

Links mercury-injection capillary pressure (MICP), described by the Thomeer
hyperbola (Pd, Bv, G), to NMR total and movable (free) porosity.  A facies-
dependent NMR<->MICP calibration (C = T2*Pc) and a fixed pore-throat-radius
cutoff partition porosity into free vs bound fluid; Thomeer parameters also
give a Swanson-style permeability.

Implements:

  - Thomeer hyperbola  Shg = Bv*exp(-G/(logPc - logPd))             (Eq. 1)
  - Normalized porosity  PhiN = Phi/(1-Phi)                         (Eq. 2)
  - RQI / FZI  FZI = sqrt(k/(1014*Phi)) / PhiN                      (Eq. 3)
  - Swanson permeability  Ka = 3.8068*G^-1.3334*(Bv/Pd)^2           (Eq. 4)
  - Inversion of Eq. 4 for G                                        (Eq. 5)
  - Washburn pore-throat radius and NMR<->MICP calibration  C=T2*Pc

Equations transcribed from the rendered article.  Pc/Pd in psia, porosity and
Bv as fractions, permeability in mD, radius in microns, T2 in ms.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

WASHBURN_AIRHG = 107.6      # r[um] ~ 107.6 / Pc[psia]  (sigma=480 dyne/cm, theta=140)
T2_CUTOFF_MS = 14.0         # bedded-facies free/bound NMR T2 cutoff
R_CUTOFF_UM = 0.3           # free/bound pore-throat-radius cutoff (all facies)


# ---------------------------------------------- Eq. 1: Thomeer ----------

def thomeer_shg(Pc, Bv, G, Pd):
    """Thomeer mercury saturation  Shg = Bv*exp(-G/(logPc-logPd))  (Eq. 1)."""
    # Base-10 Thomeer (G defined on log10); the library takes log_base explicitly.
    return petrolib.capillary_pressure.thomeer_shg(
        Pc, bv_inf=Bv, g=G, pd=Pd, log_base=10.0)


# ---------------------------------------------- Eqs. 2-3: PhiN / FZI ----

def normalized_porosity(phi):
    """Normalized porosity  PhiN = Phi/(1-Phi)  (Eq. 2)."""
    return np.asarray(phi, float) / (1.0 - np.asarray(phi, float))


def rqi(k_md, phi):
    """Reservoir quality index  RQI = sqrt(k/(1014*Phi))  (microns).  k in mD."""
    return np.sqrt(k_md / (1014.0 * phi))


def fzi(k_md, phi):
    """Flow zone indicator  FZI = RQI / PhiN  (microns)  (Eq. 3)."""
    return rqi(k_md, phi) / normalized_porosity(phi)


# ---------------------------------------------- Eqs. 4-5: Swanson k -----

def swanson_permeability(G, Bv, Pd):
    """Swanson-type air permeability  Ka = 3.8068*G^-1.3334*(Bv/Pd)^2  (Eq. 4).  mD."""
    return 3.8068 * G ** (-1.3334) * (Bv / Pd) ** 2


def g_from_permeability(Ka, Bv, Pd):
    """Invert Eq. 4 for the Thomeer shape factor G  (Eq. 5)."""
    return ((Ka / 3.8068) * (Bv / Pd) ** (-2)) ** (-1.0 / 1.3334)


# ---------------------------------------------- Washburn / calibration --

def washburn_radius(Pc_psia, coeff=WASHBURN_AIRHG):
    """Pore-throat radius  r[um] = 107.6 / Pc[psia]  (air-mercury Washburn)."""
    return coeff / np.asarray(Pc_psia, float)


def nmr_micp_constant(T2_ms, Pc_psia):
    """Facies NMR<->MICP calibration constant  C = T2 * Pc."""
    return np.asarray(T2_ms, float) * np.asarray(Pc_psia, float)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Thomeer & NMR Porosity Partitioning")
    print("=" * 60)

    # Thomeer hyperbola: 0 below entry pressure, rises to Bv as Pc grows
    Bv, G, Pd = 0.20, 0.30, 10.0
    assert thomeer_shg(5.0, Bv, G, Pd) == 0.0       # Pc < Pd
    shg = thomeer_shg([20.0, 50.0, 100.0, 1e6], Bv, G, Pd)
    print(f"  Thomeer Shg            = {np.round(shg, 4)}")
    # Shg rises monotonically toward Bv (approached only asymptotically)
    assert np.all(np.diff(shg) > 0) and 0.9 * Bv < shg[-1] < Bv

    # Eq. 4 / Eq. 5 round-trip: recover G from the Swanson permeability
    Ka = swanson_permeability(G, Bv, Pd)
    G_rec = g_from_permeability(Ka, Bv, Pd)
    print(f"  Swanson Ka             = {Ka:.3f} mD")
    print(f"  recovered G            = {G_rec:.4f}  (true {G})")
    assert abs(G_rec - G) < 1e-6

    # Normalized porosity and FZI
    assert abs(normalized_porosity(0.20) - 0.25) < 1e-9
    print(f"  FZI (k=10mD, phi=0.2)  = {fzi(10.0, 0.20):.3f} um")
    assert fzi(10.0, 0.20) > 0

    # Washburn cutoff maps r=0.3 um to Pc ~ 358.7 psia; calibrate C so the
    # 0.3-um cutoff corresponds to the 14-ms bedded-facies T2 cutoff
    Pc_cut = WASHBURN_AIRHG / R_CUTOFF_UM
    C = nmr_micp_constant(T2_CUTOFF_MS, Pc_cut)
    print(f"  Pc at 0.3um / C        = {Pc_cut:.1f} psia / {C:.0f} ms*psia")
    assert abs(Pc_cut - 358.7) < 0.5
    # round-trip: at that C, the 0.3-um cutoff Pc gives back the 14-ms T2
    assert abs(C / Pc_cut - T2_CUTOFF_MS) < 1e-9
    print("  PASS")
    return {"Ka": Ka, "G_rec": G_rec, "Pc_cut": Pc_cut, "C": C}


if __name__ == "__main__":
    test_all()
