"""
Article 1: Application of Laboratory and Field NMR to Characterize the Tuscaloosa
           Marine Shale
Besov, Tinni, Sondergeld, Rai, Paul, Ebnother, Smagala (2017)
Reference: Petrophysics Vol. 58, No. 3 (June 2017), pp. 221-231
DOI: none assigned (this issue predates SPWLA DOI assignment)

Laboratory and field NMR characterize the Tuscaloosa Marine Shale.  The matrix
porosity is the T2 distribution integrated below a 10-ms cutoff (signal above it
is microfracture/induced-fracture porosity); the irreducible-water fraction, a
T1/T2 fluid-typing ratio, a Washburn pore-throat radius, and a volumetric
recoverable-oil estimate follow.

Implements:

  - Matrix vs fracture porosity from a T2 cutoff (10 ms)
  - Irreducible-water fraction of the total porosity
  - Washburn pore-throat radius from injection pressure
  - Volumetric recoverable oil  STOOIP = 7758*A*h*phi*So/Bo

Note: this issue's PDF has a text layer but the typeset display equations were
dropped, so the relations are faithful standard-form reconstructions; the
recoverable-oil inputs (80 acres, 137 ft, phi=0.07, So=0.114, Bo=1.3 -> ~523,000
bbl) are transcribed from the paper.  Porosities fractional, lengths in m.
"""

import numpy as np

BBL_PER_ACRE_FT = 7758.0


# ---------------------------------------------- NMR partition --------------

def matrix_porosity(t2_s, amplitude, total_porosity, cutoff_s=0.010):
    """Matrix porosity = total porosity * fraction of T2 amplitude below the cutoff.

    Signal at T2 > 10 ms is treated as microfracture porosity, not matrix.
    """
    t2 = np.asarray(t2_s, float)
    a = np.asarray(amplitude, float)
    return total_porosity * a[t2 <= cutoff_s].sum() / a.sum()


def fracture_porosity(t2_s, amplitude, total_porosity, cutoff_s=0.010):
    """Microfracture porosity = total porosity * fraction of amplitude above the cutoff."""
    t2 = np.asarray(t2_s, float)
    a = np.asarray(amplitude, float)
    return total_porosity * a[t2 > cutoff_s].sum() / a.sum()


def irreducible_fraction(as_received_pu, total_pu):
    """Irreducible-water fraction of the total porosity = as-received / total."""
    return as_received_pu / total_pu


# ---------------------------------------------- pore size / volumetrics --------------

def washburn_radius(pc, sigma=0.025, theta_deg=0.0):
    """Pore-throat radius from injection pressure  r = 2*sigma*|cos(theta)|/Pc (Washburn)."""
    return 2.0 * sigma * abs(np.cos(np.radians(theta_deg))) / np.asarray(pc, float)


def recoverable_oil(area_acres, height_ft, phi, so, bo):
    """Volumetric oil  STOOIP = 7758*A*h*phi*So/Bo  (bbl)."""
    return BBL_PER_ACRE_FT * area_acres * height_ft * phi * so / bo


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Tuscaloosa Marine Shale NMR")
    print("=" * 60)

    # T2 cutoff splits matrix from microfracture porosity (sums to total)
    t2 = np.array([0.001, 0.005, 0.02, 0.1])      # s
    amp = np.array([0.4, 0.4, 0.1, 0.1])
    mat = matrix_porosity(t2, amp, 0.07)
    frac = fracture_porosity(t2, amp, 0.07)
    print(f"  matrix / fracture phi  = {mat:.4f} / {frac:.4f}")
    assert np.isclose(mat + frac, 0.07) and mat > frac

    # Irreducible fraction ~ 40.8% (2.9 of 7.1 p.u.)
    assert np.isclose(irreducible_fraction(2.9, 7.1), 0.408, atol=0.01)

    # Higher injection pressure accesses smaller throats
    assert washburn_radius(5000.0) < washburn_radius(3500.0)

    # Recoverable-oil volumetric reproduces the paper's ~523,000 bbl
    stooip = recoverable_oil(80.0, 137.0, 0.07, 0.114, 1.3)
    print(f"  recoverable oil        = {stooip:,.0f} bbl")
    assert abs(stooip - 523000.0) < 5000.0
    print("  PASS")
    return {"matrix_phi": float(mat), "STOOIP": float(stooip)}


if __name__ == "__main__":
    test_all()
