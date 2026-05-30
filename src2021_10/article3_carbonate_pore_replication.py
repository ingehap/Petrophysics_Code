"""
Article 3: Replication of Carbonate Reservoir Pores at the Original Size
           Using 3D Printing
Ishutov, Hodder, Chalaturnyk, Zambrano-Narvaez (2021)
DOI: 10.30632/PJV62N5-2021a3

A technical note: a workflow to fabricate a 1:1-scale (original-size) resin
replica of a carbonate pore network via two-photon photopolymerization
(Nanoscribe).  A CT-scanned sub-volume of Cantarell carbonate is segmented,
inscribed into a solid cylinder, scaffolded to cut print time, printed at
1-micron layers, and verified by CT and SEM.  The advance over prior work is
replication at the original pore size (no 5x upscaling).

Implements:

  - Pore-size scaling  d_model = S * d_original  (this work S = 1)
  - Equivalent (spherical) pore diameter  d_eq = (6V/pi)^(1/3)
  - Cylinder bulk volume
  - Scaffolding print-time speedup

Note: this technical note publishes no equations, no porosity numbers, and no
replication-error metric; the relations here are standard reconstructions
consistent with the prose, flagged as such.  Lengths in microns.
"""

import numpy as np


# ---------------------------------------------- scaling -----------------

def scaled_diameter(d_original, scale=1.0):
    """Printed pore diameter  d_model = scale * d_original.  This work: 1:1."""
    return scale * np.asarray(d_original, float)


# ---------------------------------------------- equivalent diameter -----

def equivalent_diameter(volume):
    """Equivalent spherical pore diameter  d_eq = (6V/pi)^(1/3)."""
    return (6.0 * np.asarray(volume, float) / np.pi) ** (1.0 / 3.0)


def sphere_volume(diameter):
    """Volume of a sphere of given diameter  V = (pi/6) d^3."""
    return np.pi / 6.0 * np.asarray(diameter, float) ** 3


# ---------------------------------------------- cylinder volume ---------

def cylinder_volume(diameter, height):
    """Cylinder bulk volume  V = pi*(d/2)^2*h."""
    return np.pi * (diameter / 2.0) ** 2 * height


# ---------------------------------------------- print time --------------

def scaffold_speedup(time_solid, time_scaffolded):
    """Print-time speedup factor from scaffolding."""
    return time_solid / time_scaffolded


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Original-Size Carbonate Pore Replication")
    print("=" * 60)

    # 1:1 replication leaves pore sizes unchanged; prior work used 5x
    d_orig = np.array([5.0, 50.0, 500.0])      # microns, pore-size range
    d_model = scaled_diameter(d_orig, scale=1.0)
    d_prior = scaled_diameter(d_orig, scale=5.0)
    print(f"  1:1 model diameters    = {d_model}")
    assert np.allclose(d_model, d_orig)
    assert np.allclose(d_prior, 5.0 * d_orig)

    # Equivalent-diameter round trip: a 50-micron sphere
    V = sphere_volume(50.0)
    d_eq = equivalent_diameter(V)
    print(f"  50-um sphere volume    = {V:.0f} um^3")
    print(f"  recovered d_eq         = {d_eq:.2f} um")
    assert abs(d_eq - 50.0) < 1e-6
    assert abs(V - 65450.0) < 5.0       # ~65,450 um^3

    # Printed VOI cylinder (1 mm dia x 2 mm tall = 1000 x 2000 um)
    Vcyl = cylinder_volume(1000.0, 2000.0)
    print(f"  VOI cylinder volume    = {Vcyl:.3e} um^3")
    assert abs(Vcyl - np.pi * 500.0 ** 2 * 2000.0) < 1.0

    # Scaffolding cuts a ~24 h solid print to ~5 h
    speedup = scaffold_speedup(24.0, 5.0)
    print(f"  scaffold speedup       = {speedup:.1f}x")
    assert speedup > 4.0
    print("  PASS")
    return {"d_eq": float(d_eq), "cyl_volume": Vcyl, "speedup": speedup}


if __name__ == "__main__":
    test_all()
