"""
Article 1: Binder Saturation as a Controlling Factor for Porosity Variation
           in 3D-Printed Sandstone
Hodder, Craplewe, Ishutov, Chalaturnyk (2021)
DOI: 10.30632/PJV62N5-2021a1

Binder-jetting 3D printing of sandstone analogs: the volume of binder
deposited per unit of powder ("binder saturation level") controls the
porosity of the printed rock.  As saturation rises, binder fills more of the
inter-grain void, so porosity falls roughly linearly from the loose-powder
packing porosity.

Implements:

  - Printed cylinder volume  V = pi * (d/2)^2 * h                 (Eq. 2)
  - Binder volume from burnout mass loss
  - Binder volume fraction  f_b = V_binder / V_total             (Eq. 4)
  - Binder saturation level  S = f_b / void_fraction             (Eq. 5)
  - Theoretical porosity trend  phi = phi0 * (1 - S)

Note: the journal's typeset equations are image-rendered and were not in the
machine-readable text; the forms here are faithful reconstructions verified
against the paper's worked example (10% saturation -> 4 vol% binder; the
porosity trend passes through 36/34/32% at 10/15/20% saturation).  Eq. 1
(printer-internal saturation <-> droplet spacing) is proprietary and only
parameterized here.  Lengths in cm, porosity/saturation as fractions.
"""

import numpy as np

POWDER_VOID = 0.40          # loose-powder packing void fraction (phi0)


# ---------------------------------------------- Eq. 2: cylinder volume --

def cylinder_volume(diameter, height):
    """Printed cylinder volume  V = pi * (d/2)^2 * h  (Eq. 2)."""
    return np.pi * (diameter / 2.0) ** 2 * height


# ---------------------------------------------- binder volume -----------

def binder_volume_from_burnout(mass_loss, rho_binder=1.05):
    """Binder volume from the mass lost on burnout  V = m_loss / rho_binder."""
    return mass_loss / rho_binder


def binder_volume_fraction(v_binder, v_total):
    """Binder volume fraction  f_b = V_binder / V_total  (Eq. 4)."""
    return v_binder / v_total


# ---------------------------------------------- Eq. 5: saturation -------

def binder_saturation(f_binder, void_fraction=POWDER_VOID):
    """Binder saturation level  S = f_b / void_fraction  (Eq. 5).

    The fraction of the available inter-grain void that the binder fills.
    """
    return f_binder / void_fraction


# ---------------------------------------------- porosity trend ----------

def porosity_from_saturation(S, phi0=POWDER_VOID):
    """Theoretical printed porosity  phi = phi0 * (1 - S).

    Reproduces the paper's 36/34/32% at S = 10/15/20%.
    """
    return phi0 * (1.0 - np.asarray(S, float))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Binder Saturation -> Porosity in 3D-Printed Sand")
    print("=" * 60)

    # Worked example: 10% saturation corresponds to ~4 vol% binder
    f_b = 0.04
    S = binder_saturation(f_b)
    print(f"  saturation @ 4 vol%    = {S:.2f}  (expect 0.10)")
    assert abs(S - 0.10) < 1e-9

    # Porosity trend through the paper's reported points
    for sat, expect in [(0.10, 0.36), (0.15, 0.34), (0.20, 0.32)]:
        phi = porosity_from_saturation(sat)
        print(f"  S={sat:.2f} -> porosity   = {phi:.3f}  (expect {expect})")
        assert abs(phi - expect) < 1e-9
    # monotonic decrease with saturation
    sats = np.linspace(0.0, 0.5, 6)
    phis = porosity_from_saturation(sats)
    assert np.all(np.diff(phis) < 0)

    # Printed cylinder volume (1-in diameter, 1-in tall: 2.54 cm)
    V = cylinder_volume(2.54, 2.54)
    print(f"  cylinder volume        = {V:.2f} cm^3")
    assert abs(V - np.pi * 1.27 ** 2 * 2.54) < 1e-9

    # Round trip: binder volume -> fraction -> saturation
    v_binder = binder_volume_from_burnout(mass_loss=0.50, rho_binder=1.05)
    f = binder_volume_fraction(v_binder, V)
    print(f"  binder fraction        = {f:.4f}")
    assert 0.0 < f < 1.0
    print("  PASS")
    return {"S_at_4pct": S, "phi_at_10pct": float(porosity_from_saturation(0.10)),
            "cyl_volume": V}


if __name__ == "__main__":
    test_all()
