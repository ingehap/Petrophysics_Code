"""
Article 2: Combining Hydraulic and Electrical Conductivity for Pore-Space
           Characterization in Carbonate Rocks
Mueller-Huber, Schoen, Boerner (2016)
Reference: Petrophysics Vol. 57, No. 3 (June 2016), pp. 233-250
DOI: none assigned (this issue predates SPWLA DOI assignment)

A modified capillary-channel model combines hydraulic permeability and electrical
formation factor for water-saturated, water-wet carbonates.  Starting from
Hagen-Poiseuille flow and Archie's formation factor, both expressed through
porosity, tortuosity and pore radius, the combination cancels tortuosity: the
permeability can be predicted from the formation factor and a characteristic
pore radius without knowing tortuosity, and pore-space characteristics (rt, the
pore-body/pore-throat ratio) can be derived.

Implements:

  - Hagen-Poiseuille volume flow through a capillary (Eq. 1)
  - Capillary-channel permeability  k = phi*r^2/(8*tau^2)  (Eq. 4)
  - Archie formation factor  F = tau^2/phi  (Eq. 6)
  - Tortuosity-free combination  k = r^2/(8*F)  (Eq. 7)
  - Variable pore radius r(x) and pore-shape factor a (Eqs. 8-9)

Note: this issue's PDF has a text layer; the capillary-channel relations
(Eqs. 1-9) are transcribed from the body, while the typeset glyphs were dropped
and reconstructed in standard form.  The variable-radius integrals are
represented by the radius profile and shape factor.  SI units unless noted;
permeability returned in m^2 (multiply by 1.01325e15 for mD).
"""

import numpy as np


# ---------------------------------------------- hydraulic --------------

def hagen_poiseuille_flow(radius, dp, viscosity, length, tau=1.0):
    """Hagen-Poiseuille volume flow through a capillary (Eq. 1)

        q = pi*r^4*dp / (8*eta*tau*L),

    for laminar flow of a viscous fluid through a tube of radius r and true
    length tau*L across a rock model of length L.
    """
    return np.pi * radius ** 4 * dp / (8.0 * viscosity * tau * length)


def capillary_permeability(phi, radius, tau):
    """Capillary-channel permeability (Eq. 4)

        k = phi*r^2 / (8*tau^2),

    linear in porosity and quadratic in pore radius and inverse tortuosity.
    """
    return phi * radius ** 2 / (8.0 * tau ** 2)


# ---------------------------------------------- electrical --------------

def formation_factor(phi, tau):
    """Archie formation factor for the capillary model (Eq. 6)

        F = tau^2/phi,

    linear in inverse porosity and quadratic in tortuosity.
    """
    return tau ** 2 / phi


def permeability_from_formation_factor(radius, f):
    """Tortuosity-free permeability from the formation factor (Eq. 7)

        k = r^2/(8*F),

    obtained by combining Eqs. 4 and 6 so tortuosity cancels (electrical and
    hydraulic current lines assumed to follow the same path).
    """
    return radius ** 2 / (8.0 * f)


# ---------------------------------------------- variable radius --------------

def pore_shape_factor(rb, rt, length):
    """Pore-shape factor of the variable-radius model (Eq. 9)

        a = ln(rb/rt)/length,

    setting the exponential radius growth from pore-throat rt to pore-body rb.
    """
    return np.log(rb / rt) / length


def pore_radius_profile(x, rt, a):
    """Variable pore radius along the channel axis (Eq. 8)

        r(x) = rt*exp(a*x),

    growing from the pore-throat radius rt; with a from `pore_shape_factor`,
    r(length) = rb (the pore-body radius).
    """
    return rt * np.exp(a * np.asarray(x, float))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Hydraulic + Electrical Pore-Space Model")
    print("=" * 60)

    # Hagen-Poiseuille flow scales as r^4 and inversely with tortuosity
    q1 = hagen_poiseuille_flow(1e-5, 1e5, 1e-3, 0.01, tau=1.0)
    q2 = hagen_poiseuille_flow(2e-5, 1e5, 1e-3, 0.01, tau=1.0)
    assert np.isclose(q2 / q1, 16.0)          # (2r)^4 = 16 r^4
    assert hagen_poiseuille_flow(1e-5, 1e5, 1e-3, 0.01, tau=2.0) < q1

    # Formation factor falls with porosity and rises with tortuosity
    assert formation_factor(0.30, 1.5) < formation_factor(0.10, 1.5)
    assert formation_factor(0.20, 2.0) > formation_factor(0.20, 1.2)

    # Tortuosity cancels: k = r^2/(8F) equals phi*r^2/(8 tau^2) with F = tau^2/phi
    phi, r, tau = 0.18, 8e-6, 1.7
    k_direct = capillary_permeability(phi, r, tau)
    f = formation_factor(phi, tau)
    k_from_f = permeability_from_formation_factor(r, f)
    print(f"  k direct / via F       = {k_direct:.3e} / {k_from_f:.3e} m^2")
    assert np.isclose(k_direct, k_from_f)

    # Variable radius grows from throat rt to body rb across the channel
    rt, rb, L = 2e-6, 1e-5, 1e-4
    a = pore_shape_factor(rb, rt, L)
    assert np.isclose(pore_radius_profile(0.0, rt, a), rt)
    assert np.isclose(pore_radius_profile(L, rt, a), rb)
    print(f"  pore-shape factor a    = {a:.1f} 1/m")
    print("  PASS")
    return {"F": float(f), "k": float(k_direct), "a": float(a)}


if __name__ == "__main__":
    test_all()
