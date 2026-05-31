"""
Article 4: Fault Block Migrations Inferred from Asphaltene Gradients
Chengli Dong, Melton P. Hows, Peter M.W. Cornelisse, Hani Elshahawi (2014)
Reference: Petrophysics Vol. 55, No. 2 (April 2014), pp. 113-123
DOI: none assigned (this issue predates SPWLA DOI assignment)

Special Issue on Deepwater.  The equilibrated asphaltene concentration vs depth
follows the Flory-Huggins-Zuo (FHZ) equation of state, controlled by gravity,
entropy and solubility terms.  Within a connected sand the profile is a single
FHZ curve; a sealing fault offsets the curve, and fault-block migration is
inferred by restoring the blocks until every sand collapses onto one curve.

Implements:

  - FHZ asphaltene EoS (gravity + entropy + solubility terms) (Eq. 1)
  - Gravity-only FHZ limit for low-GOR black oils
  - Asphaltene molar volume / particle size inferred from two depth points
  - Fault-block vertical offset between two equilibrated curves

Note: this issue's PDF dropped the display equations in extraction; the FHZ EoS
(Eq. 1) is reconstructed from the surrounding text and nomenclature with explicit
units (Zuo et al., 2010; Freed et al., 2010).  The Yen-Mullins nanoaggregate
(2.0 nm) is used in the case study.  Concentrations in wt%, depths in m,
solubility parameters in MPa^0.5.
"""

import numpy as np

R = 8.314      # universal gas constant, J/(mol*K)
G = 9.81       # m/s^2
NA = 6.022e23  # 1/mol


# ---------------------------------------------- FHZ EoS --------------

def fhz_ratio(h2, h1, molar_volume_a, delta_rho, temperature,
              va_over_v_1=0.0, va_over_v_2=0.0,
              solubility_sq_h1=0.0, solubility_sq_h2=0.0):
    """Flory-Huggins-Zuo asphaltene concentration ratio (Eq. 1)

        phi_a(h2)/phi_a(h1) = exp{ va*g*drho*(h2-h1)/(R*T)            # gravity
                                  + [(va/v)_h1 - (va/v)_h2]            # entropy
                                  - va*[(da-d)^2_h2 - (da-d)^2_h1]/(R*T) }, # solubility

    with asphaltene molar volume va (m^3/mol) and density contrast drho (kg/m^3).
    The solubility terms are the squared solubility-parameter contrasts
    (da - d)^2 at each depth.  The entropy and solubility terms vanish for
    low-GOR black oils, leaving the gravity term.
    """
    gravity = molar_volume_a * G * delta_rho * (h2 - h1) / (R * temperature)
    entropy = va_over_v_1 - va_over_v_2
    solubility = -molar_volume_a * (solubility_sq_h2 - solubility_sq_h1) / (R * temperature)
    return np.exp(gravity + entropy + solubility)


def fhz_gravity_only(h2, h1, molar_volume_a, delta_rho, temperature):
    """Gravity-only FHZ limit (low-GOR black oil)

        phi_a(h2)/phi_a(h1) = exp[va*g*drho*(h2-h1)/(R*T)].

    Note h2 below h1 (deeper) gives a ratio > 1: asphaltene increases with depth.
    """
    return np.exp(molar_volume_a * G * delta_rho * (h2 - h1) / (R * temperature))


def particle_diameter_from_molar_volume(molar_volume_a):
    """Spherical-particle diameter from the asphaltene molar volume

        d = (6*va/(pi*NA))^(1/3).
    """
    v_particle = molar_volume_a / NA
    return (6.0 * v_particle / np.pi) ** (1.0 / 3.0)


def infer_molar_volume(h2, h1, conc2, conc1, delta_rho, temperature):
    """Infer the asphaltene molar volume from two depth/concentration points
    (gravity-dominated FHZ, the single adjustable parameter)

        va = R*T*ln(conc2/conc1)/(g*drho*(h2-h1)).
    """
    return R * temperature * np.log(conc2 / conc1) / (G * delta_rho * (h2 - h1))


# ---------------------------------------------- fault-block offset --------------

def fault_block_offset(conc_block2, depth_block2, molar_volume_a, delta_rho,
                       temperature, conc_ref, depth_ref):
    """Vertical offset of an isolated fault block from the reference FHZ curve.

    Given a measured (concentration, depth) point in block 2 and the reference
    curve (conc_ref at depth_ref) with the same particle size, return the depth
    shift that places the block-2 point on the reference curve

        offset = R*T*ln(conc_block2/conc_ref)/(va*g*drho) - (depth_block2 - depth_ref).

    A nonzero offset signals a sealing fault / disequilibrium between blocks.
    """
    equilibrium_depth = depth_ref + (R * temperature * np.log(conc_block2 / conc_ref)
                                     / (molar_volume_a * G * delta_rho))
    return depth_block2 - equilibrium_depth


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Fault-Block Migration from Asphaltene Gradients")
    print("=" * 60)

    # 2.0-nm nanoaggregate molar volume
    d_nano = 2.0e-9
    v_particle = np.pi / 6.0 * d_nano ** 3
    va = v_particle * NA
    delta_rho, T = 60.0, 380.0  # kg/m^3, K

    # Gravity-only FHZ: asphaltene increases downward (deeper -> more)
    ratio = fhz_gravity_only(h2=100.0, h1=0.0, molar_volume_a=va,
                             delta_rho=delta_rho, temperature=T)
    print(f"  phi_a(100 m deeper)/phi_a(top) = {ratio:.3f}")
    assert ratio > 1.0

    # Particle size round-trips through the molar volume
    d_rec = particle_diameter_from_molar_volume(va)
    print(f"  recovered particle diameter = {d_rec*1e9:.2f} nm")
    assert np.isclose(d_rec, d_nano)

    # Infer molar volume from two equilibrated points -> recovers va
    conc1, conc2 = 5.0, 5.0 * ratio
    va_inf = infer_molar_volume(100.0, 0.0, conc2, conc1, delta_rho, T)
    assert np.isclose(va_inf, va, rtol=1e-6)
    assert np.isclose(particle_diameter_from_molar_volume(va_inf), d_nano)

    # Fault-block offset: a point on the reference curve has zero offset, a
    # depressed block has a nonzero offset
    off0 = fault_block_offset(conc2, 100.0, va, delta_rho, T, conc_ref=conc1,
                              depth_ref=0.0)
    assert np.isclose(off0, 0.0, atol=1e-6)
    off = fault_block_offset(conc2, 130.0, va, delta_rho, T, conc_ref=conc1,
                             depth_ref=0.0)
    print(f"  fault-block offset = {off:.1f} m")
    assert np.isclose(off, 30.0, atol=1e-6)

    # The full FHZ reduces to the gravity term when entropy/solubility vanish
    assert np.isclose(fhz_ratio(100.0, 0.0, va, delta_rho, T), ratio)
    print("  PASS")
    return {"FHZ_ratio": float(ratio), "va": float(va), "offset": float(off)}


if __name__ == "__main__":
    test_all()
