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
  - Forward molar volume from a Yen-Mullins particle diameter
  - Nearest Yen-Mullins form check (molecule / nanoaggregate / cluster)
  - Fault-block vertical offset between two equilibrated curves
  - FHZ connectivity test (do two stations lie on one equilibrium curve?)
  - Tar-mat onset flag from extrapolated asphaltene content (35-40 wt% cutoff)

Note: this issue's PDF dropped the display equations in extraction; the FHZ EoS
(Eq. 1) is reconstructed from the surrounding text and nomenclature with explicit
units (Zuo et al., 2010; Freed et al., 2010).  The Yen-Mullins nanoaggregate
(2.0 nm) is used in the case study.  Concentrations in wt%, depths in m,
solubility parameters in Pa^0.5 (SI), so the solubility term va*(da-d)^2/(R*T)
is dimensionless.
"""

import numpy as np

R = 8.314      # universal gas constant, J/(mol*K)
G = 9.81       # m/s^2
NA = 6.022e23  # 1/mol

# Asphaltene weight fraction at which viscosity becomes extreme and a tar mat
# is likely to form (the article's stated 35-40 wt% range).
TAR_ONSET_WT_LOW = 0.35
TAR_ONSET_WT_HIGH = 0.40

# Yen-Mullins asphaltene particle diameters (m): molecule, nanoaggregate (the
# most common form in black oils, used in the case study) and cluster.  The
# inferred molar volume must correspond to one of these forms - a strong
# internal-consistency check on the FHZ fit.
YEN_MULLINS_DIAMETERS = {"molecule": 1.5e-9, "nanoaggregate": 2.0e-9, "cluster": 5.0e-9}


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


def molar_volume_from_diameter(particle_diameter):
    """Asphaltene molar volume from a (Yen-Mullins) particle diameter

        va = (pi/6)*d^3*NA,

    the forward operation used to *fix* the FHZ curve from an assumed particle
    size when no adjustable parameter is fitted (Wells 3-4 use a fixed 2.0-nm
    nanoaggregate).
    """
    v_particle = np.pi / 6.0 * np.asarray(particle_diameter, float) ** 3
    return v_particle * NA


def nearest_yen_mullins(particle_diameter, rtol=0.25):
    """Nearest Yen-Mullins form (molecule / nanoaggregate / cluster) for a diameter

        returns (name, diameter, agrees),

    where `agrees` is True when the diameter is within `rtol` of that form's size.
    The article stresses that the FHZ-inferred size must land on one of the three
    Yen-Mullins forms - agreement is the internal check that validates the fit.
    """
    d = float(particle_diameter)
    name, ref = min(YEN_MULLINS_DIAMETERS.items(), key=lambda kv: abs(kv[1] - d))
    return name, ref, bool(np.isclose(d, ref, rtol=rtol))


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


# ---------------------------------------------- connectivity test --------------

def on_same_fhz_curve(conc1, depth1, conc2, depth2, molar_volume_a, delta_rho,
                      temperature, rtol=0.05):
    """Connectivity test: do two fluid stations lie on a single equilibrated
    (gravity-only) FHZ curve?

    Predicts the deeper concentration from the shallower one and compares to the
    measurement.  Agreement (within `rtol`) means the stations are in one
    connected, equilibrated column (the article's vertical/lateral test);
    disagreement signals a sealing fault or disequilibrium between fault blocks.
    """
    predicted = conc1 * fhz_gravity_only(depth2, depth1, molar_volume_a,
                                         delta_rho, temperature)
    return bool(np.isclose(predicted, conc2, rtol=rtol))


# ---------------------------------------------- tar-mat onset --------------

def tar_mat_likely(asphaltene_wt_fraction):
    """Flag a tar mat from the (possibly FHZ-extrapolated) asphaltene content

        tar likely when asphaltene >= 35-40 wt%.

    Returns "tar likely" at/above ~40 wt%, "tar onset" in the 35-40 wt% transition
    band, and "no tar" below.  In the case study the downdip FHZ extrapolation
    reached 40 wt%, yet no tar was found - evidence the blocks were faulted apart
    before equilibration could build the predicted gradient.
    """
    a = float(asphaltene_wt_fraction)
    if a >= TAR_ONSET_WT_HIGH:
        return "tar likely"
    if a >= TAR_ONSET_WT_LOW:
        return "tar onset"
    return "no tar"


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

    # Particle size round-trips through the molar volume (both directions)
    d_rec = particle_diameter_from_molar_volume(va)
    print(f"  recovered particle diameter = {d_rec*1e9:.2f} nm")
    assert np.isclose(d_rec, d_nano)
    assert np.isclose(molar_volume_from_diameter(d_nano), va)

    # The inferred size lands on the Yen-Mullins nanoaggregate form
    form, ref, agrees = nearest_yen_mullins(d_rec)
    print(f"  nearest Yen-Mullins form = {form} ({ref*1e9:.1f} nm), agrees={agrees}")
    assert form == "nanoaggregate" and agrees
    # a 5-nm particle instead snaps to the cluster form
    assert nearest_yen_mullins(5.0e-9)[0] == "cluster"

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

    # Connectivity test: two equilibrated points lie on one FHZ curve; a station
    # whose asphaltene decreases with depth (disequilibrium / isolated block)
    # does not
    assert on_same_fhz_curve(conc1, 0.0, conc2, 100.0, va, delta_rho, T)
    assert not on_same_fhz_curve(conc1, 0.0, conc1 * 0.8, 100.0, va, delta_rho, T)
    print("  connectivity test: equilibrated pair connected, reversed-gradient pair not")

    # Tar-mat flag across the 35-40 wt% cutoff band
    assert tar_mat_likely(0.20) == "no tar"
    assert tar_mat_likely(0.37) == "tar onset"
    assert tar_mat_likely(0.42) == "tar likely"
    print(f"  tar flag at 40 wt% = {tar_mat_likely(0.40)!r}")
    print("  PASS")
    return {"FHZ_ratio": float(ratio), "va": float(va), "offset": float(off),
            "tar_at_40wt": tar_mat_likely(0.40)}


if __name__ == "__main__":
    test_all()
