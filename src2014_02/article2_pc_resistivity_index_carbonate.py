"""
Article 2: Capillary Pressure and Resistivity Index Measurements in a Mixed-Wet
           Carbonate Reservoir
Moustafa R. Dernaika, Mohamed S. Efnik, Safouh Koronfol, Svein M. Skjaeveland,
Maisoon M. Al Mansoori, Hafez Hafez, Mohammed Z. Kalam (2014)
Reference: Petrophysics Vol. 55, No. 1 (February 2014), pp. 24-30
DOI: none assigned (this issue predates SPWLA DOI assignment)

A SCAL study measures capillary pressure and the resistivity index on the same
plugs across primary-drainage, spontaneous- and forced-imbibition cycles in a
mixed-wet carbonate, classified into rock-reservoir types (RRTs).  The
saturation exponent n is the key deliverable and increases through the
displacement cycles.

Implements:

  - Resistivity index  RI = Rt/Ro = Sw^(-n)
  - Saturation-exponent fit from a log-log RI vs Sw regression
  - Per-displacement-cycle saturation exponents (PD -> SI -> FI), the paper's
    headline result that n rises through the displacement cycles
  - Movable-oil saturation from the initial-water and residual-oil endpoints
  - Capillary-pressure unit reconciliation (the paper reports SI in bar and FI
    in psi)
  - Leverett J-function (Kalam et al., 2006) - with the paper's caveat that it
    does not reconcile Pc across the complex-carbonate RRTs
  - Capillary-tube (Washburn) pore-throat radius for the NMR-T2 vs MICP
    shielded-pore comparison
  - Archie formation resistivity factor  FRF = Ro/Rw = a/phi^m
  - Cementation-exponent fit from a log-log FRF vs phi regression
  - Permeability-based RRT grouping (Group 1 high-perm vs Group 2 tight) and
    each group's reported PD->FI saturation-exponent progression
  - NMR-vs-helium porosity QC agreement (within 1.1 p.u.)

Note: this experimental paper renders no display equations; the resistivity-
index power law and the Archie formation factor are written in standard form.
Reported saturation exponents: high-perm RRT 1-5 n = 1.99 (PD) -> 2.28 (FI);
tight RRT 6-7 n = 1.56 -> 1.82 (n differs little between PD and SI but rises in
FI).  Residual oil saturation converges to ~20% for the high-perm RRTs and ~27-
30% for the tight RRTs; SI reduces Pc from a maximum 7 bar to zero and FI applies
up to 80 psi.  Saturations as fractions, resistivities in Ohm*m.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Reservoir conditions of the study (Dernaika et al., 2014): measurements at
# reservoir temperature 121 degC; NMR T2 acquired with a 200 usec echo spacing.
RESERVOIR_TEMPERATURE_C = 121.0
NMR_ECHO_SPACING_USEC = 200.0

# Reported saturation-exponent progression (primary drainage -> forced
# imbibition) by petrophysical group (Dernaika et al., 2014): the higher-perm
# Group 1 (RRT 1-5) rises n = 1.99 -> 2.28, the tighter, more homogeneous
# Group 2 (RRT 6-7, < 1 md) rises n = 1.56 -> 1.82.  Both groups share the
# paper's headline: n increases through the displacement cycle (PD ~ SI < FI).
REPORTED_SATURATION_EXPONENTS = {
    "group1": (1.99, 2.28),
    "group2": (1.56, 1.82),
}

# NMR porosity agreed with helium porosity to within 1.1 porosity units (p.u.)
# across all samples - the study's NMR-vs-routine-core porosity QC criterion.
NMR_HELIUM_POROSITY_TOLERANCE_PU = 1.1


# ---------------------------------------------- resistivity index --------------

def resistivity_index(rt, ro):
    """Resistivity index  RI = Rt/Ro, the resistivity at saturation Sw relative
    to the fully brine-saturated resistivity Ro."""
    return petrolib.saturation_resistivity.resistivity_index(rt, ro)


def resistivity_index_from_sw(sw, n):
    """Resistivity index from water saturation  RI = Sw^(-n)."""
    return petrolib.saturation_resistivity.resistivity_index_from_sw(sw, n=n)


def water_saturation_from_ri(ri, n):
    """Water saturation from the resistivity index (inverse of RI = Sw^-n)

        Sw = RI^(-1/n),

    the application of the measured saturation exponent to logs.
    """
    return petrolib.saturation_resistivity.sw_from_resistivity_index(ri, n=n)


def archie_water_saturation(rt, rw, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation combining the formation factor and resistivity
    index

        Sw = (a*Rw/(phi^m*Rt))^(1/n).
    """
    return petrolib.saturation_resistivity.archie_sw(rt, rw, phi=phi, a=a, m=m, n=n)


def fit_saturation_exponent(sw, ri):
    """Fit the saturation exponent n from a log-log RI vs Sw regression

        log(RI) = -n*log(Sw)  ->  n = -slope.
    """
    return petrolib.saturation_resistivity.fit_saturation_exponent(sw, ri)


def saturation_exponents_by_cycle(cycles):
    """Fit the saturation exponent n for each displacement cycle.

    ``cycles`` maps a cycle label ("PD", "SI", "FI", ...) to a (Sw, RI) pair of
    arrays; each is fitted with ``fit_saturation_exponent``.  This reproduces the
    paper's central observation that n is similar for primary drainage (PD) and
    spontaneous imbibition (SI) but rises in forced imbibition (FI) - e.g. for
    the high-perm RRTs n = 1.99 (PD) -> 2.28 (FI).  Returns a dict of {label: n}.
    """
    return {label: fit_saturation_exponent(sw, ri)
            for label, (sw, ri) in cycles.items()}


# ---------------------------------------------- rock typing --------------

def rrt_group(permeability_md):
    """Group a carbonate reservoir rock type by permeability (Dernaika et al.).

    The paper splits its RRTs into two petrophysical groups:

      - Group 1 (RRT 1, 2, 4): rudstone / floatstone, higher permeability
        (RRT1 > 500 md, RRT2 100-500 md, RRT4 25-100 md), n ~ 2.0 and Sor ~ 20%;
      - Group 2 (RRT 6, 7): wackestone / mudstone, tight (< 0.1 md) with
        preserved intraparticle porosity and a higher Sor (~27-30%).

    Returns "group1" for permeability >= 0.1 md, "group2" below it.
    """
    return "group1" if permeability_md >= 0.1 else "group2"


def nmr_helium_porosity_agrees(phi_nmr_pu, phi_helium_pu,
                               tol_pu=NMR_HELIUM_POROSITY_TOLERANCE_PU):
    """NMR-vs-helium porosity QC check (Dernaika et al., 2014).

    The study reports its NMR porosity agreeing with the helium (routine-core)
    porosity to within 1.1 porosity units on every sample, validating the NMR
    pore-volume calibration before the T2/MICP pore-size comparison.  Porosities
    are given in porosity units (percentage points); returns True when the
    absolute difference is within ``tol_pu``.
    """
    return bool(abs(phi_nmr_pu - phi_helium_pu) <= tol_pu)


# ---------------------------------------------- saturation endpoints --------------

def movable_oil_saturation(swi, sor):
    """Movable-oil saturation from the initial-water and residual-oil endpoints

        So_movable = 1 - Swi - Sor,

    the oil between the initial-water saturation (after primary drainage) and the
    residual-oil saturation reached after forced imbibition.  The paper reports
    Sor converging to ~0.20 for the high-perm RRTs and ~0.27-0.30 for the tight
    RRTs.
    """
    return 1.0 - swi - sor


# ---------------------------------------------- capillary pressure --------------

def bar_to_psi(pc_bar):
    """Convert a capillary pressure from bar to psi  (1 bar = 14.5038 psi).

    The paper reports spontaneous-imbibition Pc in bar (maximum 7 bar) but
    forced-imbibition Pc in psi (up to 80 psi); this reconciles the two scales.
    """
    return np.asarray(pc_bar, float) * 14.5037738


def psi_to_bar(pc_psi):
    """Convert a capillary pressure from psi to bar  (1 psi = 0.0689476 bar)."""
    return np.asarray(pc_psi, float) / 14.5037738


def leverett_j_function(pc, sigma, contact_angle_deg, k, phi):
    """Leverett J-function - the dimensionless capillary pressure

        J(Sw) = Pc/(sigma*cos(theta)) * sqrt(k/phi),

    used to normalise Pc curves across rock types (Kalam et al., 2006).  The
    paper notes (after Masalmeh & Jing, 2004) that a single J-function does NOT
    reconcile the Pc curves across these complex carbonate RRTs, which is why
    Pc and the resistivity index are measured per RRT rather than scaled by J.
    Use consistent units: Pc and sigma in pressure / (force per length), k as an
    area (e.g. m^2); J is dimensionless.
    """
    return (np.asarray(pc, float) / (sigma * np.cos(np.radians(contact_angle_deg)))
            * np.sqrt(k / phi))


def pore_throat_radius(pc, sigma, contact_angle_deg):
    """Capillary-tube (Washburn) pore-throat radius from a capillary pressure

        r = 2*sigma*cos(theta)/Pc.

    This is the capillary-tube model underlying the paper's NMR-T2 vs MICP
    pore-size comparison: a match between the T2 distribution and the MICP pore-
    throat sizes (tight RRT 6-7) indicates simple tube-like pores, while a
    mismatch (high-perm RRT 1-5) flags large pores "shielded" behind small pore
    throats.  Consistent units (sigma in N/m, Pc in Pa) give r in metres.
    """
    return 2.0 * sigma * np.cos(np.radians(contact_angle_deg)) / np.asarray(pc, float)


# ---------------------------------------------- formation factor --------------

def formation_resistivity_factor(ro, rw):
    """Archie formation resistivity factor  FRF = Ro/Rw."""
    return ro / rw


def frf_from_porosity(phi, a=1.0, m=2.0):
    """Formation factor from porosity  FRF = a/phi^m."""
    return petrolib.saturation_resistivity.formation_factor(phi, a=a, m=m)


def fit_cementation_exponent(phi, frf):
    """Fit the cementation exponent m from a log-log FRF vs phi regression

        log(FRF) = log(a) - m*log(phi)  ->  m = -slope.

    Returns (m, a).
    """
    return petrolib.saturation_resistivity.fit_cementation_exponent(phi, frf)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Pc & Resistivity Index in Mixed-Wet Carbonate")
    print("=" * 60)

    # Resistivity index rises as water saturation falls
    sw = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    ri = resistivity_index_from_sw(sw, n=2.0)
    print(f"  RI(Sw) = {np.round(ri, 2)}")
    assert ri[0] == 1.0 and np.all(np.diff(ri) > 0)

    # Recover the saturation exponent (PD ~1.99, FI ~2.28 in the paper)
    n_pd = fit_saturation_exponent(sw, resistivity_index_from_sw(sw, 1.99))
    n_fi = fit_saturation_exponent(sw, resistivity_index_from_sw(sw, 2.28))
    print(f"  fitted n: PD={n_pd:.2f}  FI={n_fi:.2f}")
    assert np.isclose(n_pd, 1.99) and np.isclose(n_fi, 2.28) and n_fi > n_pd

    # Per-cycle exponents: n similar for PD/SI, higher for FI (paper's result)
    cycles = {
        "PD": (sw, resistivity_index_from_sw(sw, 1.99)),
        "SI": (sw, resistivity_index_from_sw(sw, 2.01)),
        "FI": (sw, resistivity_index_from_sw(sw, 2.28)),
    }
    n_by_cycle = saturation_exponents_by_cycle(cycles)
    print(f"  n by cycle: PD={n_by_cycle['PD']:.2f} SI={n_by_cycle['SI']:.2f} FI={n_by_cycle['FI']:.2f}")
    assert n_by_cycle["FI"] > n_by_cycle["SI"] > n_by_cycle["PD"]

    # Movable oil between the initial-water and residual-oil endpoints
    so_mov = movable_oil_saturation(swi=0.15, sor=0.20)
    print(f"  movable oil (Swi=0.15, Sor=0.20) = {so_mov:.2f}")
    assert np.isclose(so_mov, 0.65)
    # tighter RRTs (higher Sor) leave less movable oil
    assert movable_oil_saturation(0.15, 0.30) < so_mov

    # Capillary-pressure unit reconciliation (SI bar <-> FI psi)
    assert np.isclose(bar_to_psi(7.0), 101.5, atol=0.1)   # 7 bar SI maximum
    assert np.isclose(psi_to_bar(80.0), 5.516, atol=1e-3)  # 80 psi FI maximum
    assert np.isclose(psi_to_bar(bar_to_psi(7.0)), 7.0)

    # Leverett J-function normalises Pc; a higher-perm rock has a lower J at the
    # same Pc (the sqrt(k/phi) scaling), but the paper finds J still does not
    # collapse the carbonate RRT curves onto one another
    j_hi = leverett_j_function(pc=5e5, sigma=0.03, contact_angle_deg=0.0, k=1e-12, phi=0.30)
    j_lo = leverett_j_function(pc=5e5, sigma=0.03, contact_angle_deg=0.0, k=1e-15, phi=0.10)
    print(f"  Leverett J: high-perm={j_hi:.2f}  tight={j_lo:.2f}")
    assert j_hi > j_lo > 0

    # Washburn pore-throat radius falls as the capillary pressure rises
    r_lo_pc = pore_throat_radius(pc=1e5, sigma=0.03, contact_angle_deg=0.0)
    r_hi_pc = pore_throat_radius(pc=1e6, sigma=0.03, contact_angle_deg=0.0)
    print(f"  pore-throat radius: low Pc={r_lo_pc*1e6:.2f} um  high Pc={r_hi_pc*1e6:.2f} um")
    assert r_lo_pc > r_hi_pc > 0 and np.isclose(r_lo_pc, 2 * 0.03 / 1e5)

    # RI from measured resistivities
    assert np.isclose(resistivity_index(40.0, 10.0), 4.0)

    # Saturation inverts from the resistivity index (the application of n)
    assert np.allclose(water_saturation_from_ri(ri, n=2.0), sw)
    # a higher saturation exponent yields a higher Sw for the same RI
    assert water_saturation_from_ri(4.0, n=2.28) > water_saturation_from_ri(4.0, n=1.99)
    # full Archie Sw is consistent with FRF*RI = Rt/Rw
    sw_arch = archie_water_saturation(rt=40.0, rw=0.5, phi=0.25, m=2.0, n=2.0)
    print(f"  Archie Sw = {sw_arch:.3f}")
    assert 0 < sw_arch < 1

    # Cementation exponent from a synthetic FRF vs phi trend
    phi = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    frf = frf_from_porosity(phi, a=1.0, m=2.0)
    m_fit, a_fit = fit_cementation_exponent(phi, frf)
    print(f"  fitted m={m_fit:.3f}  a={a_fit:.3f}")
    assert np.isclose(m_fit, 2.0) and np.isclose(a_fit, 1.0)
    assert np.isclose(formation_resistivity_factor(20.0, 0.5), 40.0)

    # Permeability-based RRT grouping: high-perm RRT 1-4 vs tight RRT 6-7
    assert rrt_group(600.0) == "group1" and rrt_group(50.0) == "group1"
    assert rrt_group(0.05) == "group2"
    assert RESERVOIR_TEMPERATURE_C == 121.0

    # Reported PD->FI exponent progression: both groups rise, the tighter
    # Group 2 starting and ending lower than the high-perm Group 1
    g1_pd, g1_fi = REPORTED_SATURATION_EXPONENTS["group1"]
    g2_pd, g2_fi = REPORTED_SATURATION_EXPONENTS["group2"]
    print(f"  reported n: group1 {g1_pd}->{g1_fi}  group2 {g2_pd}->{g2_fi}")
    assert g1_fi > g1_pd and g2_fi > g2_pd and g1_pd > g2_pd and g1_fi > g2_fi

    # NMR-vs-helium porosity QC: within 1.1 p.u. passes, a larger gap fails
    assert nmr_helium_porosity_agrees(24.0, 24.8) and not nmr_helium_porosity_agrees(24.0, 26.0)
    print("  PASS")
    return {"n_PD": float(n_pd), "n_FI": float(n_fi), "m": float(m_fit),
            "movable_oil": float(so_mov)}


if __name__ == "__main__":
    test_all()
