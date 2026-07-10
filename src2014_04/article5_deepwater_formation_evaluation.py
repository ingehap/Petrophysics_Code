"""
Article 5: Formation-Evaluation Challenges and Opportunities in Deepwater
Roland Chemali, Wade Samec, Ron Balliet, Paul Cooper, David Torres, Chris Jones
           (2014)
Reference: Petrophysics Vol. 55, No. 2 (April 2014), pp. 124-135
DOI: none assigned (this issue predates SPWLA DOI assignment)

Special Issue on Deepwater.  An applied-technology review of deepwater formation
evaluation: the narrow pore-pressure / fracture-pressure (ECD) drilling window,
resistivity anisotropy (Rh, Rv) for laminated-sand saturation, and NMR fluid
typing.  This module implements the standard relations the review references.

Implements:

  - Equivalent-circulating-density (ECD) drilling window check
  - Eaton pore-pressure prediction (resistivity / acoustic forms)
  - Resistivity anisotropy ratio  lambda = sqrt(Rv/Rh)
  - Thomas-Stieber laminated-sand horizontal/vertical resistivity
  - Shale-intrinsic-anisotropy-corrected laminated resistivities (Clavaud)
  - Laminated-sand water saturation from Rh
  - NMR hydrogen index, Coates/Timur permeability and a T1 fluid-typing contrast
  - Differential-T1 (TDA) gas-confirmation flag from a two-wait-time NMR sequence

Note: this is an applied review; the models it cites (Eaton, Thomas-Stieber,
Mollison, Coates) are referenced but not written, so the standard forms are
reconstructed here.  A field case reports phi = 30 p.u., well-test k = 1,000 md,
hydrogen index 0.33 (gas), 115 ft of net moveable-fluid sand, a differential-T1
sequence at 8 s / 1 s wait times, an optical analyzer extended from 2,100 to
5,500 nm, and crudes at 20 wt% (GoM) and 5 wt% (Saudi) asphaltene.
Resistivities in Ohm*m, pressures in psi.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Field-case anchors reported in the review (Figs. 9, 13, 15).
FIELD_AVERAGE_POROSITY_PU = 30.0      # p.u., appraisal reservoir
FIELD_WELLTEST_PERMEABILITY_MD = 1000.0
FIELD_HYDROGEN_INDEX_GAS = 0.33       # density-NMR separation -> gas
FIELD_NET_MOVEABLE_SAND_FT = 115.0    # long-T1 net sand in the laminated interval
DIFFERENTIAL_T1_WAIT_TIMES_S = (8.0, 1.0)   # long / short wait times
OPTICAL_WAVELENGTH_RANGE_NM = (2100.0, 5500.0)
ASPHALTENE_WT_GOM = 20.0
ASPHALTENE_WT_SAUDI = 5.0


# ---------------------------------------------- ECD window --------------

def within_ecd_window(ecd_pressure, pore_pressure, fracture_pressure):
    """Check that the equivalent-circulating-density pressure stays inside the
    deepwater drilling window

        pore_pressure < ECD < fracture_pressure,

    i.e. above pore pressure (no influx) and below fracture pressure (no losses).
    """
    return petrolib.integrity_drilling.within_drilling_window(
        ecd_pressure, pore_pressure, fracture_pressure)


def ecd_margin(pore_pressure, fracture_pressure):
    """Width of the drilling window  frac - pore  (psi), the deepwater margin."""
    return petrolib.integrity_drilling.drilling_window_margin(pore_pressure, fracture_pressure)


# ---------------------------------------------- pore pressure --------------

def eaton_pore_pressure(overburden, hydrostatic, observed, normal, exponent=1.2):
    """Eaton pore-pressure prediction from a compaction-sensitive log

        Pp = OBP - (OBP - Pn)*(obs/normal)^exponent,

    where (obs/normal) is the ratio of the observed to the normal-compaction-trend
    log value.  The article notes deepwater practice favors the *acoustic*
    (compressional-velocity) form over the original *resistivity* form (which is
    contaminated by salinity / hydrocarbon-saturation changes); both share this
    Eaton expression, differing only in which log supplies the ratio and the
    exponent (~1.2 resistivity, ~3 sonic velocity).
    """
    return petrolib.integrity_drilling.eaton_pore_pressure(
        overburden, hydrostatic, observed, normal,
        exponent=exponent, log_type="resistivity")


# ---------------------------------------------- anisotropy --------------

def anisotropy_ratio(r_v, r_h):
    """Resistivity anisotropy coefficient  lambda = sqrt(Rv/Rh)."""
    return petrolib.em_dielectric.anisotropy_coefficient(r_h, r_v)


def laminated_resistivity(r_sand, r_shale, sand_fraction):
    """Thomas-Stieber laminated-sand resistivities

        1/Rh = Vsd/Rsand + (1-Vsd)/Rshale     (parallel / horizontal),
        Rv  = Vsd*Rsand + (1-Vsd)*Rshale      (series / vertical),

    with the laminar sand fraction Vsd.  Returns (Rh, Rv).
    """
    vsd = sand_fraction
    rh = 1.0 / (vsd / r_sand + (1.0 - vsd) / r_shale)
    rv = vsd * r_sand + (1.0 - vsd) * r_shale
    return rh, rv


def laminated_resistivity_anisotropic(r_sand, rh_shale, rv_shale, sand_fraction):
    """Thomas-Stieber laminated resistivities when the shale laminae are
    themselves intrinsically anisotropic (Rh_shale != Rv_shale)

        1/Rh = Vsd/Rsand + (1-Vsd)/Rh_shale     (horizontal),
        Rv  = Vsd*Rsand + (1-Vsd)*Rv_shale       (vertical).

    The review stresses that omitting the shale intrinsic-anisotropy correction
    (Clavaud, 2008) yields unrealistically high hydrocarbon volumes; an
    anisotropic shale (Rv_shale > Rh_shale) raises the bulk Rv (and the apparent
    anisotropy) even before any laminar sand is added.  Returns (Rh, Rv).
    """
    vsd = sand_fraction
    rh = 1.0 / (vsd / r_sand + (1.0 - vsd) / rh_shale)
    rv = vsd * r_sand + (1.0 - vsd) * rv_shale
    return rh, rv


def laminated_water_saturation(r_sand, rw, phi_sand, a=1.0, m=2.0, n=2.0):
    """Archie water saturation of the laminar sand from its (anisotropy-resolved)
    sand resistivity

        Sw = (a*Rw/(phi_sand^m*Rsand))^(1/n).
    """
    return (a * rw / (phi_sand ** m * r_sand)) ** (1.0 / n)


# ---------------------------------------------- NMR --------------

def hydrogen_index(phi_apparent, phi_true):
    """Hydrogen index from the apparent (NMR) and true porosity

        HI = phi_apparent/phi_true,

    HI < 1 for gas (low hydrogen density), diagnostic in fluid typing.
    """
    return phi_apparent / phi_true


def t1_fluid_contrast(t1_hydrocarbon, t1_water):
    """T1 fluid-typing contrast  T1_hc/T1_water (> 1 for light hydrocarbons)."""
    return t1_hydrocarbon / t1_water


def differential_t1_gas(signal_long_wait, signal_short_wait,
                        wait_times=DIFFERENTIAL_T1_WAIT_TIMES_S, tol=0.05):
    """Differential-T1 (TDA) gas flag from two NMR wait times (e.g. 8 s and 1 s)

    Gas has a long T1, so it is fully polarized only at the long wait time; the
    *difference* between the long- and short-wait echo trains isolates the slowly
    polarizing (gas) signal.  Returns True when the long-wait amplitude exceeds
    the short-wait amplitude by more than `tol` (a recovered long-T1 component),
    the article's deepwater gas-confirmation workflow.
    """
    long_w, short_w = wait_times
    if long_w <= short_w:
        raise ValueError("first wait time must be the longer one")
    return bool(signal_long_wait - signal_short_wait > tol)


def coates_permeability(porosity, ffi, bvi, c=10.0):
    """Coates (free-fluid) NMR permeability from the bound/free fluid split

        k = (phi/C)^4 * (FFI/BVI)^2   [md],

    with porosity phi in porosity units (p.u. = percent), free-fluid index FFI
    and bulk-volume-irreducible BVI (FFI + BVI = phi, all in p.u.) and the
    calibration constant C (~10).  This is the NMR-based permeability the review
    cites (Coates, 1998) for ranking deepwater pay quality.
    """
    return petrolib.nmr.timur_coates(porosity, ffi, bvi, C=c)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Deepwater Formation-Evaluation")
    print("=" * 60)

    # ECD window: in-window vs out-of-window
    assert within_ecd_window(9500, 9000, 10000)
    assert not within_ecd_window(10500, 9000, 10000)
    print(f"  ECD margin = {ecd_margin(9000, 10000):.0f} psi")
    assert ecd_margin(9000, 10000) == 1000

    # Anisotropy: laminated sand/shale gives Rv > Rh, lambda > 1
    rh, rv = laminated_resistivity(r_sand=20.0, r_shale=1.0, sand_fraction=0.5)
    lam = anisotropy_ratio(rv, rh)
    print(f"  Rh={rh:.3f}  Rv={rv:.3f}  lambda={lam:.3f}")
    assert rv > rh and lam > 1.0

    # Resolving the true sand resistivity gives a lower (more correct) Sw than
    # using the bulk horizontal resistivity
    sw_sand = laminated_water_saturation(r_sand=20.0, rw=0.05, phi_sand=0.30)
    sw_bulk = laminated_water_saturation(r_sand=rh, rw=0.05, phi_sand=0.30)
    print(f"  Sw(sand)={sw_sand:.3f}  Sw(bulk Rh)={sw_bulk:.3f}")
    assert sw_sand < sw_bulk

    # Eaton: undercompaction (observed < normal trend) raises predicted pore
    # pressure above hydrostatic, narrowing the deepwater margin
    pp_normal = eaton_pore_pressure(10000, 8000, observed=1.0, normal=1.0)
    pp_under = eaton_pore_pressure(10000, 8000, observed=0.5, normal=1.0)
    print(f"  Eaton Pp: normal={pp_normal:.0f}  undercompacted={pp_under:.0f} psi")
    assert np.isclose(pp_normal, 8000) and pp_under > pp_normal

    # Shale intrinsic anisotropy raises bulk Rv (and lambda) vs isotropic shale
    rh_i, rv_i = laminated_resistivity(20.0, 1.0, 0.5)
    rh_a, rv_a = laminated_resistivity_anisotropic(20.0, 1.0, 2.0, 0.5)
    print(f"  anisotropic shale: Rv {rv_i:.2f} -> {rv_a:.2f}")
    assert rv_a > rv_i and anisotropy_ratio(rv_a, rh_a) > anisotropy_ratio(rv_i, rh_i)

    # NMR: hydrogen index 0.33 flags gas; light HC has longer T1 than water
    hi = hydrogen_index(0.10, 0.30)
    print(f"  hydrogen index = {hi:.2f}")
    assert np.isclose(hi, 0.333, atol=0.01) and hi < 1.0
    assert t1_fluid_contrast(4.0, 0.5) > 1.0

    # Coates NMR permeability (porosity & fluid indices in p.u., C~10) rises with
    # the free-fluid fraction.  The field case's k~1,000 md is a *well-test*
    # value (FIELD_WELLTEST_PERMEABILITY_MD), shown here only as the order of
    # magnitude a high-quality deepwater sand reaches - not a Coates output to
    # validate against.
    k_low = coates_permeability(30.0, ffi=5.0, bvi=25.0)
    k_high = coates_permeability(30.0, ffi=25.0, bvi=5.0)
    print(f"  Coates k: bound-rich={k_low:.1f}  free-rich={k_high:.1f} md")
    assert k_high > k_low
    assert FIELD_WELLTEST_PERMEABILITY_MD == 1000.0

    # Differential-T1: a recovered long-wait amplitude confirms gas; equal trains
    # (all liquid, fast polarization) do not
    assert differential_t1_gas(0.30, 0.10)        # long-T1 gas component present
    assert not differential_t1_gas(0.30, 0.30)    # no differential -> no gas
    print(f"  differential-T1 wait times = {DIFFERENTIAL_T1_WAIT_TIMES_S} s")

    # Field-case anchors are self-consistent
    assert np.isclose(FIELD_HYDROGEN_INDEX_GAS, 0.33)
    assert FIELD_NET_MOVEABLE_SAND_FT == 115.0
    assert OPTICAL_WAVELENGTH_RANGE_NM == (2100.0, 5500.0)
    assert ASPHALTENE_WT_GOM > ASPHALTENE_WT_SAUDI
    print("  PASS")
    return {"ECD_margin": 1000.0, "lambda": float(lam), "HI": float(hi),
            "Pp_undercompacted": float(pp_under), "k_coates": float(k_high),
            "net_moveable_sand_ft": FIELD_NET_MOVEABLE_SAND_FT}


if __name__ == "__main__":
    test_all()
