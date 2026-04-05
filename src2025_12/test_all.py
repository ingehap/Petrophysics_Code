"""
test_all  —  Comprehensive tests for every module in petrophysics_v66n6
========================================================================

Each article module is exercised with synthetic data.  Every public
function and class is tested for:
    * correct output type and shape,
    * physical sanity (e.g. kr ∈ [0,1], Pc > 0 during drainage),
    * known analytic limits,
    * round-trip / inverse consistency where applicable.

Run with::

    python -m petrophysics_v66n6.test_all          # summary only
    python -m petrophysics_v66n6.test_all -v        # verbose

Or via pytest::

    pytest petrophysics_v66n6/test_all.py -v
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Callable, List, Tuple

import numpy as np

# ── Tolerance constants ──────────────────────────────────────────────
ATOL = 1e-8
RTOL = 1e-5


# ── Lightweight test harness ─────────────────────────────────────────
_results: List[Tuple[str, bool, str]] = []


def _register(name: str, passed: bool, detail: str = ""):
    _results.append((name, passed, detail))


def assert_close(a, b, msg="", rtol=RTOL, atol=ATOL):
    a, b = float(a), float(b)
    if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
        raise AssertionError(f"{msg}: {a} ≠ {b} (rtol={rtol}, atol={atol})")


def assert_array_close(a, b, msg="", rtol=RTOL, atol=ATOL):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(a - b))
        raise AssertionError(f"{msg}: max |diff| = {max_diff}")


def assert_in_range(arr, lo, hi, msg=""):
    arr = np.asarray(arr, float)
    if arr.min() < lo - ATOL or arr.max() > hi + ATOL:
        raise AssertionError(
            f"{msg}: values [{arr.min():.6g}, {arr.max():.6g}] "
            f"outside [{lo}, {hi}]"
        )


def run_test(func: Callable, name: str):
    """Execute *func* and record pass/fail."""
    try:
        func()
        _register(name, True)
    except Exception as e:
        _register(name, False, f"{e.__class__.__name__}: {e}")


# =====================================================================
#  1.  pgs_rock_typing  (Akbar et al., DOI: …2025a1)
# =====================================================================
def test_pgs_rock_typing():
    from petrophysics_v66n6 import pgs_rock_typing as m

    k = np.array([10, 100, 1000], dtype=float)
    phi = np.array([0.10, 0.20, 0.30])

    # Pore geometry and pore structure
    pg = m.pore_geometry(k, phi)
    ps = m.pore_structure(k, phi)
    assert pg.shape == (3,), "pore_geometry shape"
    assert ps.shape == (3,), "pore_structure shape"
    assert_close(pg[0], math.sqrt(10 / 0.10), "pore_geometry value")
    assert_close(ps[1], 100 / 0.20**3, "pore_structure value")

    # Rock-type numbers are integers in [4..15]
    rt = m.pgs_rock_type_number(k, phi)
    assert rt.dtype in (np.int32, np.int64, int), "RT dtype"
    assert_in_range(rt, 4, 15, "RT range")

    # Corey kr: endpoints and bounds
    Sw = np.linspace(0.15, 0.75, 50)
    kro = m.corey_kro(Sw, 0.15, 0.25, kro0=1.0, no=2.0)
    krw = m.corey_krw(Sw, 0.15, 0.25, krw0=0.3, nw=3.0)
    assert_in_range(kro, 0, 1, "kro ∈ [0,1]")
    assert_in_range(krw, 0, 1, "krw ∈ [0,1]")
    assert_close(kro[0], 1.0, "kro at Swir = 1.0")
    assert_close(krw[-1], 0.3, "krw at 1-Sor", rtol=0.02)

    # Corey curves helper
    Sw2, kro2, krw2 = m.corey_curves(0.15, 0.25, n_points=20)
    assert Sw2.shape == (20,)
    assert_close(Sw2[0], 0.15, "corey_curves Sw start")
    assert_close(Sw2[-1], 0.75, "corey_curves Sw end")

    # Trend models: Sorw must be positive and bounded
    Swi = np.linspace(0.05, 0.40, 30)
    Sorw = m.sorw_trend(Swi)
    assert_in_range(Sorw, 0, 1, "Sorw ∈ [0,1]")
    krw0 = m.krw_endpoint_trend(Swi)
    assert_in_range(krw0, 0, 1, "krw0 ∈ [0,1]")
    no = m.corey_exponent_oil_trend(Swi)
    nw = m.corey_exponent_water_trend(Swi)
    assert all(no > 1), "no > 1"
    assert all(nw > 1), "nw > 1"

    # Generate kr for rock types
    kr_map = m.generate_kr_for_rock_types({2: 0.10, 5: 0.25})
    assert set(kr_map.keys()) == {2, 5}
    for rt_num, rk in kr_map.items():
        assert_in_range(rk.kro, 0, 1, f"RT-{rt_num} kro")
        assert_in_range(rk.krw, 0, 1, f"RT-{rt_num} krw")

    # Recovery factor
    assert_close(m.recovery_factor(30, 100), 0.30, "RF")


# =====================================================================
#  2.  dl_permeability  (Youssef et al., DOI: …2025a2)
# =====================================================================
def test_dl_permeability():
    from petrophysics_v66n6 import dl_permeability as m

    y_true = np.array([100, 200, 300, 400, 500], dtype=float)
    y_pred = np.array([105, 198, 310, 390, 510], dtype=float)

    # R² should be close to 1 for near-perfect predictions
    r2 = m.r2_score(y_true, y_pred)
    assert 0.99 < r2 <= 1.0, f"R² = {r2}"

    # Perfect predictions → R² = 1, MAPE = 0
    assert_close(m.r2_score(y_true, y_true), 1.0, "R² perfect")
    assert_close(m.mape(y_true, y_true), 0.0, "MAPE perfect")
    assert_close(m.coefficient_of_variation(y_true, y_true), 0.0, "Cv perfect")

    # MAPE > 0 for imperfect predictions
    assert m.mape(y_true, y_pred) > 0

    # Harmonic and arithmetic means
    vals = np.array([2.0, 8.0])
    assert_close(m.harmonic_mean(vals), 3.2, "harmonic mean")
    assert_close(m.arithmetic_mean(vals), 5.0, "arithmetic mean")

    # Upscaling: isotropic case should give the constant value
    k_const = np.full((3, 3, 3), 500.0)
    for d in ("x", "y", "z"):
        assert_close(m.upscale_permeability_cartesian(k_const, d), 500.0,
                     f"upscale isotropic {d}")


# =====================================================================
#  3.  primary_drainage  (Fernandes et al., DOI: …2025a3)
# =====================================================================
def test_primary_drainage():
    from petrophysics_v66n6 import primary_drainage as m

    # rpm → omega
    assert_close(m.rpm_to_omega(60), 2 * math.pi, "60 rpm → 2π rad/s")

    # Centrifuge Pc: at r = R → Pc = 0
    omega = m.rpm_to_omega(3000)
    delta_rho = 200.0
    R = 0.15
    Pc_outlet = m.centrifuge_capillary_pressure(delta_rho, omega, np.array([R]), R)
    assert_close(Pc_outlet[0], 0.0, "Pc at outlet = 0")

    # Pc increases towards the inlet
    r = np.linspace(0.10, R, 50)
    Pc = m.centrifuge_capillary_pressure(delta_rho, omega, r, R)
    assert Pc[0] > Pc[-1], "Pc(inlet) > Pc(outlet)"
    assert all(Pc >= -ATOL), "Pc ≥ 0"

    # Porous plate: output equals input
    steps = np.array([1e5, 2e5, 5e5])
    Sw_pp = np.array([0.8, 0.5, 0.25])
    Pc_pp, Sw_ret = m.porous_plate_Pc(steps, Sw_pp)
    assert_array_close(Pc_pp, steps, "porous plate Pc")
    assert_array_close(Sw_ret, Sw_pp, "porous plate Sw")

    # Semi-dynamic Pc
    Po = np.array([3e5, 2.5e5, 2e5])
    Pw = 1e5
    Pc_sd = m.semi_dynamic_Pc(Po, Pw)
    assert_array_close(Pc_sd, Po - Pw, "semi-dynamic Pc = Po − Pw")

    # Resistivity index
    RI = m.resistivity_index(np.array([100, 200, 400]), 100.0)
    assert_array_close(RI, [1, 2, 4], "RI values")

    # Archie n recovery for exact power law
    Sw = np.array([1.0, 0.5, 0.25, 0.125])
    n_true = 2.0
    RI_exact = Sw ** (-n_true)
    n_fit = m.archie_n_exponent(Sw, RI_exact)
    assert_close(n_fit, n_true, "Archie n round-trip", rtol=0.01)

    # Average saturation
    Sw_prof = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
    r_pos = np.linspace(0.1, 0.15, 5)
    avg = m.average_saturation_from_profile(Sw_prof, r_pos)
    assert_close(avg, 0.3, "uniform profile avg")


# =====================================================================
#  4.  analog_kr  (Schembre-McCabe et al., DOI: …2025a4)
# =====================================================================
def test_analog_kr():
    from petrophysics_v66n6 import analog_kr as m

    # Capillary number
    Nca = m.capillary_number(mu=1e-3, v=1e-5, sigma=0.03)
    assert_close(Nca, 1e-3 * 1e-5 / 0.03, "Nca")

    # Phase mobility
    kr = np.array([0.5, 0.3])
    lam = m.phase_mobility(kr, 1e-3)
    assert_array_close(lam, [500, 300], "phase mobility")

    # Total mobility = sum of phases
    lam_t = m.total_mobility(np.array([0.5]), 1e-3,
                              np.array([0.5]), 2e-3)
    assert_close(lam_t[0], 500 + 250, "total mobility")

    # Fractional flow: equal mobility → f = 0.5
    f = m.fractional_flow(np.array([0.5]), 1e-3,
                           np.array([0.5]), 1e-3)
    assert_close(f[0], 0.5, "f at equal mobility")

    # Fractional flow: very mobile injected phase → f → 1
    f_high = m.fractional_flow(np.array([1.0]), 1e-5,
                                np.array([0.001]), 1e-3)
    assert f_high[0] > 0.99, "f → 1 for mobile injected phase"

    # Viscosity ratio
    assert_close(m.viscosity_ratio(1e-4, 1e-3), 0.1, "visc ratio")

    # FluidPair
    fp = m.FluidPair("test", 1e-4, 1e-3, 0.03)
    assert_close(fp.visc_ratio, 0.1, "FluidPair visc_ratio")
    assert_close(fp.Nca(1e-5), 1e-4 * 1e-5 / 0.03, "FluidPair Nca")

    # Select analog: should pick closest in log-space
    target = m.CO2_BRINE
    best = m.select_analog_pair(target, [m.N2_BRINE, m.OIL_BRINE])
    assert best.name in ("N2/brine", "mineral_oil/brine"), "analog selection"

    # Rescale: output shapes match
    Sw = np.linspace(0.2, 0.8, 20)
    kro = 1.0 - Sw
    krw = Sw
    Sw_out, lam_t, f_nw = m.rescale_kr_to_target_viscosity(
        Sw, kro, krw, m.OIL_BRINE, m.CO2_BRINE)
    assert lam_t.shape == (20,), "rescale shape"


# =====================================================================
#  5.  co2_uptake  (Chen et al., DOI: …2025a5)
# =====================================================================
def test_co2_uptake():
    from petrophysics_v66n6 import co2_uptake as m

    # Specific surface area ∝ 1/r
    S1 = m.specific_surface_area_spherical(0.1, 5e-9)
    S2 = m.specific_surface_area_spherical(0.1, 10e-9)
    assert_close(S1 / S2, 2.0, "S_sp ∝ 1/r")

    # Pore surface area: total = S_sp × V
    S_total = m.pore_surface_area_spherical(0.1, 5e-9, 1.0)
    assert_close(S_total, S1, "S_total = S_sp for V=1")

    # NMR measurement
    meas = m.NMR_CO2_Measurement("A", 4000, 60, 250.0, 1000.0, 0.02, 20.0)
    n = m.co2_sorption_moles(meas)
    assert_close(n, 250 / 1000 * 0.02, "CO2 moles")
    assert_close(m.co2_sorption_per_gram(meas), n / 20.0, "CO2 per gram")
    assert_close(m.co2_mass_from_moles(1.0), 44.01, "CO2 molar mass")

    # Storage capacity: consistent units
    cap = m.co2_storage_capacity(1e-4, 2500, 1e6)
    assert cap > 0, "storage capacity > 0"

    # Langmuir isotherm: V(0) = 0, V(∞) → V_L
    P = np.array([0.0, 1e6, 1e12])
    V = m.langmuir_isotherm(P, V_L=100, P_L=1e6)
    assert_close(V[0], 0.0, "Langmuir V(0)=0")
    assert_close(V[1], 50.0, "Langmuir V(P_L) = V_L/2")
    assert abs(V[2] - 100.0) < 0.01, "Langmuir V(∞) → V_L"


# =====================================================================
#  6.  drp_wettability  (Faisal et al., DOI: …2025a6)
# =====================================================================
def test_drp_wettability():
    from petrophysics_v66n6 import drp_wettability as m

    # Uniform contact angle range
    theta = m.uniform_contact_angle(1000, 0, 180, seed=0)
    assert theta.shape == (1000,)
    assert_in_range(theta, 0, 180, "uniform θ range")

    # Mixed-wet model: small pores WW, large pores OW
    radii = np.array([1e-6, 5e-6, 10e-6, 20e-6])
    r_thresh = 7e-6
    theta_mw = m.mixed_wet_model(radii, r_thresh, 30, 140)
    assert theta_mw[0] == 30.0, "small pore WW"
    assert theta_mw[1] == 30.0, "medium pore WW"
    assert theta_mw[2] == 140.0, "large pore OW"
    assert theta_mw[3] == 140.0, "very large pore OW"

    # Oil-wet fraction
    owf = m.oil_wet_fraction(radii, r_thresh)
    assert 0 < owf < 1, "OW fraction ∈ (0,1)"

    # Bundle of tubes kr
    rng = np.random.default_rng(42)
    radii_test = rng.lognormal(-12, 0.5, 200)
    theta_test = m.mixed_wet_model(radii_test, np.median(radii_test))
    res = m.simple_bundle_of_tubes_kr(radii_test, theta_test, sigma=0.03)
    assert_in_range(res.kro, 0, 1, "bundle kro ∈ [0,1]")
    assert_in_range(res.krw, 0, 1, "bundle krw ∈ [0,1]")
    assert_in_range(res.Sw, 0, 1, "bundle Sw ∈ [0,1]")
    assert len(res.Sw) == len(res.kro) == len(res.krw), "output lengths"

    # Compare lithotypes: returns two results
    r_A = rng.lognormal(-11, 0.8, 100)
    r_B = rng.lognormal(-12, 0.5, 100)
    res_A, res_B = m.compare_lithotypes(r_A, np.median(r_A),
                                         r_B, np.median(r_B))
    assert isinstance(res_A, m.PoreNetworkResult)
    assert isinstance(res_B, m.PoreNetworkResult)


# =====================================================================
#  7.  electrokinetic  (Halisch et al., DOI: …2025a7)
# =====================================================================
def test_electrokinetic():
    from petrophysics_v66n6 import electrokinetic as m

    # Zeta potential: sign matches sign(ΔV/ΔP)
    z = m.zeta_potential_HS(-1e-3, 1e5)
    assert z < 0, "negative ΔV → negative ζ"

    z_mV = m.zeta_potential_mV(-1e-3, 1e5)
    assert_close(z_mV, z * 1e3, "mV conversion")

    # Streaming-potential coefficient via regression
    dP = np.array([1e5, 2e5, 3e5])
    dV = np.array([-5e-4, -1e-3, -1.5e-3])
    C = m.streaming_potential_coefficient(dV, dP)
    assert_close(C, -5e-9, "C_sp", rtol=1e-3)

    # Debye length: higher ionic strength → shorter
    lam1 = m.debye_length(10)     # 10 mol/m³
    lam2 = m.debye_length(100)    # 100 mol/m³
    assert lam1 > lam2, "Debye length decreases with I"
    assert lam1 > 0 and lam2 > 0, "Debye length > 0"

    # Ionic strength
    I = m.ionic_strength_from_molarity({"Na+": 0.1, "Cl-": 0.1},
                                        {"Na+": 1, "Cl-": 1})
    assert_close(I, 0.1, "NaCl ionic strength")

    I2 = m.ionic_strength_from_molarity({"Ba2+": 0.1, "Cl-": 0.2},
                                         {"Ba2+": 2, "Cl-": 1})
    assert_close(I2, 0.5 * (0.1 * 4 + 0.2 * 1), "BaCl2 ionic strength")

    # Dopant tests
    base = m.DopantTest("base", None, -25.0)
    doped = m.DopantTest("NaI", "NaI", -10.0)
    assert_close(m.zeta_shift(base, doped), 15.0, "Δζ")
    flags = m.flag_wettability_risk([15.0, 5.0, -12.0], threshold_mV=10.0)
    assert flags == [True, False, True], "risk flags"


# =====================================================================
#  8.  dopant_wettability  (Nono et al., DOI: …2025a8)
# =====================================================================
def test_dopant_wettability():
    from petrophysics_v66n6 import dopant_wettability as m

    rng = np.random.default_rng(7)

    # PoreOccupancy
    vols = rng.uniform(1, 10, 100)
    oil = rng.random(100) > 0.6
    po = m.PoreOccupancy(100, oil, vols)
    assert 0 <= po.oil_saturation <= 1, "So ∈ [0,1]"
    assert_close(po.oil_saturation + po.water_saturation, 1.0, "So + Sw = 1")

    # Trapped-oil fraction and recovery
    before = m.PoreOccupancy(100, rng.random(100) > 0.3, vols)
    after = m.PoreOccupancy(100, rng.random(100) > 0.7, vols)
    tf = m.trapped_oil_fraction(before, after)
    assert 0 <= tf <= 1, "trapped fraction ∈ [0,1]"
    re = m.recovery_efficiency(before, after)
    assert -0.5 <= re <= 1.0, "recovery efficiency plausible"

    # Amott index bounds
    Iw = m.amott_wettability_index(0.2, 0.2, 0.5, 0.3)
    assert -1 <= Iw <= 1, "Iw ∈ [-1,1]"

    # Strongly water-wet: δw large, δo small
    Iw_ww = m.amott_wettability_index(0.1, 0.2, 0.6, 0.2)
    assert Iw_ww > 0, "water-wet Iw > 0"

    # Wettability shift
    assert_close(m.wettability_shift(0.1, 0.3), 0.2, "Δ Iw")

    # Protocol comparison and bias summary
    protocols = [
        m.ProtocolResult("Dopant-free", 0.20, 0.28, 0.35, 0.10),
        m.ProtocolResult("NaI", 0.20, 0.22, 0.45, 0.30),
    ]
    bias = m.dopant_bias_summary(protocols)
    assert "NaI" in bias, "bias summary key"
    assert_close(bias["NaI"], 0.20, "NaI Iw shift")


# =====================================================================
#  9.  low_salinity_ior  (Karoussi et al., DOI: …2025a9)
# =====================================================================
def test_low_salinity_ior():
    from petrophysics_v66n6 import low_salinity_ior as m

    # Contact angle: low salinity → low θ, high salinity → high θ
    S = np.array([100, 50000])
    theta = m.contact_angle_salinity(S, theta_high=130, theta_low=50)
    assert theta[0] < theta[1], "θ increases with salinity"
    assert theta[0] >= 50, "θ ≥ θ_low"
    assert theta[1] <= 130, "θ ≤ θ_high"

    # Imbibition recovery: monotonically increasing, bounded
    t = np.linspace(0.1, 1000, 200)
    RF = m.imbibition_recovery_expstretched(t, 0.15, 100, 0.7)
    assert_in_range(RF, 0, 0.15 + ATOL, "RF ∈ [0, RF_max]")
    assert all(np.diff(RF) >= -ATOL), "RF monotonically increasing"

    # Imbibition rate: non-negative
    rate = m.imbibition_rate(t, 0.15, 100, 0.7)
    assert all(rate >= -ATOL), "imbibition rate ≥ 0"

    # Disjoining pressure components
    h = np.array([1e-9, 5e-9, 10e-9])
    Pi_dl = m.double_layer_repulsion(h, kappa_inv=3e-9)
    assert all(Pi_dl > 0), "DL repulsion > 0"
    Pi_vdw = m.van_der_waals_attraction(h)
    assert all(Pi_vdw < 0), "vdW attraction < 0"

    # Net disjoining pressure at large distance → 0
    h_far = np.array([1e-6])
    Pi_net = m.disjoining_pressure(h_far, kappa_inv=1e-9)
    assert abs(Pi_net[0]) < 1e-3, "Π → 0 at large h"

    # RF uplift and incremental oil
    assert_close(m.rf_uplift(0.30, 0.38), 0.08, "RF uplift")
    assert_close(m.incremental_oil(1e6, 0.08), 8e4, "incremental oil")


# =====================================================================
#  10.  nanopore_adsorption  (Nguyen et al., DOI: …2025a10)
# =====================================================================
def test_nanopore_adsorption():
    from petrophysics_v66n6 import nanopore_adsorption as m

    # Washburn: h(0) = 0, h increases with time
    t = np.linspace(0, 10, 50)
    h = m.washburn_capillary_rise(0.020, 0, 1e-3, 5e-9, t)
    assert_close(h[0], 0.0, "h(0) = 0")
    assert all(np.diff(h) >= -ATOL), "h increasing"

    # Washburn round-trip: estimate r from h
    h_val = h[-1]
    r_est = m.washburn_effective_radius(h_val, t[-1], 0.020, 0, 1e-3)
    assert_close(r_est, 5e-9, "Washburn r round-trip", rtol=0.01)

    # Kelvin: hydrophilic (θ=0) condenses at lower P/P₀ than hydrophobic (θ=90)
    P_hphilic = m.kelvin_condensation_pressure(3e-9, 0.016, 5.2e-5, 0, 300)
    P_hphobic = m.kelvin_condensation_pressure(3e-9, 0.016, 5.2e-5, 90, 300)
    assert P_hphilic < P_hphobic, "hydrophilic condenses earlier"
    assert 0 < P_hphilic < 1, "P/P₀ ∈ (0,1) hydrophilic"

    # BET isotherm: V increases monotonically with P/P₀
    x = np.linspace(0.01, 0.35, 50)
    V = m.bet_isotherm(x, V_m_mono=50.0, C=80.0)
    assert all(np.diff(V) > 0), "BET V increasing"
    assert V[0] > 0, "BET V(x>0) > 0"

    # BET surface area > 0
    S = m.bet_surface_area(50.0)
    assert S > 0, "BET surface area > 0"

    # Adsorption ratio
    assert_close(m.adsorption_amount_ratio(10, 5), 2.0, "adsorption ratio")


# =====================================================================
#  11.  carbon13_mr  (Ansaribaranghar et al., DOI: …2025a11)
# =====================================================================
def test_carbon13_mr():
    from petrophysics_v66n6 import carbon13_mr as m

    # Surface relaxation: T2_obs < T2_bulk
    T2_obs = m.surface_relaxation_T2(rho2=5e-6, S_over_V=1e6, T2_bulk=2.0)
    assert T2_obs < 2.0, "surface relaxation reduces T2"
    assert T2_obs > 0, "T2 > 0"

    # No surface interaction → T2 = T2_bulk
    T2_no_surf = m.surface_relaxation_T2(rho2=0, S_over_V=1e6, T2_bulk=2.0)
    assert_close(T2_no_surf, 2.0, "zero ρ₂ → T2 = T2_bulk")

    # T1/T2 > 1 for pore-surface interactions
    ratio = m.T1_over_T2(rho1=3e-6, rho2=10e-6, S_over_V=1e6)
    assert ratio > 1, "T1/T2 > 1 with surface relaxation"

    # Relative sensitivity
    sens = m.relative_sensitivity_13C_1H()
    assert 0 < sens < 1, "13C sensitivity < 1H"

    # SNR improvement
    assert_close(m.snr_improvement_factor(4), 2.0, "√4 = 2")

    # Multi-exponential decay: correct at t=0
    t = np.linspace(0, 100, 100)
    A = np.array([0.6, 0.4])
    T2 = np.array([50.0, 10.0])
    S = m.multi_exponential_decay(t, A, T2)
    assert_close(S[0], 1.0, "S(0) = sum(A)")

    # Decay: S(t→∞) → 0
    t_long = np.array([1e6])
    S_long = m.multi_exponential_decay(t_long, A, T2)
    assert S_long[0] < 1e-10, "S(∞) → 0"

    # Log-mean T2
    T2_lm = m.log_mean_T2(A, T2)
    assert 10 < T2_lm < 50, "T2_LM between components"

    # Wettability classification
    ww = m.WettabilityMR("WW", 100, 2.0)
    assert ww.classify() == "water-wet"
    ow = m.WettabilityMR("OW", 10, 15.0)
    assert ow.classify() == "oil-wet"
    mw = m.WettabilityMR("MW", 40, 6.0)
    assert mw.classify() == "mixed-wet"


# =====================================================================
#  12.  kerogen_mr  (Zamiri et al. 2025a, DOI: …2025a12)
# =====================================================================
def test_kerogen_mr():
    from petrophysics_v66n6 import kerogen_mr as m

    # Calibrate signal
    assert_close(m.calibrate_signal(500, 1000, 0.02), 0.01, "calibration")

    # Component assignment
    T2_vals = np.array([0.3, 0.5, 15.0, 50.0])
    amps = np.array([0.2, 0.1, 0.4, 0.3])
    ker, oil = m.assign_components(T2_vals, amps, kerogen_cutoff_ms=1.0)
    assert ker.label == "kerogen"
    assert oil.label == "oil"
    assert_close(ker.amplitude, 0.3, "kerogen total amplitude")
    assert_close(oil.amplitude, 0.7, "oil total amplitude")

    # H/C ratio
    assert_close(m.hydrogen_carbon_ratio(1.2, 1.0), 1.2, "H/C")

    # Carbon moles from 13C: same natural abundance cancels
    n_C = m.carbon_moles_from_13C(200, 1000, 0.5)
    assert_close(n_C, 0.1, "carbon moles")

    # Kerogen classification
    assert m.kerogen_class_from_HC(1.8) == "I"
    assert m.kerogen_class_from_HC(1.2) == "II"
    assert m.kerogen_class_from_HC(0.7) == "III"
    assert "IV" in m.kerogen_class_from_HC(0.3)

    # Maturity
    assert "immature" in m.maturity_from_HC(1.5)
    assert "oil window" in m.maturity_from_HC(0.9)
    assert "over-mature" in m.maturity_from_HC(0.2)

    # VanKrevelenPoint
    vk = m.VanKrevelenPoint("S1", 1.1)
    assert vk.kerogen_class == "II"

    # ShaleSpeciesRelaxation
    sp = m.ShaleSpeciesRelaxation("kerogen", 300, 0.4, 92, 0.1)
    assert_close(sp.T1_T2_13C, 750, "T1/T2 13C")
    assert_close(sp.T1_T2_1H, 920, "T1/T2 1H")


# =====================================================================
#  13.  mri_rel_perm  (Zamiri et al. 2025b, DOI: …2025a13)
# =====================================================================
def test_mri_rel_perm():
    from petrophysics_v66n6 import mri_rel_perm as m

    Sw = np.linspace(0.15, 0.80, 50)

    # Corey kr: endpoints
    krw = m.corey_krw(Sw, 0.15, 0.20, 0.4, 3.0)
    kro = m.corey_kro(Sw, 0.15, 0.20, 1.0, 2.0)
    assert_in_range(krw, 0, 1, "krw ∈ [0,1]")
    assert_in_range(kro, 0, 1, "kro ∈ [0,1]")
    assert_close(krw[0], 0.0, "krw(Swir) = 0")
    assert_close(kro[-1], 0.0, "kro(1-Sor) = 0", atol=0.01)

    # Reduced saturation
    assert_close(m.reduced_saturation(0.5, 0.2, 1.0), 0.375, "Sr")
    assert_close(m.reduced_saturation(0.2, 0.2, 1.0), 0.0, "Sr at Smin")

    # Logbeta Pc: monotonically decreasing with Sw
    Sw_pc = np.linspace(0.20, 0.95, 40)
    Pc = m.logbeta_Pc_drainage(Sw_pc, Smin=0.15, po=5000, pt=1000)
    assert all(np.diff(Pc) <= ATOL), "Pc decreasing with Sw"

    # Saturation derivatives: shape check
    nx, nt = 30, 10
    x = np.linspace(0, 0.05, nx)
    t = np.linspace(0, 1000, nt)
    Sw_prof = np.outer(np.linspace(1, 0.5, nt), np.ones(nx))
    # Add spatial variation
    for j in range(nt):
        Sw_prof[j, :] += 0.1 * np.sin(np.linspace(0, np.pi, nx)) * (j / nt)
    Sw_prof = np.clip(Sw_prof, 0.15, 1.0)

    Sx, St = m.compute_saturation_derivatives(Sw_prof, x, t)
    assert Sx.shape == (nt, nx), "Sx shape"
    assert St.shape == (nt, nx), "St shape"

    # Water flux
    qw = m.compute_water_flux(St, x, phi=0.24, qt=1e-5)
    assert qw.shape == (nt, nx), "qw shape"

    # Extract fw and Dc: shape and bounds
    Sw_flat = Sw_prof.ravel()
    Sx_flat = Sx.ravel()
    qw_flat = qw.ravel()
    Sw_c, fw, Dc = m.extract_fw_Dc(Sw_flat, Sx_flat, qw_flat, qt=1e-5,
                                     n_bins=15)
    assert len(Sw_c) == 15, "fw bins"
    assert len(fw) == 15

    # Pressure drop from constant profile → finite
    Po = m.solve_pressure_profile(
        Sw_profile=np.linspace(0.8, 0.3, 30),
        x=np.linspace(0, 0.05, 30),
        krw_func=lambda s: m.corey_krw(s, 0.15, 0.20, 0.4, 3),
        kro_func=lambda s: m.corey_kro(s, 0.15, 0.20, 1.0, 2),
        Pc_func=lambda s: m.logbeta_Pc_drainage(s, 0.15, 5000, 1000),
        K=1e-12, mu_w=1e-3, mu_o=5e-3, qt=1e-5,
    )
    assert Po.shape == (30,), "pressure profile shape"
    dP = m.pressure_drop(Po)
    assert dP > 0, "pressure drop > 0 for forward flow"


# =====================================================================
#  Master runner
# =====================================================================
_ALL_TESTS = [
    ("01 pgs_rock_typing   (Akbar et al.)",            test_pgs_rock_typing),
    ("02 dl_permeability   (Youssef et al.)",           test_dl_permeability),
    ("03 primary_drainage  (Fernandes et al.)",         test_primary_drainage),
    ("04 analog_kr         (Schembre-McCabe et al.)",   test_analog_kr),
    ("05 co2_uptake        (Chen et al.)",              test_co2_uptake),
    ("06 drp_wettability   (Faisal et al.)",            test_drp_wettability),
    ("07 electrokinetic    (Halisch et al.)",           test_electrokinetic),
    ("08 dopant_wettability(Nono et al.)",              test_dopant_wettability),
    ("09 low_salinity_ior  (Karoussi et al.)",          test_low_salinity_ior),
    ("10 nanopore_adsorption(Nguyen et al.)",           test_nanopore_adsorption),
    ("11 carbon13_mr       (Ansaribaranghar et al.)",   test_carbon13_mr),
    ("12 kerogen_mr        (Zamiri et al. 2025a)",      test_kerogen_mr),
    ("13 mri_rel_perm      (Zamiri et al. 2025b)",      test_mri_rel_perm),
]


def test_all(verbose: bool = False):
    """Run every test and print a summary.

    Parameters
    ----------
    verbose : bool
        If True, print each sub-assertion as it runs.

    Returns
    -------
    bool
        True if every test passed.
    """
    global _results
    _results = []

    passed_modules = 0
    failed_modules = 0

    width = 55
    print("=" * (width + 12))
    print(f" {'petrophysics_v66n6  —  test_all':^{width + 10}}")
    print("=" * (width + 12))

    for label, func in _ALL_TESTS:
        run_test(func, label)

    # Deduplicate: one entry per run_test call
    seen = set()
    unique = []
    for name, ok, detail in _results:
        if name not in seen:
            seen.add(name)
            unique.append((name, ok, detail))

    for name, ok, detail in unique:
        status = "  PASS ✓" if ok else "**FAIL**"
        line = f"  {name:<{width}} {status}"
        print(line)
        if not ok:
            failed_modules += 1
            if verbose or True:  # always show failure detail
                for dline in detail.split("\n"):
                    print(f"      └─ {dline}")
        else:
            passed_modules += 1

    print("-" * (width + 12))
    total = passed_modules + failed_modules
    print(f"  {passed_modules}/{total} modules passed", end="")
    if failed_modules:
        print(f",  {failed_modules} FAILED")
    else:
        print("  —  ALL OK")
    print("=" * (width + 12))

    return failed_modules == 0


# ── pytest compatibility ─────────────────────────────────────────────
# Each test_* function is also directly discoverable by pytest.


# ── CLI entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    success = test_all(verbose=verbose)
    sys.exit(0 if success else 1)
