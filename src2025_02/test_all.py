#!/usr/bin/env python3
"""
Unified Test Suite for Petrophysics 66(1), February 2025 — All 12 Articles.

Each test function validates one article's module using synthetic data and
prints PASS/FAIL for every assertion.

Run as:  python test_all.py
"""

import sys
import traceback
import numpy as np

PASS = 0
FAIL = 0

def check(condition, label):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"    ✓ {label}")
    else:
        FAIL += 1
        print(f"    ✗ FAIL: {label}")


# ──────────────────────────────────────────────────────────────────────────────
def test_01_scal_model_ccs():
    """Art. 1 — Ebeltoft et al.: SCAL Model for CCS (LET correlations)."""
    print("\n[1] scal_model_ccs — LET RelPerm, Pc, Land trapping")
    from scal_model_ccs import (
        LETRelPermParams, LETCapPresParams,
        let_relative_permeability, let_capillary_pressure,
        normalized_water_saturation, leverett_j_scaling,
        land_trapping, land_coefficient, co2_storage_capacity, SCALModelCCS,
    )
    Sw = np.linspace(0.15, 1.0, 50)
    p = LETRelPermParams()
    krg, krw = let_relative_permeability(Sw, p)
    check(krg[0] > 0.5,   "krg at Swr > 0.5")
    check(krw[-1] > 0.9,  "krw at Sw=1 ≈ 1")
    check(krg[-1] < 0.01, "krg at Sw=1 ≈ 0")
    check(krw[0] < 0.01,  "krw at Swr ≈ 0")
    check(np.all(np.diff(krg) <= 1e-10), "krg monotonically decreasing")
    check(np.all(np.diff(krw) >= -1e-10), "krw monotonically increasing")

    pc = LETCapPresParams()
    Pc = let_capillary_pressure(Sw, pc)
    check(Pc[0] > Pc[-1], "Pc decreasing with Sw")
    check(Pc[-1] >= pc.Pcgw_tp - 0.01, "Pc → Pcgw_tp at Sw=1")

    Swn = normalized_water_saturation(np.array([0.15, 0.575, 1.0]), 0.15)
    check(np.allclose(Swn, [0, 0.5, 1], atol=0.01), "Swn normalization correct")

    Pc_j = leverett_j_scaling(Pc, 72, 30)
    check(np.allclose(Pc_j, Pc * 30/72, atol=1e-6), "Leverett J-scaling correct")

    C = land_coefficient(0.85, 0.35)
    check(C > 0, "Land coefficient positive")
    Sgt = land_trapping(0.85, C)
    check(abs(Sgt - 0.35) < 0.01, "Land trapping reproduces Sgt_max")
    check(land_trapping(0.0, C) == 0, "Sgt=0 when Sgi=0")

    s, r = co2_storage_capacity(1e6, 0.15, 0.35)
    check(s > r, "Structural > residual storage")
    check(s > 0 and r > 0, "Positive storage values")

    model = SCALModelCCS()
    for case in ("base", "optimistic", "pessimistic"):
        kg, kw = model.evaluate(Sw, case)
        check(kg[0] > 0 and kw[-1] > 0, f"SCAL model '{case}' valid")


def test_02_co2_brine_relperm():
    """Art. 2 — Mascle et al.: CO2/Brine SS vs USS reconciliation."""
    print("\n[2] co2_brine_relperm — Corey, fractional flow, SS/USS reconciliation")
    from co2_brine_relperm import (
        corey_relperm, fractional_flow, CoreFluidProps,
        ss_analytical_kr, capillary_end_effect_length,
        capillary_number, fit_corey, reconcile_ss_uss,
    )
    Sw = np.linspace(0.20, 0.95, 20)
    krw, krc = corey_relperm(Sw, 0.20, 0.0, 0.8, 0.6, 4, 2.5)
    check(krw[0] < 0.01 and krc[-1] < 0.01, "Corey endpoints near zero")
    check(krw[-1] > 0.5, "krw endpoint > 0.5")

    fw = fractional_flow(Sw, 0.20, 0, 0.8, 0.6, 4, 2.5, 7e-4, 5e-5)
    check(fw[0] < 0.05, "fw ≈ 0 at low Sw")
    check(fw[-1] > 0.90, "fw → 1 at high Sw")

    props = CoreFluidProps()
    krw_ss, krc_ss = ss_analytical_kr(0.5, 1e-8, 5000, props)
    check(krw_ss > 0 and krc_ss > 0, "SS kr positive")

    Nca = capillary_number(1e-8, 7e-4, 30e-3, props.A)
    check(Nca > 0, "Capillary number positive")

    cee = capillary_end_effect_length(5000, 50000, 0.020)
    check(0 < cee <= 0.020, "CEE length within core")

    np.random.seed(42)
    fitted = fit_corey(Sw, np.clip(krw*(1+0.05*np.random.randn(20)),0,1),
                       np.clip(krc*(1+0.05*np.random.randn(20)),0,1), Swir=0.20)
    check(1 < fitted['nw'] < 10, "Fitted nw in range")
    check(1 < fitted['nco2'] < 10, "Fitted nco2 in range")

    rec = reconcile_ss_uss(Sw, krw, krc, Sw[:5], krw[:5]*0.9, krc[:5]*1.1)
    check('nw' in rec and 'nco2' in rec, "Reconciliation produces parameters")


def test_03_ss_co2_brine():
    """Art. 3 — Richardson et al.: SS scCO2-brine at different pressures."""
    print("\n[3] ss_co2_brine_relperm — pressure effects, hysteresis, wettability")
    from ss_co2_brine_relperm import (
        SSTest, compute_ss_kr, pressure_effect, hysteresis_check,
        material_balance, wettability_from_kr,
    )
    t = SSTest()
    k = t.k_mD * 9.869e-16
    fw = np.array([1,.9,.7,.5,.3,.1,0.])
    Sw = np.array([1,.85,.65,.50,.38,.28,.22])
    dP = np.array([500,600,900,1200,1800,3000,5000.])
    s, krw, krg = compute_ss_kr(fw, dP, Sw, 1e-8, 7e-4, 5e-5, k, t.L, t.A)
    check(len(krw) == 7, "Correct number of kr points")
    check(np.all(krw >= 0), "krw non-negative")

    np.random.seed(42)
    eff = pressure_effect((s,krw,krg), (s,krw*(1+.02*np.random.randn(7)),krg*(1+.02*np.random.randn(7))))
    check('rmse_krw' in eff, "Pressure effect computed")

    hyst = hysteresis_check(Sw[1:], krw[1:], Sw[1:], krw[1:]*(1+0.01*np.random.randn(6)))
    check('rmse' in hyst, "Hysteresis check computed")

    mb = material_balance(1.0, 0.22, 0.78*t.PV, t.PV)
    check(mb['passes'], "Material balance passes")

    w = wettability_from_kr(0.55, 0.3, 0.6)
    check(w == "strongly water-wet", "Wettability classified correctly")


def test_04_enhanced_gas_recovery():
    """Art. 4 — Jones et al.: EGR by CO2 injection, Land trapping."""
    print("\n[4] enhanced_gas_recovery — Land, EGR efficiency, Burdine Pc")
    from enhanced_gas_recovery import (
        land_trapping, land_coefficient, compare_land_ch4_co2,
        burdine_Pc, let_kr_gas, egr_efficiency, issm_saturation,
        gravity_stable,
    )
    C = land_coefficient(0.90, 0.545)
    check(C > 0, "Land C > 0")
    check(abs(land_trapping(0.90, C) - 0.545) < 0.01, "Land reproduces endpoint")
    check(land_trapping(0.0, C) == 0, "Sgt(0)=0")

    comp = compare_land_ch4_co2(land_coefficient(0.90,0.545), 0.545,
                                 land_coefficient(0.78,0.34), 0.34)
    check(comp['reduction'] > 0.10, "CO2 trapped 15% less than CH4")
    check(comp['co2_partial_wetting'], "CO2 partial wetting detected")

    Sw = np.linspace(0.15, 1, 30)
    Pc = burdine_Pc(Sw, 0.10, 5.0, 2.0)
    check(Pc[0] > Pc[-1], "Burdine Pc decreasing")

    krg = let_kr_gas(Sw, 0.15, 0.8, 2, 1, 1.5)
    check(krg[0] > 0.5, "LET kr endpoint > 0.5")
    check(krg[-1] < 0.01, "LET kr at Sw=1 ≈ 0")

    egr = egr_efficiency(19.1, 18.6)
    check(egr['complete'], "EGR complete recovery")

    Iw = np.array([100.0]); Id = np.array([20.0]); Im = np.array([60.0])
    Sw_issm = issm_saturation(Iw, Id, Im)
    check(abs(Sw_issm[0] - 0.5) < 0.01, "ISSM Sw correct")

    check(gravity_stable(600, 1025, 122e-15, 0.5, 6.5e-4, 1e-11, 3e-4),
          "Gravity stable at low rate")


def test_05_rev_two_phase():
    """Art. 5 — McClure et al.: REV for two-phase flow."""
    print("\n[5] rev_two_phase_flow — energy dissipation, temporal REV, ergodicity")
    from rev_two_phase_flow import (
        energy_dissipation, kr_from_energy, temporal_rev_analysis,
        ergodicity_test, fluctuation_analysis, scal_duration_guide,
    )
    phi = energy_dissipation(1e-4, 1e5)
    check(phi > 0, "Energy dissipation positive")

    kr = kr_from_energy(1e-6, 1e-3, 1e-13, 1e5)
    check(0 < kr < 10, "kr from energy in range")

    np.random.seed(42)
    t = np.linspace(0, 3600, 500)
    sig = 100 + 5*np.sin(2*np.pi*t/60) + 3*np.random.randn(500)
    ws, cv, opt = temporal_rev_analysis(t, sig)
    check(opt >= 1, "Optimal window >= 1")
    check(cv[-1] < cv[0], "CV decreases with larger window")

    erg = ergodicity_test(sig[::50], np.mean(sig))
    check('ergodic' in erg, "Ergodicity test returns result")

    fl = fluctuation_analysis(sig, 0.5+0.01*np.random.randn(500), t)
    check(fl['p_cv'] > 0, "Pressure CV computed")
    check(fl['recommended_time'] > 0, "Recommended time positive")

    g = scal_duration_guide(1e-5)
    check(g['duration_hr'] > 0, "Duration guide positive")


def test_06_digital_rock_physics():
    """Art. 6 — Regaieg et al.: DRP for kr prediction."""
    print("\n[6] digital_rock_physics — pore network, wettability, ESRGAN")
    from digital_rock_physics import (
        WettabilityAnchor, pore_size_dist, assign_contacts,
        invasion_drainage_kr, esrgan_metrics, compare_drp_scal,
    )
    np.random.seed(42)
    pr = pore_size_dist(2000, 25)
    check(pr.mean() > 15, "Mean pore radius reasonable")

    anchor = WettabilityAnchor()
    ca = assign_contacts(5000, anchor)
    check(0 < ca.mean() < 180, "Contact angles in range")
    check(np.sum(ca > 90) > 0, "Some oil-wet pores exist")

    tr = pore_size_dist(5000, 10)
    Sw, krw, kro = invasion_drainage_kr(pr, tr, ca, n_steps=15)
    check(len(Sw) == 15, "Correct number of steps")
    check(krw[0] > krw[-1], "krw decreases during drainage")

    enh = esrgan_metrics(10, 2.5)
    check(enh['linear'] == 4, "4x linear enhancement")
    check(enh['volumetric'] == 64, "64x volumetric enhancement")

    comp = compare_drp_scal(Sw, krw, kro, Sw, krw*1.05, kro*0.95)
    check(comp['rmse_krw'] > 0, "RMSE computed")
    check(abs(comp['corr_krw']) > 0.5, "Good correlation")


def test_07_hybrid_drainage():
    """Art. 7 — Fernandes et al.: HDT on bimodal limestone."""
    print("\n[7] hybrid_drainage — HDT vs VOF, NMR T2, homogeneity")
    from hybrid_drainage import (
        BimodalPores, Pc_entry, bimodal_t2,
        hdt_protocol, vof_protocol, profile_homogeneity,
    )
    p = BimodalPores()
    check(p.phi_total == p.phi_macro + p.phi_meso, "Total porosity correct")

    Pc_m = Pc_entry(50, 25)
    Pc_s = Pc_entry(5, 25)
    check(Pc_s > Pc_m, "Meso Pc > macro Pc")

    T2, amp = bimodal_t2(p)
    check(len(T2) == 200, "T2 distribution has 200 points")
    check(np.all(amp >= 0), "Amplitude non-negative")

    hdt = hdt_protocol(p, 0.20)
    vof = vof_protocol(p, 0.20)
    check(hdt['final_Swi'] <= vof['final_Swi'], "HDT achieves lower Swi")

    np.random.seed(42)
    hom = profile_homogeneity(np.full(20, 0.20) + 0.005*np.random.randn(20))
    check(hom['homogeneous'], "Uniform profile is homogeneous")


def test_08_pore_scale_drainage():
    """Art. 8 — Nono et al.: PP vs OF pore-scale drainage."""
    print("\n[8] pore_scale_drainage — PP vs OF, pore occupancy, artifacts")
    from pore_scale_drainage import (
        classify_pores, pp_invasion, of_invasion,
        pore_occupancy, keff_from_occupancy, wettability_artifacts,
    )
    np.random.seed(42)
    r = np.clip(np.random.lognormal(np.log(15), 0.6, 2000), 1, 200)
    theta = np.where(np.random.rand(2000)<0.4, np.random.normal(120,15,2000),
                     np.random.normal(50,15,2000))
    theta = np.clip(theta, 0, 180)

    cls = classify_pores(r)
    check(sum(c['vf'] for c in cls.values()) > 0.99, "Volume fractions sum ~1")

    inv_pp = pp_invasion(r, 25, theta, 15000)
    inv_of = of_invasion(r, 25, theta, 15000, 0.05)
    check(inv_pp.sum() > 0, "PP invades some pores")

    occ = pore_occupancy(r, inv_pp, inv_of)
    check('Swi_pp' in occ, "Occupancy computed")
    check(occ['pp_lower'], "PP gives lower Swi (expected for non-WW)")

    ke_pp = keff_from_occupancy(r, inv_pp, 'oil')
    ke_of = keff_from_occupancy(r, inv_of, 'oil')
    check(0 < ke_pp <= 1 and 0 < ke_of <= 1, "keff in [0,1]")

    art = wettability_artifacts(theta, inv_pp, inv_of)
    check('water_wet' in art and 'oil_wet' in art, "Both wettability groups found")


def test_09_dopant_impact():
    """Art. 9 — Pairoys et al.: NaI dopant impact on SCAL."""
    print("\n[9] dopant_impact_scal — X-ray contrast, Amott, recovery")
    from dopant_impact_scal import (
        xray_attenuation, contrast_ratio, amott_index,
        recovery_comparison, imbibition_rate, sor_impact,
    )
    au = xray_attenuation('brine'); ad = xray_attenuation('brine', 0.4)
    check(ad > au, "Doped brine has higher attenuation")

    ao = xray_attenuation('oil')
    cr_u = contrast_ratio(au, ao); cr_d = contrast_ratio(ad, ao)
    check(cr_d > 3*cr_u, "Doping improves contrast > 3x")

    am = amott_index(0.15, 0.10, 0.02, 0.08)
    check(-1 <= am['IAH'] <= 1, "Amott-Harvey in [-1,1]")
    check(am['classification'] in ('water-wet','weakly water-wet','intermediate',
                                    'weakly oil-wet','oil-wet'), "Valid classification")

    pvi = np.linspace(0, 5, 50)
    rc = recovery_comparison(0.55*(1-np.exp(-1.5*pvi)), 0.45*(1-np.exp(-1.2*pvi)), pvi)
    check(rc['significant'], "Significant recovery difference detected")

    rate = imbibition_rate(np.linspace(0.1, 100, 50), np.linspace(0, 0.3, 50))
    check(len(rate) == 50, "Imbibition rate computed")

    si = sor_impact(np.array([.25,.22,.28]), np.array([.32,.35,.30]))
    check(si['increases_Sor'], "NaI increases Sor")


def test_10_dual_porosity():
    """Art. 10 — Wang & Galley: Dual matrix porosity."""
    print("\n[10] dual_porosity_sandstone — dual Brooks-Corey, Land, dual kr")
    from dual_porosity_sandstone import (
        brooks_corey_Pc, dual_porosity_Pc, imbibition_Pc_from_drainage,
        land_trapped_oil, Sot_max_from_Swdra, corey_kr_dual,
    )
    Sw = np.linspace(0.10, 1, 40)
    Pc = brooks_corey_Pc(Sw, 0.05, 2.0, 2.5)
    check(Pc[0] > Pc[-1], "Brooks-Corey Pc decreasing")
    check(Pc[-1] > 0, "Pc positive everywhere")

    PcM, Pcm, Pct = dual_porosity_Pc(Sw, 0.116, 0.075, 0.05, 0.30,
                                       2.0, 15.0, 2.5, 1.5)
    check(Pcm.max() > PcM.max(), "Meso Pc higher than macro Pc")

    Pc_imb = imbibition_Pc_from_drainage(Pct, Sw, 0.05, 0.02, 0.10, 30, 30)
    check(len(Pc_imb) == len(Sw), "Imbibition Pc computed")

    Sot = land_trapped_oil(0.5, 1.5)
    check(0 < Sot < 0.5, "Trapped oil in valid range")

    Sot_m = Sot_max_from_Swdra(0.3, 0.3, 0.05)
    check(0 < Sot_m <= 0.3, "Sot_max scaling valid")

    krw, kro = corey_kr_dual(Sw, 0.116, 0.075, 0.05, 0.30, 0.8, 0.5, 3, 4)
    check(np.all(krw >= 0) and np.all(kro >= 0), "Dual kr non-negative")
    check(krw[-1] > 0, "krw at Sw=1 > 0")


def test_11_mr_bulk_saturation():
    """Art. 11 — Ansaribaranghar et al.: 13C/1H MR bulk saturation."""
    print("\n[11] mr_bulk_saturation — CPMG, 13C oil, 1H water, Dean-Stark")
    from mr_bulk_saturation import (
        MRCalibration, cpmg_decay, oil_volume_from_C13,
        water_volume_from_H1_C13, saturation_workflow,
        compare_with_dean_stark,
    )
    cal = MRCalibration()
    t = np.linspace(0, 500, 100)
    sig = cpmg_decay(t, [60, 40], [100, 10])
    check(sig[0] == 100, "CPMG t=0 signal = sum of components")
    check(sig[-1] < sig[0], "CPMG signal decays")

    Vo = oil_volume_from_C13(3.85, cal)
    check(abs(Vo - 3.5) < 0.5, "Oil volume from 13C reasonable")

    V_oil = 3.5; V_water = 4.2
    H1 = V_oil*cal.H1_signal_per_vol_oil + V_water*cal.H1_signal_per_vol_brine
    C13 = V_oil*cal.C13_signal_per_vol_oil
    sat = saturation_workflow(H1, C13, cal)
    check(abs(sat['Sw'] - V_water/(V_oil+V_water)) < 0.02, "Sw within 2%")
    check(abs(sat['So'] - V_oil/(V_oil+V_water)) < 0.02, "So within 2%")

    ds = compare_with_dean_stark(sat['Sw'], V_water/(V_oil+V_water))
    check(ds['agreement'], "MR agrees with Dean-Stark")


def test_12_mr_saturation_imaging():
    """Art. 12 — Ansaribaranghar et al.: 13C MR saturation imaging."""
    print("\n[12] mr_saturation_imaging — profiles, CEE detection, Dean-Stark")
    from mr_saturation_imaging import (
        ImagingParams, simulate_saturation_profile,
        C13_oil_profile, H1_total_profile,
        water_profile_by_subtraction, detect_capillary_end_effect,
        dean_stark_validation, oil_wet_cee_profile,
    )
    params = ImagingParams()
    n = params.n_pixels
    phi = np.full(n, 0.22)

    Sw = simulate_saturation_profile(n, 0.25, 0.30, 'linear')
    check(len(Sw) == n, "Profile has correct length")
    check(np.all((Sw >= 0) & (Sw <= 1)), "Sw in [0,1]")

    So = 1 - Sw
    c13 = C13_oil_profile(So, phi, params.C13_sensitivity)
    h1 = H1_total_profile(Sw, So, phi)
    check(np.all(c13 >= 0), "13C signal non-negative")
    check(np.all(h1 > c13), "1H signal > 13C signal")

    w_sub = water_profile_by_subtraction(h1, c13, params.C13_sensitivity)
    check(np.allclose(w_sub, Sw * phi, atol=0.001), "Water by subtraction correct")

    cee = detect_capillary_end_effect(So)
    check('has_cee' in cee, "CEE detection returns result")

    ds = dean_stark_validation(Sw.mean(), 0.275)
    check(ds['diff_su'] < 5, "MRI within 5 s.u. of Dean-Stark")

    So_ow = oil_wet_cee_profile(n)
    check(So_ow[-1] > So_ow[0], "Oil-wet CEE: oil accumulates at outlet")


# ──────────────────────────────────────────────────────────────────────────────
def test_all():
    """Run all 12 article tests."""
    global PASS, FAIL
    PASS = FAIL = 0

    print("=" * 72)
    print("  Petrophysics Vol. 66, No. 1 (Feb 2025) — Module Test Suite")
    print("=" * 72)

    tests = [
        test_01_scal_model_ccs,
        test_02_co2_brine_relperm,
        test_03_ss_co2_brine,
        test_04_enhanced_gas_recovery,
        test_05_rev_two_phase,
        test_06_digital_rock_physics,
        test_07_hybrid_drainage,
        test_08_pore_scale_drainage,
        test_09_dopant_impact,
        test_10_dual_porosity,
        test_11_mr_bulk_saturation,
        test_12_mr_saturation_imaging,
    ]

    errors = []
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            FAIL += 1
            errors.append((test_fn.__name__, e))
            print(f"    ✗ EXCEPTION: {e}")
            traceback.print_exc()

    print("\n" + "=" * 72)
    print(f"  RESULTS:  {PASS} passed, {FAIL} failed")
    if errors:
        print(f"  Errors in: {', '.join(n for n, _ in errors)}")
    print("=" * 72)

    return FAIL == 0


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
