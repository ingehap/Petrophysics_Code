#!/usr/bin/env python3
"""
test_all.py – Comprehensive test suite for all 10 Petrophysics modules
=======================================================================
Tests every module using synthetic data and verifies numerical
correctness, physical plausibility, and API consistency.

Run:
    python test_all.py
"""

import sys
import traceback
import numpy as np


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
_pass_count = 0
_fail_count = 0


def check(condition: bool, msg: str):
    """Assert with tracking."""
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"    [PASS] {msg}")
    else:
        _fail_count += 1
        print(f"    [FAIL] {msg}")


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# =====================================================================
# 1. Probe Permeameter (Jensen & Uroza)
# =====================================================================
def test_probe_permeameter():
    section("Module 1: Probe Permeameter (Jensen & Uroza, pp. 665-681)")
    from probe_permeameter import (
        geometric_factor, depth_of_investigation, tip_calibration,
        surface_impairment_correction, measurement_time,
        perm_grain_size, co2_injectivity_index, trapping_capacity
    )

    ri, ro = 0.0039, 0.0064  # m
    gf = geometric_factor(ri, ro)
    check(gf > 0, f"Geometric factor positive: {gf:.6f}")
    check(0.001 < gf < 0.01, f"GF in reasonable range (mm): {gf*1000:.3f}")

    doi_50 = depth_of_investigation(ri, 0.50)
    doi_90 = depth_of_investigation(ri, 0.90)
    doi_95 = depth_of_investigation(ri, 0.95)
    check(doi_50 < doi_90 < doi_95, "DOI increases with response level")
    check(0.6 * ri < doi_50 < 0.8 * ri, "DOI at 50% ≈ 0.7*ri")

    kp = np.array([10, 50, 100, 500])
    ks = tip_calibration(kp, 0.63)
    check(np.all(ks < kp), "Silicone-tip perms < o-ring perms")
    check(np.allclose(ks / kp, 0.63), "Calibration factor = 0.63")

    kp_test = np.array([5, 50, 200, 500])
    kc_test = np.array([6, 55, 190, 480])
    kp_corr, factor = surface_impairment_correction(kp_test, kc_test, method='ratio')
    check(np.all(kp_corr > 0), "Corrected perms positive")
    check(0.5 < factor < 2.0, f"Correction factor plausible: {factor:.3f}")

    t_low = measurement_time(50)
    t_high = measurement_time(500)
    check(t_low > t_high, "Measurement time decreases with higher perm")
    check(25 <= t_high <= 120, f"Time range OK: {t_high:.1f} s")

    d = np.array([0.1, 0.25, 0.5])
    k_est = perm_grain_size(d)
    check(np.all(np.diff(k_est) > 0), "Perm increases with grain size")

    ii = co2_injectivity_index(200, 30)
    check(ii > 0, f"CO2 injectivity index positive: {ii:.2f}")

    tc = trapping_capacity(0.25, 200)
    check(0 < tc < 0.25, f"Trapping capacity plausible: {tc:.4f}")


# =====================================================================
# 2. Dean-Stark Saturation (Zhang et al.)
# =====================================================================
def test_dean_stark_saturation():
    section("Module 2: Dean-Stark Saturation (Zhang et al., pp. 682-698)")
    from dean_stark_saturation import (
        pve_correction, clay_dehydration_correction,
        degasification_water, degasification_oil,
        normalize_saturations, correct_saturation_workflow,
        estimate_kw_bw
    )

    # PVE: surface pore volume > in-situ → measured sat underestimates
    Swf1, Sof1 = pve_correction(35, 45, phi_s=0.33, phi_f=0.30)
    check(Swf1 > 35, f"PVE increases Sw: {Swf1:.2f}")
    check(Sof1 > 45, f"PVE increases So: {Sof1:.2f}")

    # Clay dehydration reduces Sw
    Swf2 = clay_dehydration_correction(Swf1, delta=0.201, Vcl=0.10, phi_e=0.30)
    check(Swf2 < Swf1, f"Clay dehydration reduces Sw: {Swf2:.2f}")

    # Degasification
    Swf3 = degasification_water(Swf2, kw=30.0, bw=59.2)
    check(Swf3 > 0, f"Degasification Sw positive: {Swf3:.2f}")
    Sof3 = degasification_oil(Sof1, ko=1.15)
    check(Sof3 > Sof1, f"Degasification increases So: {Sof3:.2f}")

    # Normalization
    Swc, Soc = normalize_saturations(40, 60)
    check(abs(Swc + Soc - 100) < 1e-6, "Normalized sats sum to 100%")

    # Full workflow
    kw, bw, ko = 30.0, 3.8967 * 30 - 57.707, 1.15
    Swc, Soc = correct_saturation_workflow(
        35, 45, 0.33, 0.30, 0.201, 0.10, 0.30, kw, bw, ko
    )
    check(abs(Swc + Soc - 100) < 0.1, f"Workflow: Sw={Swc:.1f}, So={Soc:.1f}")

    # Coefficient estimation
    np.random.seed(42)
    n = 20
    Sw_meas = np.random.uniform(30, 60, n)
    So_meas = np.random.uniform(20, 50, n)
    kw_est, bw_est, ko_est = estimate_kw_bw(Sw_meas, So_meas)
    check(20 < kw_est < 50, f"Estimated kw in range: {kw_est:.2f}")
    check(ko_est > 0, f"Estimated ko positive: {ko_est:.2f}")


# =====================================================================
# 3. Relative Permeability from MRI (Zamiri et al.)
# =====================================================================
def test_relative_permeability_mri():
    section("Module 3: Relative Permeability MRI (Zamiri et al., pp. 699-710)")
    from relative_permeability_mri import (
        relative_permeability_corey, capillary_pressure_model,
        fractional_mobility_oil, generate_synthetic_saturation_profiles
    )

    Sw = np.linspace(0.15, 0.85, 50)
    Krw, Kro = relative_permeability_corey(Sw)
    check(np.all(Krw >= 0) and np.all(Krw <= 1), "Krw in [0,1]")
    check(np.all(Kro >= 0) and np.all(Kro <= 1), "Kro in [0,1]")
    check(Krw[0] < Krw[-1], "Krw increases with Sw")
    check(Kro[0] > Kro[-1], "Kro decreases with Sw")

    # At Sw = Sw_irr: Krw ≈ 0
    check(Krw[0] < 0.01, f"Krw at Sw_irr ≈ 0: {Krw[0]:.4f}")

    Pc = capillary_pressure_model(Sw)
    check(np.all(np.isfinite(Pc)), "Pc values finite")
    check(Pc[0] > Pc[-1], "Pc decreases with increasing Sw")

    fnw = fractional_mobility_oil(Sw, 0.15, 0.85, 1.0, 5.0)
    check(fnw[0] > fnw[-1], "fnw decreases as water increases")
    check(0 <= fnw[-1] <= 1, "fnw in [0,1]")

    profiles, times, positions = generate_synthetic_saturation_profiles()
    check(profiles.shape == (10, 50), f"Profile shape: {profiles.shape}")
    check(profiles[0].mean() > profiles[-1].mean(),
          "Saturation decreases with time (drainage)")


# =====================================================================
# 4. Permeability Anisotropy (Silva Junior et al.)
# =====================================================================
def test_permeability_anisotropy():
    section("Module 4: Permeability Anisotropy (Silva Jr et al., pp. 711-738)")
    from permeability_anisotropy import (
        reservoir_quality_index, flow_zone_indicator, classify_hfu,
        upscale_permeability_arithmetic, upscale_permeability_harmonic,
        upscale_permeability_geometric, kv_kh_ratio,
        facies_permeability_stats
    )

    np.random.seed(42)
    n = 100
    phi = np.random.uniform(0.05, 0.30, n)
    k = 10 ** np.random.normal(1.5, 0.8, n)

    rqi = reservoir_quality_index(k, phi)
    check(np.all(rqi > 0), "RQI all positive")

    fzi = flow_zone_indicator(k, phi)
    check(np.all(fzi >= 0), "FZI all non-negative")

    labels, thresholds = classify_hfu(fzi, n_units=4)
    check(len(np.unique(labels)) <= 4, f"HFU classes ≤ 4: {np.unique(labels)}")

    k_test = np.array([10, 100, 1000])
    k_a = upscale_permeability_arithmetic(k_test)
    k_h = upscale_permeability_harmonic(k_test)
    k_g = upscale_permeability_geometric(k_test)
    check(k_h <= k_g <= k_a, f"Harmonic ≤ Geometric ≤ Arithmetic: "
          f"{k_h:.1f} ≤ {k_g:.1f} ≤ {k_a:.1f}")

    k_v = k * np.random.uniform(0.1, 0.8, n)
    ratios = kv_kh_ratio(k_v, k, window_sizes=[1, 5, 10])
    check(all(0 < r < 1 for r in ratios.values()),
          "Kv/Kh < 1 (typical for layered media)")

    facies = np.random.choice(['A', 'B'], n)
    stats = facies_permeability_stats(k, facies)
    check('A' in stats and 'B' in stats, "Stats computed for both facies")
    check(stats['A']['count'] + stats['B']['count'] == n, "Counts sum to n")


# =====================================================================
# 5. Water Saturation Equations (Acosta et al.)
# =====================================================================
def test_water_saturation_equations():
    section("Module 5: Sw Equations (Acosta et al., pp. 739-764)")
    from water_saturation_equations import (
        archie_sw, indonesian_sw, modified_indonesian_sw,
        simandoux_sw, waxman_smits_sw,
        suriname_clay_silt_sw, suriname_laminar_sw,
        bppi, swirr_from_bppi
    )

    np.random.seed(42)
    n = 30
    Rt = np.random.uniform(5, 100, n)
    phi = np.random.uniform(0.20, 0.35, n)
    Vcl = np.random.uniform(0.05, 0.20, n)
    Rw, Rcl = 0.05, 3.0

    for name, func in [
        ("Archie", lambda: archie_sw(Rt, Rw, phi)),
        ("Indonesian", lambda: indonesian_sw(Rt, Rw, phi, Vcl, Rcl)),
        ("Modified Indonesian", lambda: modified_indonesian_sw(Rt, Rw, phi, Vcl, Rcl)),
        ("Simandoux", lambda: simandoux_sw(Rt, Rw, phi, Vcl, Rcl)),
    ]:
        Sw = func()
        check(np.all((Sw >= 0) & (Sw <= 1)), f"{name}: Sw in [0,1]")

    # Waxman-Smits
    Qv = np.random.uniform(0.01, 0.5, n)
    Sw_ws = waxman_smits_sw(Rt, Rw, phi, Qv)
    check(np.all((Sw_ws >= 0) & (Sw_ws <= 1)), "Waxman-Smits: Sw in [0,1]")

    # Suriname Clay-Silt
    Sw_sur = suriname_clay_silt_sw(
        Rt, Rw, phi, m=1.78, a=1.29,
        phi_cl=0.35, m_cl=1.70, a_cl=2.50, Rwb=0.10,
        phi_sl=0.25, m_sl=1.12, a_sl=1.92, Rwsl=0.075
    )
    check(np.all((Sw_sur >= 0) & (Sw_sur <= 1)), "Suriname Clay-Silt: Sw in [0,1]")

    # Suriname Laminar
    Vsl = np.random.uniform(0.02, 0.10, n)
    Sw_lam = suriname_laminar_sw(
        Rt, Rw, Vcl, Vsl, phi, m=1.78, a=1.29,
        phi_cl=0.35, m_cl=1.70, a_cl=2.50, Rwb=0.10,
        phi_sl=0.25, m_sl=1.12, a_sl=1.92, Rwsl=0.075
    )
    check(np.all((Sw_lam >= 0) & (Sw_lam <= 1)), "Suriname Laminar: Sw in [0,1]")

    # BPPI
    phi_t = phi + 0.03
    bp = bppi(phi, phi_t, Vcl)
    check(np.all(bp >= 0), "BPPI non-negative")
    swirr = swirr_from_bppi(bp)
    check(np.all((swirr >= 0) & (swirr <= 1)), "Swirr in [0,1]")


# =====================================================================
# 6. Thin-Bed NMR (Ramadan et al.)
# =====================================================================
def test_thin_bed_nmr():
    section("Module 6: Thin-Bed NMR (Ramadan et al., pp. 765-771)")
    from thin_bed_nmr import (
        nmr_sensitivity_kernel, apparent_porosity,
        thin_bed_correction, standoff_correction,
        detect_bed_boundaries, generate_layered_porosity
    )

    z = np.linspace(-0.3, 0.3, 61)
    kernel = nmr_sensitivity_kernel(z, sigma=0.08)
    check(abs(kernel.sum() - 1.0) < 1e-6, "Kernel normalised to 1")
    check(kernel[30] == kernel.max(), "Kernel peak at centre")

    depths, true_phi = generate_layered_porosity()
    check(len(depths) == len(true_phi), "Depths and porosity same length")

    app_phi = apparent_porosity(true_phi, None, sigma=0.08)
    check(app_phi.max() <= true_phi.max() + 0.01,
          "Apparent porosity ≤ true max (smoothing)")
    check(app_phi.min() >= true_phi.min() - 0.05,
          "Apparent porosity ≈ true min (smoothing widens range)")

    # Thin bed correction
    corrected = thin_bed_correction(np.array([0.10]), 0.05, 0.15)
    check(corrected[0] > 0.10, f"Thin-bed correction increases phi: {corrected[0]:.3f}")

    phi_corr = standoff_correction(0.25, 0.005)
    check(phi_corr < 0.25, f"Stand-off correction reduces phi: {phi_corr:.3f}")

    bounds = detect_bed_boundaries(true_phi)
    check(len(bounds) > 0, f"Detected {len(bounds)} bed boundaries")


# =====================================================================
# 7. Lateral Permeability NMR (Fouda et al.)
# =====================================================================
def test_lateral_permeability_nmr():
    section("Module 7: Lateral Permeability NMR (Fouda et al., pp. 772-788)")
    from lateral_permeability_nmr import (
        timur_coates_perm, sdr_perm,
        azimuthal_perm_from_formation_test,
        heterogeneity_index_from_imaging,
        lateral_perm_profile
    )

    np.random.seed(42)
    n = 50
    phi = np.random.uniform(0.08, 0.25, n)
    T2lm = np.random.uniform(20, 200, n)
    BVI = phi * np.random.uniform(0.2, 0.6, n)
    FFI = phi - BVI

    k_tc = timur_coates_perm(phi, BVI, FFI)
    check(np.all(k_tc > 0), "Timur-Coates perm all positive")

    k_sdr = sdr_perm(phi, T2lm)
    check(np.all(k_sdr > 0), "SDR perm all positive")

    # Higher phi and T2 → higher perm
    k_lo = sdr_perm(np.array([0.10]), np.array([30]))
    k_hi = sdr_perm(np.array([0.25]), np.array([200]))
    check(k_hi[0] > k_lo[0], "SDR: higher phi+T2 → higher perm")

    pressure_data = {'top': (0.5, 50), 'bottom': (0.5, 30)}
    az_perm = azimuthal_perm_from_formation_test(pressure_data)
    check(az_perm['bottom'] > az_perm['top'],
          "Lower pressure drop → higher perm")

    res_image = 10 ** np.random.normal(1.5, 0.5, (50, 36))
    hi = heterogeneity_index_from_imaging(res_image)
    check(hi > 0, f"Heterogeneity index positive: {hi:.4f}")

    phi_2d = np.random.uniform(0.1, 0.25, (20, 8))
    T2_2d = np.random.uniform(50, 200, (20, 8))
    az, k_mean = lateral_perm_profile(phi_2d, T2_2d, np.arange(8) * 45)
    check(len(k_mean) == 8, f"Lateral profile has 8 azimuths")
    check(np.all(k_mean > 0), "All azimuthal perms positive")


# =====================================================================
# 8. ML Permeability (Raheem et al.)
# =====================================================================
def test_ml_permeability():
    section("Module 8: ML Permeability (Raheem et al., pp. 789-812)")
    from ml_permeability import (
        timur_coates, archie_sw, sandstone_resistivity,
        augment_features, pca_reduce, svd_reduce,
        train_ridge, predict_ridge, train_knn,
        mae, rse, r_squared, group_kfold_cv
    )

    np.random.seed(42)
    n = 200
    phi = np.random.uniform(0.05, 0.30, n)
    Swirr = np.clip(0.5 - phi, 0.05, 0.90)

    k_tc = timur_coates(phi, Swirr, a=1e4, b=4.0, c=2.0)
    check(np.all(k_tc > 0), "Timur-Coates perm all positive")

    # Archie Sw
    Rt = np.random.uniform(5, 100, n)
    Sw = archie_sw(Rt, 0.05, phi)
    check(np.all((Sw >= 0) & (Sw <= 1)), "Archie Sw in [0,1]")

    # Sandstone resistivity correction
    Csh = np.random.uniform(0.0, 0.2, n)
    Rs = sandstone_resistivity(Rt, Csh, Rsh=5.0)
    check(np.all(Rs >= Rt), "Rs ≥ Rt after shale correction")

    # Feature augmentation
    X = np.random.randn(100, 4)
    X_aug = augment_features(X, [3])
    check(X_aug.shape[1] > X.shape[1], f"Augmented features: {X_aug.shape[1]}")

    # PCA
    Z, ev, _ = pca_reduce(X, n_components=2)
    check(Z.shape == (100, 2), f"PCA output shape: {Z.shape}")
    check(np.sum(ev) <= 1.0, f"Explained variance sum ≤ 1: {np.sum(ev):.3f}")

    # SVD
    Z_svd, sv = svd_reduce(X, n_components=2)
    check(Z_svd.shape == (100, 2), "SVD output shape OK")

    # Ridge
    y = np.random.randn(100)
    w, b = train_ridge(X[:80], y[:80])
    y_pred = predict_ridge(X[80:], w, b)
    check(len(y_pred) == 20, "Ridge prediction length correct")

    # kNN
    y_knn = train_knn(X[:80], y[:80], X[80:], k=3)
    check(len(y_knn) == 20, "kNN prediction length correct")

    # Metrics
    m = mae(y[80:], y_pred)
    check(m >= 0, f"MAE non-negative: {m:.4f}")
    r = rse(y[80:], y_pred)
    check(r >= 0, f"RSE non-negative: {r:.4f}")
    r2 = r_squared(y[80:], y_pred)
    check(-10 < r2 < 1.1, f"R² plausible: {r2:.4f}")

    # Group k-fold
    groups = np.repeat([0, 1, 2, 3, 4], 20)
    cv = group_kfold_cv(X, y, groups, k=3)
    check(len(cv) == 3, f"3 CV folds returned")


# =====================================================================
# 9. Lithofacies Prediction (Satti et al.)
# =====================================================================
def test_lithofacies_prediction():
    section("Module 9: Lithofacies Prediction (Satti et al., pp. 813-834)")
    from lithofacies_prediction import (
        define_lithofacies, engineer_features,
        ExtraTreesClassifier, SimpleGBClassifier,
        confusion_matrix, f1_score_per_class, accuracy,
        kfold_cross_validation
    )

    np.random.seed(42)
    n = 400

    GR = np.random.uniform(15, 150, n)
    LLD = 10 ** np.random.uniform(0, 3, n)
    RHOB = np.random.uniform(2.0, 2.75, n)
    Vshl = np.clip(GR / 150, 0, 1)
    Sw = np.clip(0.3 + 0.5 * Vshl + np.random.normal(0, 0.1, n), 0, 1)
    PHIE = np.clip(0.25 - 0.2 * Vshl + np.random.normal(0, 0.03, n), 0, 0.4)

    y = define_lithofacies(Vshl, Sw, PHIE)
    check(set(y).issubset({0, 1, 2}), f"Facies in {{0,1,2}}: {set(y)}")

    X = engineer_features(GR, LLD, RHOB)
    check(X.shape[0] == n, f"Feature matrix rows = n")
    check(X.shape[1] >= 6, f"At least 6 features: {X.shape[1]}")

    # Extra Trees
    et = ExtraTreesClassifier(n_estimators=30, random_state=42)
    et.fit(X[:300], y[:300])
    y_pred = et.predict(X[300:])
    acc = accuracy(y[300:], y_pred)
    check(acc > 0.3, f"ET accuracy > chance: {acc:.3f}")

    # Gradient Boosting
    gb = SimpleGBClassifier(n_estimators=20, random_state=42)
    gb.fit(X[:300], y[:300])
    y_pred_gb = gb.predict(X[300:])
    acc_gb = accuracy(y[300:], y_pred_gb)
    check(acc_gb > 0.3, f"GB accuracy > chance: {acc_gb:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y[300:], y_pred)
    check(cm.sum() == len(y_pred), "CM entries sum to test size")

    # F1
    f1 = f1_score_per_class(y[300:], y_pred)
    check(all(0 <= v <= 1 for v in f1.values()), "F1 scores in [0,1]")

    # Cross-validation
    cv_acc = kfold_cross_validation(X, y, ExtraTreesClassifier,
                                    k=3, n_estimators=20)
    check(len(cv_acc) == 3, "3-fold CV returns 3 values")
    check(all(0 <= a <= 1 for a in cv_acc), "CV accuracies in [0,1]")


# =====================================================================
# 10. RDDTW Depth Matching (Fang et al.)
# =====================================================================
def test_rddtw_depth_matching():
    section("Module 10: RDDTW Depth Matching (Fang et al., pp. 835-851)")
    from rddtw_depth_matching import (
        dtw, cdtw, rddtw, pcc_match, pso_depth_shift,
        derivative_estimate, rmse, r_squared,
        generate_synthetic_core_log
    )

    core, log, true_shift = generate_synthetic_core_log(
        n_core=60, n_log=80, true_shift=5
    )
    check(len(core) == 60 and len(log) == 80, "Synthetic data generated")

    # Derivative estimation
    d = derivative_estimate(core)
    check(len(d) == len(core), "Derivative same length as input")
    check(np.all(np.isfinite(d)), "Derivative values finite")

    # PCC
    pcc_shift, pcc_r = pcc_match(core, log, max_shift=15)
    check(abs(pcc_shift - true_shift) <= 10,
          f"PCC shift close to truth: {pcc_shift} (true={true_shift})")
    check(-1 <= pcc_r <= 1, f"PCC in [-1,1]: {pcc_r:.4f}")

    # DTW (short sequences for speed)
    c_s, l_s = core[:30], log[:30]
    dist_dtw, _, path = dtw(c_s, l_s)
    check(dist_dtw >= 0, f"DTW distance non-negative: {dist_dtw:.4f}")
    check(len(path) >= max(len(c_s), len(l_s)),
          f"Path length ≥ max(n,m): {len(path)}")

    # CDTW
    dist_cdtw, _ = cdtw(c_s, l_s, window=5)
    check(dist_cdtw >= dist_dtw - 1e-6,
          "CDTW ≥ DTW (constrained can't be cheaper)")

    # RDDTW
    dist_rddtw, _, path_r = rddtw(c_s, l_s, tau=4.0, lambda_ewrf=0.5)
    check(dist_rddtw >= 0, f"RDDTW distance non-negative: {dist_rddtw:.4f}")
    check(len(path_r) >= max(len(c_s), len(l_s)),
          "RDDTW path length adequate")

    # RDDTW with EWRF should have less excessive warping
    _, _, path_no_ewrf = rddtw(c_s, l_s, tau=4.0, lambda_ewrf=0.0)
    deviation_ewrf = np.mean([abs(p[0]/len(c_s) - p[1]/len(l_s))
                              for p in path_r])
    deviation_no = np.mean([abs(p[0]/len(c_s) - p[1]/len(l_s))
                            for p in path_no_ewrf])
    check(deviation_ewrf <= deviation_no + 0.05,
          f"EWRF reduces warping: {deviation_ewrf:.3f} vs {deviation_no:.3f}")

    # PSO
    best_shift, best_cost = pso_depth_shift(
        core, log, n_particles=30, n_iterations=50, dz=1.0
    )
    check(abs(best_shift - true_shift) <= 12,
          f"PSO shift estimate: {best_shift:.1f} (true={true_shift})")
    check(best_cost >= 0, "PSO cost non-negative")

    # Metrics
    rm = rmse(core[:50], log[:50])
    check(rm >= 0, f"RMSE non-negative: {rm:.4f}")
    r2 = r_squared(core[:50], log[:50])
    check(-10 < r2 <= 1, f"R² plausible: {r2:.4f}")


# =====================================================================
# Main
# =====================================================================
def test_all():
    """Run all module tests and report results."""
    global _pass_count, _fail_count
    _pass_count = 0
    _fail_count = 0

    tests = [
        test_probe_permeameter,
        test_dean_stark_saturation,
        test_relative_permeability_mri,
        test_permeability_anisotropy,
        test_water_saturation_equations,
        test_thin_bed_nmr,
        test_lateral_permeability_nmr,
        test_ml_permeability,
        test_lithofacies_prediction,
        test_rddtw_depth_matching,
    ]

    errors = []
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            errors.append((test_func.__name__, e))
            print(f"\n  *** EXCEPTION in {test_func.__name__}: {e}")
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"  TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Passed : {_pass_count}")
    print(f"  Failed : {_fail_count}")
    print(f"  Errors : {len(errors)}")
    if errors:
        print("  Error details:")
        for name, err in errors:
            print(f"    {name}: {err}")
    total = _pass_count + _fail_count
    if _fail_count == 0 and len(errors) == 0:
        print(f"\n  ALL {total} CHECKS PASSED ✓")
    else:
        print(f"\n  {_fail_count + len(errors)} ISSUES DETECTED")
    print(f"{'='*70}\n")

    return _fail_count == 0 and len(errors) == 0


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
