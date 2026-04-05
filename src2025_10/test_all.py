#!/usr/bin/env python3
"""
test_all.py – Master test suite for all 11 Petrophysics Vol. 66 No. 5 modules.

Each test function generates synthetic data and validates the computational
outputs of one article module.  Run standalone:
    python test_all.py

Ref: Petrophysics, Vol. 66, No. 5 (October 2025), SPWLA.
"""

import sys
import traceback
import numpy as np

# ── module imports ──────────────────────────────────────────────────────────
import a1_log_interpretation as a1
import a2_damage_model as a2
import a3_youngs_modulus as a3
import a4_multimodal_permeability as a4
import a5_missing_log_prediction as a5
import a6_carbonate_petrophysics as a6
import a7_nmr_porosity_correction as a7
import a8_digital_core_conductivity as a8
import a9_cementing_quality as a9
import a10_neutron_log_shale as a10
import a11_fracture_identification as a11

PASS = 0
FAIL = 0


def _check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"    ✗ FAIL: {msg}")


# ============================================================================
# Article 1 – Log Interpretation (Proestakis & Fabricius)
# ============================================================================
def test_article_1():
    print("▶ Article 1: Log Interpretation for Petrophysical & Elastic Properties")

    # --- Kozeny permeability -------------------------------------------------
    phi = np.array([0.10, 0.20, 0.30])
    sb = np.array([1e5, 8e4, 5e4])
    k = a1.kozeny_permeability(phi, sb)
    _check(np.all(k > 0), "Kozeny permeability must be positive")
    _check(k[2] > k[0], "Higher porosity should give higher permeability")

    # --- Round-trip: permeability -> surface area -> permeability -------------
    skz = a1.kozeny_surface_area(phi, k)
    k2 = a1.kozeny_permeability(phi, skz)
    _check(np.allclose(k, k2, rtol=1e-6), "Kozeny round-trip consistency")

    # --- Archie m from SKZ ---------------------------------------------------
    m = a1.archie_m_from_skz(skz)
    _check(np.all(m >= 1.5), "Archie m should be >= 1.5 for real rocks")
    skz_back = a1.skz_from_archie_m(m)
    _check(np.allclose(skz, skz_back, rtol=1e-6), "m-SKZ round-trip")

    # --- Formation conductivity (parallel conduction) ------------------------
    sigma_w = 5.0
    phi_total = 0.25
    sigma_o = sigma_w * phi_total ** 1.5  # assume no bound water
    phi_w = a1.free_water_porosity(phi_total, sigma_o, sigma_w)
    _check(abs(phi_w - phi_total) < 0.01, "Free-water porosity ~ total when no AL")

    phi_al = a1.bound_water_porosity(phi_total, phi_w)
    sir = a1.irreducible_water_saturation(phi_al, phi_total)
    _check(sir >= 0.0, "Irreducible water saturation >= 0")

    # --- Elastic properties (iso-frame) --------------------------------------
    ks, gs = 70e9, 30e9
    kfl = 2.3e9
    IF = 0.5
    k_if = a1.isoframe_bulk_modulus(ks, gs, kfl, 0.20, IF)
    g_if = a1.isoframe_shear_modulus(ks, gs, kfl, 0.20, IF)
    M = a1.pwave_modulus(k_if, g_if)
    _check(M > 0, "P-wave modulus must be positive")

    alpha = a1.biot_coefficient(k_if * 0.6, ks)
    _check(0 < alpha < 1, "Biot coefficient should be in (0, 1)")

    eps = a1.vertical_elastic_strain(30e6, 15e6, M * 0.6, alpha)
    _check(eps > 0, "Elastic strain must be positive under compression")

    # --- Arps temperature correction -----------------------------------------
    s1 = 5.0
    s2 = a1.arps_temperature_correction(s1, 80, 25)
    _check(s2 < s1, "Conductivity should decrease when cooled")

    print(f"    ✓ {PASS} checks passed\n")


# ============================================================================
# Article 2 – Damage Constitutive Model (Liu et al.)
# ============================================================================
def test_article_2():
    global PASS, FAIL
    p0 = PASS
    print("▶ Article 2: M-integral Damage Constitutive Model")

    Ei, Ed, Et = 25e9, 20e9, 18e9
    D0i = a2.initial_damage(Ed, Ei)
    _check(0 < D0i < 1, "Initial damage in (0,1)")

    psi_t, psi_p = 4.5, 5.84
    D0t = a2.local_microscopic_damage(Ed, Ei, Et, psi_t, psi_p)
    _check(D0t >= 0, "Microscopic damage >= 0")

    D0 = a2.total_initial_damage(D0i, D0t)
    _check(D0 >= D0i, "Total damage >= initial damage")

    strain = np.linspace(0, 0.015, 300)
    sigma, D = a2.stress_strain_m_integral(strain, Ei, Ed, Et,
                                           psi_t, psi_p, 3.0, 0.008)
    _check(np.all(sigma >= 0), "Stress must be non-negative")
    peak_idx = np.argmax(sigma)
    _check(peak_idx > 0, "Peak stress should not be at strain=0")
    _check(D[-1] > D[0], "Damage should increase with strain")

    # M-integral on a trivial path
    N = 8
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    coords = np.column_stack([np.cos(theta), np.sin(theta)]) * 0.01
    normals = np.column_stack([np.cos(theta), np.sin(theta)])
    W = np.ones(N) * 100.0
    stress = np.zeros((N, 2, 2))
    disp_grad = np.zeros((N, 2, 2))
    M_val = a2.m_integral_2d(coords, normals, W, stress, disp_grad)
    _check(isinstance(M_val, float), "M-integral returns float")

    print(f"    ✓ {PASS - p0} checks passed\n")


# ============================================================================
# Article 3 – Young's Modulus Prediction (Al-Dousari et al.)
# ============================================================================
def test_article_3():
    global PASS, FAIL
    p0 = PASS
    print("▶ Article 3: Young's Modulus Without Shear-Wave Traveltime")

    dtc = np.array([80, 100, 120, 140])
    rho = np.array([2.4, 2.2, 2.1, 1.95])
    vsh = np.array([0.05, 0.20, 0.40, 0.80])

    es = a3.predict_es_regression(dtc, vsh, rho)
    _check(len(es) == 4, "Output length matches input")
    _check(es[0] > es[-1], "Es should decrease with increasing dtc and Vsh")

    # Steiber Vsh
    igr = np.array([0.0, 0.5, 1.0])
    vsh_st = a3.steiber_vsh(igr)
    _check(vsh_st[0] < 0.01, "Vsh ~ 0 for clean sand")
    _check(vsh_st[-1] < 1.0, "Steiber Vsh < 1.0 even at IGR=1")

    # FZI / DRT
    fzi = a3.flow_zone_indicator(np.array([100, 1]), np.array([0.20, 0.08]))
    drt = a3.discrete_rock_type(fzi)
    _check(drt[0] > drt[1], "Higher-quality rock -> higher DRT")

    # Abbas model
    es_a = a3.predict_es_abbas(np.array([0.05, 0.30]))
    _check(es_a[0] > es_a[1], "Lower porosity -> higher Es (Abbas)")

    # BPNN
    X = np.random.randn(50, 3)
    y = np.random.rand(50)
    nn = a3.SimpleBPNN(3, 6, 0.01)
    nn.fit(X, y, epochs=50)
    pred = nn.predict(X)
    _check(len(pred) == 50, "BPNN prediction shape OK")

    print(f"    ✓ {PASS - p0} checks passed\n")


# ============================================================================
# Article 4 – Multimodal (ETIN) Permeability (Fang et al.)
# ============================================================================
def test_article_4():
    global PASS, FAIL
    p0 = PASS
    print("▶ Article 4: ETIN Multimodal Permeability Prediction")

    np.random.seed(0)
    N = 10
    seqs = [np.random.randn(15, 4) for _ in range(N)]
    imgs = [np.random.randn(32, 1) for _ in range(N)]
    txts = [np.random.randn(3) for _ in range(N)]

    model = a4.ETIN(seq_len=15, seq_feat=4, img_len=32, img_chan=1,
                    text_dim=3, hidden=4)
    preds = model.predict(seqs, imgs, txts)
    _check(preds.shape == (N,), "Predictions shape matches N samples")
    _check(np.all(np.isfinite(preds)), "All predictions are finite")

    # Tensor interaction dimensions
    f1, f2, f3 = model.extract_features(seqs[0], imgs[0], txts[0])
    fusion = model.tensor_interaction(f1, f2, f3)
    h = model.hidden
    expected_dim = 3 * h + 3 * h ** 2 + h ** 3
    _check(fusion.shape[0] == expected_dim,
           f"Fusion vector dim = {expected_dim}")

    print(f"    ✓ {PASS - p0} checks passed\n")


# ============================================================================
# Article 5 – Missing Well-Log Prediction (Oppong et al.)
# ============================================================================
def test_article_5():
    global PASS, FAIL
    p0 = PASS
    print("▶ Article 5: Hybrid U-Net + LSTM Missing-Log Prediction")

    np.random.seed(1)
    n_depths = 100
    n_logs = 4
    X = np.random.randn(n_depths, n_logs)
    y_true = np.random.randn(n_depths)

    model = a5.HybridUNetLSTM(n_input_logs=n_logs, hidden=4)
    y_pred = model.predict(X)
    _check(y_pred.shape == (n_depths,), "Output shape matches depth count")
    _check(np.all(np.isfinite(y_pred)), "All predictions finite")

    r = a5.rmse(y_true, y_pred)
    _check(r >= 0, "RMSE is non-negative")

    r2 = a5.r_squared(y_true, y_pred)
    _check(isinstance(r2, float), "R-squared is a float")

    print(f"    ✓ {PASS - p0} checks passed\n")


# ============================================================================
# Article 6 – Carbonate Petrophysics Workflow (Fadhil)
# ============================================================================
def test_article_6():
    global PASS, FAIL
    p0 = PASS
    print("▶ Article 6: Carbonate Petrophysics Workflow")

    gr = np.array([20, 60, 100])
    vsh_lin = a6.vshale_linear(gr, 20, 100)
    _check(abs(vsh_lin[0]) < 0.01, "Clean Vsh ~ 0")
    _check(abs(vsh_lin[-1] - 1.0) < 0.01, "Shale Vsh ~ 1")

    vsh_lar = a6.vshale_larionov_older(gr, 20, 100)
    _check(np.all(vsh_lar <= vsh_lin), "Larionov Vsh <= linear Vsh")

    dphi = a6.density_porosity(np.array([2.2, 2.5]), 2.71, 1.0)
    _check(dphi[0] > dphi[1], "Lower density -> higher porosity")

    # Water saturation models
    rt = np.array([10, 50, 200])
    phi = np.array([0.15, 0.15, 0.15])
    sw_a = a6.sw_archie(rt, 0.03, phi)
    _check(sw_a[0] > sw_a[-1], "Higher Rt -> lower Sw")
    _check(np.all((sw_a >= 0) & (sw_a <= 1)), "Sw in [0,1]")

    sw_i = a6.sw_indonesian(rt, 0.03, phi, vsh=np.array([0.1]*3), rsh=5.0)
    _check(np.all((sw_i >= 0) & (sw_i <= 1)), "Indonesian Sw in [0,1]")

    sw_s = a6.sw_simandoux(rt, 0.03, phi, vsh=np.array([0.1]*3), rsh=5.0)
    _check(np.all((sw_s >= 0) & (sw_s <= 1)), "Simandoux Sw in [0,1]")

    # Permeability
    k = a6.perm_timur(np.array([0.15, 0.25]), np.array([0.3, 0.2]))
    _check(k[1] > k[0], "Higher phi, lower Swir -> higher perm")

    # Net pay
    flags = a6.net_pay_flag(
        phi=np.array([0.03, 0.12, 0.20]),
        sw=np.array([0.8, 0.3, 0.2]),
        vsh=np.array([0.6, 0.1, 0.05])
    )
    _check(flags[0] == 0 and flags[2] == 1, "Net-pay flagging logic")

    print(f"    ✓ {PASS - p0} checks passed\n")


# ============================================================================
# Article 7 – NMR Porosity Correction (Zhu et al.)
# ============================================================================
def test_article_7():
    global PASS, FAIL
    p0 = PASS
    print("▶ Article 7: NMR T2 Spectrum & Porosity Correction")

    minerals = {'quartz': 0.40, 'illite': 0.15, 'chlorite': 0.08,
                'ankerite': 0.10, 'calcite': 0.20, 'pyrite': 0.03}
    chi_g = a7.rock_magnetic_susceptibility(minerals)
    _check(chi_g > 0, "Rock with iron minerals should have positive chi")

    dchi = a7.delta_chi(chi_g)
    _check(dchi > 0, "Delta-chi positive for iron-rich rock")

    # T2 calculation
    B0, r = 0.047, 1e-6
    G = a7.internal_gradient(B0, dchi * 1e-6, r)
    _check(G > 0, "Internal gradient > 0")

    D, gamma, TE = 2.3e-9, 2.675e8, 100e-6
    T2D = a7.t2_diffusion(D, gamma, G, TE)
    _check(T2D > 0, "T2D > 0")

    rho2, sv = 5e-6, 2e6
    T2 = a7.t2_total(rho2, sv, D, gamma, G, TE)
    _check(0 < T2 < 10, "T2 in reasonable range (seconds)")

    # Decay correction
    echo_times = np.arange(1, 100) * TE
    M = np.exp(-echo_times / T2)
    M_corr = a7.correct_t2_decay(M, echo_times, T2D)
    _check(np.all(M_corr >= M), "Corrected signal >= measured signal")

    # Porosity correction model
    dphi = a7.porosity_correction(15.0, 10.0, 8.0)
    _check(dphi > 0, "Porosity correction is positive for iron-bearing shale")

    phi_corr = a7.corrected_porosity(2.5, dphi)
    _check(phi_corr > 2.5, "Corrected porosity > NMR porosity")

    print(f"    ✓ {PASS - p0} checks passed\n")


# ============================================================================
# Article 8 – Digital Core Conductivity (Feng & Zou)
# ============================================================================
def test_article_8():
    global PASS, FAIL
    p0 = PASS
    print("▶ Article 8: Digital-Core Conductivity Simulation")

    phi = np.array([0.05, 0.15, 0.30])
    F = a8.formation_factor(phi, a=1.0, m=2.0)
    _check(F[0] > F[-1], "F decreases with increasing porosity")

    # Directional anisotropy
    Fx = a8.formation_factor_directional(0.10, 'x')
    Fz = a8.formation_factor_directional(0.10, 'z')
    _check(Fx != Fz, "Directional F values differ (anisotropy)")

    # Resistivity index
    sw = np.array([0.3, 0.5, 1.0])
    I = a8.resistivity_index(sw, n=2.0)
    _check(I[0] > I[-1], "I decreases with increasing Sw")
    _check(abs(I[-1] - 1.0) < 0.01, "I = 1 at Sw = 1")

    # Bimodal
    I_bi = a8.resistivity_index_bimodal(np.array([0.3, 0.6]))
    n_eff = a8.bimodal_saturation_exponent(np.array([0.3, 0.6]))
    _check(n_eff[0] > n_eff[1], "n is higher below breakpoint Sw")

    # Wettability
    n_ww = a8.saturation_exponent_wettability(60)
    n_ow = a8.saturation_exponent_wettability(120)
    _check(n_ow > n_ww, "Oil-wet n > water-wet n")

    # Salinity -> Rw
    rw1 = a8.rw_from_salinity(10000, 25)
    rw2 = a8.rw_from_salinity(100000, 25)
    _check(rw1 > rw2, "Higher salinity -> lower Rw")

    # Digital core
    core, phi_actual = a8.generate_digital_core(20, 20, 20, 0.15)
    _check(0.05 < phi_actual < 0.30, "Digital core porosity in range")

    print(f"    ✓ {PASS - p0} checks passed\n")


# ============================================================================
# Article 9 – Cementing Quality (Pan et al.)
# ============================================================================
def test_article_9():
    global PASS, FAIL
    p0 = PASS
    print("▶ Article 9: Slip Interface Cementing Quality Evaluation")

    # Amplitude vs stiffness
    amp_free = a9.relative_amplitude_slip(0.1)
    amp_bond = a9.relative_amplitude_slip(1e4)
    _check(amp_free > amp_bond, "Free casing amplitude > bonded")

    # Monotonicity
    eta = np.logspace(-1, 4, 50)
    amps = a9.relative_amplitude_slip(eta)
    _check(np.all(np.diff(amps) <= 1e-10), "Amplitude decreases with stiffness")

    # USA model
    amp0 = a9.relative_amplitude_usa(0)
    amp360 = a9.relative_amplitude_usa(360)
    _check(amp360 > amp0, "Larger USA -> higher amplitude")

    # Inversion round-trip
    for test_amp in [0.10, 0.20, 0.35]:
        eta_inv = a9.invert_coupling_stiffness(test_amp)
        amp_check = a9.relative_amplitude_slip(eta_inv)
        _check(abs(amp_check - test_amp) < 0.02,
               f"Inversion round-trip at amp={test_amp}")

    # Classification
    q = a9.classify_cement_quality(np.array([0.05, 0.20, 0.40]))
    _check(q[0] == 'Good' and q[1] == 'Medium' and q[2] == 'Poor',
           "Quality classification thresholds")

    # Coupling matrix
    M = a9.coupling_stiffness_matrix(1e10, 100)
    _check(M.shape == (6, 6), "Matrix is 6x6")

    print(f"    ✓ {PASS - p0} checks passed\n")


# ============================================================================
# Article 10 – Neutron Log in Shales (Rasmus)
# ============================================================================
def test_article_10():
    global PASS, FAIL
    p0 = PASS
    print("▶ Article 10: Neutron Log Response in Shales")

    # Migration length: water < quartz
    Lm_water = a10.migration_length({'quartz': 0.0}, 1.0)
    Lm_qtz = a10.migration_length({'quartz': 1.0}, 0.0)
    _check(Lm_water < Lm_qtz, "Lm(water) < Lm(quartz)")

    # Neutron porosity transforms
    phi_ls = a10.neutron_porosity_limestone(6.0)
    _check(abs(phi_ls - 1.0) < 0.05, "Lm*=6 -> phi~1 for limestone")

    phi_ls0 = a10.neutron_porosity_limestone(15.5)
    _check(abs(phi_ls0) < 0.05, "Lm*=15.5 -> phi~0 for limestone")

    # Effective Lm*
    Lm_star = a10.effective_lm_star(10.0)
    _check(Lm_star > 0, "Lm* is positive")

    # Shale nonlinear response
    phi_nl = a10.nonlinear_shale_response(0.5, 0.10)
    _check(0 < phi_nl < 1, "Shale apparent porosity in (0,1)")

    # Linear vs nonlinear difference
    phi_lin = 0.10 + a10.apparent_shale_porosity_ss(0.5)
    _check(abs(phi_nl - phi_lin) > 0.001,
           "Nonlinear and linear shale responses differ")

    print(f"    ✓ {PASS - p0} checks passed\n")


# ============================================================================
# Article 11 – Fracture Identification (Lee et al.)
# ============================================================================
def test_article_11():
    global PASS, FAIL
    p0 = PASS
    print("▶ Article 11: Automatic Fracture Identification from Image Logs")

    image, true_depths = a11.generate_borehole_image(
        n_depths=300, n_fractures=10, seed=99)
    _check(image.shape == (300, 360), "Image shape correct")
    _check(len(true_depths) == 10, "Correct number of fractures")

    features = a11.extract_features(image)
    _check(features.shape == (300, 5), "Feature matrix shape")

    pred = a11.detect_fractures_threshold(features, grad_threshold=0.20)
    _check(len(pred) > 0, "At least one fracture detected")

    f1 = a11.f1_score_with_tolerance(true_depths, pred, alpha=5)
    _check(0 <= f1 <= 1, "F1 score in [0,1]")

    rmse_val = a11.rmse_depths(true_depths, pred)
    _check(rmse_val >= 0, "RMSE >= 0")

    # Perfect prediction
    f1_perfect = a11.f1_score_with_tolerance(true_depths, true_depths, alpha=0)
    _check(abs(f1_perfect - 1.0) < 1e-9, "Perfect prediction -> F1 = 1")

    # CNN detector
    det = a11.SimpleFractureDetector(n_features=5)
    proba = det.predict_proba(features)
    _check(proba.shape == (300,), "CNN proba shape")
    _check(np.all((proba >= 0) & (proba <= 1)), "Probabilities in [0,1]")

    print(f"    ✓ {PASS - p0} checks passed\n")


# ============================================================================
# Main
# ============================================================================
def test_all():
    """Run all tests and report summary."""
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    tests = [
        test_article_1,
        test_article_2,
        test_article_3,
        test_article_4,
        test_article_5,
        test_article_6,
        test_article_7,
        test_article_8,
        test_article_9,
        test_article_10,
        test_article_11,
    ]

    print("=" * 72)
    print("  Petrophysics Vol. 66 No. 5 (October 2025) – Full Test Suite")
    print("=" * 72 + "\n")

    errors = []
    for test_fn in tests:
        try:
            test_fn()
        except Exception:
            FAIL += 1
            name = test_fn.__name__
            errors.append(name)
            print(f"    ✗ EXCEPTION in {name}:")
            traceback.print_exc()
            print()

    print("=" * 72)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    if errors:
        print(f"  Errors in: {', '.join(errors)}")
    else:
        print("  All modules validated successfully.")
    print("=" * 72)
    return FAIL == 0


if __name__ == "__main__":
    ok = test_all()
    sys.exit(0 if ok else 1)
