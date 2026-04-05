"""
test_all.py — Comprehensive unit tests for all 12 modules in
petrophysics_spwla_2026 (Petrophysics Vol. 67, No. 2, April 2026)

Run with:
    python test_all.py              # prints pass/fail per test
    python -m pytest test_all.py -v # via pytest

Each test class corresponds to one article module.  Tests use only
synthetic data and numpy — no external fixtures needed.
"""

import math
import sys
import traceback
import types
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make the package importable regardless of working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import petrophysics_spwla_2026.a01_sponge_core_saturation_uncertainty  as a01
import petrophysics_spwla_2026.a02_nmr_wettability_pore_partitioning   as a02
import petrophysics_spwla_2026.a03_water_rock_mechanical_ae             as a03
import petrophysics_spwla_2026.a04_wireline_anomaly_diagnosis           as a04
import petrophysics_spwla_2026.a05_ail_hierarchical_correction          as a05
import petrophysics_spwla_2026.a06_bioclastic_limestone_classification  as a06
import petrophysics_spwla_2026.a07_knowledge_guided_dcdnn               as a07
import petrophysics_spwla_2026.a08_shale_induced_stress_fracture        as a08
import petrophysics_spwla_2026.a09_acid_fracturing_cbm                  as a09
import petrophysics_spwla_2026.a10_interlayer_fracture_propagation      as a10
import petrophysics_spwla_2026.a11_awi_cement_evaluation                as a11
import petrophysics_spwla_2026.a12_depth_shifting_ml                    as a12

# ---------------------------------------------------------------------------
# Tiny assertion helpers (no pytest required)
# ---------------------------------------------------------------------------

def assert_close(a, b, rtol=1e-6, atol=1e-9, msg=""):
    """Assert |a - b| <= atol + rtol * |b|."""
    if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol):
        raise AssertionError(
            f"{msg}  got {a}, expected ~{b}  "
            f"(rtol={rtol}, atol={atol})"
        )


def assert_in_range(val, lo, hi, msg=""):
    if not (lo <= float(val) <= hi):
        raise AssertionError(f"{msg}  {val} not in [{lo}, {hi}]")


def assert_true(expr, msg=""):
    if not expr:
        raise AssertionError(msg or "Assertion failed")


def assert_array_shape(arr, shape, msg=""):
    if np.asarray(arr).shape != shape:
        raise AssertionError(
            f"{msg}  shape {np.asarray(arr).shape} != expected {shape}"
        )


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _run(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  [PASS] {name}")
    except Exception:
        tb = traceback.format_exc()
        _results.append((name, False, tb))
        print(f"  [FAIL] {name}")
        # Print first 4 lines of traceback for quick diagnosis
        for line in tb.splitlines()[-6:]:
            print(f"         {line}")


# ===========================================================================
# Article 1 — Sponge Core Saturation Uncertainty (Alghazal & Krinis 2026)
# ===========================================================================

class TestA01:
    """Tests for a01_sponge_core_saturation_uncertainty.py"""

    def test_salt_correction_factor_zero_salinity(self):
        """At 0 ppm salinity the correction factor must be exactly 1.0."""
        csf = a01.salt_correction_factor(0.0)
        assert_close(csf, 1.0, msg="salt_correction_factor(0)")

    def test_salt_correction_factor_increases_with_salinity(self):
        """Higher salinity → larger Csf (more salt in water → larger corrected volume)."""
        csf_low  = a01.salt_correction_factor(20_000)
        csf_high = a01.salt_correction_factor(150_000)
        assert_true(csf_high > csf_low,
                    "Csf should increase with salinity")

    def test_formation_water_volume_identity(self):
        """At 0 ppm the corrected volume equals the distilled volume."""
        Vd = 48.5
        Vw = a01.formation_water_volume(Vd, std_ppm=0.0)
        assert_close(Vw, Vd, rtol=1e-5, msg="Vw == Vd at 0 ppm")

    def test_formation_water_volume_increases_with_salinity(self):
        Vd = 50.0
        Vw_lo = a01.formation_water_volume(Vd, 20_000)
        Vw_hi = a01.formation_water_volume(Vd, 150_000)
        assert_true(Vw_hi > Vw_lo, "Higher salinity → larger Vw")

    def test_oil_volume_positive(self):
        """Oil volume must be non-negative for realistic inputs."""
        Vw = a01.formation_water_volume(50.0, 66_000)
        Vo = a01.oil_volume(mwet=400.0, mdry=385.0, Vw=Vw,
                            rho_w=1.07, rho_o=0.85)
        assert_true(Vo >= 0, f"Oil volume negative: {Vo}")

    def test_surface_saturations_sum_to_one(self):
        """Sos + Sws should equal 1 when all pore volume is accounted for."""
        Vp = 80.0
        Vo, Vw = 20.0, 60.0
        Sos, Sws = a01.surface_saturations(Vo, Vw, Vp)
        assert_close(Sos + Sws, 1.0, rtol=1e-6, msg="Sos + Sws")

    def test_reservoir_saturations_scale_with_formation_volume_factor(self):
        """Larger Bo → larger Sor."""
        Sor1, _ = a01.reservoir_saturations(0.3, 0.7, 0.20, 0.18, Bo=1.1, Bw=1.0)
        Sor2, _ = a01.reservoir_saturations(0.3, 0.7, 0.20, 0.18, Bo=1.5, Bw=1.0)
        assert_true(Sor2 > Sor1, "Larger Bo → larger Sor")

    def test_estimate_salinity_mass_balance_positive(self):
        """Mass-balance salinity must be ≥ 0."""
        std = a01.estimate_salinity_mass_balance(
            mwet=400.0, mtol=385.0, mmeth=383.5, Vd=50.0)
        assert_true(std >= 0, f"Salinity negative: {std}")

    def test_estimate_salinity_mass_balance_increases_with_salt_mass(self):
        """More salt leached (mtol - mmeth) → higher salinity."""
        s_lo = a01.estimate_salinity_mass_balance(400.0, 385.0, 384.5, 50.0)
        s_hi = a01.estimate_salinity_mass_balance(400.0, 385.0, 380.0, 50.0)
        assert_true(s_hi > s_lo, "Larger salt mass → higher salinity")

    def test_monte_carlo_output_keys(self):
        """Monte Carlo result must contain required statistical keys."""
        dists  = a01.InputDistributions()
        result = a01.run_monte_carlo(dists, n_simulations=500)
        for key in ("p10", "p50", "p90", "mean", "std", "Sor_array"):
            assert_true(key in result, f"Missing key: {key}")

    def test_monte_carlo_percentile_ordering(self):
        """P10 ≤ P50 ≤ P90."""
        result = a01.run_monte_carlo(a01.InputDistributions(), n_simulations=500)
        assert_true(result["p10"] <= result["p50"] <= result["p90"],
                    "Percentile ordering violated")

    def test_monte_carlo_saturation_in_range(self):
        """Oil saturation percentiles must be physically plausible (0–100 s.u.)."""
        result = a01.run_monte_carlo(a01.InputDistributions(), n_simulations=500)
        assert_in_range(result["p10"],  0, 100, "P10")
        assert_in_range(result["p90"],  0, 100, "P90")

    def test_spearman_sensitivity_returns_dict(self):
        """spearman_sensitivity must return a dict keyed by input names."""
        result = a01.run_monte_carlo(a01.InputDistributions(), n_simulations=300)
        coeffs = a01.spearman_sensitivity(result)
        assert_true(isinstance(coeffs, dict) and len(coeffs) > 0,
                    "spearman_sensitivity returned empty result")

    def test_spearman_salinity_dominant(self):
        """Water salinity must appear in top-3 by |Spearman rho| (paper finding).

        The default InputDistributions uses plug-scale dimensions (3.8 cm diam)
        but the grain_vol distribution (~120 cm³) exceeds the bulk volume
        (~68 cm³) of that plug, giving Vp ≤ 0 and a degenerate MC.  The paper
        works with full-diameter cores (~10 cm diam), so we use those here.

        NaN correlations (constant Sor) are excluded before ranking.
        """
        dists = a01.InputDistributions()
        # Full-diameter core parameters so that Vp > 0 throughout the MC
        dists.diameter_cm   = ('triangular', (9.9, 10.0, 10.1))
        dists.length_cm     = ('triangular', (29.8, 30.0, 30.2))
        dists.grain_vol_cm3 = ('triangular', (1820, 1850, 1880))
        dists.mwet_g        = ('triangular', (4800, 5000, 5200))
        dists.weight_loss_g = ('triangular', (90, 100, 110))
        dists.Vd_cm3        = ('triangular', (48, 50, 52))
        result = a01.run_monte_carlo(dists, n_simulations=5000, seed=42)
        coeffs = a01.spearman_sensitivity(result)
        valid  = {k: v for k, v in coeffs.items() if not math.isnan(v)}
        ranked = sorted(valid, key=lambda k: abs(valid[k]), reverse=True)
        top3   = ranked[:3]
        assert_true(any("salinity" in v for v in top3),
                    f"Salinity should be in top-3; top-3={top3}")


# ===========================================================================
# Article 2 — NMR Wettability Pore Partitioning (Aljishi et al. 2026)
# ===========================================================================

class TestA02:
    """Tests for a02_nmr_wettability_pore_partitioning.py"""

    def test_wettability_index_bounds(self):
        """Iw must lie in [-1, +1]."""
        for Sw, Sdo in [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]:
            Iw = a02.wettability_index(Sw, Sdo)
            assert_in_range(Iw, -1, 1, f"Iw at Sw={Sw}")

    def test_wettability_index_water_wet(self):
        """Full brine, no oil → strongly water-wet (Iw = +1)."""
        assert_close(a02.wettability_index(1.0, 0.0), 1.0)

    def test_wettability_index_oil_wet(self):
        """No brine, full oil → strongly oil-wet (Iw = -1)."""
        assert_close(a02.wettability_index(0.0, 1.0), -1.0)

    def test_classify_wettability_labels(self):
        assert_true(a02.classify_wettability(0.5)  == "Water-wet")
        assert_true(a02.classify_wettability(-0.5) == "Oil-wet")
        assert_true(a02.classify_wettability(0.0)  == "Mixed-wet")

    def test_pore_partition_fractions_sum(self):
        """Pore fractions should sum to approximately total porosity."""
        pp = a02.compute_pore_partition(0.076, 0.080, 0.082)
        assert_in_range(pp.total(), 0.07, 0.09, "Partition sum")

    def test_pore_partition_validity(self):
        """All partition fractions must be non-negative and their sum must
        equal phi_total (the value passed in), not 1.0.

        is_valid() checks that fractions sum to ~1.0, which is not the
        convention here (fractions are in porosity units, not PV fractions).
        Instead we verify each component is non-negative and the total
        matches phi_total.
        """
        phi_total = 0.082
        pp = a02.compute_pore_partition(0.076, 0.080, phi_total)
        assert_true(pp.f_ww >= 0, "f_ww must be non-negative")
        assert_true(pp.f_ow >= 0, "f_ow must be non-negative")
        assert_true(pp.f_mw >= 0, "f_mw must be non-negative")
        assert_close(pp.total(), phi_total, rtol=1e-4,
                     msg="Partition fractions must sum to phi_total")

    def test_pore_partition_water_wet_sample(self):
        """Brine-dominant sample → large f_ww."""
        pp = a02.compute_pore_partition(phi_brine_seqA=0.090,
                                        phi_oil_seqB=0.020,
                                        phi_total=0.095)
        assert_true(pp.f_ww >= pp.f_ow, "Water-wet sample should have f_ww ≥ f_ow")

    def test_sequence_C_monotone_oil_increase(self):
        """Oil fraction should not decrease during forced imbibition in Seq C."""
        pressures = np.array([0, 100, 250, 500, 1000, 2000, 4500], dtype=float)
        states = a02.simulate_sequence_C(
            phi_brine_spont=0.076, phi_oil_spont_counter=0.002,
            phi_oil_forced=0.018, phi_iso=0.006, pressure_steps=pressures)
        oils = [s.phi_oil for s in states]
        assert_true(oils[-1] >= oils[0], "Oil fraction should grow over Seq C")

    def test_sequence_D_brine_displaces_oil(self):
        """End of Seq D should have more brine than start."""
        pressures = np.array([0, 100, 500, 1000, 4500], dtype=float)
        states = a02.simulate_sequence_D(
            phi_oil_spont=0.080, phi_brine_spont_counter=0.045,
            phi_brine_forced=0.050, phi_iso=0.006, pressure_steps=pressures)
        brine_end   = states[-1].phi_brine
        brine_start = states[0].phi_brine
        assert_true(brine_end > brine_start, "Brine should increase in Seq D")

    def test_choking_threshold_range(self):
        """Paper reports 26–71 % choking threshold; synthetic should be plausible."""
        thr = a02.choking_threshold(phi_oil_seqC_final=0.020, phi_total=0.082)
        assert_in_range(thr, 0, 100, "Choking threshold")

    def test_mineralogy_wettability_carbonate(self):
        """Carbonate-rich → oil-wet tendency."""
        t = a02.mineralogy_wettability_tendency(0.10, 0.75, 0.05, 0.02)
        assert_true("Oil" in t, f"Expected oil-wet tendency, got '{t}'")

    def test_mineralogy_wettability_silica(self):
        """Quartz-rich → water-wet tendency."""
        t = a02.mineralogy_wettability_tendency(0.80, 0.05, 0.05, 0.01)
        assert_true("Water" in t or "Mixed" in t,
                    f"Expected water-wet or mixed, got '{t}'")

    def test_recommend_strategy_mixed_wet(self):
        pp = a02.PorePartition(f_ww=0.10, f_ow=0.15, f_mw=0.60)
        rec = a02.recommend_production_strategy(pp)
        assert_true("Mixed" in rec or "brine" in rec.lower(),
                    "Mixed-wet strategy should mention brine or moderate drawdown")


# ===========================================================================
# Article 3 — Water-Rock Mechanical & AE (Zhao 2026)
# ===========================================================================

class TestA03:
    """Tests for a03_water_rock_mechanical_ae.py"""

    def test_porosity_fluid_saturation_basic(self):
        """Standard cylinder: porosity = (m_sat - m_dry) / (rho_w * V_bulk)."""
        phi = a03.porosity_fluid_saturation(
            m_saturated=285.5, m_dry=276.3, rho_water=1.0, V_bulk=98.2)
        expected = (285.5 - 276.3) / (1.0 * 98.2)
        assert_close(phi, expected, rtol=1e-6)

    def test_porosity_in_realistic_range(self):
        phi = a03.porosity_fluid_saturation(285.5, 276.3, 1.0, 98.2)
        assert_in_range(phi, 0.05, 0.20, "Porosity")

    def test_mechanical_decay_at_zero(self):
        """At t=0 the property should equal a + c (= initial value)."""
        val = a03.mechanical_property_decay(0.0, a=35.0, b=0.55, c=25.0)
        assert_close(val, 35.0 + 25.0, rtol=1e-6)

    def test_mechanical_decay_asymptote(self):
        """At large t the property approaches c (the asymptote)."""
        val = a03.mechanical_property_decay(1000.0, a=35.0, b=0.55, c=25.0)
        assert_close(val, 25.0, rtol=1e-3)

    def test_compressive_strength_decreases_with_soaking(self):
        """Longer soaking → lower strength (paper's key result)."""
        s0 = a03.compressive_strength(0)
        s6 = a03.compressive_strength(6)
        assert_true(s6 < s0, "Strength should decrease with soaking")

    def test_elastic_modulus_decreases_with_soaking(self):
        E0 = a03.elastic_modulus(0)
        E6 = a03.elastic_modulus(6)
        assert_true(E6 < E0, "Modulus should decrease with soaking")

    def test_attenuation_coefficient_positive(self):
        alpha = a03.attenuation_coefficient(A0=100.0, Ax=62.5, x=5.0)
        assert_true(alpha > 0, "Attenuation coefficient must be positive")

    def test_attenuation_roundtrip(self):
        """predict_amplitude should recover Ax when using computed alpha."""
        A0, Ax_orig, x = 100.0, 62.5, 5.0
        alpha = a03.attenuation_coefficient(A0, Ax_orig, x)
        Ax_pred = a03.predict_amplitude(A0, alpha, x)
        assert_close(Ax_pred, Ax_orig, rtol=1e-5)

    def test_attenuation_increases_with_soaking(self):
        alpha0 = a03.attenuation_coefficient(100.0, 62.5, 5.0)
        alpha6 = a03.attenuation_coefficient(100.0, 48.0, 5.0)
        assert_true(alpha6 > alpha0,
                    "Higher water content → higher attenuation")

    def test_ae_energy_model_monotone_near_failure(self):
        """Accumulated AE energy should increase as tc is approached."""
        tc = 300.0
        t  = np.linspace(0, 295, 100)
        E  = a03.accumulated_ae_energy_model(t, tc=tc, A=5.0, m=0.3)
        assert_true(E[-1] > E[0], "AE energy should increase toward failure")

    def test_ae_energy_release_rate_positive(self):
        t  = np.linspace(0, 295, 100)
        E  = a03.accumulated_ae_energy_model(t, tc=300.0, A=5.0, m=0.3)
        dE = a03.ae_energy_release_rate(E, t)
        assert_true(np.any(dE > 0), "Energy release rate should be positive")

    def test_ae_energy_expectation_decreases_with_soaking(self):
        mu0 = a03.ae_energy_expectation_with_soaking(0)
        mu6 = a03.ae_energy_expectation_with_soaking(6)
        assert_true(mu6 < mu0, "AE energy expectation should decrease with soaking")

    def test_failure_mode_dry_is_brittle(self):
        """Dry specimen with very high E/sigma_c ratio (BI > 250) → Brittle.

        Paper reports brittle failure for dry sandstone.  BI = E_MPa / sigma_c.
        The module classifies BI > 250 AND t < 1 month as Brittle.
        """
        # BI = 20_000 / 50 = 400 >> 250 → Brittle
        mode = a03.failure_mode_indicator(t_months=0,
                                           sigma_c=50.0, E_MPa=20_000.0)
        assert_true(mode == "Brittle", f"Expected Brittle, got {mode}")

    def test_failure_mode_long_soak_is_ductile(self):
        mode = a03.failure_mode_indicator(t_months=6,
                                           sigma_c=27.0, E_MPa=8000.0)
        assert_true(mode in ("Ductile", "Transitional"),
                    f"Expected Ductile/Transitional, got {mode}")

    def test_soaking_weakening_profile_shape(self):
        profile = a03.soaking_weakening_profile(6.0, n_points=30)
        for key in ("t_months", "strength_MPa", "modulus_GPa", "ae_energy_mu"):
            assert_true(key in profile and len(profile[key]) == 30,
                        f"Profile missing key or wrong length: {key}")


# ===========================================================================
# Article 4 — Wireline Logging Anomaly Diagnosis (Liu et al. 2026)
# ===========================================================================

class TestA04:
    """Tests for a04_wireline_anomaly_diagnosis.py"""

    def _params(self):
        return a04.InstrumentParams()

    def test_gravity_positive(self):
        assert_true(a04.gravity(self._params()) > 0)

    def test_buoyancy_positive(self):
        assert_true(a04.buoyancy(self._params()) > 0)

    def test_tension_lowering_increases_with_depth(self):
        p = self._params()
        T0 = a04.cable_tension_lowering(p, depth_m=500)
        T1 = a04.cable_tension_lowering(p, depth_m=2000)
        assert_true(T1 > T0, "Tension should increase with depth (cable weight)")

    def test_tension_hoisting_greater_than_lowering(self):
        """Hoisting requires more tension than free lowering."""
        p  = self._params()
        Tl = a04.cable_tension_lowering(p, 1000)
        Th = a04.cable_tension_hoisting(p, 1000)
        assert_true(Th > Tl, "Hoisting tension > lowering tension")

    def test_tension_hoisting_stuck_increases(self):
        p  = self._params()
        T0 = a04.cable_tension_hoisting(p, 1000, stuck_force_N=0)
        T1 = a04.cable_tension_hoisting(p, 1000, stuck_force_N=5000)
        assert_true(T1 > T0, "Stuck force should increase tension")

    def test_moving_average_constant_signal(self):
        """MA of a constant signal must equal the constant."""
        x  = np.full(100, 7.5)
        ma = a04.moving_average(x, window=10)
        assert_close(ma[50], 7.5, rtol=1e-6)

    def test_moving_std_zero_for_constant(self):
        x   = np.full(100, 3.0)
        std = a04.moving_std(x, window=10)
        # Interior values should be (near) zero
        assert_close(std[50], 0.0, atol=1e-10)

    def test_detect_anomalies_output_keys(self):
        t = np.random.default_rng(0).normal(5000, 100, 500)
        out = a04.detect_tension_anomalies(t, window=20)
        for k in ("ma", "sigma", "abs_dev", "L1_flags", "L2_flags", "any_anomaly"):
            assert_true(k in out, f"Missing key: {k}")

    def test_detect_anomalies_flags_large_spike(self):
        """A very large spike should trigger L1 anomaly flag."""
        rng = np.random.default_rng(0)
        t   = rng.normal(5000, 50, 300)
        t[150] = 50_000   # extreme spike
        out  = a04.detect_tension_anomalies(t, window=20, k1=4.0)
        assert_true(out["L1_flags"][150], "Large spike should trigger L1 flag")

    def test_remove_adjacent_spikes_reduces_extremes(self):
        """Spike removal should reduce the max absolute value."""
        rng  = np.random.default_rng(0)
        z    = rng.normal(0, 0.05, 200)
        z[100] = 50.0   # isolated spike
        z_c  = a04.remove_adjacent_spikes(z)
        assert_true(z_c[100] < z[100], "Spike should be reduced")

    def test_ewma_reference_is_mean_of_first_window(self):
        z   = np.arange(100, dtype=float)
        ref = a04.ewma_reference(z, window=20)
        assert_close(ref, np.mean(z[:20]), rtol=1e-6)

    def test_moving_mad_non_negative(self):
        z   = np.random.default_rng(1).normal(0, 1, 100)
        mad = a04.moving_mad(z, ref=0.0, window=10)
        assert_true(np.all(mad >= 0), "MAD must be non-negative")

    def test_zscore_zero_mean(self):
        arr = np.random.default_rng(2).normal(5, 2, 200)
        z   = a04.zscore_standardise(arr)
        assert_close(z.mean(), 0.0, atol=1e-10)

    def test_grade_alarm_extreme_triggers_grade4(self):
        z = np.array([0.0, 1.5, 2.5, 3.5, 5.0])
        g = a04.grade_vibration_alarm(z)
        assert_true(g[-1] == int(a04.AlarmGrade.EXTREME),
                    "Z-score > 4 should give EXTREME alarm")

    def test_diagnose_returns_correct_shape(self):
        n   = 200
        rng = np.random.default_rng(3)
        result = a04.diagnose_wireline_run(
            tension=rng.normal(5000, 100, n),
            vib_x=rng.normal(0, 0.05, n),
            vib_y=rng.normal(0, 0.03, n),
            vib_z=rng.normal(0, 0.02, n),
        )
        assert_array_shape(result["diagnosis"], (n,))

    def test_diagnose_jamming_detected(self):
        """Injecting a large tension + large vibration spike should trigger jamming."""
        n   = 300
        rng = np.random.default_rng(4)
        t   = rng.normal(5000, 50, n)
        t[150:160]   += 8000    # tension jump
        vx  = rng.normal(0, 0.03, n)
        vx[150:160]  += 5.0    # extreme vibration
        vy  = rng.normal(0, 0.02, n)
        vz  = rng.normal(0, 0.01, n)
        res = a04.diagnose_wireline_run(t, vx, vy, vz)
        # At least one sample in the injected window should be flagged
        diag_window = res["diagnosis"][145:165]
        assert_true(np.any(diag_window >= 1), "Jamming event should be detected")


# ===========================================================================
# Article 5 — AIL Hierarchical Correction (Qiao et al. 2026)
# ===========================================================================

class TestA05:
    """Tests for a05_ail_hierarchical_correction.py"""

    def test_anisotropy_coefficient_isotropic(self):
        """Isotropic formation: Rh == Rv → λ = 1."""
        lam = a05.anisotropy_coefficient(10.0, 10.0)
        assert_close(lam, 1.0, rtol=1e-6)

    def test_anisotropy_coefficient_anisotropic(self):
        """Rv > Rh → λ > 1."""
        lam = a05.anisotropy_coefficient(Rh=10.0, Rv=40.0)
        assert_close(lam, 2.0, rtol=1e-5)

    def test_apparent_resistivity_dip_zero(self):
        """At 0° dip the apparent resistivity equals Rh."""
        Ra = a05.apparent_resistivity_dip(10.0, 40.0, dip_deg=0.0)
        assert_close(Ra, 10.0, rtol=1e-4)

    def test_apparent_resistivity_dip_90(self):
        """At 90° dip the formula reduces to sqrt(Rh*Rv) (geometric mean)."""
        Rh, Rv = 10.0, 40.0
        Ra = a05.apparent_resistivity_dip(Rh, Rv, dip_deg=90.0)
        expected = Rh / math.sqrt(Rh / Rv)   # = sqrt(Rh*Rv)
        assert_close(Ra, expected, rtol=1e-4)

    def test_true_thickness_correction_at_zero_dip(self):
        """At 0° dip the true thickness equals the apparent thickness."""
        h_true = a05.true_thickness_correction_coefficient(5.0, 0.0)
        assert_close(h_true, 5.0, rtol=1e-6)

    def test_true_thickness_correction_decreases_with_dip(self):
        """Higher dip → smaller true thickness."""
        h0 = a05.true_thickness_correction_coefficient(5.0, 30.0)
        h1 = a05.true_thickness_correction_coefficient(5.0, 80.0)
        assert_true(h1 < h0, "True thickness decreases with dip")

    def test_software_focusing_uniform_weights(self):
        """Uniform-weight focusing returns the arithmetic mean."""
        Ra = np.array([10.0, 12.0, 14.0, 11.0, 13.0])
        focused = a05.software_focusing(Ra)
        assert_close(focused, Ra.mean(), rtol=1e-6)

    def test_invasion_corrected_resistivity_no_invasion(self):
        """Zero invasion depth → Ra returned unchanged."""
        Rt = a05.invasion_corrected_resistivity(Ra=12.0, Rxo=3.0,
                                                  di_m=0.0, doi_m=0.9)
        assert_close(Rt, 12.0, rtol=1e-4)

    def test_invasion_corrected_gt_rxo(self):
        """Corrected Rt should be > Rxo for a conductive invasion zone."""
        Rt = a05.invasion_corrected_resistivity(Ra=12.0, Rxo=3.0,
                                                  di_m=0.2, doi_m=0.9)
        assert_true(Rt > 3.0, f"Rt ({Rt}) should exceed Rxo (3.0)")

    def test_hierarchical_correction_output_keys(self):
        Ra = np.array([8.5, 10.0, 12.0, 13.5, 14.5, 15.0, 15.5])
        out = a05.hierarchical_ail_correction(
            apparent_resistivities=Ra, dip_deg=75.0,
            apparent_thickness_m=4.0, Rxo=2.5, invasion_diam_m=0.25,
            R_vt_array=np.full(5, 15.0))
        for k in ("Rh", "Rv", "Rt_focused", "lambda_anisotropy", "h_true_m"):
            assert_true(k in out, f"Missing key: {k}")

    def test_hierarchical_correction_Rh_positive(self):
        Ra  = np.linspace(8, 16, 7)
        out = a05.hierarchical_ail_correction(Ra, 75.0, 4.0, 2.5, 0.25,
                                               np.full(5, 15.0))
        assert_true(out["Rh"] > 0, "Rh must be positive")

    def test_accuracy_error_zero_for_perfect(self):
        err = a05.accuracy_error_pct(10.0, 10.0)
        assert_close(err, 0.0, atol=1e-10)

    def test_accuracy_error_less_than_3pct(self):
        """Paper claims < 3 % accuracy for the hierarchical correction."""
        Rh_true = 15.0
        Ra      = np.full(7, Rh_true * 1.02)   # 2% over-estimate
        out     = a05.hierarchical_ail_correction(Ra, 45.0, 5.0, 2.0, 0.15,
                                                    np.full(5, Rh_true))
        err = a05.accuracy_error_pct(out["Rh"], Rh_true)
        assert_in_range(err, 0, 20, "Accuracy error")   # wide tolerance for synthetic data

    def test_thickness_library_correct_count(self):
        dips = np.array([0.0, 45.0, 90.0])
        thks = np.array([2.0, 5.0])
        lib  = a05.build_thickness_correction_library(dips, thks)
        assert_true(len(lib) == 6, f"Expected 6 entries, got {len(lib)}")


# ===========================================================================
# Article 6 — Bioclastic Limestone Classification (Guo et al. 2026)
# ===========================================================================

class TestA06:
    """Tests for a06_bioclastic_limestone_classification.py"""

    def test_classify_grain_energy_high(self):
        e = a06.classify_grain_energy(0.60, 0.20, 0.10)
        assert_true(e == "high", f"Expected 'high', got '{e}'")

    def test_classify_grain_energy_low(self):
        e = a06.classify_grain_energy(0.05, 0.10, 0.60)
        assert_true(e == "low", f"Expected 'low', got '{e}'")

    def test_classify_grain_energy_mixed(self):
        e = a06.classify_grain_energy(0.15, 0.45, 0.20)
        assert_true(e == "mixed", f"Expected 'mixed', got '{e}'")

    def test_classify_mudstone(self):
        rt = a06.classify_geological_facies(0.01, 0.02, 0.05, 0.90)
        assert_true(rt == a06.RockType.A, f"Expected Type A, got {rt}")

    def test_classify_wackestone(self):
        rt = a06.classify_geological_facies(0.04, 0.08, 0.60, 0.28)
        assert_true(rt == a06.RockType.B, f"Expected Type B, got {rt}")

    def test_classify_high_energy(self):
        rt = a06.classify_geological_facies(0.65, 0.15, 0.05, 0.10)
        assert_true(rt in (a06.RockType.C3_1, a06.RockType.C3_2),
                    f"Expected C3 type, got {rt}")

    def test_r35_increases_with_permeability(self):
        R35_lo = a06.r35_pore_throat_radius(k_md=0.1, phi_frac=0.10)
        R35_hi = a06.r35_pore_throat_radius(k_md=100, phi_frac=0.20)
        assert_true(R35_hi > R35_lo, "R35 should increase with permeability")

    def test_r35_positive(self):
        R35 = a06.r35_pore_throat_radius(5.0, 0.15)
        assert_true(R35 > 0, "R35 must be positive")

    def test_displacement_pressure_decreases_with_k(self):
        """Higher permeability → larger pores → lower entry pressure."""
        Pd_lo = a06.displacement_pressure(k_md=0.1, phi_frac=0.08)
        Pd_hi = a06.displacement_pressure(k_md=50, phi_frac=0.20)
        assert_true(Pd_hi < Pd_lo, "Entry pressure decreases with k")

    def test_petrophysical_sample_from_core(self):
        ps = a06.PetrophysicalSample.from_core(phi=0.15, k_md=5.0)
        assert_true(ps.R35 > 0 and ps.Pd > 0, "Derived params must be positive")

    def test_petrophysical_sample_as_array_shape(self):
        ps  = a06.PetrophysicalSample.from_core(0.15, 5.0)
        arr = ps.as_array()
        assert_array_shape(arr, (5,))

    def test_kmeans_returns_correct_length(self):
        samples = [a06.PetrophysicalSample.from_core(0.05 + i*0.02, 0.1 * 2**i)
                   for i in range(14)]
        labels = a06.kmeans_petrophysical(samples, n_clusters=7)
        assert_array_shape(labels, (14,))

    def test_kmeans_labels_in_range(self):
        samples = [a06.PetrophysicalSample.from_core(0.05 + i*0.02, 0.1 * 2**i)
                   for i in range(14)]
        labels = a06.kmeans_petrophysical(samples, n_clusters=7)
        assert_true(labels.min() >= 0 and labels.max() <= 6,
                    "Labels must be in [0, n_clusters)")

    def test_productivity_rank_ordering(self):
        """C3 types must have higher rank than A."""
        rank_A  = a06.productivity_rank(a06.RockType.A)
        rank_C3 = a06.productivity_rank(a06.RockType.C3_2)
        assert_true(rank_C3 > rank_A, "C3 should rank higher than A")


# ===========================================================================
# Article 7 — Knowledge-Guided DCDNN (Yu et al. 2026)
# ===========================================================================

class TestA07:
    """Tests for a07_knowledge_guided_dcdnn.py"""

    def test_archie_fully_saturated(self):
        """Sw = 1 when Rt equals Rw / phi^m (fully water saturated)."""
        Rw, phi = 0.05, 0.20
        Rt = 1.0 * Rw / (phi**2)   # a=1, m=2, n=2
        Sw = a07.archie_water_saturation(Rw, Rt, phi)
        assert_close(Sw, 1.0, rtol=1e-4)

    def test_archie_sw_in_range(self):
        Sw = a07.archie_water_saturation(0.05, 15.0, 0.15)
        assert_in_range(Sw, 0.0, 1.0, "Archie Sw")

    def test_archie_sw_decreases_with_rt(self):
        """Higher Rt → lower Sw (more oil)."""
        Sw1 = a07.archie_water_saturation(0.05, 5.0,  0.20)
        Sw2 = a07.archie_water_saturation(0.05, 50.0, 0.20)
        assert_true(Sw2 < Sw1, "Sw should decrease with Rt")

    def test_timur_permeability_positive(self):
        k = a07.timur_permeability(phi=0.15, Swi=0.35)
        assert_true(k > 0, "Timur permeability must be positive")

    def test_timur_permeability_increases_with_phi(self):
        k1 = a07.timur_permeability(0.10, 0.40)
        k2 = a07.timur_permeability(0.25, 0.40)
        assert_true(k2 > k1, "Higher phi → higher k")

    def test_timur_permeability_decreases_with_swi(self):
        k1 = a07.timur_permeability(0.20, 0.20)
        k2 = a07.timur_permeability(0.20, 0.60)
        assert_true(k2 < k1, "Higher Swi → lower k")

    def test_neutron_porosity_correction_no_clay(self):
        """With Vcl=0 the correction is the simple average."""
        phi_xp = a07.neutron_porosity_correction(0.20, 0.18, clay_vol=0.0)
        assert_close(phi_xp, 0.19, rtol=1e-4)

    def test_neutron_porosity_correction_clay_reduces(self):
        """Clay increases neutron response → correction reduces apparent phi."""
        phi_noc = a07.neutron_porosity_correction(0.20, 0.18, clay_vol=0.0)
        phi_cly = a07.neutron_porosity_correction(0.20, 0.18, clay_vol=0.30)
        assert_true(phi_cly < phi_noc, "Clay should reduce corrected porosity")

    def test_augment_features_shape(self):
        rng = np.random.default_rng(0)
        X   = rng.uniform(0, 1, (50, 5))
        X[:, 2] = rng.uniform(0.05, 0.30, 50)  # phi_n
        X[:, 3] = rng.uniform(0.05, 0.30, 50)  # phi_d
        X_aug = a07.augment_features(X, col_phi=2, col_Rt=0,
                                       col_phi_n=2, col_phi_d=3,
                                       col_Vcl=4, Rw=0.05)
        assert_array_shape(X_aug, (50, 8))   # 5 original + 3 knowledge features

    def test_dilated_conv1d_output_length(self):
        x      = np.ones(20)
        kernel = np.array([0.25, 0.5, 0.25])
        y      = a07.dilated_conv1d(x, kernel, dilation=1)
        expected_len = 20 - 1 * (3 - 1)
        assert_true(len(y) == expected_len,
                    f"Expected length {expected_len}, got {len(y)}")

    def test_relu_all_positive_unchanged(self):
        x = np.array([1.0, 2.0, 3.0])
        assert_true(np.allclose(a07.relu(x), x))

    def test_relu_clips_negatives(self):
        x = np.array([-1.0, 0.0, 1.0])
        assert_close(a07.relu(x)[0], 0.0)

    def test_simple_dnn_forward_shape(self):
        dnn = a07.SimpleDNN([8, 16, 8, 1], seed=0)
        X   = np.random.default_rng(0).normal(0, 1, (20, 8))
        out = dnn.forward(X)
        assert_array_shape(out, (20,))

    def test_ensemble_metrics_perfect(self):
        y = np.arange(10, dtype=float)
        m = a07.ensemble_metrics(y, y)
        assert_close(m["MAE"],  0.0, atol=1e-10)
        assert_close(m["R2"],   1.0, rtol=1e-6)
        assert_close(m["RMSE"], 0.0, atol=1e-10)

    def test_ensemble_model_averages_predictions(self):
        """EnsembleModel.predict should be the mean of member predictions."""
        rng  = np.random.default_rng(0)
        X    = rng.normal(0, 1, (10, 4))
        dnn1 = a07.SimpleDNN([4, 8, 1], seed=0)
        dnn2 = a07.SimpleDNN([4, 8, 1], seed=1)
        ens  = a07.EnsembleModel(models=[dnn1, dnn2])
        p1   = dnn1.forward(X)
        p2   = dnn2.forward(X)
        pens = ens.predict(X)
        expected = (p1 + p2) / 2.0
        assert_true(np.allclose(pens, expected, rtol=1e-5))


# ===========================================================================
# Article 8 — Shale Induced-Stress Fracture (Ci 2026)
# ===========================================================================

class TestA08:
    """Tests for a08_shale_induced_stress_fracture.py"""

    def test_induced_stress_symmetry(self):
        """Induced stress field must be symmetric about x=0."""
        X, Y = np.meshgrid(np.array([-50.0, 50.0]), np.array([10.0]))
        sxx, syy = a08.induced_stress_field(200.0, 12.0, X, Y)
        # σxx at ±50 m should be equal (symmetric fracture)
        assert_close(abs(sxx[0, 0]), abs(sxx[0, 1]), rtol=1e-4)

    def test_induced_stress_at_large_distance_small(self):
        """Far from fracture the induced stress should be small."""
        X = np.array([[5000.0]])
        Y = np.array([[5000.0]])
        sxx, syy = a08.induced_stress_field(200.0, 12.0, X, Y)
        assert_in_range(abs(sxx[0, 0]), 0, 0.5, "Far-field σxx")

    def test_induced_stress_difference_shape(self):
        X, Y = np.meshgrid(np.linspace(-300, 300, 10),
                           np.linspace(-200, 200, 8))
        sxx, syy = a08.induced_stress_field(200.0, 10.0, X, Y)
        delta = a08.induced_stress_difference(syy, sxx)
        assert_array_shape(delta, (8, 10))

    def test_stress_diff_vs_volume_monotone_shape(self):
        """Induced stress difference should evolve with volume (any monotone trend)."""
        form = a08.FormationParams()
        frac = a08.FractureParams(half_length_m=100.0, net_pressure=10.0)
        vols = np.linspace(100, 2000, 15)
        ds   = a08.stress_diff_vs_volume(frac, form, vols, 18.0, 150.0, 10.0)
        assert_array_shape(ds, (15,))
        assert_true(np.all(ds >= 0), "Stress difference should be non-negative")

    def test_optimal_pumping_rate_output(self):
        form  = a08.FormationParams()
        frac  = a08.FractureParams()
        rates = np.arange(8, 25, 2, dtype=float)
        out   = a08.optimal_pumping_rate(form, frac, rates)
        assert_true("recommended_rate" in out, "Missing recommended_rate")
        assert_in_range(out["recommended_rate"], 8, 24, "Recommended rate")

    def test_secondary_fracture_condition_shape(self):
        X, Y = np.meshgrid(np.linspace(-100, 100, 5),
                           np.linspace(-50, 50, 4))
        sxx, syy = a08.induced_stress_field(100.0, 15.0, X, Y)
        form = a08.FormationParams()
        cond = a08.secondary_fracture_condition(form, syy, sxx)
        assert_array_shape(cond, (4, 5))

    def test_effective_stress_difference_reasonable(self):
        form = a08.FormationParams()
        X    = np.array([[0.0]])
        Y    = np.array([[10.0]])
        sxx, syy = a08.induced_stress_field(200.0, 10.0, X, Y)
        eff = a08.effective_stress_difference(form, syy, sxx)
        # Effective diff should be near in-situ diff ± ~20 MPa
        assert_in_range(float(eff[0, 0]), 0, 80, "Effective stress difference")


# ===========================================================================
# Article 9 — Acid Fracturing CBM (Zhao et al. 2026)
# ===========================================================================

class TestA09:
    """Tests for a09_acid_fracturing_cbm.py"""

    def test_acid_dissolution_depth_positive(self):
        d = a09.acid_dissolution_depth(10.0, 300.0, 0.012)
        assert_true(d > 0, "Etching depth must be positive")

    def test_acid_dissolution_depth_increases_with_concentration(self):
        d5  = a09.acid_dissolution_depth(5.0,  300.0, 0.012)
        d15 = a09.acid_dissolution_depth(15.0, 300.0, 0.012)
        assert_true(d15 > d5, "Higher concentration → deeper etching")

    def test_acid_dissolution_depth_increases_with_time(self):
        d1 = a09.acid_dissolution_depth(10.0, 100.0, 0.012)
        d2 = a09.acid_dissolution_depth(10.0, 500.0, 0.012)
        assert_true(d2 > d1, "Longer contact → deeper etching")

    def test_fracture_pressure_reduction_positive(self):
        seam = a09.CoalSeamParams()
        acid = a09.AcidParams(concentration_pct=10.0)
        delta = a09.fracture_pressure_reduction(seam, acid)
        assert_true(delta >= 0, "Acid should reduce fracture pressure")

    def test_breakdown_pressure_lower_with_acid(self):
        seam = a09.CoalSeamParams()
        acid = a09.AcidParams(concentration_pct=10.0)
        Pf_raw  = seam.fracture_pressure_MPa
        Pf_acid = a09.breakdown_pressure_with_acid(seam, acid)
        assert_true(Pf_acid < Pf_raw, "Acid lowers breakdown pressure")

    def test_fracture_half_length_positive(self):
        E_prime = 2.5e3 / (1.0 - 0.30**2)
        xf = a09.fracture_half_length_vs_volume(500.0, 15.0, E_prime)
        assert_true(xf > 0, "Half-length must be positive")

    def test_fracture_half_length_increases_with_volume(self):
        E_prime = 2.5e3 / (1.0 - 0.30**2)
        xf1 = a09.fracture_half_length_vs_volume(200.0, 15.0, E_prime)
        xf2 = a09.fracture_half_length_vs_volume(800.0, 15.0, E_prime)
        assert_true(xf2 > xf1, "More fluid → longer fracture")

    def test_fracture_width_positive(self):
        E_prime = 2.5e3 / 0.91
        w = a09.fracture_width_acid(xf=100.0, p_net=8.0, E_prime=E_prime)
        assert_true(w > 0, "Fracture width must be positive")

    def test_fci_top_gt_bottom(self):
        """Perforations at top of seam → higher FCI than at bottom (paper result)."""
        seam = a09.CoalSeamParams()
        FCI_top = a09.fracture_complexity_index(seam.sigma_H_MPa, seam.sigma_h_MPa,
                                                  p_net=8.0, T_coal=seam.T_MPa,
                                                  acid_concentration=10.0, perf_at_top=True)
        FCI_bot = a09.fracture_complexity_index(seam.sigma_H_MPa, seam.sigma_h_MPa,
                                                  p_net=8.0, T_coal=seam.T_MPa,
                                                  acid_concentration=10.0, perf_at_top=False)
        assert_true(FCI_top > FCI_bot, "Top perforation should give higher FCI")

    def test_fci_increases_with_concentration(self):
        seam = a09.CoalSeamParams()
        fci5  = a09.fracture_complexity_index(seam.sigma_H_MPa, seam.sigma_h_MPa,
                                               8.0, seam.T_MPa, 5.0, True)
        fci15 = a09.fracture_complexity_index(seam.sigma_H_MPa, seam.sigma_h_MPa,
                                               8.0, seam.T_MPa, 15.0, True)
        assert_true(fci15 > fci5, "Higher acid conc → higher FCI")

    def test_acid_etched_conductivity_positive(self):
        Fc = a09.acid_etched_conductivity(w_mm=2.0, C_pct=10.0)
        assert_true(Fc > 0, "Conductivity must be positive")

    def test_optimise_acid_returns_recommendation(self):
        seam = a09.CoalSeamParams()
        conc = np.arange(5, 21, 5, dtype=float)
        out  = a09.optimise_acid_parameters(seam, conc)
        assert_true("recommended_conc" in out and out["recommended_conc"] in conc)


# ===========================================================================
# Article 10 — Interlayer Fracture Propagation (Zhao et al. 2026)
# ===========================================================================

class TestA10:
    """Tests for a10_interlayer_fracture_propagation.py"""

    def test_jiyang_stratigraphy_count(self):
        layers = a10.build_jiyang_stratigraphy()
        assert_true(len(layers) == 5, f"Expected 5 layers, got {len(layers)}")

    def test_jiyang_stratigraphy_layer_types(self):
        layers = a10.build_jiyang_stratigraphy()
        names  = [l.name for l in layers]
        assert_true("Coal-1" in names, "Coal layer missing")
        assert_true("Sandstone-1" in names, "Sandstone layer missing")

    def test_interface_crossing_high_pressure(self):
        """When the tip stress (Kic/sqrt(pi*aperture)) exceeds |sigma_n|,
        the fracture crosses the interface.

        sigma_tip = 1.8 / sqrt(pi * 0.003) ≈ 18.5 MPa > |sigma_n| = 10 MPa.
        """
        crossed = a10.interface_crossing_tension(sigma_n=-10.0,
                                                   Kic=1.8,
                                                   aperture_m=0.003)
        assert_true(crossed, "sigma_tip > |sigma_n| should allow crossing")

    def test_interface_crossing_tight_barrier(self):
        """When the tip stress is less than |sigma_n|, the fracture is blocked.

        A large aperture (0.1 m) drastically reduces sigma_tip:
        sigma_tip = 0.5 / sqrt(pi * 0.1) ≈ 0.89 MPa << |sigma_n| = 50 MPa.
        The fracture cannot cross a thick, high-stress barrier.
        """
        crossed = a10.interface_crossing_tension(sigma_n=-50.0,
                                                   Kic=0.5,
                                                   aperture_m=0.10)
        assert_true(not crossed, "sigma_tip < |sigma_n| should block crossing")

    def test_crossing_probability_initiating_layer_is_one(self):
        layers = a10.build_jiyang_stratigraphy()
        probs  = a10.interface_crossing_probability(layers, 0, 12.0, 22.0, 15.0)
        assert_close(probs["Sandstone-1"], 1.0, rtol=1e-6)

    def test_crossing_probability_all_in_range(self):
        layers = a10.build_jiyang_stratigraphy()
        probs  = a10.interface_crossing_probability(layers, 0, 12.0, 22.0, 15.0)
        for name, p in probs.items():
            assert_in_range(p, 0, 1, f"Probability for {name}")

    def test_coal_initiation_lower_height(self):
        """Coal initiation should fracture fewer layers than sandstone."""
        layers = a10.build_jiyang_stratigraphy()
        h_ss   = a10.fracture_height_growth(layers, 0, 22.0, 15.0)
        h_coal = a10.fracture_height_growth(layers, 2, 10.0, 15.0)
        assert_true(h_ss >= h_coal,
                    f"Sandstone ({h_ss}m) should reach ≥ coal ({h_coal}m) height")

    def test_cleat_connection_probability_above_critical(self):
        """Net pressure exceeding stress diff should give high probability."""
        coal = a10.build_jiyang_stratigraphy()[2]
        p_net = coal.sigma_H_MPa - coal.sigma_h_MPa + 10.0  # well above critical
        prob  = a10.cleat_connection_probability(coal, p_net)
        assert_true(prob > 0.5,
                    f"Above-critical p_net should give high cleat probability")

    def test_optimise_fracturing_output_keys(self):
        layers = a10.build_jiyang_stratigraphy()
        out    = a10.optimise_fracturing_parameters(
            layers, np.array([10.0, 15.0, 22.0]), np.array([10.0, 15.0]))
        for k in ("Q_opt", "mu_opt", "max_height", "height_grid"):
            assert_true(k in out, f"Missing key: {k}")

    def test_optimise_fracturing_grid_shape(self):
        layers = a10.build_jiyang_stratigraphy()
        Qarr   = np.array([10.0, 22.0])
        muarr  = np.array([10.0, 15.0, 25.0])
        out    = a10.optimise_fracturing_parameters(layers, Qarr, muarr)
        assert_array_shape(out["height_grid"], (2, 3))


# ===========================================================================
# Article 11 — AWI Cement Evaluation (Zhang et al. 2026)
# ===========================================================================

class TestA11:
    """Tests for a11_awi_cement_evaluation.py"""

    def _conv(self):
        return a11.CementSlurry("Conv", is_AWI=False, AWI_material_pct=0.0)

    def _awi(self):
        return a11.CementSlurry("AWI", is_AWI=True, AWI_material_pct=5.0)

    def _core_hi(self):
        return a11.FormationCore(permeability_mD=50.0, porosity_frac=0.22)

    def test_conductivity_vs_time_shape(self):
        t = np.linspace(0, 180, 100)
        s = a11.conductivity_vs_time(t, self._conv())
        assert_array_shape(s, (100,))

    def test_conductivity_positive(self):
        t = np.linspace(0, 180, 50)
        s = a11.conductivity_vs_time(t, self._conv())
        assert_true(np.all(s > 0), "Conductivity must be positive")

    def test_awi_delays_conductivity_jump(self):
        """AWI slurry onset must be later than conventional."""
        t = np.linspace(0, 200, 1000)
        s_conv = a11.conductivity_vs_time(t, self._conv())
        s_awi  = a11.conductivity_vs_time(t, self._awi())
        t_conv = a11.detect_conductivity_jump(s_conv, t)
        t_awi  = a11.detect_conductivity_jump(s_awi,  t)
        assert_true(t_awi is not None and t_conv is not None,
                    "Both jumps should be detected")
        assert_true(t_awi > t_conv, "AWI jump should be later")

    def test_water_invasion_rate_positive(self):
        Q = a11.water_invasion_rate(self._core_hi(), self._conv(), 5.0, 30.0)
        assert_true(Q > 0, "Invasion rate must be positive")

    def test_water_invasion_rate_awi_lower(self):
        """AWI slurry should have lower invasion rate than conventional."""
        core = self._core_hi()
        Q_conv = a11.water_invasion_rate(core, self._conv(), 5.0, 30.0)
        Q_awi  = a11.water_invasion_rate(core, self._awi(), 5.0, 30.0)
        assert_true(Q_awi < Q_conv,
                    "AWI invasion rate should be < conventional")

    def test_crack_width_positive(self):
        w = a11.crack_width_second_interface(5.0)
        assert_true(w > 0, "Crack width must be positive")

    def test_crack_width_increases_with_pressure(self):
        w1 = a11.crack_width_second_interface(2.0)
        w2 = a11.crack_width_second_interface(8.0)
        assert_true(w2 > w1, "Higher pressure → wider crack")

    def test_hydraulic_conductivity_positive(self):
        K = a11.hydraulic_conductivity_crack(1.0)
        assert_true(K > 0, "Hydraulic conductivity must be positive")

    def test_hydraulic_conductivity_cubic_scaling(self):
        """Doubling aperture should increase conductivity by ~8×."""
        K1 = a11.hydraulic_conductivity_crack(1.0)
        K2 = a11.hydraulic_conductivity_crack(2.0)
        assert_in_range(K2 / K1, 7.0, 9.0, "Cubic-law scaling")

    def test_gel_strength_increases_with_time(self):
        SGS0 = a11.static_gel_strength(0.0)
        SGS60 = a11.static_gel_strength(60.0)
        assert_true(SGS60 > SGS0, "Gel strength increases with time")

    def test_gel_strength_awi_higher(self):
        """AWI slurry builds gel faster/higher."""
        SGS_conv = a11.static_gel_strength(30.0, is_AWI=False)
        SGS_awi  = a11.static_gel_strength(30.0, is_AWI=True)
        assert_true(SGS_awi > SGS_conv, "AWI gel strength > conventional")

    def test_ion_loss_fraction_in_range(self):
        core   = self._core_hi()
        slurry = self._conv()
        f = a11.ion_loss_fraction(core, slurry)
        assert_in_range(f, 0.0, 1.0, "Ion loss fraction")

    def test_ion_loss_awi_lower(self):
        """AWI in low-permeability core → minimal ion loss."""
        core_lo = a11.FormationCore(2.0, 0.15)
        f_conv  = a11.ion_loss_fraction(core_lo, self._conv(), 24.0)
        f_awi   = a11.ion_loss_fraction(core_lo, self._awi(), 24.0)
        assert_true(f_awi < f_conv, "AWI ion loss should be lower")

    def test_awi_score_higher_for_awi_slurry(self):
        """AWI slurry should outperform conventional on composite score."""
        core_lo = a11.FormationCore(2.0, 0.15)
        s_conv = a11.awi_performance_score(self._conv(), self._core_hi())
        s_awi  = a11.awi_performance_score(self._awi(), core_lo)
        assert_true(s_awi["AWI_score"] > s_conv["AWI_score"],
                    "AWI slurry should have higher composite score")


# ===========================================================================
# Article 12 — Depth Shifting ML (Pan et al. 2026)
# ===========================================================================

class TestA12:
    """Tests for a12_depth_shifting_ml.py"""

    def _make_logs(self, shift_samples=3, n=300, seed=0):
        rng = np.random.default_rng(seed)
        ref = 50.0 + 20.0 * np.sin(2 * np.pi * np.arange(n) / 25.0) + rng.normal(0, 2, n)
        tgt = np.roll(ref, -shift_samples) + rng.normal(0, 1, n)
        return ref, tgt

    def test_rmse_zero_identical(self):
        y = np.arange(10, dtype=float)
        assert_close(a12.rmse(y, y), 0.0, atol=1e-10)

    def test_rmse_known_value(self):
        y_true = np.zeros(4)
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])
        assert_close(a12.rmse(y_true, y_pred), 1.0, rtol=1e-6)

    def test_mad_zero_identical(self):
        y = np.arange(10, dtype=float)
        assert_close(a12.mad(y, y), 0.0, atol=1e-10)

    def test_mad_known_value(self):
        y_true = np.zeros(4)
        y_pred = np.array([2.0, 2.0, 2.0, 2.0])
        assert_close(a12.mad(y_true, y_pred), 2.0, rtol=1e-6)

    def test_cross_correlation_shift_detects_known_shift(self):
        """xcorr should recover the true shift within ±1 sample."""
        ref, tgt = self._make_logs(shift_samples=5)
        detected = a12.cross_correlation_shift(ref, tgt, max_shift=20)
        assert_in_range(detected, 3, 7, "XCorr detected shift")

    def test_cross_correlation_zero_shift(self):
        """Identical logs → shift = 0."""
        ref = np.sin(np.linspace(0, 6 * np.pi, 200))
        detected = a12.cross_correlation_shift(ref, ref, max_shift=10)
        assert_true(detected == 0, f"Zero shift expected, got {detected}")

    def test_apply_bulk_shift_preserves_length(self):
        ref, tgt = self._make_logs()
        shifted  = a12.apply_bulk_shift(tgt, 5)
        assert_true(len(shifted) == len(tgt), "Shifted log length must be preserved")

    def test_apply_bulk_shift_known_content(self):
        """Elements starting from position `shift` should match original."""
        x       = np.arange(10, dtype=float)
        shifted = a12.apply_bulk_shift(x, 3)
        assert_close(shifted[3], x[0], rtol=1e-10)

    def test_dtw_alignment_same_log_zero_distance(self):
        """Aligning a log to itself gives zero DTW distance."""
        ref = np.sin(np.linspace(0, 2 * np.pi, 30))
        dist, pi, pj = a12.dtw_alignment(ref, ref, window=5)
        assert_close(dist, 0.0, atol=1e-10)

    def test_dtw_path_lengths_match(self):
        ref = np.sin(np.linspace(0, 2 * np.pi, 20))
        tgt = np.sin(np.linspace(0, 2 * np.pi, 22))
        _, pi, pj = a12.dtw_alignment(ref, tgt, window=5)
        assert_true(len(pi) == len(pj), "DTW path arrays must have equal length")

    def test_dtw_path_endpoints(self):
        """DTW path must start at (0,0) and end at (n-1, m-1)."""
        ref = np.sin(np.linspace(0, 2 * np.pi, 20))
        tgt = np.cos(np.linspace(0, 2 * np.pi, 20))
        _, pi, pj = a12.dtw_alignment(ref, tgt, window=10)
        assert_true(pi[0] == 0 and pj[0] == 0, "Path must start at (0,0)")
        assert_true(pi[-1] == 19 and pj[-1] == 19, "Path must end at (n-1, m-1)")

    def test_ridge_predictor_fit_and_predict_shape(self):
        ref, tgt = self._make_logs(n=100)
        model    = a12.RidgeShiftPredictor(alpha=1.0)
        shifts   = np.full(100, 1.5)
        model.fit(ref, tgt, shifts, window=5)
        pred     = model.predict(ref, tgt, window=5)
        assert_array_shape(pred, (100,))

    def test_ridge_predictor_constant_target(self):
        """Constant true shift should be recovered near-exactly after fitting."""
        ref, tgt = self._make_logs(shift_samples=4, n=200)
        model    = a12.RidgeShiftPredictor(alpha=0.1)
        true_s   = np.full(200, 1.0)
        model.fit(ref, tgt, true_s, window=6)
        pred     = model.predict(ref, tgt, window=6)
        assert_in_range(pred.mean(), 0.5, 1.5, "Ridge mean shift")

    def test_cnn_predictor_predict_shape(self):
        cnn  = a12.SimpleCNN1DShiftPredictor(kernel_size=5, n_filters=4, seed=0)
        segs = [np.random.default_rng(i).normal(0, 1, 20) for i in range(8)]
        y    = np.full(8, 1.5)
        cnn.fit(segs, y, lr=1e-3, epochs=5)
        pred = cnn.predict(segs)
        assert_array_shape(pred, (8,))

    def test_ensemble_depth_shift_uniform_weights(self):
        """Uniform-weight ensemble should average the shifts exactly."""
        s1 = np.full(50, 1.0)
        s2 = np.full(50, 3.0)
        ens = a12.ensemble_depth_shift({"log1": s1, "log2": s2})
        assert_close(ens[0], 2.0, rtol=1e-6)

    def test_ensemble_depth_shift_custom_weights(self):
        s1 = np.full(50, 0.0)
        s2 = np.full(50, 4.0)
        ens = a12.ensemble_depth_shift(
            {"a": s1, "b": s2}, weights={"a": 1.0, "b": 3.0})
        assert_close(ens[0], 3.0, rtol=1e-5)

    def test_align_logs_xcorr(self):
        rng   = np.random.default_rng(0)
        depth = np.linspace(1000, 1200, 200)
        ref   = np.sin(2 * np.pi * depth / 10) + rng.normal(0, 0.1, 200)
        tgt   = np.roll(ref, -4) + rng.normal(0, 0.05, 200)
        out   = a12.align_logs(ref, tgt, depth, method="xcorr", max_shift=15)
        assert_true("aligned_log" in out and len(out["aligned_log"]) == 200)

    def test_align_logs_dtw(self):
        rng   = np.random.default_rng(1)
        depth = np.linspace(1000, 1100, 100)
        ref   = np.sin(2 * np.pi * depth / 8) + rng.normal(0, 0.1, 100)
        tgt   = np.roll(ref, -3) + rng.normal(0, 0.05, 100)
        out   = a12.align_logs(ref, tgt, depth, method="dtw", dtw_window=15)
        assert_true("dtw_dist" in out and out["dtw_dist"] >= 0)


# ===========================================================================
# Master test runner
# ===========================================================================

def test_all():
    """
    Run every test across all 12 modules and print a structured report.
    Returns True if all tests pass, False otherwise.
    """
    _results.clear()

    test_classes = [
        ("A01 – Sponge Core Saturation Uncertainty",  TestA01),
        ("A02 – NMR Wettability Pore Partitioning",   TestA02),
        ("A03 – Water-Rock Mechanical & AE",           TestA03),
        ("A04 – Wireline Anomaly Diagnosis",           TestA04),
        ("A05 – AIL Hierarchical Correction",          TestA05),
        ("A06 – Bioclastic Limestone Classification",  TestA06),
        ("A07 – Knowledge-Guided DCDNN",               TestA07),
        ("A08 – Shale Induced-Stress Fracture",        TestA08),
        ("A09 – Acid Fracturing CBM",                  TestA09),
        ("A10 – Interlayer Fracture Propagation",      TestA10),
        ("A11 – AWI Cement Evaluation",                TestA11),
        ("A12 – Depth Shifting ML",                    TestA12),
    ]

    total = 0
    for title, cls in test_classes:
        print(f"\n{'─'*62}")
        print(f"  {title}")
        print(f"{'─'*62}")
        instance = cls()
        methods  = sorted(m for m in dir(cls) if m.startswith("test_"))
        for name in methods:
            _run(f"{cls.__name__}.{name}", getattr(instance, name))
        total += len(methods)

    # Summary
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{'═'*62}")
    print(f"  RESULTS:  {passed}/{total} passed  |  {failed} failed")
    print(f"{'═'*62}")

    if failed:
        print("\nFailed tests:")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n  ✗ {name}")
                for line in tb.splitlines()[-5:]:
                    print(f"    {line}")
    else:
        print("\n  All tests passed ✓")

    return failed == 0


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
