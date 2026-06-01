"""
test_all.py — Comprehensive unit tests for all 10 modules in
src2026_06 (Petrophysics Vol. 67, No. 3, June 2026).

Run with:
    python test_all.py              # prints pass/fail per test
    python -m pytest test_all.py -v # via pytest

Each test class corresponds to one article module.  Tests use only synthetic
data and numpy — no external fixtures needed.
"""

import math
import sys
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make the modules importable regardless of working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import a01_carbonate_pore_type_dielectric      as a01
import a02_mf_dielectric_fracture_sensitivity  as a02
import a03_pore_size_fluid_movement            as a03
import a04_cretaceous_depositional_model       as a04
import a05_udar_anisotropy_sensitivity         as a05
import a06_deterministic_inversion_uncertainty as a06
import a07_3d_lookahead_em_inversion           as a07
import a08_mud_gas_ratio_fluid_id              as a08
import a09_mf_dielectric_emulsion              as a09
import a10_acoustic_emission_multiphase        as a10


# ---------------------------------------------------------------------------
# Tiny assertion helpers (no pytest required)
# ---------------------------------------------------------------------------

def assert_close(a, b, rtol=1e-6, atol=1e-9, msg=""):
    if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol):
        raise AssertionError(f"{msg}  got {a}, expected ~{b}")


def assert_true(expr, msg=""):
    if not expr:
        raise AssertionError(msg or "Assertion failed")


def assert_in_range(val, lo, hi, msg=""):
    if not (lo <= float(val) <= hi):
        raise AssertionError(f"{msg}  {val} not in [{lo}, {hi}]")


# ===========================================================================
# Article 1 — Carbonate Pore Type vs Dielectric Permittivity
# ===========================================================================

class TestA01:
    def test_complex_permittivity_sign(self):
        z = a01.complex_permittivity(18.0, 5.0)
        assert_close(z.real, 18.0)
        assert_true(z.imag < 0, "imaginary part should be negative (e' - i e'')")

    def test_loss_tangent(self):
        assert_close(a01.loss_tangent(20.0, 4.0), 0.2)

    def test_imag_from_conductivity_positive(self):
        epp = a01.imag_from_conductivity(0.5, 12e6)
        assert_true(epp > 0, "e'' must be positive")

    def test_normalized_permittivity(self):
        assert_close(a01.normalized_permittivity(6.0, 0.30), 20.0)

    def test_normalized_permittivity_bad_phi(self):
        try:
            a01.normalized_permittivity(6.0, 0.0)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_perimeter_over_area(self):
        assert_close(a01.perimeter_over_area([10, 20], [5, 5]), 3.0)

    def test_dominant_pore_size_midpoint(self):
        dom = a01.dominant_pore_size([1, 2, 3, 4], [1, 1, 1, 1])
        assert_in_range(dom, 2, 3, "DOMSize at 50% cum area")

    def test_aspect_ratio_bounds(self):
        ar = a01.aspect_ratio(10.0, 4.0)
        assert_close(ar, 0.4)
        assert_in_range(ar, 0.0, 1.0)

    def test_classify_moldic(self):
        s = a01.CarbonateSample("m", 0.30, 1468, 13.5, 45.0, 320.0, 0.34)
        assert_true(a01.classify_pore_type(s) == "moldic")

    def test_classify_intercrystalline(self):
        s = a01.CarbonateSample("i", 0.18, 3.9, 18.0, 250.0, 35.0, 0.58)
        assert_true(a01.classify_pore_type(s) == "intercrystalline")


# ===========================================================================
# Article 2 — MF Dielectric Fracture Sensitivity
# ===========================================================================

class TestA02:
    def test_debye_limits(self):
        lo = a02.cole_cole_permittivity(1.0, eps_inf=6.0, d_eps=40.0, tau=1e-8)
        hi = a02.cole_cole_permittivity(1e12, eps_inf=6.0, d_eps=40.0, tau=1e-8)
        assert_close(lo.real, 46.0, rtol=1e-3, msg="static limit e_inf+d_eps")
        assert_close(hi.real, 6.0, rtol=1e-2, msg="high-freq limit e_inf")

    def test_conductivity_positive(self):
        sig = a02.effective_conductivity(1e8, 5.0)
        assert_true(float(sig) > 0)

    def test_relaxation_regime(self):
        assert_true(a02.relaxation_regime(1e3, 4e-8) == "interfacial")
        assert_true(a02.relaxation_regime(2e9, 4e-8) == "bulk")

    def test_area_between_curves_zero(self):
        f = [1e6, 1e7, 1e8]
        a = a02.area_between_curves(f, [10, 10, 10], [10, 10, 10])
        assert_close(a, 0.0, atol=1e-9)

    def test_area_between_curves_positive(self):
        f = [1e6, 1e7, 1e8]
        a = a02.area_between_curves(f, [10, 10, 10], [20, 20, 20])
        assert_true(a > 0)

    def test_percent_increase(self):
        assert_close(a02.percent_increase(50.0, 60.0), 20.0)

    def test_case_gain(self):
        c = a02.FractureCase("carbonate", "vertical", "near",
                             45.0, 70.0, 0.30, 0.40)
        assert_close(c.eps_gain, 25.0)
        assert_true(c.cond_pct > 0)

    def test_orientation_rule(self):
        assert_true(a02.orientation_more_sensitive("carbonate") == "vertical")
        assert_true(a02.orientation_more_sensitive("sandstone") == "ambiguous")


# ===========================================================================
# Article 3 — Pore-Size Distribution & Fluid Movement
# ===========================================================================

class TestA03:
    def test_archie_monotonic(self):
        sw_lo = a03.archie_sw(0.05, 50.0, 0.2, 2.0)
        sw_hi = a03.archie_sw(0.05, 5.0, 0.2, 2.0)
        assert_true(sw_hi > sw_lo, "lower Rt -> higher Sw")

    def test_archie_clipped(self):
        sw = a03.archie_sw(0.05, 0.01, 0.2, 2.0)
        assert_in_range(sw, 0.0, 1.0)

    def test_rw_decreases_with_salinity(self):
        assert_true(a03.rw_from_salinity(200_000) < a03.rw_from_salinity(20_000))

    def test_partition_sums_to_one(self):
        t2 = np.logspace(-1, 3, 50)
        amp = np.ones_like(t2)
        p = a03.partition_porosity(t2, amp)
        assert_close(p["micro"] + p["meso"] + p["macro"], 1.0, rtol=1e-9)

    def test_t2_logmean(self):
        t2 = np.array([10.0, 100.0])
        lm = a03.t2_logmean(t2, np.array([1.0, 1.0]))
        assert_close(lm, math.sqrt(1000.0), rtol=1e-6)

    def test_sdr_increases_with_porosity(self):
        assert_true(a03.sdr_permeability(0.25, 100) >
                    a03.sdr_permeability(0.15, 100))

    def test_timur_coates(self):
        k = a03.timur_coates_permeability(0.2, 0.15, 0.05)
        assert_true(k > 0)

    def test_forward_model_porosity(self):
        m = a03.FormationModel(0.15, 0.05, 100_000, 2.0)
        assert_close(m.porosity, 0.20)
        assert_close(m.sw, 0.75)

    def test_gaussian_peak(self):
        x = np.array([0.0, 1.0, 2.0])
        modes = [a03.GaussianMode(2.0, 1.0, 0.5)]
        y = a03.gaussian_t2_model(x, modes)
        assert_close(y[1], 2.0, rtol=1e-6)


# ===========================================================================
# Article 4 — Cretaceous Depositional Model
# ===========================================================================

class TestA04:
    def test_dunham_grainstone(self):
        assert_true(a04.dunham_class(0.7, True, has_mud=False) == "grainstone")

    def test_dunham_mudstone(self):
        assert_true(a04.dunham_class(0.02, False) == "mudstone")

    def test_dunham_rank_order(self):
        assert_true(a04.dunham_energy_rank("grainstone") >
                    a04.dunham_energy_rank("mudstone"))

    def test_synthetic_resistivity_inverse(self):
        sres = a04.synthetic_resistivity([0.0, 1.0], 2.0, 200.0)
        assert_true(sres[0] > sres[1], "high conductivity -> low resistivity")

    def test_electrofacies_roundtrip(self):
        core = {"GR": [25, 80], "RHOB": [2.68, 2.45]}
        labels = ["grain", "mud"]
        cents = a04.train_electrofacies(core, labels)
        pred = a04.classify_electrofacies(
            cents, {"GR": [26, 78], "RHOB": [2.67, 2.46]})
        assert_true(pred == ["grain", "mud"])

    def test_flag_unconformity(self):
        poro = np.array([0.08, 0.20, 0.10])
        gr = np.array([60, 25, 55])
        flags = a04.flag_unconformity(poro, gr)
        assert_true(bool(flags[1]))

    def test_trend(self):
        assert_true(a04.trend_regressive([1, 2, 3, 4]) == "regressive")
        assert_true(a04.trend_regressive([4, 3, 2, 1]) == "transgressive")

    def test_rank_eod(self):
        assert_true(a04.rank_eod("rudist buildup") >
                    a04.rank_eod("lagoonal mudstone"))


# ===========================================================================
# Article 5 — UDAR Anisotropy Sensitivity
# ===========================================================================

class TestA05:
    def test_skin_depth_scaling(self):
        d1 = a05.skin_depth(10.0, 1e4)
        d2 = a05.skin_depth(40.0, 1e4)
        assert_true(d2 > d1, "higher resistivity -> deeper skin depth")

    def test_induction_number(self):
        ln = a05.induction_number(10.0, 10.0, 1e4)
        assert_true(ln > 0)

    def test_field_regime(self):
        assert_true(a05.field_regime(1.0, 100.0, 1e3) == "near-field")
        assert_true(a05.field_regime(500.0, 1.0, 1e6) == "far-field")

    def test_anisotropy_ratio(self):
        assert_close(a05.anisotropy_ratio(10.0, 40.0), 2.0)

    def test_sensitivity_increases_with_spacing(self):
        s_short = a05.anisotropy_sensitivity(10.0, 1.5, 5.0, 4e4)
        s_long = a05.anisotropy_sensitivity(10.0, 1.5, 40.0, 4e4)
        assert_true(s_long > s_short, "longer spacing -> more sensitivity")

    def test_sensitivity_increases_with_frequency(self):
        s_lo = a05.anisotropy_sensitivity(10.0, 1.5, 20.0, 2e3)
        s_hi = a05.anisotropy_sensitivity(10.0, 1.5, 20.0, 4e4)
        assert_true(s_hi > s_lo, "higher frequency -> more sensitivity")

    def test_dominant_factor_resistivity(self):
        df = a05.dominant_factor([2.0, 10.0, 50.0, 200.0],
                                 [5.0, 20.0, 40.0], [2e3, 1e4, 4e4])
        assert_true(df == "resistivity")


# ===========================================================================
# Article 6 — Deterministic Inversion Uncertainty
# ===========================================================================

class TestA06:
    @staticmethod
    def _forward(m):
        offsets = np.linspace(1.0, 10.0, 12)
        return np.log(m[0]) * np.exp(-offsets / 6.0) + \
            np.log(m[1]) * (1.0 - np.exp(-offsets / 6.0))

    def test_single_inversion_recovers(self):
        data = self._forward(np.array([5.0, 50.0]))
        m, rms = a06.deterministic_inversion(
            self._forward, data, [4.0, 40.0], [(1, 20), (10, 200)])
        assert_close(m[0], 5.0, rtol=0.1)
        assert_close(m[1], 50.0, rtol=0.1)
        assert_true(rms < 1e-2)

    def test_multistart_average(self):
        data = self._forward(np.array([5.0, 50.0]))
        ms = a06.multistart_inversion(
            self._forward, data, [(1, 20), (10, 200)], n_starts=20)
        assert_true(ms["models"].shape == (20, 2))
        assert_close(ms["average"][0], 5.0, rtol=0.2)

    def test_feasible_set_brackets(self):
        data = self._forward(np.array([5.0, 50.0]))
        ms = a06.multistart_inversion(
            self._forward, data, [(1, 20), (10, 200)], n_starts=20)
        fs = a06.feasible_set_sampling(
            self._forward, data, ms["average"], [(1, 20), (10, 200)],
            noise_level=0.05, n_samples=500)
        assert_true(fs["p5"][0] <= fs["p50"][0] <= fs["p95"][0])
        assert_true(fs["n_accepted"] >= 1)


# ===========================================================================
# Article 7 — 3D Look-Ahead EM Inversion
# ===========================================================================

class TestA07:
    def test_lookahead_limits(self):
        far = a07.lookahead_apparent_resistivity(20.0, 2.0, 1e6, 15.0)
        near = a07.lookahead_apparent_resistivity(20.0, 2.0, 0.0, 15.0)
        assert_close(far, 20.0, rtol=1e-3, msg="far away -> rho_here")
        assert_close(near, 2.0, rtol=1e-3, msg="at boundary -> rho_ahead")

    def test_dtb_inverse(self):
        ra = a07.lookahead_apparent_resistivity(20.0, 2.0, 8.0, 15.0)
        dtb = a07.distance_to_boundary(ra, 20.0, 2.0, 15.0)
        assert_close(dtb, 8.0, rtol=1e-3)

    def test_boundary_azimuth(self):
        az = np.arange(0, 360, 30)
        sig = 1.0 + 0.4 * np.cos(np.radians(az - 120.0))
        b = a07.boundary_from_azimuthal(sig, az, dtb=8.0)
        assert_in_range(b.azimuth_deg, 110, 130)
        assert_true(b.dip_deg > 0)

    def test_tracker_converges(self):
        tr = a07.DTBTracker(dtb=30.0)
        for meas in (24.0, 17.0, 11.0, 6.0):
            tr.update(meas)
        assert_true(tr.dtb < 30.0, "tracked DTB should decrease toward measurements")


# ===========================================================================
# Article 8 — Mud Gas Ratio Fluid ID
# ===========================================================================

class TestA08:
    def test_wetness_ratio(self):
        g = a08.GasReading(90, 5, 3, 1, 1)
        assert_close(a08.wetness_ratio(g), 10.0, rtol=1e-6)

    def test_balance_ratio(self):
        g = a08.GasReading(90, 5, 3, 1, 1)
        assert_close(a08.balance_ratio(g), 95.0 / 5.0)

    def test_character_ratio(self):
        g = a08.GasReading(90, 5, 4, 1, 1)
        assert_close(a08.character_ratio(g), 0.5)

    def test_dry_gas(self):
        g = a08.GasReading(9800, 120, 40, 25, 15)
        assert_true("gas" in a08.classify_fluid(g))

    def test_heavy_oil(self):
        g = a08.GasReading(2500, 1600, 1400, 1800, 2700)
        assert_true("oil" in a08.classify_fluid(g))

    def test_eight_types(self):
        assert_true(len(a08.FLUID_TYPES) == 8)

    def test_density_index_monotonic(self):
        dry = a08.GasReading(9900, 50, 20, 10, 10)
        oily = a08.GasReading(3000, 1500, 1300, 1600, 2000)
        assert_true(a08.fluid_density_index(oily) >
                    a08.fluid_density_index(dry))

    def test_normalize_reading(self):
        g = a08.normalize_reading({"C1": 100, "iC4": 5, "nC4": 5})
        assert_close(g.c4, 10.0)


# ===========================================================================
# Article 9 — MF Dielectric Emulsion
# ===========================================================================

class TestA09:
    def test_maxwell_garnett_limit(self):
        eh = complex(2.3, 0.0)
        z = a09.maxwell_garnett(eh, complex(70.0, 0.0), 0.0)
        assert_close(z.real, 2.3, rtol=1e-6)

    def test_mixing_increases_with_water(self):
        eh, ei = complex(2.3, 0.0), complex(70.0, 0.0)
        lo = a09.maxwell_garnett(eh, ei, 0.1).real
        hi = a09.maxwell_garnett(eh, ei, 0.4).real
        assert_true(hi > lo, "more water -> higher permittivity")

    def test_bruggeman_runs(self):
        z = a09.bruggeman(complex(2.3, 0.0), complex(70.0, 0.0), 0.3)
        assert_true(z.real > 2.3)

    def test_emulsion_type_ow(self):
        s = a09.EmulsionState(55.0, 0.4, 0.7)
        assert_true("O/W" in a09.emulsion_type(s))

    def test_emulsion_type_wo(self):
        s = a09.EmulsionState(8.0, 1e-4, 0.25)
        assert_true("W/O" in a09.emulsion_type(s))

    def test_inversion_proximity(self):
        assert_close(a09.inversion_point_proximity(0.5, 0.5), 0.0)

    def test_separation_monotonic(self):
        s = a09.separation_fraction(np.array([0.0, 10.0, 100.0]), 0.02)
        assert_true(s[0] < s[1] < s[2])
        assert_close(s[0], 0.0, atol=1e-9)

    def test_droplet_radius_positive(self):
        r = a09.droplet_radius_from_relaxation(1e-7, 70.0, 0.4)
        assert_true(r > 0)


# ===========================================================================
# Article 10 — Acoustic Emission Multiphase
# ===========================================================================

class TestA10:
    def _wave(self, freq, amp, n=2048, fs=1e6):
        t = np.arange(n) / fs
        return amp * np.sin(2 * np.pi * freq * t)

    def test_rms(self):
        w = self._wave(1e4, 1.0)
        assert_close(a10.ae_rms(w), 1.0 / math.sqrt(2.0), rtol=1e-2)

    def test_energy_scales_with_amplitude(self):
        lo = a10.acoustic_energy(self._wave(1e4, 1.0))
        hi = a10.acoustic_energy(self._wave(1e4, 2.0))
        assert_close(hi / lo, 4.0, rtol=1e-2)

    def test_ring_down_count(self):
        w = self._wave(1e4, 1.0)
        assert_true(a10.ring_down_count(w, 0.5) > 0)

    def test_spectral_centroid(self):
        # Higher-frequency content -> higher spectral centroid (monotonic).
        c_lo = a10.spectral_centroid(self._wave(1e4, 1.0), 1e6)
        c_hi = a10.spectral_centroid(self._wave(8e4, 1.0), 1e6)
        assert_true(c_hi > c_lo, "higher tone -> higher centroid")
        assert_true(c_lo > 0)

    def test_extract_features(self):
        f = a10.extract_features(self._wave(1e4, 1.0), 1e6)
        assert_true(f.energy > 0 and f.rms > 0)

    def test_energy_rate_calibration(self):
        rates = [50, 100, 200, 400]
        energies = [a := 2.0 * (q ** 1.8) for q in rates]
        coef_a, b = a10.fit_energy_rate(rates, energies)
        assert_close(b, 1.8, rtol=1e-3)
        q = a10.rate_from_energy(energies[1], coef_a, b)
        assert_close(q, 100.0, rtol=1e-2)

    def test_classify_phase_gas(self):
        f = a10.extract_features(self._wave(8e4, 1.5), 1e6)
        assert_true("gas" in a10.classify_phase(f))

    def test_breakthrough_detection(self):
        flags = a10.detect_breakthrough([100, 105, 110, 600, 650])
        assert_true(3 in flags)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

_results: list = []


def _run(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  [PASS] {name}")
    except Exception:
        tb = traceback.format_exc()
        _results.append((name, False, tb))
        print(f"  [FAIL] {name}")
        for line in tb.splitlines()[-4:]:
            print(f"         {line}")


def test_all():
    test_classes = [
        ("A01 – Carbonate Pore Type Dielectric",      TestA01),
        ("A02 – MF Dielectric Fracture Sensitivity",  TestA02),
        ("A03 – Pore-Size Distribution & Flow",       TestA03),
        ("A04 – Cretaceous Depositional Model",       TestA04),
        ("A05 – UDAR Anisotropy Sensitivity",         TestA05),
        ("A06 – Deterministic Inversion Uncertainty", TestA06),
        ("A07 – 3D Look-Ahead EM Inversion",          TestA07),
        ("A08 – Mud Gas Ratio Fluid ID",              TestA08),
        ("A09 – MF Dielectric Emulsion",              TestA09),
        ("A10 – Acoustic Emission Multiphase",        TestA10),
    ]

    total = 0
    for title, cls in test_classes:
        print(f"\n{'-'*62}")
        print(f"  {title}")
        print(f"{'-'*62}")
        instance = cls()
        methods = sorted(m for m in dir(cls) if m.startswith("test_"))
        for name in methods:
            _run(f"{cls.__name__}.{name}", getattr(instance, name))
        total += len(methods)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print(f"\n{'='*62}")
    print(f"  RESULTS:  {passed}/{total} passed  |  {failed} failed")
    print(f"{'='*62}")

    if failed:
        print("\nFailed tests:")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n  x {name}")
                for line in tb.splitlines()[-4:]:
                    print(f"    {line}")
    else:
        print("\n  All tests passed")

    return failed == 0


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
