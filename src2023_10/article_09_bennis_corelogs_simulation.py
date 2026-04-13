"""
article_09_bennis_corelogs_simulation.py
========================================
Implementation of ideas from:

    Bennis, M. and Torres-Verdin, C. (2023).  "Numerical Simulation of
    Well Logs Based on Core Measurements: An Effective Method for Data
    Quality Control and Improved Petrophysical Interpretation."
    Petrophysics, 64(5), 753-772.  DOI: 10.30632/PJV64N5-2023a9

The paper proposes computing *synthetic* well logs from high-resolution
core data via depth-resolved volumetric mixing models, then comparing
the synthetic logs to the recorded wireline logs.  Discrepancies flag
data-quality issues (bad borehole environmental corrections, biased
core measurements, depth shifts, etc.).

Concepts implemented:

    * Volumetric model: V_qtz + V_clay + V_calc + phi*Sw + phi*(1-Sw) = 1
    * Forward log models (in the order used by Bennis & Torres-Verdin):
          GR    = sum(V_i * GR_i)
          rhoB  = sum(V_i * rho_i) + phi*(Sw*rho_w + (1-Sw)*rho_hc)
          NPHI  = sum(V_i * nphi_i) + phi*(Sw + (1-Sw)*HI_hc)
          DTC   = time-average:  1/Vp = sum(V_i / Vp_i) + phi/Vp_fluid
    * Upscaling: convolve high-resolution synthetic logs with a vertical
      response function (Gaussian) to mimic the wireline tool aperture
    * Bias detection: chi-square misfit of measured vs synthetic logs and
      shift / scale corrections
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# Mineral / fluid library
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MineralLog:
    rho: float        # g/cc
    nphi: float       # pu (limestone-equivalent)
    GR: float         # API
    Vp: float         # m/s


MINS: Dict[str, MineralLog] = {
    "qtz":   MineralLog(2.65, -2.0,   8.0, 5950.0),
    "clay":  MineralLog(2.45, 38.0, 250.0, 3200.0),
    "calc":  MineralLog(2.71,  0.0,   5.0, 6400.0),
    "dol":   MineralLog(2.87,  2.0,   8.0, 7050.0),
}
RHO_W = 1.05
RHO_HC = 0.75
HI_HC = 0.65
VP_FLUID = 1500.0


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------
@dataclass
class CoreSample:
    z: np.ndarray
    V_qtz: np.ndarray
    V_clay: np.ndarray
    V_calc: np.ndarray
    phi: np.ndarray
    Sw: np.ndarray

    def matrix_total(self) -> np.ndarray:
        return self.V_qtz + self.V_clay + self.V_calc


def synthetic_logs(s: CoreSample) -> dict:
    """Forward-model GR, rhoB, NPHI, Vp from core mineralogy and phi/Sw."""
    Vm = 1.0 - s.phi
    # Renormalise mineral fractions to sum to (1 - phi)
    sum_min = s.matrix_total()
    sum_min = np.where(sum_min < 1e-9, 1e-9, sum_min)
    fq = s.V_qtz / sum_min * Vm
    fc = s.V_clay / sum_min * Vm
    fk = s.V_calc / sum_min * Vm

    GR = fq * MINS["qtz"].GR + fc * MINS["clay"].GR + fk * MINS["calc"].GR

    rhoB = (fq * MINS["qtz"].rho + fc * MINS["clay"].rho
            + fk * MINS["calc"].rho
            + s.phi * (s.Sw * RHO_W + (1.0 - s.Sw) * RHO_HC))

    NPHI = (fq * MINS["qtz"].nphi / 100.0
            + fc * MINS["clay"].nphi / 100.0
            + fk * MINS["calc"].nphi / 100.0
            + s.phi * (s.Sw + (1.0 - s.Sw) * HI_HC))

    inv_vp = (fq / MINS["qtz"].Vp + fc / MINS["clay"].Vp
              + fk / MINS["calc"].Vp
              + s.phi / VP_FLUID)
    Vp = 1.0 / inv_vp
    return {"GR": GR, "RHOB": rhoB, "NPHI": NPHI, "Vp": Vp}


# ---------------------------------------------------------------------------
# Upscaling (vertical response function)
# ---------------------------------------------------------------------------
def vertical_response(x: np.ndarray, dz_m: float, fwhm_m: float) -> np.ndarray:
    """Convolve x with a Gaussian whose FWHM matches the tool aperture."""
    sigma = fwhm_m / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    n = max(3, int(round(6 * sigma / dz_m)) | 1)
    t = (np.arange(n) - n // 2) * dz_m
    g = np.exp(-0.5 * (t / sigma) ** 2)
    g /= g.sum()
    return np.convolve(x, g, mode="same")


# ---------------------------------------------------------------------------
# QC: chi-square misfit
# ---------------------------------------------------------------------------
def chi_square(measured: np.ndarray, synthetic: np.ndarray,
               sigma: float) -> float:
    return float(np.mean(((measured - synthetic) / sigma) ** 2))


def detect_bias(measured: np.ndarray, synthetic: np.ndarray
                ) -> tuple[float, float]:
    """Linear regression measured = a*synthetic + b - returns (a, b).
    a != 1 or b != 0 indicates an environmental-correction bias."""
    a, b = np.polyfit(synthetic, measured, 1)
    return float(a), float(b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_all() -> None:
    rng = np.random.default_rng(9)
    # Synthetic 50 m of clastic sequence at 1 cm core resolution
    z = np.arange(0.0, 50.0, 0.01)
    n = z.size
    base = 0.5 + 0.4 * np.sin(2 * np.pi * z / 8.0)
    V_qtz = np.clip(0.6 * base + 0.05 * rng.standard_normal(n), 0.01, None)
    V_clay = np.clip(0.4 * (1 - base) + 0.05 * rng.standard_normal(n),
                     0.01, None)
    V_calc = np.full(n, 0.05) + 0.02 * rng.standard_normal(n)
    V_calc = np.clip(V_calc, 0.01, None)
    phi = 0.05 + 0.18 * base + 0.01 * rng.standard_normal(n)
    phi = np.clip(phi, 0.02, 0.32)
    Sw = np.clip(0.3 + 0.5 * (1 - base) + 0.05 * rng.standard_normal(n),
                 0.1, 1.0)
    core = CoreSample(z, V_qtz, V_clay, V_calc, phi, Sw)
    synth = synthetic_logs(core)

    # Core-scale logs should respect physical bounds
    assert np.all(synth["GR"] >= 0)
    assert np.all((synth["RHOB"] > 1.5) & (synth["RHOB"] < 3.0))
    assert np.all(synth["NPHI"] > -0.1)
    assert np.all(synth["Vp"] > 1500)

    # Upscale to 30 cm wireline
    GR_log = vertical_response(synth["GR"], dz_m=0.01, fwhm_m=0.30)
    RHOB_log = vertical_response(synth["RHOB"], dz_m=0.01, fwhm_m=0.30)
    assert GR_log.shape == synth["GR"].shape

    # Simulate a *biased* measurement (e.g. bad borehole correction):
    GR_meas = 1.10 * GR_log + 5.0 + 1.0 * rng.standard_normal(n)
    chi0 = chi_square(GR_meas, GR_log, sigma=5.0)
    a, b = detect_bias(GR_meas, GR_log)
    # Bias parameters should be recovered close to (1.10, 5.0)
    assert abs(a - 1.10) < 0.03, a
    assert abs(b - 5.0) < 2.0, b
    # Removing the bias greatly reduces misfit
    GR_corr = (GR_meas - b) / a
    chi1 = chi_square(GR_corr, GR_log, sigma=5.0)
    assert chi1 < chi0 / 5.0, (chi1, chi0)

    print(f"article_09_bennis_corelogs_simulation: OK  "
          f"(chi^2 before={chi0:.2f}, after={chi1:.2f}, "
          f"bias=(a={a:.3f}, b={b:.2f}))")


if __name__ == "__main__":
    test_all()
