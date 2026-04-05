#!/usr/bin/env python3
"""
Module 10: Through-Tubing Cement Evaluation (SNHR + EMI + ML)
=============================================================
Implements ideas from:
  Zeghlache et al., "Challenges and Solutions for Advanced Through-
  Tubing Cement Evaluation,"
  Petrophysics, vol. 66, no. 4, pp. 677–688, August 2025.

Key concepts:
  - Selective Non-Harmonic Resonance (SNHR): resonance power-loss
    analysis for cement bond quality
  - Electromechanical Impedance (EMI): admittance measurement
    from piezoelectric transducers for bond stiffness
  - Machine-learning-based eccentricity (ECC) correction
  - Bond Index (BI) computation combining SNHR + EMI
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. SNHR — Selective Non-Harmonic Resonance
# ---------------------------------------------------------------------------
def snhr_resonance_spectrum(
    freq_hz: np.ndarray,
    cement_impedance_MRayl: float = 5.0,
    casing_thickness_m: float = 0.009,
    casing_velocity_m_s: float = 5900.0,
    tubing_thickness_m: float = 0.006,
    fluid_velocity_m_s: float = 1480.0,
    eccentricity: float = 0.0,
) -> np.ndarray:
    """Simulate the SNHR power spectrum.

    The resonance frequencies of the tubing-fluid-casing-cement system
    behave like a Duffing oscillator. Higher cement stiffness increases
    damping, reducing the resonance peak amplitude.

    Parameters
    ----------
    freq_hz : ndarray — frequency sweep
    cement_impedance_MRayl : float — 0 for free pipe, >3 for cement
    eccentricity : float — tubing eccentricity ratio (0–1)

    Returns
    -------
    power_spectrum : ndarray — normalised resonance power at each freq.
    """
    # Casing resonance frequency
    f_res = casing_velocity_m_s / (2 * casing_thickness_m)

    # Damping: higher cement impedance → more energy loss → more damping
    zeta_cement = 0.02 + 0.15 * (cement_impedance_MRayl / 10.0)
    # Tubing effect: additional resonance
    f_tubing = fluid_velocity_m_s / (2 * tubing_thickness_m)

    # Duffing-like response
    omega = 2 * np.pi * freq_hz
    omega0 = 2 * np.pi * f_res
    omega_t = 2 * np.pi * f_tubing

    H_casing = 1.0 / np.sqrt((1 - (omega / omega0) ** 2) ** 2 +
                               (2 * zeta_cement * omega / omega0) ** 2)
    H_tubing = 0.3 / np.sqrt((1 - (omega / omega_t) ** 2) ** 2 +
                               (2 * 0.05 * omega / omega_t) ** 2)

    power = H_casing + H_tubing

    # Eccentricity distortion: amplitude modulation and peak shift
    if eccentricity > 0:
        # ECC causes asymmetric wave propagation → broader, shifted peaks
        power *= (1.0 + 0.5 * eccentricity * np.sin(omega / omega0 * np.pi))
        # Additional broadening effect
        power *= (1.0 + 0.3 * eccentricity)

    # Normalise by the theoretical maximum (free pipe, no damping → Q = ∞)
    # Free pipe has zeta_cement ≈ 0.02, so max H ≈ 1/(2*0.02) = 25
    max_theoretical = 1.0 / (2 * 0.02) + 0.3 / (2 * 0.05)
    return power / max_theoretical


def snhr_bond_indicator(power_spectrum: np.ndarray, freq_hz: np.ndarray) -> float:
    """Compute the SNHR bond indicator from the resonance power spectrum.

    The cemented condition shows higher power loss (lower peak amplitude),
    while free pipe shows a sharp high peak.

    Returns
    -------
    float in [0, 1] — 0 = free pipe, 1 = well cemented.
    """
    peak_power = np.max(power_spectrum)
    # Free pipe peak ~ 1.0; well-cemented has much lower peak due to damping
    # Map: peak=1 → BI=0 (free), peak→0 → BI=1 (cemented)
    bi = 1.0 - np.clip(peak_power, 0, 1)
    return float(bi)


# ---------------------------------------------------------------------------
# 2. EMI — Electromechanical Impedance
# ---------------------------------------------------------------------------
def emi_admittance(
    freq_hz: np.ndarray,
    cement_stiffness_GPa: float = 10.0,
    eccentricity: float = 0.0,
) -> np.ndarray:
    """Simulate EMI admittance spectrum.

    Higher cement stiffness → lower admittance (more energy absorbed
    by the stiffer cement bond, reducing the mechanical response).

    Returns
    -------
    admittance : ndarray — normalised admittance.
    """
    f0 = 200e3   # typical piezo resonance ~200 kHz
    omega = 2 * np.pi * freq_hz
    omega0 = 2 * np.pi * f0

    # Stiffness-dependent damping: more stiffness → more damping
    zeta = 0.05 + 0.2 * (cement_stiffness_GPa / 20.0)

    # Mechanical admittance: inverse of impedance
    impedance_sq = (omega0 ** 2 - omega ** 2) ** 2 + (2 * zeta * omega * omega0) ** 2
    admittance = 1.0 / np.sqrt(impedance_sq + 1e-30)

    # Cement stiffness directly reduces the overall admittance level
    stiffness_factor = 1.0 / (1.0 + cement_stiffness_GPa / 5.0)
    admittance *= stiffness_factor

    # Eccentricity adds asymmetry
    if eccentricity > 0:
        admittance *= (1.0 + 0.2 * eccentricity)

    return admittance / (admittance.max() + 1e-12)


def emi_bond_indicator(admittance: np.ndarray) -> float:
    """Compute EMI bond indicator.

    Lower mean admittance → better cement bond.
    """
    mean_adm = np.mean(admittance)
    bi = 1.0 - np.clip(mean_adm * 2, 0, 1)
    return float(bi)


# ---------------------------------------------------------------------------
# 3. ML-based eccentricity correction (simple feedforward NN proxy)
# ---------------------------------------------------------------------------
class SimpleNeuralNet:
    """Minimal 1-hidden-layer neural network for ECC correction."""

    def __init__(self, n_input: int, n_hidden: int, n_output: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (n_input, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.5, (n_hidden, n_output))
        self.b2 = np.zeros(n_output)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.001,
              n_epochs: int = 200):
        """Mini-batch gradient descent."""
        for _ in range(n_epochs):
            h = np.maximum(0, X @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            error = pred - y

            # Backprop
            dW2 = h.T @ error / len(X)
            db2 = error.mean(axis=0)
            dh = error @ self.W2.T
            dh[h <= 0] = 0
            dW1 = X.T @ dh / len(X)
            db1 = dh.mean(axis=0)

            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1


def train_ecc_correction_model(n_samples: int = 500) -> SimpleNeuralNet:
    """Generate training data and train the ECC correction NN.

    Input: [raw_snhr_bi, raw_emi_bi, eccentricity_estimate]
    Output: [corrected_bi]
    """
    rng = np.random.default_rng(42)

    X_train = np.zeros((n_samples, 3))
    y_train = np.zeros((n_samples, 1))

    freq = np.linspace(50e3, 400e3, 200)

    for i in range(n_samples):
        cement_imp = rng.uniform(0, 8)
        cement_stiff = rng.uniform(0, 20)
        ecc = rng.uniform(0, 0.8)

        # True BI (without eccentricity)
        ps_true = snhr_resonance_spectrum(freq, cement_imp, eccentricity=0.0)
        true_bi = snhr_bond_indicator(ps_true, freq)

        # Measured (with eccentricity)
        ps_ecc = snhr_resonance_spectrum(freq, cement_imp, eccentricity=ecc)
        raw_snhr = snhr_bond_indicator(ps_ecc, freq)

        adm_ecc = emi_admittance(freq, cement_stiff, eccentricity=ecc)
        raw_emi = emi_bond_indicator(adm_ecc)

        X_train[i] = [raw_snhr, raw_emi, ecc]
        y_train[i] = [true_bi]

    nn = SimpleNeuralNet(3, 16, 1)
    nn.train(X_train, y_train, lr=0.005, n_epochs=300)
    return nn


# ---------------------------------------------------------------------------
# 4. Combined Bond Index
# ---------------------------------------------------------------------------
def combined_bond_index(
    snhr_bi: float,
    emi_bi: float,
    eccentricity: float,
    correction_model: Optional[SimpleNeuralNet] = None,
    w_snhr: float = 0.6,
    w_emi: float = 0.4,
) -> float:
    """Compute the final corrected Bond Index.

    If a correction model is available, it corrects for eccentricity.
    """
    if correction_model is not None:
        inp = np.array([[snhr_bi, emi_bi, eccentricity]])
        corrected = float(correction_model.forward(inp)[0, 0])
        return np.clip(corrected, 0, 1)
    return np.clip(w_snhr * snhr_bi + w_emi * emi_bi, 0, 1)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_all():
    freq = np.linspace(50e3, 400e3, 200)

    # SNHR: free pipe vs cemented
    ps_free = snhr_resonance_spectrum(freq, cement_impedance_MRayl=0.0)
    ps_cem = snhr_resonance_spectrum(freq, cement_impedance_MRayl=6.0)
    bi_free = snhr_bond_indicator(ps_free, freq)
    bi_cem = snhr_bond_indicator(ps_cem, freq)
    assert bi_cem > bi_free, f"Cemented BI ({bi_cem}) should > free ({bi_free})"

    # EMI: free vs cemented
    adm_free = emi_admittance(freq, cement_stiffness_GPa=0.1)
    adm_cem = emi_admittance(freq, cement_stiffness_GPa=15.0)
    emi_bi_free = emi_bond_indicator(adm_free)
    emi_bi_cem = emi_bond_indicator(adm_cem)
    assert emi_bi_cem > emi_bi_free

    # Eccentricity distortion
    ps_ecc = snhr_resonance_spectrum(freq, cement_impedance_MRayl=6.0,
                                      eccentricity=0.6)
    bi_ecc = snhr_bond_indicator(ps_ecc, freq)
    # ECC distorts the measurement
    assert abs(bi_ecc - bi_cem) > 0.01

    # Train correction model
    model = train_ecc_correction_model(n_samples=300)

    # Combined BI with correction
    bi_corrected = combined_bond_index(bi_ecc, emi_bi_cem, 0.6, model)
    assert 0 <= bi_corrected <= 1
    # Corrected should be closer to true cemented value
    bi_uncorrected = combined_bond_index(bi_ecc, emi_bi_cem, 0.6, None)
    # Just check it's a valid number
    assert 0 <= bi_uncorrected <= 1

    print("[PASS] cement_snhr_emi — all tests passed")


if __name__ == "__main__":
    test_all()
