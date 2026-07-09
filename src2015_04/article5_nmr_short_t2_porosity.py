"""
Article 5: New Method to Estimate Porosity More Accurately from NMR Data with
           Short Relaxation Times
Venkataramanan, Gruber, LaVigne, Habashy, Iglesias, Cohorn, Anand, Rampurawala,
Jain, Heaton, Akkurt, Rylander, Lewis (2015)
Reference: Petrophysics Vol. 56, No. 2 (April 2015), pp. 147-157
DOI: none assigned (this issue predates SPWLA DOI assignment)

The regularized inverse Laplace transform (ILT) of NMR echo data systematically
biases the recovered porosity at short relaxation times (the regularization
smooths fast-decaying signal away).  This method scans a Dirac-delta through the
T2 spectrum to map the bias B(T2), then applies a correction factor Cf(T2) =
1/(1 + B(T2)) (optionally SNR-weighted) so the corrected binned and total
porosity are accurate over the whole T2 range.

Implements:

  - NMR kernel and Tikhonov (regularized) T2 inversion  Q = ||G - Lf||^2 + a||f||^2  (Eq. 1)
  - Porosity bias  B(T2) = <phi(T2)> - 1  (Eq. 2)
  - Correction factor  Cf = 1/(1 + B)  and corrected porosity (Eqs. 3-4)
  - SNR-weighted correction factor (Eq. 5) and total porosity (Eq. 6)

Note: this issue's PDF has a text layer; the cost function, bias and
correction-factor relations (Eqs. 1-6) are transcribed from the body, while the
typeset glyphs were dropped and reconstructed in standard form.  Times in s,
porosity in p.u. (or fraction), amplitudes dimensionless.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- inversion --------------

def nmr_kernel(times, t2_bins):
    """NMR relaxation kernel  L[i,j] = exp(-t_i/T2_j)  relating the T2
    distribution to the measured magnetization decay."""
    return petrolib.nmr.cpmg_kernel(times, t2_bins)


def tikhonov_t2_inversion(g, l, alpha):
    """Regularized (Tikhonov) T2-distribution inversion (Eq. 1)

        f = argmin ||G - L f||^2 + alpha*||f||^2 = (L'L + alpha I)^-1 L'G,

    the smoothed inverse-Laplace estimate of the T2 distribution (the second
    term suppresses noise but biases short-T2 amplitudes).
    """
    l = np.asarray(l, float)
    g = np.asarray(g, float)
    n = l.shape[1]
    return np.linalg.solve(l.T @ l + alpha * np.eye(n), l.T @ g)


# ---------------------------------------------- bias / correction --------------

def porosity_bias(phi_recovered, phi_true=1.0):
    """Relative porosity bias from a unit-area delta input (Eq. 2)

        B = (sum phi_recovered)/phi_true - 1,

    negative where the ILT underestimates porosity (short T2).
    """
    return float(np.sum(phi_recovered) / phi_true - 1.0)


def correction_factor(bias):
    """Porosity correction factor  Cf = 1/(1 + B)  (Eq. 4),

    > 1 where porosity is underestimated (B < 0), < 1 where overestimated.
    """
    return 1.0 / (1.0 + bias)


def corrected_porosity(phi, cf):
    """Corrected binned porosity  phi_c = Cf*phi  (Eq. 3)."""
    return cf * np.asarray(phi, float)


def snr_correction_factor(r_t2, r_mean, beta=1.0):
    """SNR-weighted correction factor component (Eq. 5)

        w = R(T2)^beta / (R(T2)^beta + <R>^beta),

    in [0, 1]: near 0 where the per-bin SNR R(T2) is small relative to the mean,
    near 1 where it is large, tempering the correction in noisy bins.
    """
    return r_t2 ** beta / (r_t2 ** beta + r_mean ** beta)


def total_porosity(phi_bins):
    """Total NMR porosity  phi = sum over T2 bins  (Eq. 6)."""
    return petrolib.nmr.total_porosity(phi_bins)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: NMR Short-T2 Porosity Correction")
    print("=" * 60)

    # Build a short-T2 synthetic decay and invert with regularization
    times = np.linspace(2e-4, 1.0, 200)
    t2_bins = np.logspace(np.log10(3e-4), np.log10(3.0), 30)
    true_f = np.zeros(t2_bins.size)
    true_f[3] = 1.0                               # a short-T2 component
    g = nmr_kernel(times, t2_bins) @ true_f
    f_hat = tikhonov_t2_inversion(g, nmr_kernel(times, t2_bins), alpha=0.1)

    # The regularized inversion underestimates the short-T2 porosity
    bias = porosity_bias(f_hat, phi_true=1.0)
    print(f"  porosity bias          = {bias:.3f}")
    assert bias < 0.0                              # underestimated

    # Correction factor exceeds 1 and restores the total porosity
    cf = correction_factor(bias)
    print(f"  correction factor      = {cf:.3f}")
    assert cf > 1.0
    phi_c = corrected_porosity(f_hat, cf)
    assert np.isclose(total_porosity(phi_c), 1.0)

    # When overestimated (B > 0), the factor reduces porosity
    assert correction_factor(0.2) < 1.0

    # SNR weighting is in [0, 1] and rises with the per-bin SNR
    w_lo = snr_correction_factor(0.5, 5.0)
    w_hi = snr_correction_factor(20.0, 5.0)
    print(f"  SNR weight low/high    = {w_lo:.3f} / {w_hi:.3f}")
    assert 0.0 < w_lo < w_hi < 1.0
    print("  PASS")
    return {"bias": bias, "Cf": float(cf), "phi_corrected": total_porosity(phi_c)}


if __name__ == "__main__":
    test_all()
