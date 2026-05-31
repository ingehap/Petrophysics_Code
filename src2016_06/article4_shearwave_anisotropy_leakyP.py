"""
Article 4: An Improved Multiscale and Leaky P-Wave Removal Analysis for
           Shear-Wave Anisotropy Inversion with Crossed-Dipole Logs
Li, Tao, Wang, Zhang, Vega (2016)
Reference: Petrophysics Vol. 57, No. 3 (June 2016), pp. 270-293
DOI: none assigned (this issue predates SPWLA DOI assignment)

Crossed-dipole shear-wave anisotropy is inverted from the four-component
(XX, XY, YX, YY) flexural waveforms.  Alford rotation projects the data onto the
fast and slow principal directions; in slow formations a leaky P-wave
contaminates the flexural waveforms and biases the inversion.  This paper uses a
shift-invariant dual-tree complex wavelet transform (DTCWT) to separate the
leaky-P from the flexural wave by scale, and a multi-objective function over
levels to stabilize the fast-azimuth/slowness inversion.

Implements:

  - Analytic (complex) wavelet  psi_c = psi_r + j*psi_j  (Hilbert pair, Eq. 1)
  - Alford rotation to fast/slow principal waveforms (Eqs. 3-4)
  - Leaky-P contamination of the principal waveforms (Eqs. 5-6)
  - Fast-shear azimuth by cross-component energy minimization
  - Anisotropy objective: fast/slow waveform misfit after a slowness time shift (Eq. 7)

Note: this issue's PDF has a text layer; the DTCWT (Eq. 1), Alford rotation /
leaky-P relations (Eqs. 3-6) and the inversion objective (Eq. 7) are transcribed
from the body, while the typeset glyphs were dropped and reconstructed in
standard form (the full DTCWT filter bank is represented by the analytic-signal
Hilbert pair).  Angles in radians unless noted; slowness in us/ft.
"""

import numpy as np


# ---------------------------------------------- complex wavelet --------------

def analytic_wavelet(real_wavelet):
    """Analytic (complex) wavelet  psi_c = psi_r + j*psi_j  (Eq. 1),

    with psi_j the Hilbert transform of psi_r (a Hilbert pair, as the two DTCWT
    trees form).  The magnitude |psi_c| is shift-invariant.  Implemented via the
    FFT-based analytic signal.
    """
    x = np.asarray(real_wavelet, float)
    n = x.size
    xf = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(xf * h)


# ---------------------------------------------- Alford rotation --------------

def alford_rotation(xx, xy, yx, yy, theta):
    """Rotate the four-component tensor to the principal (fast/slow) frame
    (Eqs. 3-4)

        FP = cos^2 t*XX + sin t cos t*(XY+YX) + sin^2 t*YY
        SP = sin^2 t*XX - sin t cos t*(XY+YX) + cos^2 t*YY
        cross = sin t cos t*(YY-XX) + cos^2 t*XY - sin^2 t*YX,

    where t is the fast-shear azimuth.  Returns (FP, SP, cross).
    """
    c, s = np.cos(theta), np.sin(theta)
    xx, xy, yx, yy = (np.asarray(v, float) for v in (xx, xy, yx, yy))
    fp = c ** 2 * xx + s * c * (xy + yx) + s ** 2 * yy
    sp = s ** 2 * xx - s * c * (xy + yx) + c ** 2 * yy
    cross = s * c * (yy - xx) + c ** 2 * xy - s ** 2 * yx
    return fp, sp, cross


def add_leaky_p(fp, sp, leaky_p, theta):
    """Leaky-P contamination of the principal waveforms (Eqs. 5-6)

        FP_L = FP + (sin t + cos t)^2 * LeakyP
        SP_L = SP + (sin t - cos t)^2 * LeakyP.
    """
    c, s = np.cos(theta), np.sin(theta)
    return fp + (s + c) ** 2 * leaky_p, sp + (s - c) ** 2 * leaky_p


def fast_shear_azimuth(xx, xy, yx, yy, n_theta=361):
    """Estimate the fast-shear azimuth by minimizing the rotated cross-component
    energy over theta in [0, pi)."""
    thetas = np.linspace(0.0, np.pi, n_theta)
    energy = [np.sum(alford_rotation(xx, xy, yx, yy, t)[2] ** 2) for t in thetas]
    return float(thetas[int(np.argmin(energy))])


# ---------------------------------------------- anisotropy objective --------------

def anisotropy_objective(fast, slow, shift_samples):
    """Waveform-similarity objective for anisotropy inversion (Eq. 7)

        E = sum_t [ FP(t) - SP(t - shift) ]^2,

    minimized when the slow waveform, advanced by the slowness-difference delay
    (a positive `shift_samples`), matches the fast waveform.
    """
    slow_shifted = np.roll(np.asarray(slow, float), -int(shift_samples))
    return float(np.sum((np.asarray(fast, float) - slow_shifted) ** 2))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Shear-Wave Anisotropy + Leaky-P Removal")
    print("=" * 60)

    # Analytic wavelet is a Hilbert pair: real part recovers the input
    w = np.cos(np.linspace(0, 6 * np.pi, 128)) * np.exp(-np.linspace(0, 3, 128))
    wc = analytic_wavelet(w)
    assert np.allclose(wc.real, w, atol=1e-9) and np.any(np.abs(wc.imag) > 1e-3)

    # Synthesize a fast/slow medium with a known azimuth and recover it
    t = np.linspace(0, 1, 200)
    fast_src = np.sin(2 * np.pi * 5 * t) * np.exp(-3 * t)
    slow_src = np.zeros_like(t)               # pure fast polarization for the test
    az = np.radians(30.0)
    c, s = np.cos(az), np.sin(az)
    # forward: project the principal waves back onto the tool XX/XY/YX/YY frame
    xx = c ** 2 * fast_src + s ** 2 * slow_src
    yy = s ** 2 * fast_src + c ** 2 * slow_src
    xy = yx = s * c * (fast_src - slow_src)
    az_fit = fast_shear_azimuth(xx, xy, yx, yy)
    print(f"  fast azimuth true/fit  = 30.0 / {np.degrees(az_fit):.1f} deg")
    assert np.isclose(np.degrees(az_fit), 30.0, atol=1.0)

    # The rotation recovers the fast waveform and a near-zero cross component
    fp, sp, cross = alford_rotation(xx, xy, yx, yy, az_fit)
    assert np.allclose(fp, fast_src, atol=1e-6) and np.sum(cross ** 2) < 1e-6

    # Leaky-P adds energy to both principal waveforms
    fp_l, sp_l = add_leaky_p(fp, sp, leaky_p=0.5 * np.cos(2 * np.pi * 8 * t), theta=az_fit)
    assert np.sum(fp_l ** 2) > np.sum(fp ** 2)

    # Anisotropy objective is minimized at the correct slow-wave time shift
    fast_w = np.sin(2 * np.pi * 5 * t) * np.exp(-3 * t)
    slow_w = np.roll(fast_w, 8)               # slow arrives 8 samples later
    e_correct = anisotropy_objective(fast_w, slow_w, shift_samples=8)
    e_wrong = anisotropy_objective(fast_w, slow_w, shift_samples=0)
    print(f"  objective correct/wrong = {e_correct:.2e} / {e_wrong:.2e}")
    assert e_correct < e_wrong
    print("  PASS")
    return {"azimuth_deg": float(np.degrees(az_fit)), "E_correct": e_correct}


if __name__ == "__main__":
    test_all()
