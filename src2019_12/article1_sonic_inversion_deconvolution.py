"""
Article 1: Inversion of High-Resolution High-Quality Sonic Compressional and
           Shear Logs for Unconventional Reservoirs
Lei, Zeroug, Bose, Prioul, Donald (2019)
DOI: 10.30632/PJV60N6-2019a1

Borehole-sonic slowness logs from different receiver-subarray apertures are each
a convolution (blurring) of the true slowness profile with an aperture response
function: a wide aperture is robust but low-resolution, a narrow one is
high-resolution but noisy.  Stacking several apertures builds an overdetermined
linear system that is deconvolved (Moore-Penrose pseudoinverse) into one
high-resolution slowness log, with the residual mismatch as a QC flag.

Implements:

  - Aperture response (symmetric quadratic) kernel                 (Appendix A1)
  - Average-slowness convolution  d_N = conv(F_N, s)               (Eqs. 1, 8)
  - Stacked multiaperture system  D = G @ S                        (Eqs. 10-12)
  - Deconvolution by pseudoinverse  S = G^+ @ D                    (Eq. 13)
  - QC mismatch  ||G_N @ S - D_N||                                 (Eq. 14)

Note: this issue's PDF text layer kept the equation numbers and variable
definitions but dropped the typeset glyphs, so these are the standard
convolution / least-squares-deconvolution forms anchored to those definitions.
Slowness in us/ft.
"""

import numpy as np


# ---------------------------------------------- kernels -----------------

def aperture_kernel(N, curvature=0.3):
    """Normalized symmetric (quadratic) aperture response of length N.

    w_k = 1 - curvature*((k-c)/c)^2, normalized to sum 1 (curvature=0 -> boxcar).
    A wider aperture averages over more samples (lower resolution).
    """
    k = np.arange(N)
    c = (N - 1) / 2.0
    w = 1.0 - curvature * ((k - c) / max(c, 1.0)) ** 2
    w = np.clip(w, 1e-6, None)
    return w / w.sum()


def convolution_matrix(kernel, n):
    """(n x n) convolution (blurring) matrix for a centered, zero-padded kernel."""
    m = len(kernel)
    half = m // 2
    G = np.zeros((n, n))
    for i in range(n):
        for j, w in enumerate(kernel):
            col = i + (j - half)
            if 0 <= col < n:
                G[i, col] += w
    # renormalize rows truncated at the edges
    G /= G.sum(axis=1, keepdims=True)
    return G


# ---------------------------------------------- forward / inverse -------

def blur(slowness, kernel):
    """Average-slowness log  d_N = conv(F_N, s)  (Eq. 1)."""
    G = convolution_matrix(kernel, len(slowness))
    return G @ np.asarray(slowness, float)


def deconvolve(D_stack, G_stack):
    """High-resolution slowness  S = G^+ @ D  (Eq. 13, Moore-Penrose)."""
    return np.linalg.pinv(G_stack) @ D_stack


def qc_mismatch(G_N, S, D_N):
    """QC mismatch for one aperture  ||G_N @ S - D_N||  (Eq. 14)."""
    return float(np.linalg.norm(G_N @ S - D_N))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Sonic Slowness Deconvolution")
    print("=" * 60)

    # True slowness profile: 60 us/ft background with a thin fast (low-slowness)
    # bed and a thin slow bed
    n = 60
    s_true = np.full(n, 90.0)
    s_true[20:24] = 70.0
    s_true[40:42] = 120.0

    # Kernels are normalized (unit area)
    apertures = [3, 5, 7, 9]
    kernels = [aperture_kernel(N) for N in apertures]
    assert all(abs(k.sum() - 1.0) < 1e-12 for k in kernels)

    # Each aperture blurs the log; the widest blurs the most
    blurred = [blur(s_true, k) for k in kernels]
    err_narrow = np.max(np.abs(blurred[0] - s_true))
    err_wide = np.max(np.abs(blurred[-1] - s_true))
    print(f"  blur error narrow/wide = {err_narrow:.1f} / {err_wide:.1f} us/ft")
    assert err_wide > err_narrow                       # wider aperture, more blur

    # Stack the apertures and deconvolve to recover the high-resolution log
    Gs = [convolution_matrix(k, n) for k in kernels]
    G_stack = np.vstack(Gs)
    D_stack = np.concatenate([G @ s_true for G in Gs])
    s_hat = deconvolve(D_stack, G_stack)
    err_deconv = np.max(np.abs(s_hat[2:-2] - s_true[2:-2]))
    print(f"  deconvolved max error  = {err_deconv:.3f} us/ft")
    assert err_deconv < err_wide                       # better than any blurred log
    assert err_deconv < 1.0                            # near-perfect (noise-free)

    # QC mismatch is small for the recovered solution
    q = qc_mismatch(Gs[0], s_hat, D_stack[:n])
    print(f"  QC mismatch            = {q:.3e}")
    assert q < 1e-6
    print("  PASS")
    return {"err_wide": float(err_wide), "err_deconv": float(err_deconv)}


if __name__ == "__main__":
    test_all()
