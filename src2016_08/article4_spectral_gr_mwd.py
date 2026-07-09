"""
Article 4: Spectral Gamma-Ray Measurement While Drilling
Xu, Huiszoon, Wang, Adolph, Yi, Cavin, Laughlin, Tollefsen, Jacobsen, Boyce (2016)
Reference: Petrophysics Vol. 57, No. 4 (August 2016), pp. 377-389
DOI: none assigned (this issue predates SPWLA DOI assignment)

A low-cost spectral gamma-ray tool is built into an MWD collar.  A NaI detector
records a 128-channel spectrum (to 3.3 MeV); the potassium, uranium and thorium
concentrations are derived from the spectral windows.  Detector gain is
stabilized without a radioactive source by holding a form factor (FF) of two
low-energy windows at zero, and the K/U/Th concentrations are obtained by a
weighted-least-squares fit of standard spectra, then converted to total (SGR)
and uranium-free (CGR) gamma-ray logs in API units.

Implements:

  - Sourceless-gain-regulation form factor  FF = (W1 - W2)/(W1 + W2)  (Eq. 1)
  - Weighted-least-squares spectral fit for K, U, Th concentrations
  - Total (SGR) and uranium-free (CGR) gamma ray in API units

Note: this issue's PDF has a text layer; the form factor (Eq. 1) and the
spectral-fit / API-conversion workflow are transcribed from the body, while the
typeset glyphs were dropped and the window-stripping inversion is implemented in
the standard weighted-least-squares form.  Concentrations: K in wt%, U and Th in
ppm; GR in API.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Standard API sensitivity coefficients (API per unit concentration)
API_K = 16.0          # per wt% K
API_U = 8.0           # per ppm U
API_TH = 4.0          # per ppm Th


# ---------------------------------------------- gain regulation --------------

def form_factor(w1, w2):
    """Sourceless-gain-regulation form factor (Eq. 1)

        FF = (W1 - W2)/(W1 + W2),

    from counts in two windows straddling the low-energy hump.  FF < 0 (too many
    counts in W2) means the gain is too high and the PMT high voltage must be
    lowered; FF > 0 means it must be raised; gain is stable at FF = 0.
    """
    return (w1 - w2) / (w1 + w2)


# ---------------------------------------------- spectral fit --------------

def spectral_wls(counts, standards, weights=None):
    """Weighted-least-squares elemental concentrations from a measured spectrum.

    Solves  counts ~ standards' @ [K, U, Th]  in a weighted sense, where
    `standards` is (3, n_channels) of the K/U/Th standard spectra.  Returns the
    fitted [K, U, Th] concentrations (Poisson weights 1/counts by default).
    """
    counts = np.asarray(counts, float)
    a = np.asarray(standards, float).T          # (n_channels, 3)
    if weights is None:
        weights = 1.0 / np.maximum(counts, 1.0)
    w = np.sqrt(np.asarray(weights, float))
    aw = a * w[:, None]
    bw = counts * w
    conc, *_ = np.linalg.lstsq(aw, bw, rcond=None)
    return conc


# ---------------------------------------------- API gamma ray --------------

def total_gamma_ray(k, u, th, api_k=API_K, api_u=API_U, api_th=API_TH):
    """Total spectral gamma ray (SGR) in API units

        SGR = api_k*K + api_u*U + api_th*Th.
    """
    return petrolib.nuclear.gr_api(k, u, th, coeff=(api_k, api_u, api_th))


def uranium_free_gamma_ray(k, th, api_k=API_K, api_th=API_TH):
    """Uranium-free (computed) gamma ray (CGR) in API units

        CGR = api_k*K + api_th*Th,

    computed from the K and Th concentrations only.
    """
    return petrolib.nuclear.cgr_api(k, th, coeff=(api_k, api_th))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Spectral Gamma-Ray While Drilling")
    print("=" * 60)

    # Form factor is zero at balanced windows, signed otherwise
    assert np.isclose(form_factor(1000.0, 1000.0), 0.0)
    assert form_factor(900.0, 1100.0) < 0.0    # too many counts in W2 -> lower HV
    assert form_factor(1100.0, 900.0) > 0.0

    # Build synthetic K/U/Th standards and recover known concentrations by WLS
    ch = np.linspace(0.5, 3.0, 64)
    std_k = np.exp(-((ch - 1.46) ** 2) / 0.02)                 # 40K line at 1.46 MeV
    std_u = np.exp(-((ch - 1.76) ** 2) / 0.05) + 0.3 * np.exp(-((ch - 0.61) ** 2) / 0.03)
    std_th = np.exp(-((ch - 2.62) ** 2) / 0.04) + 0.4 * np.exp(-((ch - 0.91) ** 2) / 0.03)
    standards = np.vstack([std_k, std_u, std_th])
    true_conc = np.array([2.5, 4.0, 12.0])                     # K(wt%), U(ppm), Th(ppm)
    counts = standards.T @ true_conc
    fit = spectral_wls(counts, standards, weights=np.ones_like(counts))
    print(f"  fitted K/U/Th          = {np.round(fit, 2)}")
    assert np.allclose(fit, true_conc, atol=1e-6)

    # Total GR exceeds the uranium-free GR by the uranium contribution
    sgr = total_gamma_ray(*true_conc)
    cgr = uranium_free_gamma_ray(true_conc[0], true_conc[2])
    print(f"  SGR / CGR              = {sgr:.1f} / {cgr:.1f} API")
    assert np.isclose(sgr - cgr, API_U * true_conc[1]) and sgr > cgr
    print("  PASS")
    return {"K": float(fit[0]), "U": float(fit[1]), "Th": float(fit[2]),
            "SGR": float(sgr), "CGR": float(cgr)}


if __name__ == "__main__":
    test_all()
