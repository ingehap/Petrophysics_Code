"""
Article 4: Kerogen Content and Maturity, Mineralogy and Clay Typing from DRIFTS
           Analysis of Cuttings or Core
Michael M. Herron, MaryEllen Loan, Alyssa M. Charsky, Susan L. Herron,
Andrew E. Pomerantz, Marina Polyakov (2014)
Reference: Petrophysics Vol. 55, No. 5 (October 2014), pp. 435-446
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best of the 2014 SPWLA Annual Logging Symposium.  Diffuse-reflectance infrared
Fourier-transform spectroscopy (DRIFTS) of crushed cuttings or core gives
mineralogy and organic content.  Spectra are taken in Kubelka-Munk units and
inverted as a weighted least-squares linear combination of pure mineral-standard
spectra; the mineral-stripped residual over the aliphatic C-H band gives an
organic signal that, scaled by maturity, yields TOC.

Implements:

  - Kubelka-Munk transform  f(R) = (1-R)^2/(2R)
  - Weighted least-squares mineral inversion (non-negative)
  - Mineral-stripped organic signal over the 2800-3000 cm^-1 aliphatic band
  - TOC from the organic signal scaled by maturity  TOC ~ ratio*signal, ratio~Ro
  - Cation exchange capacity as a linear combination of clay abundances

Note: this paper carries no numbered display equations; the Kubelka-Munk
relation, the linear-combination inversion, the aliphatic-band organic signal
and the maturity-scaled TOC relation are transcribed from the prose and written
in standard form (Kubelka & Munk, 1931; Charsky & Herron, 2012).  Wavenumbers
in cm^-1, abundances and TOC as weight fractions, CEC in meq/100 g.
"""

import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------- Kubelka-Munk --------------

def kubelka_munk(reflectance):
    """Kubelka-Munk remission function (Kubelka & Munk, 1931)

        f(R) = (1 - R)^2/(2*R),

    the diffuse-reflectance intensity that is approximately linear in
    concentration (unlike raw reflectance).
    """
    r = np.asarray(reflectance, float)
    return (1.0 - r) ** 2 / (2.0 * r)


# ---------------------------------------------- mineral inversion --------------

def mineral_inversion(spectrum, standards, weights=None):
    """Weighted, non-negative least-squares mineral inversion

        min_c || W*(spectrum - standards^T c) ||^2,  c >= 0,

    decomposing a measured spectrum into pure mineral-standard spectra.
    ``standards`` has shape (n_minerals, n_channels); returns the (normalized)
    mineral weight fractions c.
    """
    a = np.asarray(standards, float).T            # (channels, minerals)
    b = np.asarray(spectrum, float)
    if weights is not None:
        w = np.sqrt(np.asarray(weights, float))[:, None]
        a, b = a * w, b * w.ravel()
    c, *_ = np.linalg.lstsq(a, b, rcond=None)
    c = np.clip(c, 0.0, None)                      # non-negative abundances
    total = c.sum()
    return c / total if total > 0 else c


# ---------------------------------------------- organic signal & TOC --------------

def organic_signal(wavenumbers, residual_spectrum, band=(2800.0, 3000.0)):
    """Integrate the mineral-stripped residual over the aliphatic C-H band

        signal = integral_{2800}^{3000} residual(nu) d(nu).
    """
    nu = np.asarray(wavenumbers, float)
    res = np.asarray(residual_spectrum, float)
    mask = (nu >= band[0]) & (nu <= band[1])
    order = np.argsort(nu[mask])
    return float(_trapezoid(res[mask][order], nu[mask][order]))


def toc_from_drifts(signal, maturity_ro, calibration=1.0):
    """TOC from the organic signal scaled by thermal maturity

        TOC = calibration*Ro*signal,

    where the proportionality (signal-per-TOC) decreases as maturity Ro (%)
    increases, so a higher Ro maps a given signal to a higher TOC.
    """
    return calibration * maturity_ro * np.asarray(signal, float)


# ---------------------------------------------- clay typing --------------

def cec_from_clays(clay_abundances, clay_cec):
    """Cation exchange capacity from a clay-mineral assemblage

        CEC = sum_i abundance_i*CEC_i,

    a linear combination dominated by smectite.  CEC_i in meq/100 g.
    """
    a = np.asarray(clay_abundances, float)
    c = np.asarray(clay_cec, float)
    return float(a @ c)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: DRIFTS Kerogen, Mineralogy & Clay Typing")
    print("=" * 60)

    # Kubelka-Munk: lower reflectance (stronger absorption) gives larger f(R)
    assert kubelka_munk(0.2) > kubelka_munk(0.8) > 0

    # Mineral inversion recovers a known two-mineral mixture
    nu = np.linspace(400, 4000, 200)
    quartz = np.exp(-((nu - 1080) / 40) ** 2)        # Si-O stretch
    calcite = np.exp(-((nu - 1430) / 50) ** 2)       # carbonate
    truth = np.array([0.7, 0.3])
    spectrum = truth[0] * quartz + truth[1] * calcite
    c = mineral_inversion(spectrum, [quartz, calcite])
    print(f"  recovered mineralogy = {np.round(c, 3)}")
    assert np.allclose(c, truth, atol=1e-6)

    # Organic signal: integrate an aliphatic C-H residual band
    residual = 0.5 * np.exp(-((nu - 2920) / 25) ** 2)
    sig = organic_signal(nu, residual)
    print(f"  organic signal = {sig:.3f}")
    assert sig > 0

    # TOC scales with maturity for the same signal
    toc_low = toc_from_drifts(sig, maturity_ro=0.5)
    toc_high = toc_from_drifts(sig, maturity_ro=1.35)
    print(f"  TOC(Ro=0.5)={toc_low:.3f}  TOC(Ro=1.35)={toc_high:.3f}")
    assert toc_high > toc_low > 0

    # CEC dominated by smectite
    cec = cec_from_clays([0.1, 0.05, 0.02], [10.0, 90.0, 5.0])  # illite, smectite, kaolinite
    print(f"  CEC = {cec:.2f} meq/100g")
    assert np.isclose(cec, 0.1 * 10 + 0.05 * 90 + 0.02 * 5)
    print("  PASS")
    return {"TOC": float(toc_high), "CEC": float(cec)}


if __name__ == "__main__":
    test_all()
