"""
Article 1: Reservoir Characterization Challenges Due to the Multiscale Spatial
           Heterogeneity in the Presalt Carbonate Sag Formation, North Campos
           Basin, Brazil
Chitale, Alabi, Gramin, Lepley, Piccoli (2015)
Reference: Petrophysics Vol. 56, No. 6 (December 2015), pp. 552-576
DOI: none assigned (this issue predates SPWLA DOI assignment)

A core-to-log study of the texturally heterogeneous presalt Sag carbonates
(Macabu Formation).  NMR is central: the T2 distribution is split at a T2 cutoff
into bound (BVI) and free (FFI) fluid, and permeability is estimated from the
Coates (Timur-Coates) and SDR transforms.  Carbonate (limestone) T2 cutoffs are
high (~90-200 ms) compared with sandstones (~33 ms), and heterogeneity makes a
single constant cutoff only an approximation.

Implements:

  - NMR T2-distribution partition into BVI and FFI at a T2 cutoff
  - Bound- and free-fluid porosities and the total-porosity model
  - T2 logarithmic-mean relaxation time
  - Coates (Timur-Coates) permeability  k = (phi/C)^4*(FFI/BVI)^2
  - SDR permeability  k = a*phi^4*T2lm^2

Note: this is a core/NMR carbonate case study; the relations below are the
standard NMR petrophysics it relies on (Coates et al., 1999).  The typeset
glyphs were dropped in extraction, so the transforms are standard-form
reconstructions.  Permeability in mD, times in ms, porosity as a fraction.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

T2_CUTOFF_LIMESTONE_MS = 100.0    # typical carbonate cutoff (90-200 ms range)


# ---------------------------------------------- NMR partition --------------

def t2_partition(t2_bins, amplitudes, t2_cutoff=T2_CUTOFF_LIMESTONE_MS):
    """Split a T2 distribution at the cutoff into bound (BVI) and free (FFI) fluid

        BVI = sum of amplitudes with T2 <  cutoff   (capillary-bound),
        FFI = sum of amplitudes with T2 >= cutoff   (free / movable).

    Returns (BVI, FFI) in the amplitude (porosity) units of the distribution.
    """
    return petrolib.nmr.bvi_ffi(t2_bins, amplitudes, cutoff_ms=t2_cutoff)


def total_porosity(bvi, ffi):
    """Total (NMR) porosity in the total-porosity model  phi = BVI + FFI."""
    return bvi + ffi


def t2_logmean(t2_bins, amplitudes):
    """Logarithmic-mean T2  T2lm = exp(sum(A*ln T2)/sum(A))."""
    return petrolib.nmr.t2_logmean(t2_bins, amplitudes)


# ---------------------------------------------- NMR permeability --------------

def coates_permeability(phi, ffi, bvi, c=10.0):
    """Coates (Timur-Coates) permeability  k = (phi/C)^4*(FFI/BVI)^2."""
    return petrolib.nmr.timur_coates(phi, ffi, bvi, C=c)


def sdr_permeability(phi, t2lm, a=4.0):
    """SDR permeability  k = a*phi^4*T2lm^2  (T2lm the log-mean relaxation time)."""
    return petrolib.nmr.sdr(phi, t2lm, a=a)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Presalt Sag NMR Petrophysics")
    print("=" * 60)

    # Partition a synthetic carbonate T2 distribution at the limestone cutoff
    bins = np.array([1.0, 10.0, 50.0, 100.0, 300.0, 1000.0])
    amps = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.04])    # porosity units
    bvi, ffi = t2_partition(bins, amps, t2_cutoff=100.0)
    phi = total_porosity(bvi, ffi)
    print(f"  BVI / FFI / phi        = {bvi:.3f} / {ffi:.3f} / {phi:.3f}")
    assert np.isclose(phi, amps.sum())
    assert np.isclose(bvi, 0.02 + 0.03 + 0.04)       # T2 < 100 ms

    # A lower (sandstone-like) cutoff moves more signal into FFI
    bvi_lo, ffi_lo = t2_partition(bins, amps, t2_cutoff=33.0)
    assert ffi_lo > ffi and bvi_lo < bvi

    # T2 log-mean lies within the distribution range
    t2lm = t2_logmean(bins, amps)
    print(f"  T2 log-mean            = {t2lm:.2f} ms")
    assert bins[0] < t2lm < bins[-1]

    # Coates and SDR permeability are positive and rise with porosity / free fluid
    k_coates = coates_permeability(phi, ffi, bvi)
    k_sdr = sdr_permeability(phi, t2lm)
    print(f"  Coates / SDR k         = {k_coates:.3f} / {k_sdr:.3f} mD")
    assert k_coates > 0 and k_sdr > 0
    assert coates_permeability(phi, ffi * 1.5, bvi) > k_coates
    print("  PASS")
    return {"BVI": bvi, "FFI": ffi, "k_coates": float(k_coates), "k_sdr": float(k_sdr)}


if __name__ == "__main__":
    test_all()
