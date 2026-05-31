"""
Article 3: Water-Wet or Oil-Wet: is it Really That Simple in Shales?
Gupta, Jernigen, Curtis, Rai, Sondergeld (2018)
DOI: 10.30632/PJV59N3-2018a2

An NMR spontaneous-imbibition experiment measures how much brine vs. dodecane a
shale plug imbibes, and turns the contrast into an NMR wettability index (-1
oil-wet to +1 water-wet).  The paper shows wettability in shales is not binary:
it is controlled by the connectivity of the organic (kerogen) and clay networks,
with percolation thresholds near 5 wt% TOC and 10 wt% clay.

Implements:

  - NMR wettability index  Iw = (NMR_w - NMR_do)/(NMR_w + NMR_do)
  - Average wettability index over the two imbibition sequences
  - Clay-content wettability classification (cutoffs 10 / 65 wt%)
  - TOC percolation flag for an oil-wet pathway (5 wt% threshold)
  - API gamma ray from spectral Th/U/K

Note: this issue's PDF has a text layer but its typeset display-equation glyphs
were dropped in extraction, so the wettability index (Eq. 1) is a faithful
standard-form reconstruction (Looyestijn-Hofman / Sulucarnain); the 5 wt% TOC
and 10/65 wt% clay thresholds are transcribed from the paper's findings.
"""

import numpy as np

TOC_PERCOLATION = 5.0        # wt% TOC for connected oil-wet pathway
CLAY_OILWET = 10.0           # wt% clay below which oil-wet
CLAY_WATERWET = 65.0         # wt% clay above which water-wet


# ---------------------------------------------- wettability index --------------

def nmr_wettability_index(nmr_water, nmr_dodecane):
    """NMR wettability index  Iw = (NMR_w - NMR_do)/(NMR_w + NMR_do)  (Eq. 1).

    NMR_w / NMR_do = incremental porosity imbibed as brine / as dodecane.
    Iw = +1 fully water-wet, -1 fully oil-wet, 0 neutral.
    """
    w = np.asarray(nmr_water, float)
    do = np.asarray(nmr_dodecane, float)
    return (w - do) / (w + do)


def average_wettability(iw_seq1, iw_seq2):
    """Average wettability index over the two imbibition sequences."""
    return 0.5 * (iw_seq1 + iw_seq2)


def wettability_from_clay(clay_wt):
    """Classify wettability from clay content (wt%) using the paper's cutoffs."""
    if clay_wt < CLAY_OILWET:
        return "oil-wet"
    if clay_wt > CLAY_WATERWET:
        return "water-wet"
    return "mixed-wet"


def oil_wet_pathway(toc_wt):
    """True if TOC exceeds the ~5 wt% percolation threshold for an oil-wet path."""
    return np.asarray(toc_wt, float) >= TOC_PERCOLATION


def gr_api(th_ppm, u_ppm, k_wt):
    """API gamma ray from spectral Th/U/K  GR = 4*Th + 8*U + 16*K."""
    return 4.0 * th_ppm + 8.0 * u_ppm + 16.0 * k_wt


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: NMR Wettability Index in Shales")
    print("=" * 60)

    # Index sign tracks which fluid is preferentially imbibed
    assert nmr_wettability_index(0.08, 0.02) > 0       # more brine -> water-wet
    assert nmr_wettability_index(0.02, 0.08) < 0       # more oil   -> oil-wet
    assert np.isclose(nmr_wettability_index(0.05, 0.05), 0.0)

    iw = average_wettability(nmr_wettability_index(0.07, 0.03),
                             nmr_wettability_index(0.06, 0.03))
    print(f"  average wettability    = {iw:+.3f}")
    assert -1.0 <= iw <= 1.0

    # Clay cutoffs at 10 and 65 wt%
    labels = [wettability_from_clay(c) for c in (5.0, 30.0, 80.0)]
    print(f"  clay 5/30/80 wt%       = {labels}")
    assert labels == ["oil-wet", "mixed-wet", "water-wet"]

    # TOC percolation threshold
    assert oil_wet_pathway(6.0) and not oil_wet_pathway(3.0)

    # Spectral GR synthesis
    assert np.isclose(gr_api(10.0, 5.0, 2.0), 4 * 10 + 8 * 5 + 16 * 2)
    print("  PASS")
    return {"avg_wettability": float(iw)}


if __name__ == "__main__":
    test_all()
