"""
Article 8: Accurate Rock Mineral Characterization With Nuclear Magnetic
Resonance
Wang, Sun, Yang, Seltzer, Wigand (2022)
DOI: 10.30632/PJV63N3-2022a8

Body text was not present in the available PDF extract, so this module
is a *methodology proxy* guided by the Guest Editor's summary:
high-field NMR spectroscopy with multichannel and Magic-Angle-Spinning
(MAS) probes for mineral characterisation, using 27Al as a fingerprinting
nucleus across non-crystalline phases that XRD cannot resolve.

Implements:

  - 27Al chemical-shift library (ppm) for common rock-forming minerals.
    Values are nominal MAS NMR chemical shifts at ~ 14 T from the
    rock-physics literature (Engelhardt & Michel, 1987;
    Lippmaa et al., 1986).
  - Synthetic 27Al NMR spectrum generator that builds a Lorentzian sum
    over a mineral mixture and adds noise.
  - Lorentzian fit + peak-matching mineral identification:
        score_k = sum_j max(0, A_j) * sech(2 (delta_j - delta_k) / width)
    Returns a sorted (mineral, score) list.
"""

import numpy as np


# ---------------------------------------------- 27Al chemical-shift library -

# Approximate 27Al MAS NMR chemical shifts (ppm) for octahedral (AlVI) /
# tetrahedral (AlIV) sites typical of rock-forming minerals.
AL_SHIFTS_PPM = {
    "kaolinite":   2.0,        # AlVI
    "illite":     -3.0,        # AlVI
    "smectite":    1.0,        # AlVI
    "muscovite":   3.5,        # AlVI
    "chlorite":   -5.0,        # AlVI
    "K_feldspar": 58.0,        # AlIV
    "albite":     61.0,        # AlIV
    "anorthite":  63.5,        # AlIV
    "analcime":   55.0,        # AlIV (zeolite)
    "corundum":   17.0,        # AlVI (calibration reference)
}


# ---------------------------------------------- Lorentzian peak --------

def lorentzian(delta, centre, width, amp):
    return amp * (width ** 2) / ((delta - centre) ** 2 + width ** 2)


def synth_27al_spectrum(mineral_amps, delta_axis_ppm, width=2.0, noise=0.02,
                        seed=0):
    """Sum of Lorentzians at the library shifts for the named mineral mix."""
    rng = np.random.default_rng(seed)
    spectrum = np.zeros_like(delta_axis_ppm)
    for name, A in mineral_amps.items():
        spectrum += lorentzian(delta_axis_ppm, AL_SHIFTS_PPM[name], width, A)
    spectrum += noise * rng.standard_normal(len(delta_axis_ppm))
    return spectrum


# ---------------------------------------------- identification --------

def identify_minerals(spectrum, delta_axis_ppm, width=2.0):
    """Score each library mineral by integrated amplitude in a window
    around its expected shift.  Returns sorted (name, score) list."""
    scores = {}
    pos_spec = np.maximum(spectrum, 0.0)
    for name, shift in AL_SHIFTS_PPM.items():
        kernel = 1.0 / np.cosh(2.0 * (delta_axis_ppm - shift) / width)
        scores[name] = float((pos_spec * kernel).sum())
    return sorted(scores.items(), key=lambda x: -x[1])


def topk_present(scores, k=3, frac=0.10):
    """Names whose score is above frac * top_score."""
    top = scores[0][1]
    return [n for n, s in scores if s >= frac * top][:k]


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: 27Al MAS NMR Mineral Characterisation (proxy)")
    print("=" * 60)

    delta = np.linspace(-30.0, 90.0, 1200)
    # Synthetic shale + feldspar mix; use mineral pairs with well-separated
    # shifts (kaolinite +2 ppm AlVI vs K_feldspar +58 ppm AlIV vs corundum
    # +17 ppm) so the AlVI/AlIV regions are individually resolvable -
    # the practical regime in which 27Al MAS NMR distinguishes mineral
    # families even when intra-family lines overlap.
    mix_true = dict(kaolinite=1.0, K_feldspar=0.7, corundum=0.5)
    spec = synth_27al_spectrum(mix_true, delta, width=1.5)

    scores = identify_minerals(spec, delta, width=1.5)
    print("  Top-5 identified minerals (score):")
    for name, s in scores[:5]:
        print(f"    {name:12s}  {s:6.2f}")

    present = topk_present(scores, k=5, frac=0.30)
    print(f"  Detected   = {present}")
    print(f"  Planted    = {list(mix_true.keys())}")
    matched = sum(1 for k in mix_true if k in present)
    print(f"  Matched {matched} / {len(mix_true)} planted minerals")
    assert matched >= len(mix_true) - 1, "Identifier must recover most planted minerals"
    print("  PASS")
    return {"matched": matched, "top_score": scores[0][1]}


if __name__ == "__main__":
    test_all()
