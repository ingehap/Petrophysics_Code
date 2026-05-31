"""
Article 3: Characterizing Natural Gamma-Ray Tools Without the API Calibration
           Formation
Moake (2017)
Reference: Petrophysics Vol. 58, No. 5 (October 2017), pp. 485-500
DOI: none assigned (this issue predates SPWLA DOI assignment)

Instead of the physical University-of-Houston API calibration formation, a tool
is characterized by Monte Carlo modeling plus a single calibration point,
defining a "digital API formation."  The radioactive sources (K-40, the Th
chain, and the U chains in secular equilibrium) emit photons whose spectrum is
binned into representative energies; the modeled count rate equals a Monte Carlo
tally times a source multiplier, and the tool sensitivity is the count rate per
200 API.

Implements:

  - Representative (centroid) bin energy  Er = sum(Ei*Pi)/sum(Pi)
  - Source emission rate  n = (n/C)*C  and isotope sampling fractions
  - U-235 folded into U-238 with weight 0.04604
  - Count rate  = tally*multiplier  and tool sensitivity  = count_rate/200 API

Note: this issue's PDF has a text layer; the representative-energy relation
survived, while other equations lost their glyphs and are faithful standard-form
reconstructions.  The digital API source mix (35% Th, 49% U, 16% K) and the
U-235 weight (0.04604) are reproduced.  Energies in MeV.
"""

import numpy as np

U235_WEIGHT = 0.04604        # weight folding U-235 photon probabilities into U-238
API_REFERENCE = 200.0        # API value of the calibration (digital) formation
SOURCE_MIX = {"Th": 0.35, "U": 0.49, "K": 0.16}   # digital-API source fractions


# ---------------------------------------------- spectrum --------------

def representative_energy(energies, probabilities):
    """Representative bin energy  Er = sum(Ei*Pi)/sum(Pi); also returns sum(Pi)  (Eq. 1)."""
    e = np.asarray(energies, float)
    p = np.asarray(probabilities, float)
    return float((e * p).sum() / p.sum()), float(p.sum())


def source_rate(n_per_concentration, concentration):
    """Photon emission rate  n = (n/C)*C  (linear in element concentration)."""
    return n_per_concentration * np.asarray(concentration, float)


def isotope_fractions(rates):
    """Isotope sampling fractions  n_i/sum(n_j)."""
    r = np.asarray(rates, float)
    return r / r.sum()


def uranium_yield(u238_probs, u235_probs, weight=U235_WEIGHT):
    """Fold the U-235 chain into U-238  yield = U238 + weight*U235."""
    return np.asarray(u238_probs, float) + weight * np.asarray(u235_probs, float)


# ---------------------------------------------- count rate --------------

def count_rate(tally, multiplier):
    """Modeled count rate  = Monte Carlo tally * source multiplier."""
    return np.asarray(tally, float) * multiplier


def tool_sensitivity(count_rate_value, api=API_REFERENCE):
    """Tool sensitivity  = count_rate / 200 API  (cps/API)."""
    return count_rate_value / api


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Gamma-Ray Tool API Characterization")
    print("=" * 60)

    # Representative energy is a probability-weighted mean within the bin
    er, ptot = representative_energy([0.5, 0.6, 0.7], [0.2, 0.5, 0.3])
    print(f"  representative energy  = {er:.3f} MeV (sum P = {ptot:.2f})")
    assert 0.5 < er < 0.7 and np.isclose(ptot, 1.0)

    # Source rate is linear; isotope fractions sum to 1
    assert np.isclose(source_rate(3.0, 4.0), 12.0)
    assert np.isclose(isotope_fractions([0.35, 0.49, 0.16]).sum(), 1.0)

    # U-235 adds a small weighted contribution to U-238
    y = uranium_yield([1.0], [2.0])
    assert np.isclose(y[0], 1.0 + U235_WEIGHT * 2.0)

    # Count rate and tool sensitivity (cps/API)
    cr = count_rate(0.005, 1.7505)            # tally x multiplier (per cm^3, 200 API)
    sens = tool_sensitivity(cr * 200.0)       # scale tally-product to 200 API basis
    print(f"  count rate / sensitivity = {cr:.4f} / {sens:.4f}")
    assert cr > 0 and np.isclose(sens, cr)
    assert np.isclose(sum(SOURCE_MIX.values()), 1.0)
    print("  PASS")
    return {"Er": er, "sensitivity": float(sens)}


if __name__ == "__main__":
    test_all()
