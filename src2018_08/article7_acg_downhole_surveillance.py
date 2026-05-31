"""
Article 7: ACG - 20 Years of Downhole Surveillance History
Sheydayev, Atakishiyev, Zett, Schoepf, Thiruvenkatanathan (2018)
DOI: 10.30632/PJV59V4-2018a6

A field-history paper on the Azeri-Chirag-Guneshli (ACG) giant Caspian field,
tracing 20 years of surveillance from conventional production logging to
permanent downhole pressure/temperature gauges (PDHG) and permanent
distributed-fiber sensing.  The paper is qualitative, so this module is a
*methodology proxy*: it implements the standard surveillance computations the
described workflow relies on - a productivity index from gauge pressures, a
moving-average transient-event detector on a P/T stream, and the data-rate
budget that motivated the streaming/visualization architecture.

Implements:

  - Productivity index  PI = q/(Pr - Pwf)  and inflow from PI
  - Drawdown from a target rate  dP = q/PI
  - Transient-event detection on a gauge stream (moving-average residual)
  - Distributed-sensing data-rate budget (channels x rate x sample size)

Note: this article contains no published equations (it is an operations /
field-history review), so - as with the methodology proxies elsewhere in this
repository - the relations below are the standard surveillance formulas the
described PDHG/DFO workflow uses, not formulas transcribed from the paper.
SI / field-consistent units.
"""

import numpy as np


# ---------------------------------------------- productivity --------------

def productivity_index(q, p_res, p_wf):
    """Productivity index  PI = q/(Pr - Pwf)  (rate per unit drawdown)."""
    return q / (p_res - p_wf)


def inflow_rate(pi, p_res, p_wf):
    """Inflow rate from PI and drawdown  q = PI*(Pr - Pwf)."""
    return pi * (p_res - p_wf)


def required_drawdown(q, pi):
    """Drawdown needed for a target rate  dP = q/PI."""
    return q / pi


# ---------------------------------------------- surveillance --------------

def detect_transients(stream, window=10, n_sigma=4.0):
    """Flag transient events in a gauge stream by moving-average residual.

    A sample whose deviation from the trailing moving average exceeds n_sigma
    times the residual standard deviation is flagged as a transient (e.g. a
    choke change, slug, or shut-in seen by a permanent gauge).  Returns the
    integer indices of the flagged samples.
    """
    x = np.asarray(stream, float)
    pad = window // 2
    kernel = np.ones(window) / window
    # Edge-pad before smoothing so the trend is not pulled toward zero at the
    # ends (which would flag the boundaries instead of true transients).
    trend = np.convolve(np.pad(x, pad, mode="edge"), kernel, mode="same")[pad:pad + len(x)]
    resid = x - trend
    thresh = n_sigma * resid.std()
    return np.where(np.abs(resid) > thresh)[0]


def data_rate(n_channels, sample_hz, bytes_per_sample):
    """Distributed-sensing data rate  = channels*rate*bytes  (bytes/s).

    The ~1 TB/hr streams the paper cites come from tens of thousands of fiber
    channels sampled fast; this sizes the acquisition/visualization pipeline.
    """
    return n_channels * sample_hz * bytes_per_sample


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: ACG Downhole Surveillance (proxy)")
    print("=" * 60)

    # PI round-trips: inflow at the same drawdown recovers the rate
    pi = productivity_index(q=5000.0, p_res=4000.0, p_wf=3500.0)
    print(f"  productivity index     = {pi:.2f} bbl/d/psi")
    assert np.isclose(inflow_rate(pi, 4000.0, 3500.0), 5000.0)
    assert np.isclose(required_drawdown(5000.0, pi), 500.0)

    # Transient detector flags a planted shut-in spike, not the quiet baseline
    rng = np.random.default_rng(6)
    stream = 3500.0 + rng.normal(0, 1.0, 200)
    stream[120:123] += 60.0                            # planted transient
    flags = detect_transients(stream, window=10, n_sigma=4.0)
    print(f"  transient samples      = {flags.tolist()}")
    assert any(118 <= i <= 125 for i in flags)
    assert len(flags) < 15                             # baseline stays quiet

    # Data-rate budget: ~10k channels at 10 kHz, 2 bytes -> ~0.7 TB/hr
    tb_per_hr = data_rate(10000, 10000, 2) * 3600 / 1e12
    print(f"  fiber data rate        = {tb_per_hr:.2f} TB/hr")
    assert tb_per_hr > 0.5
    print("  PASS")
    return {"PI": float(pi), "n_transient": int(len(flags))}


if __name__ == "__main__":
    test_all()
