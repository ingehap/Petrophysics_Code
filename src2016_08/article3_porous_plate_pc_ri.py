"""
Article 3: Drainage Capillary Pressure and Resistivity Index from Short-Wait
           Porous-Plate Experiments
Dernaika, Wilson, Skjaeveland, Ebeltoft (2016)
Reference: Petrophysics Vol. 57, No. 4 (August 2016), pp. 369-376
DOI: none assigned (this issue predates SPWLA DOI assignment)

The porous-plate desaturation method gives capillary pressure and resistivity
index (RI) vs. saturation history without model assumptions, but is slow because
equilibrium at each Pc step takes a long time.  This paper "short-waits" each
step and predicts the equilibrium water saturation (and 1/RI) by fitting an
exponential-decay model to the early transient, cutting the experiment time by
~3x while reproducing the same Sw-RI relationship and Archie saturation
exponent n.

Implements:

  - Exponential-decay saturation model  Sw(t) = Sweqm + (Sws - Sweqm)*exp(-(t-ts)/tc)  (Eq. 1)
  - Exponential-decay 1/RI model with the same form (Eq. 2)
  - Equilibrium extraction from an early transient (Guggenheim three-point method)
  - Archie resistivity index  RI = Rt/Ro = Sw^(-n)  and saturation-exponent fit

Note: this issue's PDF has a text layer; the exponential-decay relations (Eqs.
1-2) are transcribed from the body, while the typeset glyphs were dropped and
reconstructed in standard form.  Saturations as fractions, times in arbitrary
consistent units, resistivities in ohm-m.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- decay models --------------

def exp_decay_saturation(t, ts, sws, sweqm, tc):
    """Exponential-decay water saturation during a Pc step (Eq. 1)

        Sw(t) = Sweqm + (Sws - Sweqm)*exp(-(t - ts)/tc),

    decaying from the start saturation Sws (at ts) toward the equilibrium Sweqm
    with characteristic time tc.
    """
    return sweqm + (sws - sweqm) * np.exp(-(np.asarray(t, float) - ts) / tc)


def exp_decay_inv_ri(t, ts, inv_ri_s, inv_ri_eqm, tc):
    """Exponential-decay reciprocal resistivity index 1/RI (Eq. 2)

        1/RI(t) = 1/RIeqm + (1/RIs - 1/RIeqm)*exp(-(t - ts)/tc),

    the same form applied to 1/RI; tc is typically smaller than the saturation
    tc (RI changes faster than Sw during primary drainage).
    """
    return inv_ri_eqm + (inv_ri_s - inv_ri_eqm) * np.exp(-(np.asarray(t, float) - ts) / tc)


def fit_equilibrium(times, values):
    """Extract the equilibrium asymptote and characteristic time of an
    exponential decay from an early, non-equilibrated transient.

    Uses the Guggenheim method on three equally time-spaced samples (the basis
    of the "short-wait" prediction): for y(t) = y_inf + (y0 - y_inf)*exp(-t/tc),
    samples at t, t+dt, t+2dt give
        tc      = -dt / ln((y2 - y1)/(y1 - y0))
        y_inf   = y0 - (y1 - y0)^2/((y2 - y1) - (y1 - y0)).
    Returns (asymptote, tc).
    """
    t = np.asarray(times, float)
    y = np.asarray(values, float)
    # pick three equally spaced indices spanning the series
    i0, i1, i2 = 0, len(t) // 2, len(t) - 1
    dt = t[i1] - t[i0]
    y0, y1, y2 = y[i0], y[i1], y[i2]
    d1, d2 = y1 - y0, y2 - y1
    tc = -dt / np.log(d2 / d1)
    asymptote = y0 - d1 ** 2 / (d2 - d1)
    return float(asymptote), float(tc)


# ---------------------------------------------- Archie RI --------------

def resistivity_index(rt, ro):
    """Resistivity index  RI = Rt/Ro  (Ro = resistivity at 100% water)."""
    return petrolib.saturation_resistivity.resistivity_index(rt, ro)


def archie_saturation_exponent(sw, ri):
    """Fit the Archie saturation exponent n from RI = Sw^(-n)

        log10(RI) = -n*log10(Sw),

    by least-squares through the (log Sw, log RI) points.  Returns n.
    """
    return petrolib.saturation_resistivity.fit_saturation_exponent(sw, ri)


def resistivity_index_from_sw(sw, n):
    """Archie resistivity index  RI = Sw^(-n)."""
    return petrolib.saturation_resistivity.resistivity_index_from_sw(sw, n=n)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Short-Wait Porous-Plate Pc & RI")
    print("=" * 60)

    # Saturation decays from start toward equilibrium
    sw0 = exp_decay_saturation(0.0, 0.0, sws=0.9, sweqm=0.4, tc=5.0)
    sw_inf = exp_decay_saturation(1000.0, 0.0, sws=0.9, sweqm=0.4, tc=5.0)
    print(f"  Sw(0) / Sw(inf)        = {sw0:.3f} / {sw_inf:.3f}")
    assert np.isclose(sw0, 0.9) and np.isclose(sw_inf, 0.4, atol=1e-3)

    # Guggenheim fit recovers the equilibrium and tc from an early transient
    t = np.linspace(0.0, 15.0, 7)            # short-wait window (< equilibrium)
    y = exp_decay_saturation(t, 0.0, sws=0.9, sweqm=0.4, tc=5.0)
    sweqm_fit, tc_fit = fit_equilibrium(t, y)
    print(f"  fitted Sweqm / tc      = {sweqm_fit:.3f} / {tc_fit:.2f}")
    assert np.isclose(sweqm_fit, 0.4, atol=1e-3) and np.isclose(tc_fit, 5.0, atol=1e-2)

    # 1/RI uses the same decay form
    inv_ri0 = exp_decay_inv_ri(0.0, 0.0, inv_ri_s=0.9, inv_ri_eqm=0.25, tc=3.0)
    assert np.isclose(inv_ri0, 0.9)

    # Archie: RI = Sw^(-n); fit recovers the exponent used to synthesize the data
    sw = np.array([1.0, 0.8, 0.6, 0.4, 0.3])
    ri = resistivity_index_from_sw(sw, n=2.1)
    n_fit = archie_saturation_exponent(sw, ri)
    print(f"  fitted Archie n        = {n_fit:.3f}")
    assert np.isclose(n_fit, 2.1)
    assert np.isclose(resistivity_index(2.1, 1.0), 2.1)
    print("  PASS")
    return {"Sweqm": sweqm_fit, "tc": tc_fit, "n": float(n_fit)}


if __name__ == "__main__":
    test_all()
