"""
Article 6: Reminiscences on the Development of the First Commercial
           Array-Induction Measurement
Peter Elkington (2014)
Reference: Petrophysics Vol. 55, No. 6 (December 2014), pp. 618-623
DOI: none assigned (this issue predates SPWLA DOI assignment)

A historical account of BPB's Digital Induction System (~1981-82).  The essay is
narrative and carries no display equations, but it describes a concrete signal-
processing workflow: coils were modelled (point dipoles -> rings -> cylinders),
the four coil-triplet outputs were combined by weighted addition to match the
public-domain 6FF40 deep and dual-induction medium radial geometric factors,
vertical response was shaped with a sparse depth-shift filter (a digital filter
whose coefficients are mostly zero), and three or four independent measurements
were inverted on a conductivity-domain plot for (Rxo, Rt, Di).

Implements (the documented workflow, in standard induction-log form):

  - Doll cumulative radial geometric factor for a two-coil sonde
  - Apparent conductivity as a geometric-factor-weighted mix of invaded and
    virgin zones (the conductivity-domain combination)
  - Weighted combination of array sub-responses to match a target geometric
    factor (the brute-force weighted-addition step)
  - Sparse depth-shift vertical filter (mostly-zero-coefficient filter)
  - Three-measurement step-profile invasion inversion for (Rxo, Rt, Di)

Note: this essay's PDF has a text layer but no equations; the relations below
are the standard Doll induction-response forms (Doll, 1949; Moran & Kunz, 1962)
that the described workflow rests on.  Conductivities in S/m, depths/radii in m.
"""

import numpy as np


# ---------------------------------------------- radial geometric factor --------------

def radial_geometric_factor(r, spacing):
    """Doll cumulative radial geometric factor for a two-coil sonde of the given
    coil spacing L: the fraction of the signal originating within radius r.

        G(r) = r^2 / (r^2 + (L/2)^2),

    a monotone form rising from 0 on the axis to 1 far from the borehole, with
    the half-spacing setting the radial depth of investigation.
    """
    r = np.asarray(r, float)
    return r ** 2 / (r ** 2 + (spacing / 2.0) ** 2)


def apparent_conductivity(sigma_xo, sigma_t, r_invasion, spacing):
    """Apparent conductivity as a geometric-factor-weighted mix of the invaded
    (sigma_xo) and virgin (sigma_t) zones (the conductivity-domain combination)

        Ca = G(Di)*sigma_xo + (1 - G(Di))*sigma_t,

    with G the cumulative radial geometric factor out to the invasion radius.
    """
    g = radial_geometric_factor(r_invasion, spacing)
    return g * sigma_xo + (1.0 - g) * sigma_t


# ---------------------------------------------- weighted combination --------------

def weighted_geometric_factor(sub_factors, weights):
    """Weighted addition of array sub-responses to synthesize a target radial
    geometric factor (the brute-force weighted-addition step)

        G_target(r) = sum_i w_i*G_i(r),

    with the per-channel cumulative geometric factors G_i and weights w_i.
    """
    sub = np.asarray(sub_factors, float)
    w = np.asarray(weights, float).reshape(-1, 1)
    return np.sum(w * sub, axis=0)


# ---------------------------------------------- vertical filter --------------

def sparse_vertical_filter(log, shifts, weights):
    """Sparse depth-shift vertical filter: a weighted addition of a few depth-
    shifted copies of the radially processed log, equivalent to a digital filter
    whose coefficients are mostly zero (memory-limited 1981 hardware).

        out[k] = sum_j w_j * log[k - shift_j].

    Shifts in samples; edges handled by clamping to the array bounds.
    """
    log = np.asarray(log, float)
    out = np.zeros_like(log)
    n = log.size
    for s, w in zip(shifts, weights):
        idx = np.clip(np.arange(n) - s, 0, n - 1)
        out += w * log[idx]
    return out


# ---------------------------------------------- invasion inversion --------------

def invert_invasion(ca_short, ca_long, spacing_short, spacing_long,
                    sigma_xo, di_grid=None):
    """Step-profile invasion inversion: solve for (Rxo, Rt, Di) from two array
    apparent conductivities measured at different spacings, given a known flushed-
    zone conductivity sigma_xo (e.g. from a shallow measurement).

    For each candidate invasion radius Di the long-spacing reading fixes a virgin
    conductivity sigma_t; the radius that makes the short-spacing model match its
    reading is selected.  Returns (Rxo, Rt, Di).
    """
    if di_grid is None:
        di_grid = np.linspace(0.05, 1.5, 400)
    best = None
    for di in di_grid:
        g_long = radial_geometric_factor(di, spacing_long)
        sigma_t = (ca_long - g_long * sigma_xo) / (1.0 - g_long)
        if sigma_t <= 0:
            continue
        ca_short_model = apparent_conductivity(sigma_xo, sigma_t, di, spacing_short)
        resid = abs(ca_short_model - ca_short)
        if best is None or resid < best[0]:
            best = (resid, sigma_t, di)
    _, sigma_t, di = best
    return 1.0 / sigma_xo, 1.0 / sigma_t, di


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Array-Induction Geometric Factors")
    print("=" * 60)

    # Radial geometric factor rises monotonically from 0 toward 1
    g = radial_geometric_factor([0.0, 0.5, 5.0, 50.0], spacing=1.0)
    print(f"  G(r) = {np.round(g, 3)}")
    assert g[0] == 0.0 and g[-1] > 0.99 and np.all(np.diff(g) > 0)

    # Apparent conductivity lies between the invaded and virgin values
    ca = apparent_conductivity(sigma_xo=1.0, sigma_t=0.1, r_invasion=0.3, spacing=1.0)
    print(f"  Ca = {ca:.4f} S/m")
    assert 0.1 < ca < 1.0

    # Weighted combination can reproduce one channel exactly (weights -> selector)
    r = np.linspace(0.01, 5, 50)
    subs = [radial_geometric_factor(r, L) for L in (0.5, 1.0, 2.0)]
    combo = weighted_geometric_factor(subs, [0.0, 1.0, 0.0])
    assert np.allclose(combo, subs[1])

    # Sparse vertical filter with a single zero-shift unit weight is identity
    log = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    assert np.allclose(sparse_vertical_filter(log, [0], [1.0]), log)
    smoothed = sparse_vertical_filter(log, [-1, 0, 1], [0.25, 0.5, 0.25])
    assert smoothed.max() < log.max()  # symmetric smoothing reduces the peak

    # Invasion inversion recovers a synthetic step profile
    sigma_xo, sigma_t, di_true = 1.0, 0.1, 0.4
    Ls, Ll = 0.5, 2.0
    ca_s = apparent_conductivity(sigma_xo, sigma_t, di_true, Ls)
    ca_l = apparent_conductivity(sigma_xo, sigma_t, di_true, Ll)
    rxo, rt, di = invert_invasion(ca_s, ca_l, Ls, Ll, sigma_xo)
    print(f"  inverted: Rxo={rxo:.3f}  Rt={rt:.3f}  Di={di:.3f} m")
    assert np.isclose(rt, 1.0 / sigma_t, rtol=0.05) and np.isclose(di, di_true, atol=0.05)
    print("  PASS")
    return {"Ca": float(ca), "Rt": float(rt), "Di": float(di)}


if __name__ == "__main__":
    test_all()
