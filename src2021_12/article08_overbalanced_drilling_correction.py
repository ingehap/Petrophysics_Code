"""
Article 8: The Impact of Overbalanced Drilling From Exploration/Appraisal
           Wells to Field Development Plan
Mohammadlou, Reppert, Del Negro, Jones (2021)
DOI: 10.30632/PJV62N6-2021a8

Case study (Melke Formation, Norwegian Sea): two early wells drilled with very
high mud overbalance (~220 and ~170 bar) yield log- and core-derived
porosity/permeability that under-represent true reservoir quality (NMR read
~12% where ~18% was expected, a ~33% undercall).  A low-overbalance geopilot
well plus a Brae Field analog confirm that the high-quality rock above
~12 p.u. / ~100 md was damaged.

Implements:

  - Overbalance pressure  dP = P_mud - P_pore                  (INF-1)
  - Mud hydrostatic pressure  P = 0.0980665 * rho_sg * TVD     (INF-2)
  - Additive porosity correction to core/overburden truth      (INF-4)
  - Porosity-permeability (k-phi) semilog transform & fit       (INF-7)
  - Klinkenberg gas-slippage correction                        (INF-8)
  - Fraction-of-original overburden (FOO) porosity correction  (INF-6)
  - Damage flag (phi > 12 p.u. AND k > 100 md)

Note: the paper is an observational case study with no equations; the forms
here are standard petrophysical relations (flagged INF, inferred) consistent
with its prose and numbers.  Mud weight in sg, pressure in bar, depth in m,
permeability in md, porosity in fraction.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

BAR_PER_SG_M = 0.0980665    # bar per (sg * metre): rho_water*g in bar/m
DAMAGE_PHI = 0.12           # porosity damage threshold (p.u. -> fraction)
DAMAGE_K = 100.0            # permeability damage threshold (md)


# ---------------------------------------------- INF-1/2: pressure -------

def mud_hydrostatic_bar(rho_sg, tvd_m):
    """Mud hydrostatic pressure  P = 0.0980665 * rho_sg * TVD  (INF-2).  bar."""
    return petrolib.integrity_drilling.hydrostatic_pressure_bar(tvd_m, sg=rho_sg)


def overbalance_pressure(p_mud_bar, p_pore_bar):
    """Overbalance  dP = P_mud - P_pore  (INF-1).  bar."""
    return p_mud_bar - p_pore_bar


# ---------------------------------------------- INF-4: porosity corr ----

def correct_porosity(phi_measured, delta_phi):
    """Additive porosity correction toward overburden-corrected core truth."""
    return np.asarray(phi_measured, float) + delta_phi


# ---------------------------------------------- INF-7: k-phi ------------

def fit_kphi(phi, k):
    """Fit semilog poro-perm trend  log10(k) = a*phi + b.  Returns (a, b)."""
    lf = petrolib.inversion_numerics.fitting.fit_line(phi, k, yform="log10")
    return float(lf.slope), float(lf.intercept)


def kphi_permeability(phi, a, b):
    """Predict permeability from porosity:  k = 10^(a*phi + b)."""
    return 10.0 ** (a * np.asarray(phi, float) + b)


# ---------------------------------------------- INF-8: Klinkenberg ------

def klinkenberg(k_gas, b_slip, p_mean_bar):
    """Liquid-equivalent permeability  k_inf = k_gas / (1 + b/P_mean)  (INF-8)."""
    return petrolib.flow_transport.klinkenberg_corrected(k_gas, b=b_slip, p_mean=p_mean_bar)


# ---------------------------------------------- INF-6: FOO --------------

def fraction_of_original(prop_at_ncp, prop_at_ref):
    """Fraction-of-original overburden correction  FOO = prop(NCP)/prop(ref)."""
    return np.asarray(prop_at_ncp, float) / prop_at_ref


# ---------------------------------------------- damage flag -------------

def damage_flag(phi, k, phi_cut=DAMAGE_PHI, k_cut=DAMAGE_K):
    """Flag overbalance-damaged high-quality rock: phi > cut AND k > cut."""
    return (np.asarray(phi, float) > phi_cut) & (np.asarray(k, float) > k_cut)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Overbalanced-Drilling Impact & Correction")
    print("=" * 60)

    # Exploration well: 1.74 sg mud, near-hydrostatic pore pressure (~1.03 sg)
    tvd = 2600.0
    p_mud = mud_hydrostatic_bar(1.74, tvd)
    p_pore = mud_hydrostatic_bar(1.03, tvd)
    dP = overbalance_pressure(p_mud, p_pore)
    print(f"  mud / pore pressure    = {p_mud:.0f} / {p_pore:.0f} bar")
    print(f"  overbalance dP         = {dP:.0f} bar (paper ~220)")
    assert 170 < dP < 220

    # NMR undercall: measured 12 p.u. -> corrected toward 18 p.u. truth
    phi_corr = correct_porosity(0.12, delta_phi=0.06)
    undercall = (0.18 - 0.12) / 0.18
    print(f"  corrected porosity     = {phi_corr:.2f} (undercall {100*undercall:.0f}%)")
    assert abs(phi_corr - 0.18) < 1e-9
    assert abs(undercall - 0.333) < 0.01

    # Brae-like undamaged k-phi trend through (12%, 100 md) and (15%, 2000 md)
    a, b = fit_kphi([0.12, 0.15], [100.0, 2000.0])
    k_at_13 = kphi_permeability(0.13, a, b)
    print(f"  k-phi fit a/b          = {a:.1f} / {b:.2f}")
    print(f"  k at 13 p.u.           = {k_at_13:.0f} md")
    assert 100.0 < k_at_13 < 2000.0

    # Klinkenberg correction lowers gas permeability
    k_inf = klinkenberg(k_gas=120.0, b_slip=15.0, p_mean_bar=20.0)
    assert k_inf < 120.0

    # FOO: porosity drops under net confining pressure
    foo = fraction_of_original(0.155, 0.18)
    assert 0.0 < foo < 1.0

    # Damage flag isolates the anomalous high-perm low-porosity cloud
    phi = np.array([0.10, 0.13, 0.15, 0.20])
    k = np.array([5.0, 500.0, 2000.0, 50.0])
    flags = damage_flag(phi, k)
    print(f"  damage flags           = {flags.tolist()}")
    assert flags.tolist() == [False, True, True, False]
    print("  PASS")
    return {"overbalance": dP, "phi_corrected": float(phi_corr),
            "kphi": (a, b)}


if __name__ == "__main__":
    test_all()
