"""
Article 2: Integrated Petrofacies Characterization and Interpretation of
           Depositional Environment of the Bakken Shale in the Williston Basin,
           North America
Bhattacharya, Carr (2016)
Reference: Petrophysics Vol. 57, No. 2 (April 2016), pp. 96-111
DOI: none assigned (this issue predates SPWLA DOI assignment)

Core and advanced logs are combined into a stochastic multimineral solution and
a petrofacies classification for the Bakken shale.  TOC is estimated from the
density log (Schmoker-Hester), clay volume from neutron-density and uranium-free
gamma, and mineral volumes from a linear log-response inversion.  The Bakken
shale members are then classified into five petrofacies from the
quartz-to-carbonate ratio and clay content (a ternary scheme), and the link
between petrofacies and depositional environment is tested with a chi-square
statistic.

Implements:

  - Schmoker-Hester density TOC  TOC = 154.497/rho_b - 57.261  (Eq. 1)
  - Averaged clay volume  Vclay = (Vclay_ND + Vclay_CGR)/2  (Eq. 2)
  - Stochastic multimineral linear inversion (log responses -> volumes)
  - Petrofacies classification from quartz/carbonate ratio and clay
  - Chi-square statistic for petrofacies vs. depositional environment

Note: this issue's PDF has a text layer; the Schmoker TOC and clay-volume
relations (Eqs. 1-2) and the ternary classification thresholds (Q/C = 3 and 1/3,
clay = 30%) are transcribed from the body, while the multimineral inversion and
chi-square are the standard methods the workflow relies on.  Densities in
g/cm^3, volumes/fractions dimensionless.
"""

import numpy as np


# ---------------------------------------------- log transforms --------------

def toc_schmoker(rho_b):
    """Schmoker-Hester (1983) density TOC  TOC = 154.497/rho_b - 57.261  [wt%]."""
    return 154.497 / np.asarray(rho_b, float) - 57.261


def clay_volume(vclay_nd, vclay_cgr):
    """Averaged clay volume  Vclay = (Vclay_ND + Vclay_CGR)/2  (Eq. 2)."""
    return 0.5 * (np.asarray(vclay_nd, float) + np.asarray(vclay_cgr, float))


# ---------------------------------------------- multimineral inversion --------------

def multimineral_inversion(log_responses, endpoints):
    """Stochastic multimineral solution by constrained linear inversion.

    Solves  endpoints @ volumes = log_responses  with the unity constraint
    sum(volumes) = 1, in a least-squares sense.  `endpoints` is (n_logs,
    n_minerals) of the pure-mineral tool responses.  Returns the mineral
    volume fractions.
    """
    a = np.asarray(endpoints, float)
    b = np.asarray(log_responses, float)
    n_min = a.shape[1]
    # append the unity (closure) equation with a large weight
    w = 1e3
    a_aug = np.vstack([a, w * np.ones(n_min)])
    b_aug = np.concatenate([b, [w * 1.0]])
    vols, *_ = np.linalg.lstsq(a_aug, b_aug, rcond=None)
    return vols


# ---------------------------------------------- petrofacies --------------

def petrofacies(quartz, carbonate, clay, clay_cut=0.30, qc_high=3.0, qc_low=1.0 / 3.0):
    """Classify a Bakken shale petrofacies from composition.

    Uses the quartz-to-carbonate ratio (thresholds 3 and 1/3) and the clay
    content (30% cutoff) to assign one of five petrofacies:
      'argillaceous'         (clay > clay_cut)
      'siliceous'            (Q/C > 3)
      'mixed-siliceous'      (1 <= Q/C <= 3)
      'mixed-calcareous'     (1/3 <= Q/C < 1)
      'calcareous'           (Q/C < 1/3)
    """
    if clay > clay_cut:
        return "argillaceous"
    qc = quartz / carbonate
    if qc > qc_high:
        return "siliceous"
    if qc >= 1.0:
        return "mixed-siliceous"
    if qc >= qc_low:
        return "mixed-calcareous"
    return "calcareous"


def chi_square_statistic(contingency):
    """Pearson chi-square statistic of a contingency table

        chi2 = sum (O - E)^2/E,   E_ij = row_i*col_j/total,

    used to test the association between petrofacies and depositional
    environment.  Returns (chi2, degrees_of_freedom).
    """
    o = np.asarray(contingency, float)
    row = o.sum(axis=1, keepdims=True)
    col = o.sum(axis=0, keepdims=True)
    e = row @ col / o.sum()
    chi2 = float(np.sum((o - e) ** 2 / e))
    dof = (o.shape[0] - 1) * (o.shape[1] - 1)
    return chi2, dof


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Bakken Shale Petrofacies")
    print("=" * 60)

    # Schmoker TOC decreases with bulk density and is positive for organic shale
    toc = toc_schmoker(2.35)
    print(f"  Schmoker TOC @2.35      = {toc:.2f} wt%")
    assert toc > 0 and toc_schmoker(2.6) < toc

    # Averaged clay volume
    assert np.isclose(clay_volume(0.30, 0.40), 0.35)

    # Multimineral inversion recovers known volumes (quartz/carbonate/clay)
    # endpoints: rows = [density, neutron, GR]; cols = [quartz, carbonate, clay]
    endpoints = np.array([[2.65, 2.71, 2.45],     # grain density
                          [-0.02, 0.00, 0.30],    # neutron porosity
                          [15.0, 10.0, 120.0]])   # gamma ray
    true_vols = np.array([0.5, 0.3, 0.2])
    logs = endpoints @ true_vols
    vols = multimineral_inversion(logs, endpoints)
    print(f"  inverted volumes        = {np.round(vols, 3)}")
    assert np.allclose(vols, true_vols, atol=1e-3) and np.isclose(vols.sum(), 1.0)

    # Petrofacies classification across the ternary scheme
    assert petrofacies(0.7, 0.1, 0.2) == "siliceous"
    assert petrofacies(0.5, 0.4, 0.1) == "mixed-siliceous"
    assert petrofacies(0.1, 0.7, 0.2) == "calcareous"
    assert petrofacies(0.4, 0.2, 0.4) == "argillaceous"
    print(f"  facies (Q.7/C.1/cl.2)   = {petrofacies(0.7, 0.1, 0.2)}")

    # Chi-square is large when petrofacies and environment are strongly associated
    table = np.array([[40, 2, 1], [3, 35, 2], [1, 4, 38]])
    chi2, dof = chi_square_statistic(table)
    print(f"  chi-square / dof        = {chi2:.1f} / {dof}")
    assert chi2 > 0 and dof == 4
    print("  PASS")
    return {"TOC": float(toc), "chi2": chi2}


if __name__ == "__main__":
    test_all()
