"""
xiong_productivity_factors.py
Implementation of ideas from:
Xiong et al., "Quantitative Evaluation of High-Productivity Controlling Factors
for Ultradeep Gas Wells in the Qixia Formation",
Petrophysics, Vol. 65, No. 2 (April 2024), pp. 194-214.

The paper builds an evaluation index system from geological/petrophysical
indicators (degree of dolomitization, distribution of high-energy shoal-mound
complexes, fracture development, porosity, permeability...) and ranks their
relative importance using grey relational analysis (GRA) and analytic hierarchy
process (AHP)-style weighting.
"""
import numpy as np


def normalize(X):
    """Min-max normalize each column to [0, 1]."""
    X = np.asarray(X, dtype=float)
    mn, mx = X.min(0), X.max(0)
    return (X - mn) / (mx - mn + 1e-30)


def grey_relational_grade(reference, candidates, rho=0.5):
    """Grey relational analysis between a reference series and several candidates."""
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidates, dtype=float)
    diff = np.abs(cand - ref[None, :])
    dmin, dmax = diff.min(), diff.max()
    xi = (dmin + rho * dmax) / (diff + rho * dmax + 1e-30)
    return xi.mean(axis=1)


def ahp_weights(pairwise):
    """Eigenvector method for AHP-style weights from a pairwise comparison matrix."""
    M = np.asarray(pairwise, dtype=float)
    w, v = np.linalg.eig(M)
    idx = np.argmax(w.real)
    weights = np.abs(v[:, idx].real)
    return weights / weights.sum()


def productivity_score(features, weights):
    """Composite weighted productivity score per well."""
    F = normalize(features)
    return F @ np.asarray(weights)


def rank_factors(names, weights):
    order = np.argsort(weights)[::-1]
    return [(names[i], float(weights[i])) for i in order]


def test_all():
    rng = np.random.default_rng(2)
    n_wells, n_factors = 25, 5
    names = ["dolomitization", "shoal_mound", "fractures", "porosity", "permeability"]
    X = rng.uniform(0, 1, size=(n_wells, n_factors))
    productivity = (
        0.35 * X[:, 0] + 0.25 * X[:, 1] + 0.20 * X[:, 2] + 0.10 * X[:, 3] + 0.10 * X[:, 4]
        + 0.05 * rng.standard_normal(n_wells)
    )
    grades = grey_relational_grade(productivity, X.T)
    pairwise = np.array([
        [1, 2, 3, 5, 5],
        [1/2, 1, 2, 4, 4],
        [1/3, 1/2, 1, 3, 3],
        [1/5, 1/4, 1/3, 1, 1],
        [1/5, 1/4, 1/3, 1, 1],
    ])
    w = ahp_weights(pairwise)
    assert abs(w.sum() - 1) < 1e-9
    score = productivity_score(X, w)
    assert score.shape == (n_wells,)
    ranked = rank_factors(names, w)
    assert ranked[0][0] == "dolomitization"
    print("xiong_productivity_factors OK  top=", ranked[0])


if __name__ == "__main__":
    test_all()
