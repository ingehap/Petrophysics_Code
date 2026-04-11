"""
Article 4: Evaluating the Usefulness of Least Squares Regression in
Petrophysics Interpretation
Etnyre (Petrophysics, Vol. 65, No. 1, Feb 2024, pp. 70-94)

Compares ordinary least squares (OLS), reverse regression, and the
reduced major axis (RMA / geometric mean) regression — the article's
central message is that OLS slopes are biased toward zero in the presence
of measurement error in X, and RMA is often more appropriate for
petrophysical crossplots (e.g., porosity vs. permeability log-log).
"""
import numpy as np


def ols(x, y):
    x = np.asarray(x); y = np.asarray(y)
    a = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
    b = y.mean() - a * x.mean()
    return a, b


def reverse_ols(x, y):
    a_inv, b_inv = ols(y, x)  # x = a_inv*y + b_inv
    a = 1.0 / a_inv
    b = -b_inv / a_inv
    return a, b


def rma(x, y):
    """Reduced major axis: slope = sign(r) * sy/sx."""
    sx = np.std(x); sy = np.std(y)
    r = np.corrcoef(x, y)[0, 1]
    a = np.sign(r) * sy / sx
    b = y.mean() - a * x.mean()
    return a, b


def test_all():
    rng = np.random.default_rng(1)
    n = 500
    x_true = rng.uniform(0.05, 0.30, n)        # porosity
    y_true = 3.0 * x_true + 0.5                # log10(k)
    x_obs = x_true + rng.normal(0, 0.02, n)    # noisy x
    y_obs = y_true + rng.normal(0, 0.10, n)
    a_ols, _ = ols(x_obs, y_obs)
    a_rev, _ = reverse_ols(x_obs, y_obs)
    a_rma, _ = rma(x_obs, y_obs)
    # OLS biased low, reverse biased high, RMA in between, all positive
    assert a_ols < a_rma < a_rev
    assert a_ols < 3.0 < a_rev
    print(f"article4 OK | OLS={a_ols:.2f} RMA={a_rma:.2f} REV={a_rev:.2f} (true=3.00)")


if __name__ == "__main__":
    test_all()
