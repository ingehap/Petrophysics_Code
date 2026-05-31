"""
Article 3: Estimation of Permeability in the McMurray Formation Using
           High-Resolution Data Sources
Manchuk, Garner, Deutsch (2015)
Reference: Petrophysics Vol. 56, No. 2 (April 2015), pp. 125-139
DOI: none assigned (this issue predates SPWLA DOI assignment)

High-resolution microresistivity image logs are turned into fine-scale (mm)
permeability micromodels and upscaled by flow-based averaging.  Within the sand
(above the percolation threshold) the permeability follows a log-linear function
of shale volume Vsh (clean-sand permeability ksa, and k25 at Vsh = 0.25);
mud pixels are below the threshold.  The effective (upscaled) permeability lies
between the harmonic (series, lower bound) and arithmetic (parallel, upper
bound) averages, with the geometric average in between.

Implements:

  - Log-linear Vsh-permeability transform  log10(k) = alpha + y'*Vsh  (Eqs. 2-4)
  - Permeability averaging: arithmetic / geometric / harmonic
  - Flow-based effective-permeability bounds (harmonic <= geometric <= arithmetic)

Note: this issue's PDF has a text layer; the Vsh-k log-linear relation (Eqs. 2-4)
and the averaging-based upscaling are transcribed from the body, while the
typeset glyphs were dropped and reconstructed in standard form.  Permeability in
mD, Vsh and porosity as fractions.
"""

import numpy as np


# ---------------------------------------------- Vsh-permeability --------------

def vsh_permeability(vsh, k_sand, k25):
    """Log-linear shale-volume permeability transform (Eqs. 2-4)

        log10(k) = log10(k_sand) + y'*Vsh,   y' = 4*log10(k25/k_sand),

    so k = k_sand*(k25/k_sand)^(Vsh/0.25); k_sand is the clean-sand permeability
    and k25 the permeability at Vsh = 0.25.  Applied to sand pixels (above the
    percolation threshold).
    """
    vsh = np.asarray(vsh, float)
    return k_sand * (k25 / k_sand) ** (vsh / 0.25)


# ---------------------------------------------- averaging / upscaling --------------

def permeability_average(k_values, method="geometric"):
    """Average a permeability field (arithmetic / geometric / harmonic)."""
    k = np.asarray(k_values, float)
    if method == "arithmetic":
        return float(np.mean(k))
    if method == "geometric":
        return float(np.exp(np.mean(np.log(k))))
    if method == "harmonic":
        return float(k.size / np.sum(1.0 / k))
    raise ValueError(f"unknown method: {method}")


def effective_permeability_bounds(k_values):
    """Flow-based effective-permeability bounds for a heterogeneous field

        harmonic (series, lower) <= geometric <= arithmetic (parallel, upper).

    The true flow-based effective permeability lies within these bounds; returns
    (harmonic, geometric, arithmetic).
    """
    return (permeability_average(k_values, "harmonic"),
            permeability_average(k_values, "geometric"),
            permeability_average(k_values, "arithmetic"))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: McMurray Permeability Upscaling")
    print("=" * 60)

    # Vsh-k transform: clean sand at Vsh=0, k25 at Vsh=0.25, decreasing with Vsh
    assert np.isclose(vsh_permeability(0.0, 5000.0, 500.0), 5000.0)
    assert np.isclose(vsh_permeability(0.25, 5000.0, 500.0), 500.0)
    k_mid = vsh_permeability(0.5, 5000.0, 500.0)
    print(f"  k(Vsh=0.5)             = {k_mid:.1f} mD")
    assert k_mid < 500.0

    # Averages are ordered harmonic <= geometric <= arithmetic
    field = np.array([10.0, 100.0, 1000.0, 5000.0])
    h, g, a = effective_permeability_bounds(field)
    print(f"  harmonic/geo/arith     = {h:.1f} / {g:.1f} / {a:.1f} mD")
    assert h <= g <= a
    assert np.isclose(g, permeability_average(field, "geometric"))

    # A layered (high-contrast) field: harmonic is dominated by the tight layers
    layered = np.array([0.1, 0.1, 0.1, 2000.0])
    h2, g2, a2 = effective_permeability_bounds(layered)
    assert h2 < 1.0 < a2
    print("  PASS")
    return {"k_mid": float(k_mid), "harmonic": float(h), "arithmetic": float(a)}


if __name__ == "__main__":
    test_all()
