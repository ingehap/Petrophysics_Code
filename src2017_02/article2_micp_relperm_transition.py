"""
Article 2: Relative Permeability Effects Overlooked in MICP Measurements;
           Transition Zones Likely to be Smaller
Maas, Springer, Hebing (2017)
Reference: Petrophysics Vol. 58, No. 1 (February 2017), pp. 19-27
DOI: none assigned (this issue predates SPWLA DOI assignment)

Mercury-injection capillary pressure (MICP) is biased by mercury relative
permeability: at the standard short equilibration time the wetting (gas)
saturation has not reached equilibrium, so the apparent capillary-pressure curve
undershoots and transition zones look larger than they are.  This module
implements the equilibration shortfall, the (phi/K)^0.5 capillary-pressure
scaling, a two-sample t-test for comparing equilibration times, and the
homogeneity-number sample filter.

Implements:

  - Equilibration shortfall  Sw_apparent = Sw_eq*(1 - exp(-t/tau))
  - Capillary-pressure scaling  Pc_scaled = Pc*sqrt(phi/K)
  - Two-sample Student t statistic
  - Homogeneity-number sample-quality filter (V <= 0.25)

Note: this issue's PDF has a text layer; the (phi/K)^0.5 scaling and the t-test
survived as prose, so the relations below are standard-form reconstructions of
the methods the paper applies.  Pc in Pa, time arbitrary-consistent, fractions.
"""

import numpy as np

HOMOGENEITY_CUTOFF = 0.25


# ---------------------------------------------- equilibration --------------

def apparent_saturation(sw_equilibrium, time, tau):
    """Apparent mercury saturation at finite equilibration time

        Sw_apparent = Sw_eq*(1 - exp(-t/tau)),

    which undershoots the equilibrium value and approaches it as t -> infinity.
    """
    return sw_equilibrium * (1.0 - np.exp(-np.asarray(time, float) / tau))


def pc_scaling(pc, phi, k):
    """Capillary-pressure normalization  Pc_scaled = Pc*sqrt(phi/K)."""
    return np.asarray(pc, float) * np.sqrt(phi / k)


# ---------------------------------------------- statistics / QC --------------

def two_sample_t(mean1, mean2, std1, std2, n1, n2):
    """Two-sample Student t statistic  (m1 - m2)/sqrt(s1^2/n1 + s2^2/n2)."""
    return (mean1 - mean2) / np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)


def is_homogeneous(homogeneity_number, cutoff=HOMOGENEITY_CUTOFF):
    """Sample passes the homogeneity filter if V <= cutoff (Maas & Hebing 2013)."""
    return np.asarray(homogeneity_number, float) <= cutoff


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: MICP Relative-Permeability Effects")
    print("=" * 60)

    # Finite equilibration undershoots; longer time approaches equilibrium
    s_short = apparent_saturation(0.80, 10.0, tau=1000.0)
    s_long = apparent_saturation(0.80, 1e5, tau=1000.0)
    print(f"  Sw 10s / long          = {s_short:.3f} / {s_long:.3f} (eq 0.80)")
    assert s_short < s_long and abs(s_long - 0.80) < 1e-3

    # (phi/K)^0.5 scaling: tighter rock (lower K) scales Pc up
    assert pc_scaling(1e6, 0.25, 1e-15) > pc_scaling(1e6, 0.25, 1e-12)

    # Two-sample t statistic is nonzero for separated means
    t = two_sample_t(75.0, 60.0, 3.0, 3.0, 4, 4)
    print(f"  t statistic            = {t:.2f}")
    assert t > 0

    # Homogeneity filter
    flags = is_homogeneous(np.array([0.18, 0.30]))
    assert flags.tolist() == [True, False]
    print("  PASS")
    return {"Sw_short": float(s_short), "t": float(t)}


if __name__ == "__main__":
    test_all()
