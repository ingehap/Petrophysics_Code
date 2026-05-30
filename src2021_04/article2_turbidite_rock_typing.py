"""
Article 2: Challenges in the Petrophysical and Dynamic Characterization of
           Deepwater Turbidite Deposits of the Colombian Caribbean Offshore -
           A Case Study
Angel Restrepo, Gomez-Moncada, Mora Sanchez, Bueno Silva (2021)
DOI: 10.30632/PJV62N2-2021a2

A field case study of deepwater turbidite sandstones that integrates core,
MICP, NMR, and electrical (Co-Cw / Waxman-Smits) data into a pore-throat-radius
(Winland R35) rock-type model driving the log-based evaluation.

Implements:

  - R35 from core CT  log10(R35) = -6.06*RHOB + 0.83*PEF + 10.94     (Eq. 1)
  - R35 from logs     log10(R35) = 0.23*PHITD - 2.47*VSH - 1.18      (Eq. 2)
  - Rock-type classification by R35 pore-throat radius
  - Waxman-Smits conductivity  Co = (1/F*)*(Cw + B*Qv)
  - Irreducible-water-saturation lookup by rock type

Equations transcribed from the rendered article (the only two numbered
equations; both are field-specific empirical regressions).  This is largely a
descriptive case study - Archie/Brooks-Corey/relative-perm are cited by name
only and supplied here in standard form where needed.  R35 in microns.
"""

import numpy as np

# Rock-type pore-throat-radius cutoffs (microns) and Swirr ranges (Table 2)
ROCK_TYPES = [
    ("RT-1", 5.0, np.inf, (0.10, 0.235)),
    ("RT-2", 2.0, 5.0,    (0.215, 0.364)),
    ("RT-3", 0.5, 2.0,    (0.36, 0.554)),
    ("RT-4", 0.0, 0.5,    (None, None)),     # clay-rich, non-reservoir
]
B_MAX = 3.83        # ohm^-1 m^-1 equiv^-1 liter @ 25 C (Waxman-Smits)


# ---------------------------------------------- Eqs. 1-2: R35 -----------

def r35_from_ct(rhob_ct, pef_ct):
    """R35 from CT density & PEF  log10(R35) = -6.06*RHOB + 0.83*PEF + 10.94 (Eq. 1)."""
    return 10.0 ** (-6.06 * np.asarray(rhob_ct, float)
                    + 0.83 * np.asarray(pef_ct, float) + 10.94)


def r35_from_logs(phitd, vsh):
    """R35 from logs  log10(R35) = 0.23*PHITD - 2.47*VSH - 1.18  (Eq. 2).

    Note: as printed, the coefficients yield sub-micron R35 over normal ranges;
    the coefficients are exposed here verbatim and flagged in the README.
    """
    return 10.0 ** (0.23 * np.asarray(phitd, float)
                    - 2.47 * np.asarray(vsh, float) - 1.18)


# ---------------------------------------------- rock typing -------------

def rock_type(r35):
    """Classify a pore-throat radius (microns) into a rock type."""
    for name, lo, hi, _ in ROCK_TYPES:
        if lo <= r35 < hi:
            return name
    return "RT-4"


def swirr_range(rt_name):
    """Irreducible-water-saturation (min, max) for a rock type (Table 2)."""
    for name, _, _, sw in ROCK_TYPES:
        if name == rt_name:
            return sw
    return (None, None)


# ---------------------------------------------- Waxman-Smits ------------

def waxman_smits_co(cw, f_star, qv, B=B_MAX):
    """Waxman-Smits  Co = (1/F*)*(Cw + B*Qv)  (mho/m)."""
    return (1.0 / f_star) * (cw + B * qv)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Deepwater Turbidite Rock Typing (case study)")
    print("=" * 60)

    # Eq. 1: a clean, low-density / high-PEF point -> large R35
    r1 = r35_from_ct(2.0, 3.0)
    print(f"  R35 (RHOB=2.0,PEF=3.0) = {r1:.1f} um  (expect ~20.4)")
    assert abs(r1 - 10 ** 1.31) < 0.5
    # a denser, higher-PEF point -> sub-micron R35
    r2 = r35_from_ct(2.4, 4.0)
    assert r2 < 1.0 and r1 > r2

    # Eq. 2 runs and is monotone in porosity / clay (verbatim coefficients)
    assert r35_from_logs(0.30, 0.05) > r35_from_logs(0.25, 0.10)

    # Rock-type classifier on R35 cutoffs
    assert rock_type(8.0) == "RT-1"
    assert rock_type(3.0) == "RT-2"
    assert rock_type(1.0) == "RT-3"
    assert rock_type(0.2) == "RT-4"
    print("  rock types 8/3/1/0.2um = RT-1/RT-2/RT-3/RT-4 OK")

    # Swirr increases as rock quality degrades
    assert swirr_range("RT-1")[1] < swirr_range("RT-3")[1]

    # Waxman-Smits Co line
    co = waxman_smits_co(cw=2.0, f_star=9.04, qv=0.5)
    print(f"  Waxman-Smits Co        = {co:.3f} mho/m  (expect 0.433)")
    assert abs(co - 0.433) < 1e-3
    print("  PASS")
    return {"r35_ct": r1, "co": co}


if __name__ == "__main__":
    test_all()
