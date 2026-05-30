"""
Article 11: Loading Effects on Gas Relative Permeability of a Low-Permeability
            Sandstone
Agostini, Egermann, Jeannin, Portier, Skoczylas, Wang (2019)
DOI: 10.30632/PJV60N2-2019a9

In a low-permeability sandstone the gas permeability depends strongly on the
effective stress (loading): increasing confining stress closes microcracks and
reduces permeability, while at low pore pressure gas slippage (Klinkenberg)
raises the apparent permeability.  The gas relative permeability follows a Corey
model whose absolute permeability is stress-dependent.

Implements:

  - Effective stress  sigma_eff = sigma_conf - alpha*P_pore (Biot)
  - Stress-dependent permeability  k(sigma_eff) = k0*exp(-c*sigma_eff)
  - Klinkenberg apparent gas permeability  k_app = k_l*(1 + b/Pm)
  - Corey gas relative permeability

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard stress-dependent-permeability / Klinkenberg /
Corey relations the loading study applies.
"""

import numpy as np


# ---------------------------------------------- stress / permeability ---

def effective_stress(sigma_conf, p_pore, biot=1.0):
    """Biot effective stress  sigma_eff = sigma_conf - alpha*P_pore."""
    return sigma_conf - biot * np.asarray(p_pore, float)


def stress_permeability(k0, sigma_eff, c=0.05):
    """Stress-dependent permeability  k = k0*exp(-c*sigma_eff)  (microcrack closure)."""
    return k0 * np.exp(-c * np.asarray(sigma_eff, float))


def klinkenberg(k_l, b, p_mean):
    """Klinkenberg apparent gas permeability  k_app = k_l*(1 + b/Pm)."""
    return k_l * (1.0 + b / np.asarray(p_mean, float))


# ---------------------------------------------- Corey kr ----------------

def corey_krg(sg, sgc, swr, ng=2.0, krg_max=1.0):
    """Corey gas relative permeability  krg = krg_max*Sg*^ng."""
    sg = np.asarray(sg, float)
    sge = np.clip((sg - sgc) / (1.0 - sgc - swr), 0.0, 1.0)
    return krg_max * sge ** ng


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 11: Loading Effects on Gas Relative Permeability")
    print("=" * 60)

    # Effective stress rises with confining stress, falls with pore pressure
    assert effective_stress(40.0, 10.0) == 30.0
    assert effective_stress(40.0, 20.0) < effective_stress(40.0, 10.0)

    # Loading (higher effective stress) reduces permeability
    k_lo = stress_permeability(1.0, 10.0)
    k_hi = stress_permeability(1.0, 40.0)
    print(f"  k at sigma_eff 10/40   = {k_lo:.3f} / {k_hi:.3f}")
    assert k_hi < k_lo

    # Klinkenberg: apparent gas permeability exceeds liquid value, falls with Pm
    assert klinkenberg(0.5, 8.0, 2.0) > klinkenberg(0.5, 8.0, 20.0) > 0.5

    # Corey gas kr: zero below critical gas saturation, rises with Sg
    sgc, swr = 0.05, 0.25
    assert corey_krg(sgc, sgc, swr) < 1e-12
    sg = np.linspace(sgc, 1 - swr, 20)
    assert np.all(np.diff(corey_krg(sg, sgc, swr)) >= -1e-12)
    assert abs(corey_krg(1 - swr, sgc, swr) - 1.0) < 1e-9

    # Combined: at higher loading the (stress-reduced) absolute permeability
    # scales the whole gas-kr*k curve down
    keff_lo = stress_permeability(1.0, 10.0) * corey_krg(0.5, sgc, swr)
    keff_hi = stress_permeability(1.0, 40.0) * corey_krg(0.5, sgc, swr)
    print(f"  k*krg at low/high load = {keff_lo:.3f} / {keff_hi:.3f}")
    assert keff_hi < keff_lo
    print("  PASS")
    return {"k_highstress": float(k_hi), "keff_highload": float(keff_hi)}


if __name__ == "__main__":
    test_all()
