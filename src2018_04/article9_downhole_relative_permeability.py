"""
Article 9: Downhole Estimation of Relative Permeability With Integration of
           Formation-Tester Measurements and Advanced Well Logs
Hadibeik, Azari, Kalawina, Ramakrishna, Eyuboglu, Khan, Al-Rushaid, Al-Rashidi,
Ahmad (2018)
DOI: 10.30632/PJV59N2-2018a8  (inferred; body beyond extraction)

Formation-tester transient pressure data, combined with saturation logs, can be
inverted for two-phase relative permeability downhole.  This *methodology proxy*
implements the standard Corey/Brooks-Corey relative-permeability model the
workflow fits: normalized water saturation between the irreducible and residual
endpoints drives power-law water and oil relative permeabilities, from which the
fractional flow and the mobility ratio follow.

Implements:

  - Normalized water saturation between Swirr and Sor
  - Corey water/oil relative permeability  krw, kro
  - Fractional flow of water  fw = (krw/mu_w)/(krw/mu_w + kro/mu_o)
  - End-point mobility ratio

Note: this article's body was beyond this issue's machine extraction, so - as
with the other methodology proxies in this repository - the relations below are
the standard relative-permeability formulas the described inversion fits, not
formulas transcribed from the paper.  The DOI suffix (a8) is inferred from the
issue's confirmed pattern.  Saturations/relperms fractional; viscosities in cP.
"""

import numpy as np


# ---------------------------------------------- Corey model --------------

def normalized_sw(sw, swirr=0.15, sor=0.20):
    """Normalized (effective) water saturation  Swn = (Sw-Swirr)/(1-Swirr-Sor)."""
    swn = (np.asarray(sw, float) - swirr) / (1.0 - swirr - sor)
    return np.clip(swn, 0.0, 1.0)


def krw(sw, swirr=0.15, sor=0.20, krw_max=0.4, nw=3.0):
    """Corey water relative permeability  krw = krw_max*Swn^nw."""
    return krw_max * normalized_sw(sw, swirr, sor) ** nw


def kro(sw, swirr=0.15, sor=0.20, kro_max=1.0, no=2.0):
    """Corey oil relative permeability  kro = kro_max*(1-Swn)^no."""
    return kro_max * (1.0 - normalized_sw(sw, swirr, sor)) ** no


def fractional_flow(sw, mu_w=0.5, mu_o=2.0, **corey):
    """Water fractional flow  fw = (krw/mu_w)/(krw/mu_w + kro/mu_o)."""
    lam_w = krw(sw, **{k: corey[k] for k in corey if k in ("swirr", "sor", "krw_max", "nw")}) / mu_w
    lam_o = kro(sw, **{k: corey[k] for k in corey if k in ("swirr", "sor", "kro_max", "no")}) / mu_o
    return lam_w / (lam_w + lam_o)


def endpoint_mobility_ratio(krw_max=0.4, kro_max=1.0, mu_w=0.5, mu_o=2.0):
    """End-point mobility ratio  M = (krw_max/mu_w)/(kro_max/mu_o)."""
    return (krw_max / mu_w) / (kro_max / mu_o)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 9: Downhole Relative Permeability (proxy)")
    print("=" * 60)

    # Endpoints: at Swirr only oil flows; at 1-Sor only water flows
    assert np.isclose(krw(0.15), 0.0) and np.isclose(kro(0.15), 1.0)
    assert np.isclose(kro(0.80), 0.0) and np.isclose(krw(0.80), 0.4)

    # krw rises and kro falls monotonically across the saturation range
    sw = np.linspace(0.15, 0.80, 14)
    assert np.all(np.diff(krw(sw)) >= 0) and np.all(np.diff(kro(sw)) <= 0)

    # Fractional flow is S-shaped from 0 to 1
    fw = fractional_flow(sw)
    print(f"  fw(Swirr) / fw(1-Sor)  = {fw[0]:.3f} / {fw[-1]:.3f}")
    assert np.isclose(fw[0], 0.0) and np.isclose(fw[-1], 1.0) and np.all(np.diff(fw) >= 0)

    # End-point mobility ratio
    m = endpoint_mobility_ratio()
    print(f"  end-point mobility ratio = {m:.2f}")
    assert np.isclose(m, (0.4 / 0.5) / (1.0 / 2.0))
    print("  PASS")
    return {"M": float(m), "fw_mid": float(fractional_flow(0.5))}


if __name__ == "__main__":
    test_all()
