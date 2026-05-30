"""
Article 10: How the Invasion Zone Can Contribute to the Estimation of
            Petrophysical Properties From Log Inversion at Well Scale?
Vandamme, Caroli, Gratton (2019)
DOI: 10.30632/PJV60N2-2019a8

Mud-filtrate invasion creates a radial saturation profile: a flushed (invaded)
zone with filtrate-altered saturation near the borehole and the virgin zone
beyond.  Multi-depth-of-investigation resistivity logs see different mixtures of
the two zones; jointly inverting them recovers both the virgin water saturation
and the invasion radius - the invasion zone adds information rather than just
noise.

Implements:

  - Archie saturation in the flushed (Rxo, Sxo) and virgin (Rt, Sw) zones
  - Radial (step) two-zone resistivity model vs depth of investigation
  - Least-squares inversion of multi-DOI apparent resistivity for (Rt, Rxo, ri)

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard invaded-zone Archie / radial-inversion method
the paper analyzes.
"""

import numpy as np


# ---------------------------------------------- Archie ------------------

def archie_sw(Rt, Rw, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation  Sw = (a*Rw/(phi^m*Rt))^(1/n)."""
    return (a * Rw / (np.asarray(phi, float) ** m * Rt)) ** (1.0 / n)


def two_zone_apparent_resistivity(Rxo, Rt, ri, doi):
    """Pseudo-radial apparent resistivity seen by a tool of given DOI.

    Conductivity mix weighted by an invaded-zone geometric factor that falls off
    as (ri/doi)^2 (a radial tool weights the far field heavily), so a shallow
    tool reads ~Rxo and a deep tool reads ~Rt:
        1/Ra = G/Rxo + (1-G)/Rt,  G = min((ri/doi)^2, 1)
    """
    G = np.clip((ri / np.asarray(doi, float)) ** 2, 0.0, 1.0)
    return 1.0 / (G / Rxo + (1.0 - G) / Rt)


# ---------------------------------------------- inversion ---------------

def invert_invasion(doi, Ra_obs, grid_rt, grid_rxo, grid_ri):
    """Grid-search (Rt, Rxo, ri) minimizing misfit to multi-DOI resistivities."""
    Ra_obs = np.asarray(Ra_obs, float)
    best = None; best_err = np.inf
    for Rt in grid_rt:
        for Rxo in grid_rxo:
            for ri in grid_ri:
                pred = two_zone_apparent_resistivity(Rxo, Rt, ri, doi)
                err = np.sum((np.log(pred) - np.log(Ra_obs)) ** 2)
                if err < best_err:
                    best_err, best = err, (Rt, Rxo, ri)
    return best


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 10: Invasion Zone in Log Inversion")
    print("=" * 60)

    # Archie: oil in the virgin zone (high Rt) -> low Sw; flushed zone has Sxo
    sw = archie_sw(40.0, 0.05, 0.20)
    sxo = archie_sw(8.0, 0.30, 0.20)              # Rmf > Rw, residual oil
    print(f"  Sw virgin / Sxo flushed = {sw:.2f} / {sxo:.2f}")
    assert sw < sxo                                # invasion raises near-well Sw

    # Two-zone response: a shallow tool reads near Rxo, a deep tool near Rt
    Rxo, Rt, ri = 5.0, 40.0, 0.4                   # ri = 0.4 m invasion radius
    doi = np.array([0.1, 0.3, 0.6, 1.5])           # m
    Ra = two_zone_apparent_resistivity(Rxo, Rt, ri, doi)
    print(f"  Ra vs DOI              = {np.array2string(Ra, precision=1)}")
    assert Ra[0] < Ra[-1]                           # shallow ~ Rxo, deep ~ Rt
    assert abs(Ra[0] - Rxo) < 1.0 and Ra[-1] > 20.0

    # Inversion recovers the planted (Rt, Rxo, ri) from the multi-DOI logs
    Rt_g = np.array([20, 30, 40, 50.0])
    Rxo_g = np.array([3, 5, 7.0])
    ri_g = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    rt_hat, rxo_hat, ri_hat = invert_invasion(doi, Ra, Rt_g, Rxo_g, ri_g)
    print(f"  recovered Rt/Rxo/ri    = {rt_hat} / {rxo_hat} / {ri_hat}")
    assert rt_hat == Rt and rxo_hat == Rxo and abs(ri_hat - ri) < 1e-9
    print("  PASS")
    return {"sw_virgin": float(sw), "Rt_hat": rt_hat, "ri_hat": ri_hat}


if __name__ == "__main__":
    test_all()
