"""
grader_digital_rock.py
Implementation of ideas from:
Grader et al., "Digital Rock Method for Relative Permeability in Valhall Chalk",
Petrophysics, Vol. 65, No. 2 (April 2024), pp. 149-157.

The paper presents a digital-rock workflow that derives drainage and imbibition
relative permeability curves for high-porosity / low-permeability chalk much
faster than the laboratory. We mimic the essentials with a Brooks-Corey /
Corey-type parametric model whose endpoints and exponents are derived from a
synthetic pore-network "digital rock" represented by a pore-size distribution.
"""
import numpy as np


def corey_relperm(Sw, Swi=0.2, Sor=0.15, krw_max=0.4, kro_max=1.0, nw=2.5, no=2.0):
    """Brooks-Corey two-phase relative permeabilities."""
    Sw = np.clip(Sw, Swi, 1 - Sor)
    Swn = (Sw - Swi) / (1 - Swi - Sor)
    krw = krw_max * Swn ** nw
    kro = kro_max * (1 - Swn) ** no
    return krw, kro


def digital_rock_endpoints(pore_radii, wettability="water-wet"):
    """Estimate endpoint saturations from a pore-size distribution.

    Smaller pores hold irreducible water; larger pores hold residual oil.
    Wettability shifts these endpoints, mimicking the paper's Valhall chalk study.
    """
    r = np.asarray(pore_radii, dtype=float)
    r_sorted = np.sort(r)
    n = len(r_sorted)
    Swi = (r_sorted[: n // 5] ** 3).sum() / (r_sorted ** 3).sum()
    Sor = (r_sorted[-n // 5 :] ** 3).sum() / (r_sorted ** 3).sum() * 0.4
    if wettability == "oil-wet":
        Swi, Sor = Sor * 0.8, Swi * 1.2
    return float(Swi), float(Sor)


def fractional_flow(Sw, mu_w=1.0, mu_o=2.0, **kw):
    krw, kro = corey_relperm(Sw, **kw)
    return krw / mu_w / (krw / mu_w + kro / mu_o + 1e-30)


def test_all():
    rng = np.random.default_rng(0)
    pores = rng.lognormal(mean=0.0, sigma=0.6, size=2000)
    Swi, Sor = digital_rock_endpoints(pores, "water-wet")
    assert 0 < Swi < 0.5 and 0 < Sor < 0.5
    Sw = np.linspace(0, 1, 50)
    krw, kro = corey_relperm(Sw, Swi=Swi, Sor=Sor)
    assert krw.min() >= 0 and kro.min() >= 0
    assert krw[-1] >= krw[0] and kro[0] >= kro[-1]
    fw = fractional_flow(Sw, Swi=Swi, Sor=Sor)
    assert 0 <= fw.min() and fw.max() <= 1
    print("grader_digital_rock OK  Swi=%.3f Sor=%.3f" % (Swi, Sor))


if __name__ == "__main__":
    test_all()
