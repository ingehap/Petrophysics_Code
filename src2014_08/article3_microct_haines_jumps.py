"""
Article 3: Multiphase Flow in Porous Rock Imaged Under Dynamic Flow Conditions
           with Fast X-Ray Computed Microtomography
S. Berg, R. Armstrong, H. Ott, A. Georgiadis, S. A. Klapp, A. Schwing,
R. Neiteler, N. Brussee, A. Makurat, L. Leu, F. Enzmann, J.-O. Schwarz, M. Wolf,
F. Khan, M. Kersten, S. Irvine, M. Stampanoni (2014)
Reference: Petrophysics Vol. 55, No. 4 (August 2014), pp. 304-312
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best of the 2013 SCA Symposium.  Fast synchrotron X-ray microtomography images
two-phase displacement during drainage.  The pore-scale displacement proceeds by
abrupt Haines jumps; their pressure-volume work, the split between reversible
interfacial energy and irreversible dissipation, and the power-law distribution
of jump (event) sizes are quantified.

Implements:

  - Capillary number  Ncap = v*mu/sigma
  - Haines-jump pressure-volume work  W = integral(p dV)  (Eq. 1)
  - Pore-scale energy balance: interfacial-energy fraction sigma*dA/(p*dV) and
    the dissipated remainder (Eq. 2)
  - Invasion-percolation event-size power law  N ~ (dV/Vpore)^(-n)  (Eq. 3)
  - Imaging-derived porosity and saturation from voxel counts

Note: this issue's PDF has a text layer; Eq. 1 survived, while the Eq. 2 energy
balance and the Eq. 3 power-law bodies were dropped in extraction and
reconstructed from the surrounding text (measured exponent n = 1.0; interfacial
energy 36% of p*dV, ~64% dissipated).  SI units; volumes in m^3.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# np.trapz was renamed np.trapezoid in NumPy 2.0; support both.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------- capillary number --------------

def capillary_number(velocity, viscosity, ift):
    """Capillary number  Ncap = v*mu/sigma.

    Experimental values ranged 4e-9 to 4e-8; the critical Ncap for desaturation
    in sintered glass was about 3e-8.
    """
    return petrolib.relperm_wettability.capillary_number(mu=viscosity, v=velocity, sigma=ift)


# ---------------------------------------------- Haines-jump work --------------

def haines_jump_work(pressure, volume):
    """Pressure-volume work of a displacement event (Eq. 1)

        W = integral(p dV),

    the area under the pressure-volume curve over the subision/rision cycle.
    """
    return float(_trapezoid(np.asarray(pressure, float), np.asarray(volume, float)))


def energy_balance(pressure_work, ift, delta_area):
    """Pore-scale energy balance (Eq. 2)

        p*dV = sigma*dA + dissipation,

    returning the reversible interfacial-energy fraction sigma*dA/(p*dV) and the
    irreversibly dissipated fraction (measured ~36% interfacial, ~64% dissipated).
    """
    interfacial = ift * delta_area
    frac_interfacial = interfacial / pressure_work
    return {"interfacial_energy": float(interfacial),
            "fraction_interfacial": float(frac_interfacial),
            "fraction_dissipated": float(1.0 - frac_interfacial)}


# ---------------------------------------------- event-size power law --------------

def event_size_distribution(event_volumes, pore_volume, exponent, c=1.0):
    """Invasion-percolation event-size frequency (Eq. 3)

        N(dV/Vpore) = c*(dV/Vpore)^(-n),

    the heavy-tailed distribution of Haines-jump sizes (measured n = 1.0).
    """
    x = np.asarray(event_volumes, float) / pore_volume
    return c * x ** (-exponent)


def fit_event_exponent(event_volumes, pore_volume, counts):
    """Fit the event-size power-law exponent from a log-log regression

        log N = log c - n*log(dV/Vpore)  ->  returns n.
    """
    x = np.log10(np.asarray(event_volumes, float) / pore_volume)
    y = np.log10(np.asarray(counts, float))
    slope, _ = np.polyfit(x, y, 1)
    return -slope


# ---------------------------------------------- imaging diagnostics --------------

def imaged_porosity(pore_voxels, total_voxels):
    """Porosity from segmented voxel counts  phi = pore_voxels/total_voxels."""
    return petrolib.porosity_lithology.porosity_from_voxel_count(pore_voxels, total_voxels)


def imaged_saturation(phase_voxels, pore_voxels):
    """Phase saturation from segmented voxels  S = phase_voxels/pore_voxels."""
    return phase_voxels / pore_voxels


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Micro-CT Haines Jumps & Energy Balance")
    print("=" * 60)

    # Capillary number matches the reported experimental scale
    ncap = capillary_number(1.54e-6, 1e-3, 0.035)
    print(f"  Ncap = {ncap:.2e}")
    assert 1e-9 < ncap < 1e-7

    # Haines-jump work: constant pressure over a volume change gives p*dV
    v = np.linspace(0, 5.9e-9, 50)
    p = np.full_like(v, 1e4)
    w = haines_jump_work(p, v)
    print(f"  W = {w:.3e} J")
    assert np.isclose(w, 1e4 * 5.9e-9, rtol=1e-3)

    # Energy balance reproduces the ~36% interfacial / ~64% dissipated split
    eb = energy_balance(pressure_work=0.035 * 5.7e-7 / 0.36, ift=0.035, delta_area=5.7e-7)
    print(f"  interfacial fraction = {eb['fraction_interfacial']:.2f}")
    assert np.isclose(eb["fraction_interfacial"], 0.36, atol=0.01)
    assert np.isclose(eb["fraction_dissipated"], 0.64, atol=0.01)

    # Event-size power law: recover the exponent n = 1.0 from synthetic counts
    ev = np.logspace(-4, -1, 25)
    counts = event_size_distribution(ev, pore_volume=1.0, exponent=1.0, c=1e3)
    n_fit = fit_event_exponent(ev, 1.0, counts)
    print(f"  event exponent n = {n_fit:.3f}")
    assert np.isclose(n_fit, 1.0)

    # Imaging diagnostics
    assert np.isclose(imaged_porosity(2380, 10000), 0.238)
    assert np.isclose(imaged_saturation(595, 2380), 0.25)
    print("  PASS")
    return {"Ncap": float(ncap), "W": float(w), "n": float(n_fit)}


if __name__ == "__main__":
    test_all()
