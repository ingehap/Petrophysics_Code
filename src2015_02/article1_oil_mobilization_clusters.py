"""
Article 1: Onset of Oil Mobilization and Nonwetting-Phase Cluster-Size
           Distribution
Berg, Armstrong, Georgiadis, Ott, Schwing, Neiteler, Brussee, Makurat, Rucker,
Leu, Wolf, Khan, Enzmann, Kersten (2015)
Reference: Petrophysics Vol. 56, No. 1 (February 2015), pp. 15-22
DOI: none assigned (this issue predates SPWLA DOI assignment)

Best Papers of the 2014 SCA Symposium.  Fast X-ray microtomography images the
nonwetting (oil) phase during imbibition.  A macroscopic (cluster-based)
capillary number characterizes the capillary-vs-viscous flow regime, and the
oil-cluster sizes follow a power-law distribution analyzed with logarithmic
binning (where the measured exponent is the true exponent minus one).

Implements:

  - Macroscopic (cluster-based) capillary number (Eq. 1)
  - Microscopic capillary number  Ca = mu*v/sigma
  - Logarithmic bin edges for a cluster-size histogram (Eq. 2)
  - Power-law cluster-size distribution and a log-binned exponent fit

Note: this issue's PDF has a text layer; the macroscopic-Ca and log-binning
relations are transcribed from the body, while the typeset glyphs were dropped
and reconstructed in standard form (Armstrong et al., 2014; Newman, 2005).  SI
units; cluster sizes/voxels in consistent units.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- capillary number --------------

def macroscopic_capillary_number(mu_nw, velocity, sigma, cluster_length, pore_radius, phi):
    """Macroscopic (cluster-based) capillary number (Armstrong et al., 2014; Eq. 1)

        Ca_macro = mu_nw*v*l_cl / (sigma*rp*phi),

    with the averaged cluster length l_cl, the pore-throat radius rp, porosity
    phi, nonwetting viscosity mu_nw, velocity v and interfacial tension sigma.
    """
    return mu_nw * velocity * cluster_length / (sigma * pore_radius * phi)


def microscopic_capillary_number(mu, velocity, sigma):
    """Microscopic capillary number  Ca = mu*v/sigma."""
    return petrolib.relperm_wettability.capillary_number(mu=mu, v=velocity, sigma=sigma)


# ---------------------------------------------- cluster-size distribution --------------

def logarithmic_bins(x_min, x_max, n):
    """Logarithmic bin edges (Newman, 2005; Eq. 2)

        x_k = x_min * a^k,   a = (x_max/x_min)^(1/n),

    used to bin a heavy-tailed cluster-size distribution without small-count
    artifacts.  Returns n+1 edges.
    """
    a = (x_max / x_min) ** (1.0 / n)
    return x_min * a ** np.arange(n + 1)


def cluster_size_distribution(s, alpha, c=1.0):
    """Power-law cluster-size distribution  N(s) = c*s^(-alpha)."""
    return c * np.asarray(s, float) ** (-alpha)


def fit_power_law_exponent(sizes, counts, log_binned=True):
    """Fit the cluster-size power-law exponent from a log-log regression.

    When the counts come from a logarithmically binned distribution, the slope
    gives (alpha - 1), so the true exponent alpha is the fitted exponent + 1
    (Newman, 2005).  Returns the true exponent alpha.
    """
    x = np.log10(np.asarray(sizes, float))
    y = np.log10(np.asarray(counts, float))
    slope, _ = np.polyfit(x, y, 1)
    exponent = -slope
    return exponent + 1.0 if log_binned else exponent


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Oil Mobilization & Cluster-Size Distribution")
    print("=" * 60)

    # Macroscopic Ca rises with velocity and viscosity, falls with IFT
    ca = macroscopic_capillary_number(mu_nw=1e-3, velocity=1e-5, sigma=0.03,
                                      cluster_length=1e-3, pore_radius=5e-5, phi=0.2)
    print(f"  macroscopic Ca         = {ca:.2e}")
    assert ca > 0 and macroscopic_capillary_number(1e-3, 1e-4, 0.03, 1e-3, 5e-5, 0.2) > ca
    assert microscopic_capillary_number(1e-3, 1e-5, 0.06) < microscopic_capillary_number(1e-3, 1e-5, 0.03)

    # Logarithmic bins increase exponentially and span [x_min, x_max]
    edges = logarithmic_bins(1.0, 1e5, 40)
    assert edges.size == 41 and np.isclose(edges[0], 1.0) and np.isclose(edges[-1], 1e5)
    assert np.all(np.diff(edges) > 0)

    # Power-law fit recovers the true exponent from a log-binned distribution
    alpha_true = 2.1
    s = np.logspace(0, 5, 30)
    # synthesize log-binned counts: N(s) ~ s^-(alpha-1) per log bin
    counts = cluster_size_distribution(s, alpha_true - 1.0, c=1e6)
    alpha_fit = fit_power_law_exponent(s, counts, log_binned=True)
    print(f"  cluster exponent       = {alpha_fit:.3f}")
    assert np.isclose(alpha_fit, alpha_true)
    print("  PASS")
    return {"Ca_macro": float(ca), "alpha": float(alpha_fit)}


if __name__ == "__main__":
    test_all()
