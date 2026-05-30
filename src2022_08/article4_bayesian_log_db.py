"""
Article 4: A Fast and Transparent Bayesian Log Interpretation Method
Spalburg (2022)
DOI: 10.30632/PJV63N4-2022a4

Precomputes a database of synthetic formation realisations - each storing
property values and the calculated tool responses - and reduces log
interpretation to a database search returning realisations whose forward
responses match the borehole logs within tool error margins.  Posterior
probabilities follow Bayes' theorem (Eq. 1) with optional Gaussian
likelihood weighting (Appendix 1, Eq. A1.1).

Implements the Appendix-3 forward operators:

  - Volume-weighted GR                                        (Eq. A3.1)
  - Bulk density                                              (Eq. A3.2)
  - Photoelectric factor                                      (Eq. A3.3)
  - Neutron porosity with excavation correction (simplified)  (Eq. A3.4)
  - Wyllie compressional travel-time (critical-porosity)      (Eqs. A3.5-A3.13)
  - Waxman-Smits / Dual-Water merged resistivity (Juhasz B)   (Eqs. A3.14-A3.16)
"""

import numpy as np


# -------------------------------------------- end-member catalogue --------

# (rho, GR, PEF, neutron, dt_us_per_ft)
MINERAL = {
    "qtz":  (2.65, 10.0, 1.81, 0.000, 55.5),
    "calc": (2.71, 12.0, 5.08, 0.000, 47.5),
    "dolo": (2.87,  8.0, 3.14, 0.020, 43.5),
    "clay": (2.60, 120.0, 3.42, 0.350, 90.0),
}
# (rho, neutron, dt)
FLUID = {
    "water": (1.00, 1.000, 189.0),
    "oil":   (0.80, 0.900, 230.0),
    "gas":   (0.20, 0.050, 600.0),
}


def volume_weights_legal(volumes):
    return np.isclose(sum(volumes.values()), 1.0, atol=1e-6)


# -------------------------------------------- forward operators ----------

def gr_forward(mineral_fractions, phi, vfluid=None):
    """Eq. A3.1 - volume-weighted GR of the solid skeleton."""
    matrix_total = 1.0 - phi
    return sum(MINERAL[k][1] * v * matrix_total
               for k, v in mineral_fractions.items())


def rhob_forward(mineral_fractions, phi, sw, fluid="water"):
    """Eq. A3.2 - bulk density.  Single-fluid pore (water/oil/gas mixed by sw)."""
    rho_min = sum(MINERAL[k][0] * v for k, v in mineral_fractions.items())
    rho_fluid = FLUID["water"][0] * sw + FLUID[fluid][0] * (1.0 - sw)
    return (1.0 - phi) * rho_min + phi * rho_fluid


def pef_forward(mineral_fractions, phi):
    """Eq. A3.3 - photoelectric factor (volume-weighted on solids)."""
    return sum(MINERAL[k][2] * v * (1.0 - phi)
               for k, v in mineral_fractions.items())


def nphi_forward(mineral_fractions, phi, sw, fluid="water"):
    """Eq. A3.4 - neutron porosity with a crude excavation correction.

        nphi = (1-phi)*sum(v_i * N_i) + phi*(sw*N_w + (1-sw)*N_fluid)
               - 0.04 * (1-sw) * phi               (excavation term)
    """
    matrix = sum(MINERAL[k][3] * v for k, v in mineral_fractions.items())
    fluid_n = sw * FLUID["water"][1] + (1.0 - sw) * FLUID[fluid][1]
    return (1.0 - phi) * matrix + phi * fluid_n - 0.04 * (1.0 - sw) * phi


def dt_forward(mineral_fractions, phi, sw, fluid="water"):
    """Eqs. A3.5-A3.13 - Wyllie / critical-porosity compressional travel time.

    Falls back to Wyllie time-average for tractability:
        dt = (1 - phi) * dt_matrix + phi * dt_fluid
    """
    dt_matrix = sum(MINERAL[k][4] * v for k, v in mineral_fractions.items())
    dt_fluid = sw * FLUID["water"][2] + (1.0 - sw) * FLUID[fluid][2]
    return (1.0 - phi) * dt_matrix + phi * dt_fluid


def rt_forward(phi, sw, vsh, rw=0.05, m=2.0, n=2.0, a=1.0,
               B_juhasz_per_S_m=0.045, Qv=0.10):
    """Eqs. A3.14-A3.16 - merged Waxman-Smits / Dual-Water with Juhasz B."""
    # Waxman-Smits-like
    sigma_w = 1.0 / rw
    sigma_o = (phi ** m) * (sw ** n) * (sigma_w + B_juhasz_per_S_m * Qv * vsh / sw)
    return 1.0 / max(sigma_o, 1e-12)


# -------------------------------------------- database builder ----------

def build_database(n=20_000, fluid="oil", seed=0):
    """Random multiphase grid.  Returns (X_props, Y_response) tuples."""
    rng = np.random.default_rng(seed)
    props = []
    resp = []
    for _ in range(n):
        # Mineral volumes - Dirichlet (1, 1, 0.1, 0.5) so qtz/calc dominate
        v = rng.dirichlet([3.0, 1.0, 0.5, 1.0])
        mins = dict(zip(["qtz", "calc", "dolo", "clay"], v))
        phi = float(rng.uniform(0.03, 0.30))
        sw = float(rng.uniform(0.10, 1.00))
        vsh = float(mins["clay"])
        props.append([phi, sw, vsh] + list(v))
        resp.append([
            gr_forward(mins, phi),
            rhob_forward(mins, phi, sw, fluid),
            pef_forward(mins, phi),
            nphi_forward(mins, phi, sw, fluid),
            dt_forward(mins, phi, sw, fluid),
            np.log10(rt_forward(phi, sw, vsh)),
        ])
    return np.array(props), np.array(resp)


# -------------------------------------------- Bayesian search ----------

def bayesian_match(props, resp, obs, sigma):
    """Eq. 1 / A1.1 Gaussian-likelihood posterior weights.

    Returns the property-vector posterior mean and uncertainty.
    """
    chi2 = (((resp - obs) / sigma) ** 2).sum(1)
    log_lik = -0.5 * chi2
    log_lik -= log_lik.max()
    w = np.exp(log_lik)
    w /= w.sum()
    mu = (w[:, None] * props).sum(0)
    var = (w[:, None] * (props - mu) ** 2).sum(0)
    return mu, np.sqrt(var)


# -------------------------------------------- tests --------------------

def test_all():
    print("=" * 60)
    print("Article 4: Database-Driven Bayesian Log Interpretation")
    print("=" * 60)

    props, resp = build_database(n=20_000, fluid="oil")
    print(f"  Database size: {len(props)} realisations x {props.shape[1]} props "
          f"and {resp.shape[1]} tool responses")

    # Truth: oil-bearing carbonate analogue
    true_phi, true_sw = 0.18, 0.30
    mins_true = dict(qtz=0.10, calc=0.75, dolo=0.10, clay=0.05)
    obs = np.array([
        gr_forward(mins_true, true_phi),
        rhob_forward(mins_true, true_phi, true_sw, "oil"),
        pef_forward(mins_true, true_phi),
        nphi_forward(mins_true, true_phi, true_sw, "oil"),
        dt_forward(mins_true, true_phi, true_sw, "oil"),
        np.log10(rt_forward(true_phi, true_sw, mins_true["clay"])),
    ])
    obs *= (1.0 + 0.01 * np.random.default_rng(0).standard_normal(len(obs)))
    sigma = np.array([3.0, 0.02, 0.10, 0.03, 2.0, 0.05])

    mu, std = bayesian_match(props, resp, obs, sigma)
    print(f"  Posterior phi  = {mu[0]:.3f}  +/- {std[0]:.3f}   (true {true_phi})")
    print(f"  Posterior Sw   = {mu[1]:.3f}  +/- {std[1]:.3f}   (true {true_sw})")
    print(f"  Posterior Vsh  = {mu[2]:.3f}  +/- {std[2]:.3f}   (true {mins_true['clay']})")
    print(f"  Posterior calc = {mu[4]:.3f}  +/- {std[4]:.3f}   (true {mins_true['calc']})")

    assert abs(mu[0] - true_phi) < 0.04, "Posterior phi must be within 0.04"
    assert abs(mu[1] - true_sw) < 0.20, "Posterior Sw must be within 0.20"
    print("  PASS")
    return {"phi_mean": float(mu[0]), "sw_mean": float(mu[1]),
            "phi_std": float(std[0]), "sw_std": float(std[1])}


if __name__ == "__main__":
    test_all()
