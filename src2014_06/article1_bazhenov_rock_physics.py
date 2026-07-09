"""
Article 1: A Case Study about Formation Evaluation and Rock Physics Modeling of
           the Bazhenov Shale
Pavel Kulyapin and Tatiana F. Sokolova (2014)
Reference: Petrophysics Vol. 55, No. 3 (June 2014), pp. 211-218
DOI: none assigned (this issue predates SPWLA DOI assignment)

A multimineral statistical inversion evaluates the organic-rich Bazhenov shale,
and a Kuster-Toksoz inclusion rock-physics model relates the mineral/porosity
volumes to elastic moduli and velocities.  Effective and secondary (vug/
fracture) porosity are separated, and a shear-wave splitting index quantifies
anisotropy.

Implements:

  - Multimineral statistical inversion with unit-sum closure (Eq. 1)
  - Effective porosity  phi_ef = phi_total/3  (Eq. 2)
  - Block resistivity  Rblock = 10^(1.201 + 0.0427*GR - 0.0002*GR^2)  (Eq. 6)
  - Matrix-block and secondary porosity (Eqs. 3-5)
  - Shear-wave splitting/anisotropy index  Sp = (Vfast - Vslow)/Vfast  (Eq. 7)
  - Kuster-Toksoz spherical-inclusion effective moduli and acoustic impedance

Note: this issue's PDF dropped most display equations in extraction; the block
resistivity (Eq. 6) and effective porosity (Eq. 2) survived, while the inversion
(Eq. 1), secondary-porosity (Eqs. 3-5) and splitting (Eq. 7) forms are
reconstructed in standard form (Kuster & Toksoz, 1974; Wendelstein & Rezvanov,
1978).  Resistivities in Ohm*m, moduli in GPa, density in g/cm^3.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- multimineral inversion --------------

def multimineral_inversion(log_readings, response_matrix, closure_weight=100.0):
    """Multimineral statistical inversion (Eq. 1)

        f_i = sum_j e_ij*V_j,   subject to  sum_j V_j = 1,

    solved by weighted least squares with a heavily weighted unit-sum closure
    row.  ``response_matrix`` has shape (n_logs, n_minerals); returns the
    mineral volume fractions V_j.
    """
    return petrolib.porosity_lithology.multimineral_solve(
        log_readings, response_matrix, closure_weight=closure_weight, method="lstsq")


# ---------------------------------------------- porosity --------------

def effective_porosity(phi_total):
    """Effective porosity from total porosity (Eq. 2, Salym empirical)

        phi_ef = phi_total/3.
    """
    return np.asarray(phi_total, float) / 3.0


def block_resistivity(gr):
    """Matrix-block resistivity from gamma ray (Eq. 6)

        Rblock = 10^(1.201 + 0.0427*GR - 0.0002*GR^2),

    with GR in microR/h.
    """
    gr = np.asarray(gr, float)
    return 10.0 ** (1.201 + 0.0427 * gr - 0.0002 * gr ** 2)


def block_porosity(r_lld, rbw=0.2):
    """Archie matrix-block porosity for a fully water-saturated block (m=2)

        phi_block = sqrt(Rbw/RLLD),

    with bound-water resistivity Rbw = 0.2 Ohm*m.
    """
    return np.sqrt(rbw / np.asarray(r_lld, float))


def secondary_porosity(phi_ef, phi_block):
    """Secondary (vug/fracture) porosity (Eqs. 3-5)

        phi_sec = phi_ef - phi_block,

    the excess of effective over matrix-block porosity (positive where vugs or
    fractures contribute).
    """
    return phi_ef - phi_block


# ---------------------------------------------- anisotropy --------------

def shear_splitting_index(v_fast, v_slow):
    """Shear-wave splitting / anisotropy index (Eq. 7)

        Sp = (Vfast - Vslow)/Vfast.
    """
    return petrolib.acoustic_geomech.shear_wave_splitting(v_fast, v_slow)


# ---------------------------------------------- Kuster-Toksoz --------------

def kuster_toksoz_spheres(k_m, mu_m, k_i, mu_i, concentration):
    """Kuster-Toksoz effective moduli for spherical inclusions (Kuster & Toksoz,
    1974)

        (K* - K_m)/(K* + 4/3 mu_m) = c*(K_i - K_m)/(K_i + 4/3 mu_m),
        (mu* - mu_m)/(mu* + zeta_m) = c*(mu_i - mu_m)/(mu_i + zeta_m),
        zeta_m = mu_m/6*(9 K_m + 8 mu_m)/(K_m + 2 mu_m).

    Returns (K*, mu*) in the units of the inputs.
    """
    beta = 4.0 / 3.0 * mu_m
    tk = concentration * (k_i - k_m) / (k_i + beta)
    k_star = (k_m + tk * beta) / (1.0 - tk)
    zeta = mu_m / 6.0 * (9.0 * k_m + 8.0 * mu_m) / (k_m + 2.0 * mu_m)
    tm = concentration * (mu_i - mu_m) / (mu_i + zeta)
    mu_star = (mu_m + tm * zeta) / (1.0 - tm)
    return k_star, mu_star


def p_wave_velocity(k, mu, rho):
    """P-wave velocity  Vp = sqrt((K + 4/3 mu)/rho), K, mu in GPa, rho in g/cm^3,
    returning m/s."""
    return np.sqrt((k + 4.0 / 3.0 * mu) * 1e9 / (rho * 1e3))


def acoustic_impedance(rho, vp):
    """Acoustic impedance  AI = rho*Vp."""
    return petrolib.acoustic_geomech.acoustic_impedance(rho, vp)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Bazhenov Shale Rock Physics")
    print("=" * 60)

    # Multimineral inversion recovers a known composition obeying closure
    # minerals: [quartz, clay, kerogen]; logs: [GR-proxy, density, neutron]
    e = np.array([[20.0, 150.0, 2000.0],   # GR endpoints
                  [2.65, 2.70, 1.10],       # density endpoints
                  [0.02, 0.35, 0.65]])      # neutron endpoints
    v_true = np.array([0.55, 0.30, 0.15])
    logs = e @ v_true
    v = multimineral_inversion(logs, e)
    print(f"  inverted volumes = {np.round(v, 3)}  sum={v.sum():.3f}")
    assert np.allclose(v, v_true, atol=1e-3) and np.isclose(v.sum(), 1.0)

    # Effective porosity is a third of the total
    assert np.isclose(effective_porosity(0.30), 0.10)

    # Block resistivity rises then is read against deep resistivity for secondary
    rb = block_resistivity(40.0)
    pb = block_porosity(50.0)
    psec = secondary_porosity(effective_porosity(0.30), pb)
    print(f"  Rblock={rb:.2f}  phi_block={pb:.3f}  phi_sec={psec:.3f}")
    assert rb > 0 and pb > 0

    # Shear splitting index in [0, 1]
    sp = shear_splitting_index(2500.0, 2300.0)
    assert np.isclose(sp, 0.08)

    # Kuster-Toksoz: adding compliant pores lowers the moduli and velocity
    k0, mu0, rho0 = 37.0, 44.0, 2.65         # quartz-like matrix
    k_eff, mu_eff = kuster_toksoz_spheres(k0, mu0, 2.2, 0.0, concentration=0.1)
    print(f"  K*={k_eff:.2f}  mu*={mu_eff:.2f} GPa")
    assert k_eff < k0 and mu_eff < mu0
    vp = p_wave_velocity(k_eff, mu_eff, rho0)
    ai = acoustic_impedance(rho0, vp)
    print(f"  Vp={vp:.0f} m/s  AI={ai:.0f}")
    assert 2000 < vp < 7000 and ai > 0
    print("  PASS")
    return {"phi_sec": float(psec), "K_eff": float(k_eff), "Vp": float(vp)}


if __name__ == "__main__":
    test_all()
