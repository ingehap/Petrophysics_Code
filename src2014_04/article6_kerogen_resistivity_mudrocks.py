"""
Article 6: Quantifying the Effect of Kerogen on Resistivity Measurements in
           Organic-Rich Mudrocks
Nikhil Kethireddy, Huangye Chen, Zoya Heidari (2014)
Reference: Petrophysics Vol. 55, No. 2 (April 2014), pp. 136-146
DOI: none assigned (this issue predates SPWLA DOI assignment)

A regular submission.  Electrically conductive kerogen lowers the measured
resistivity of organic-rich mudrocks, so conventional saturation models
overestimate water saturation.  A pore-scale model solves Laplace's conductivity
equation on a rock with conductive kerogen to quantify the effect and correct it.

Implements:

  - Water/gas saturation and kerogen volumetric concentration (Eqs. 1-3)
  - Kerogen porosity and total porosity (Eqs. 4, 6)
  - Plagioclase-quartz mineral constraint (Eq. 8, linear)
  - Kerogen volume from TOC (Eq. 9, linear)
  - Effective resistivity of gas-bearing kerogen  R_effK = R_K*exp(c*phi_K) (Eq. 10)
  - Laplace conductivity solver  div(sigma*grad V) = 0  (Eq. 7) for the effective
    resistivity of a heterogeneous (kerogen-bearing) medium
  - Kerogen-conductivity threshold (~1000 Ohm*m) above which rock R is unaffected
  - Archie water saturation and its overestimation when kerogen is ignored

Note: this issue's PDF dropped the display equations in extraction; the
saturation/porosity definitions (Eqs. 1-6) and the Laplace equation (Eq. 7) are
reconstructed from the surviving text and nomenclature, and Eq. 10 is the stated
exponential form.  The TOC->kerogen and plagioclase-quartz slopes (Eqs. 8, 9)
lived in figures and are left as inputs.  Resistivities in Ohm*m.
"""

import numpy as np


# ---------------------------------------------- saturations & volumes --------------

def water_saturation(v_water, v_gas):
    """Water saturation of the fluid system (Eq. 1)  Sw = Vw/(Vw + Vg)."""
    return v_water / (v_water + v_gas)


def gas_saturation(v_water, v_gas):
    """Gas saturation (Eq. 2)  Sg = Vg/(Vw + Vg) = 1 - Sw."""
    return v_gas / (v_water + v_gas)


def kerogen_concentration(v_kerogen, v_total):
    """Kerogen volumetric concentration (Eq. 3)  C_k = V_k/V_t."""
    return v_kerogen / v_total


def kerogen_porosity(v_gas_kerogen, v_kerogen):
    """Kerogen porosity (Eq. 4)  phi_k = V_gk/V_k."""
    return v_gas_kerogen / v_kerogen


def total_porosity(matrix_pore_volume, kerogen_pore_volume, v_total):
    """Total porosity of the rock-fluid system (Eq. 6)

        phi_t = (matrix pore volume + kerogen pore volume)/V_t.
    """
    return (matrix_pore_volume + kerogen_pore_volume) / v_total


def plagioclase_from_quartz(c_quartz, slope, intercept=0.0):
    """Plagioclase volumetric concentration from quartz (Eq. 8, linear)

        C_plagioclase = slope*C_quartz + intercept,

    the mineral correlation (Fig. 4) used as a constraint to reduce the
    non-uniqueness of the multi-mineral log inversion (coefficients from core).
    """
    return slope * np.asarray(c_quartz, float) + intercept


def kerogen_from_toc(toc, slope, intercept=0.0):
    """Kerogen volumetric concentration from TOC (Eq. 9, linear)

        C_k = slope*TOC + intercept,

    with TOC in wt% and C_k in vol% (coefficients from the paper's Fig. 5).
    """
    return slope * np.asarray(toc, float) + intercept


# ---------------------------------------------- kerogen resistivity --------------

def effective_kerogen_resistivity(r_kerogen, phi_kerogen, c):
    """Effective resistivity of the gas-bearing kerogen system (Eq. 10)

        R_effK = R_K*exp(c*phi_K),

    rising with the gas-filled kerogen porosity phi_K (an insulating gas phase
    raises the kerogen-domain resistivity); c is fitted (Fig. 12).
    """
    return r_kerogen * np.exp(c * np.asarray(phi_kerogen, float))


# Above this kerogen resistivity the kerogen behaves as an insulator and the
# bulk rock resistivity is essentially unaffected (synthetic Case 1).
KEROGEN_CONDUCTIVE_THRESHOLD_OHMM = 1000.0


def kerogen_affects_resistivity(r_kerogen):
    """Whether conductive kerogen materially lowers the bulk rock resistivity

        affects rock R only when R_kerogen <~ 1000 Ohm*m.

    Per synthetic Case 1, for kerogen resistivity above ~1000 Ohm*m the rock
    resistivity is unaffected (kerogen acts as an insulator like the matrix);
    below it the rock resistivity falls with increasing kerogen conductivity and
    Archie overestimates water saturation.
    """
    return bool(float(r_kerogen) < KEROGEN_CONDUCTIVE_THRESHOLD_OHMM)


# ---------------------------------------------- Laplace conductivity solver --------------

def effective_conductivity_2d(sigma_map, n_iter=5000, tol=1e-7):
    """Effective conductivity of a 2D heterogeneous medium by solving Laplace's
    conductivity equation (Eq. 7)

        div(sigma*grad V) = 0,

    with a unit potential drop applied top-to-bottom (V=1 at row 0, V=0 at the
    last row) and no-flux side walls.  Uses a harmonic-mean face conductivity
    finite-difference (Jacobi) iteration; returns the effective conductivity
    (current per unit applied field), so resistivity = 1/sigma_eff.
    """
    sigma = np.asarray(sigma_map, float)
    ny, nx = sigma.shape
    v = np.linspace(1.0, 0.0, ny)[:, None] * np.ones((1, nx))

    def face(a, b):  # harmonic mean of adjacent cell conductivities
        return 2.0 * a * b / (a + b)

    sn = face(sigma[:-1, :], sigma[1:, :])   # north-south faces (ny-1, nx)
    se = face(sigma[:, :-1], sigma[:, 1:])   # east-west faces  (ny, nx-1)
    for _ in range(n_iter):
        v_old = v.copy()
        num = np.zeros_like(v)
        den = np.zeros_like(v)
        # vertical neighbours
        num[1:, :] += sn * v[:-1, :]; den[1:, :] += sn
        num[:-1, :] += sn * v[1:, :]; den[:-1, :] += sn
        # horizontal neighbours
        num[:, 1:] += se * v[:, :-1]; den[:, 1:] += se
        num[:, :-1] += se * v[:, 1:]; den[:, :-1] += se
        v_new = num / den
        v_new[0, :] = 1.0      # top electrode
        v_new[-1, :] = 0.0     # bottom electrode
        v = v_new
        if np.max(np.abs(v - v_old)) < tol:
            break
    # current across the top row from the applied unit drop
    current = np.sum(sn[0, :] * (v[0, :] - v[1, :]))
    sigma_eff = current * (ny - 1) / nx     # normalize by geometry (unit drop)
    return float(sigma_eff)


# ---------------------------------------------- Archie --------------

def archie_sw(rt, rw, phi, a=1.0, m=2.0, n=2.0):
    """Archie water saturation  Sw = (a*Rw/(phi^m*Rt))^(1/n)."""
    return (a * rw / (phi ** m * rt)) ** (1.0 / n)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Kerogen Effect on Resistivity")
    print("=" * 60)

    # Saturations and volumes
    assert np.isclose(water_saturation(0.6, 0.4), 0.6)
    assert np.isclose(gas_saturation(0.6, 0.4), 0.4)
    assert np.isclose(kerogen_concentration(0.20, 1.0), 0.20)
    ck = kerogen_from_toc(5.0, slope=2.0, intercept=1.0)
    print(f"  kerogen concentration (TOC=5%) = {ck:.1f} vol%")
    assert np.isclose(ck, 11.0)
    # plagioclase-quartz mineral constraint (Eq. 8)
    assert np.isclose(plagioclase_from_quartz(30.0, slope=0.5, intercept=2.0), 17.0)

    # Kerogen-conductivity threshold: conductive kerogen lowers rock R only
    # below ~1000 Ohm*m
    assert kerogen_affects_resistivity(100.0)
    assert not kerogen_affects_resistivity(1e7)
    print(f"  kerogen affects R: 100 Ohm*m={kerogen_affects_resistivity(100.0)}, "
          f"1e7 Ohm*m={kerogen_affects_resistivity(1e7)}")

    # Effective kerogen resistivity rises with gas-filled kerogen porosity
    r1 = effective_kerogen_resistivity(100.0, 0.1, c=1.5)
    r2 = effective_kerogen_resistivity(100.0, 0.5, c=1.5)
    print(f"  R_effK(phi=0.1)={r1:.1f}  R_effK(phi=0.5)={r2:.1f} Ohm*m")
    assert r2 > r1 > 100.0

    # Laplace solver: a homogeneous medium returns its own conductivity
    homo = np.full((40, 40), 0.5)
    sigma_eff = effective_conductivity_2d(homo, n_iter=2000)
    print(f"  homogeneous sigma_eff = {sigma_eff:.4f} S/m (true 0.5)")
    assert np.isclose(sigma_eff, 0.5, rtol=0.02)

    # Conductive kerogen raises the rock conductivity (lowers resistivity)
    rock = np.full((40, 40), 0.1)            # matrix conductivity (10 Ohm*m)
    rock[:, 15:25] = 1.0                      # connected conductive kerogen band
    sigma_kero = effective_conductivity_2d(rock, n_iter=3000)
    print(f"  with conductive kerogen: sigma_eff = {sigma_kero:.4f} S/m")
    assert sigma_kero > 0.1                    # more conductive than matrix alone

    # Ignoring the conductive kerogen (lower Rt) overestimates Sw
    rt_true = 1.0 / 0.1                         # matrix-only resistivity
    rt_meas = 1.0 / sigma_kero                  # measured (kerogen-lowered)
    sw_true = archie_sw(rt_true, 0.02, 0.10)
    sw_apparent = archie_sw(rt_meas, 0.02, 0.10)
    overestimate = (sw_apparent - sw_true) / sw_true
    print(f"  Sw(true)={sw_true:.3f}  Sw(ignoring kerogen)={sw_apparent:.3f}"
          f"  (+{overestimate*100:.0f}%)")
    assert sw_apparent > sw_true               # overestimation
    print("  PASS")
    return {"C_k": float(ck), "R_effK": float(r2), "sigma_kerogen": float(sigma_kero),
            "Sw_overestimate": float(sw_apparent - sw_true),
            "kerogen_conductive": kerogen_affects_resistivity(100.0)}


if __name__ == "__main__":
    test_all()
