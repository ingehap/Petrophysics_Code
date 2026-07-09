"""
Article 2: Geomechanics of Orthorhombic Media
Far, Quirein, Mekic (2016)
Reference: Petrophysics Vol. 57, No. 6 (December 2016), pp. 588-596
DOI: none assigned (this issue predates SPWLA DOI assignment)

A geomechanical model for orthorhombic media (a VTI background plus two
orthogonal vertical fracture sets).  Hooke's law links stress and strain through
the stiffness tensor; the compliance is its inverse; and a simplified horizontal-
stress model (with pore pressure) predicts the two horizontal stresses from the
anisotropic Young's moduli and Poisson's ratios.  It reduces to the standard VTI
model when the two horizontal directions are equal.  Shear-wave splitting
measures the anisotropy.

Implements:

  - Hooke's law  sigma = C*epsilon  and compliance  S = inv(C)
  - Direction-dependent Young's moduli and Poisson's ratios from the compliance
    matrix (Eqs. 9-12)
  - Compliance-symmetry condition  v_ij/E_i = v_ji/E_j  (Eq. 7)
  - Simplified orthorhombic horizontal stresses with pore pressure (Eqs. 25-26)
  - VTI reduction of the horizontal-stress model
  - Shear-wave splitting  SWS = (Vs_fast - Vs_slow)/Vs_fast

Note: this issue's PDF has a text layer and this article's key equations
survived as ASCII; the relations below are transcribed.  Stresses in psi, moduli
in psi, ratios/strains dimensionless.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- elasticity --------------

def hookes_stress(stiffness, strain):
    """Hooke's law in Voigt notation  sigma = C*epsilon  (Eq. 1)."""
    return np.asarray(stiffness, float) @ np.asarray(strain, float)


def compliance(stiffness):
    """Compliance matrix  S = inv(C)  (Eq. 3)."""
    return np.linalg.inv(np.asarray(stiffness, float))


# ---------------------------------------------- engineering moduli --------------

def engineering_moduli(stiffness):
    """Direction-dependent Young's moduli and Poisson's ratios from the
    compliance matrix S = inv(C) (Eqs. 9-12)

        E_i  = 1/S_ii            (i = 1,2,3)
        v_ij = -S_ij/S_ii        (stress in i, strain measured in j)
        mu_23 = 1/S44, mu_13 = 1/S55, mu_12 = 1/S66.

    With Voigt axes 1='h' (sigma_h), 2='H' (sigma_H), 3='V' (sigma_V), returns a
    dict of the engineering constants {E1,E2,E3, v12,v21,v13,v31,v23,v32,
    mu23,mu13,mu12}.
    """
    s = compliance(stiffness)
    e = {f"E{i+1}": 1.0 / s[i, i] for i in range(3)}
    nu = {}
    for i in range(3):
        for j in range(3):
            if i != j:
                nu[f"v{i+1}{j+1}"] = -s[j, i] / s[i, i]
    shear = {"mu23": 1.0 / s[3, 3], "mu13": 1.0 / s[4, 4], "mu12": 1.0 / s[5, 5]}
    return {**e, **nu, **shear}


def compliance_symmetry_residual(moduli):
    """Compliance-symmetry residual  v_ij/E_i - v_ji/E_j  (Eq. 7).

    For a physically plausible (symmetric-compliance) medium this is ~0 for every
    pair.  Returns the maximum absolute residual over the three index pairs.
    """
    pairs = [(1, 2), (1, 3), (2, 3)]
    res = [moduli[f"v{i}{j}"] / moduli[f"E{i}"] - moduli[f"v{j}{i}"] / moduli[f"E{j}"]
           for i, j in pairs]
    return float(np.max(np.abs(res)))


# ---------------------------------------------- horizontal stress --------------

def horizontal_stress(sigma_v, pore_pressure, biot, e_strike, eps_strike,
                      nu_cross, eps_cross, nu_vertical):
    """Orthorhombic horizontal stress with pore pressure (Eqs. 25-26)

        sigma = nu_V*(sigma_v - alpha*P) + E*(eps + nu_cross*eps_cross) + alpha*P.

    Apply with the H-direction parameters for sigma_H and the h-direction
    parameters for sigma_h; setting the two cross ratios equal recovers VTI.
    """
    return (nu_vertical * (sigma_v - biot * pore_pressure)
            + e_strike * (eps_strike + nu_cross * eps_cross)
            + biot * pore_pressure)


def sigma_H(sigma_v, pore_pressure, biot, e_h_dir, nu_hv, nu_hh, eps_h, eps_hh):
    """Maximum horizontal stress, simplified orthorhombic model (Eq. 25)

        sigma_H = vHV*(sigma_v - alpha*P) + EH*(epsH + vhH*epsh) + alpha*P.
    """
    return horizontal_stress(sigma_v, pore_pressure, biot, e_h_dir, eps_h,
                             nu_hh, eps_hh, nu_hv)


def sigma_h(sigma_v, pore_pressure, biot, e_h_dir, nu_hv, nu_hh, eps_h, eps_hh):
    """Minimum horizontal stress, simplified orthorhombic model (Eq. 26)

        sigma_h = vhV*(sigma_v - alpha*P) + Eh*(epsh + vHh*epsH) + alpha*P.

    Same functional form as sigma_H with the h-direction parameters; with
    vhH = vHh = vh and vVh = vVH = vV the pair reduces to the VTI model.
    """
    return horizontal_stress(sigma_v, pore_pressure, biot, e_h_dir, eps_h,
                             nu_hh, eps_hh, nu_hv)


def shear_wave_splitting(vs_fast, vs_slow):
    """Shear-wave splitting  SWS = (Vs_fast - Vs_slow)/Vs_fast  (Eq. 27)."""
    return petrolib.acoustic_geomech.shear_wave_splitting(vs_fast, vs_slow)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 2: Geomechanics of Orthorhombic Media")
    print("=" * 60)

    # Compliance is the inverse of the stiffness (S*C = I)
    c = np.diag([30.0, 30.0, 25.0, 8.0, 8.0, 12.0]) + 0.0
    c[0, 1] = c[1, 0] = 10.0
    c[0, 2] = c[2, 0] = c[1, 2] = c[2, 1] = 9.0
    s = compliance(c)
    assert np.allclose(s @ c, np.eye(6), atol=1e-9)

    # Hooke's law returns a 6-vector of stresses
    sig = hookes_stress(c, [1e-4, 1e-4, 2e-4, 0, 0, 0])
    assert sig.shape == (6,)

    # Engineering moduli from compliance: E_i = 1/S_ii, and the medium is
    # physically plausible (compliance-symmetry residual ~ 0, Eq. 7)
    mod = engineering_moduli(c)
    assert np.isclose(mod["E3"], 1.0 / s[2, 2])
    assert np.isclose(mod["v12"], -s[1, 0] / s[0, 0])
    assert compliance_symmetry_residual(mod) < 1e-9
    print(f"  E1 / E2 / E3           = {mod['E1']:.2f} / {mod['E2']:.2f} / {mod['E3']:.2f}")

    # Case-study inputs: the larger applied strain gives the larger horizontal stress
    sv, p, alpha = 7250.0, 3600.0, 0.7
    sH = horizontal_stress(sv, p, alpha, e_strike=4.0e6, eps_strike=4e-4,
                           nu_cross=0.2, eps_cross=2e-4, nu_vertical=0.25)
    sh = horizontal_stress(sv, p, alpha, e_strike=4.0e6, eps_strike=2e-4,
                           nu_cross=0.2, eps_cross=4e-4, nu_vertical=0.25)
    print(f"  sigma_H / sigma_h      = {sH:.0f} / {sh:.0f} psi")
    assert sH > sh

    # The named sigma_H/sigma_h helpers match the generic stress model
    sH2 = sigma_H(sv, p, alpha, e_h_dir=4.0e6, nu_hv=0.25, nu_hh=0.2, eps_h=4e-4, eps_hh=2e-4)
    sh2 = sigma_h(sv, p, alpha, e_h_dir=4.0e6, nu_hv=0.25, nu_hh=0.2, eps_h=2e-4, eps_hh=4e-4)
    assert np.isclose(sH2, sH) and np.isclose(sh2, sh)

    # Shear-wave splitting is zero with no anisotropy, positive otherwise
    assert shear_wave_splitting(3000.0, 3000.0) == 0.0
    sws = shear_wave_splitting(3000.0, 2700.0)
    print(f"  shear-wave splitting   = {sws * 100:.1f} %")
    assert 0 < sws < 1
    print("  PASS")
    return {"sigma_H": float(sH), "SWS": float(sws), "E1": float(mod["E1"])}


if __name__ == "__main__":
    test_all()
