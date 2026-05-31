"""
Article 4: Recovering Elastic Properties From Rock Fragments
Dang, Gupta, Chakravarty, Bhoumick, Taneja, Sondergeld, Rai (2017)
Reference: Petrophysics Vol. 58, No. 3 (June 2017), pp. 270-280
DOI: none assigned (this issue predates SPWLA DOI assignment)

Elastic properties are recovered from irregular rock fragments (no plugs needed):
the bulk modulus from the bulk compressibility of a mercury-injection
(non-intrusion) curve evaluated at 5,000 psi, and Young's modulus from
nanoindentation via the reduced modulus.  A dynamic bulk modulus from ultrasonic
velocities provides cross-validation.

Implements:

  - Bulk compressibility  C = -(1/Vb)*dVb/dP  and  K = 1/C at 5,000 psi
  - Reduced modulus  E* = (sqrt(pi)/2)*S/sqrt(A)
  - Young's modulus from the reduced modulus (indenter correction)
  - Nanoindentation hardness  H = P/A  and dynamic K = rho*(Vp^2 - 4/3*Vs^2)

Note: this issue's PDF has a text layer; the K = rho(Vp^2 - 4/3 Vs^2) and
reduced-modulus forms survived, while the MICP-compressibility equation lost its
glyph and is a faithful standard-form reconstruction.  Berkovich diamond:
E_i = 1141 GPa, v_i = 0.07.
"""

import numpy as np

E_DIAMOND = 1141e9           # Berkovich indenter Young's modulus (Pa)
V_DIAMOND = 0.07             # Berkovich indenter Poisson's ratio


# ---------------------------------------------- bulk modulus (MICP) --------------

def bulk_compressibility(vb, pressure):
    """Bulk compressibility  C = -(1/Vb)*dVb/dP  from a bulk-volume vs pressure curve."""
    vb = np.asarray(vb, float)
    p = np.asarray(pressure, float)
    return -np.gradient(vb, p) / vb


def bulk_modulus_micp(vb, pressure, p_eval=5000.0):
    """Bulk modulus from MICP  K = 1/C, with C evaluated at 5,000 psi."""
    c = bulk_compressibility(vb, pressure)
    return 1.0 / float(np.interp(p_eval, pressure, c))


# ---------------------------------------------- Young's modulus (nanoindentation) --------------

def reduced_modulus(stiffness, contact_area):
    """Reduced modulus  E* = (sqrt(pi)/2)*S/sqrt(A)  (S = contact stiffness)."""
    return (np.sqrt(np.pi) / 2.0) * stiffness / np.sqrt(contact_area)


def youngs_modulus(e_star, v_sample=0.25, e_indenter=E_DIAMOND, v_indenter=V_DIAMOND):
    """Young's modulus from the reduced modulus

        1/E* = (1 - v_s^2)/E_s + (1 - v_i^2)/E_i  ->  solve E_s.
    """
    return (1.0 - v_sample ** 2) / (1.0 / e_star - (1.0 - v_indenter ** 2) / e_indenter)


def hardness(peak_load, contact_area):
    """Nanoindentation hardness  H = P_peak/A."""
    return peak_load / contact_area


# ---------------------------------------------- ultrasonic --------------

def dynamic_bulk_modulus(rho, vp, vs):
    """Dynamic bulk modulus  K = rho*(Vp^2 - 4/3*Vs^2)."""
    return rho * (vp ** 2 - 4.0 / 3.0 * vs ** 2)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Recovering Elastic Properties From Fragments")
    print("=" * 60)

    # MICP bulk modulus: Vb decreases with pressure -> positive K
    p = np.linspace(20.0, 10000.0, 200)
    vb = 1.0 - 1.0e-5 * p
    k = bulk_modulus_micp(vb, p)
    print(f"  K from MICP @5000psi   = {k:.3e} (psi units)")
    assert k > 0

    # Reduced modulus -> Young's modulus round-trips
    e_s_true, v_s = 50e9, 0.25
    e_star = 1.0 / ((1 - v_s ** 2) / e_s_true + (1 - V_DIAMOND ** 2) / E_DIAMOND)
    e_s = youngs_modulus(e_star, v_s)
    print(f"  recovered E            = {e_s / 1e9:.2f} GPa (true 50)")
    assert np.isclose(e_s, e_s_true, rtol=1e-9)

    # Hardness and a reduced modulus from raw indentation quantities
    assert hardness(0.5, 1e-12) > 0 and reduced_modulus(2e4, 1e-12) > 0

    # Dynamic K = rho(Vp^2 - 4/3 Vs^2)
    kd = dynamic_bulk_modulus(2500.0, 4000.0, 2300.0)
    assert np.isclose(kd, 2500.0 * (4000.0 ** 2 - 4.0 / 3.0 * 2300.0 ** 2))
    print("  PASS")
    return {"E_GPa": float(e_s / 1e9), "K_dynamic": float(kd)}


if __name__ == "__main__":
    test_all()
