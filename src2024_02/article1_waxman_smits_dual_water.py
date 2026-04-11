"""
Article 1: The Fundamental Flaws of the Waxman-Smits and Dual Water Formulations
Rasmus, Kennedy, Homan (Petrophysics, Vol. 65, No. 1, Feb 2024, pp. 5-31)

Implements the classic shaly-sand conductivity models (Waxman-Smits and
Dual Water) used to relate brine conductivity Cw to rock conductivity Co
in the presence of clay (CEC -> Qv).
"""
import numpy as np


def waxman_smits_co(Cw, Qv, phi, Sw=1.0, m_star=2.0, n_star=2.0, B=None, T=25.0):
    """Waxman-Smits: Co = (phi^m* * Sw^n* / F*) * (Cw + B*Qv/Sw)
    where F* = phi^-m*. B is the equivalent counter-ion conductance."""
    if B is None:
        # Waxman-Thomas temperature dependence (S/m per mol/L)
        B = (1.0 - 0.83 * np.exp(-np.exp(-0.5 + 0.0577 * T) * Cw))
        B *= (-1.28 + 0.225 * T - 0.0004059 * T * T)
        B = np.maximum(B, 0.0)
    Co = (phi ** m_star) * (Sw ** n_star) * (Cw + B * Qv / np.maximum(Sw, 1e-9))
    return Co


def dual_water_co(Cw, Cwb, Swb, phi, Sw=1.0, m=2.0, n=2.0):
    """Dual Water: Co = phi^m * Sw^n * [(1 - Swb/Sw)*Cw + (Swb/Sw)*Cwb]"""
    frac = np.clip(Swb / np.maximum(Sw, 1e-9), 0.0, 1.0)
    Co = (phi ** m) * (Sw ** n) * ((1.0 - frac) * Cw + frac * Cwb)
    return Co


def test_all():
    Cw = np.linspace(0.1, 20.0, 25)  # S/m
    phi, Qv, Sw = 0.25, 0.3, 1.0
    co_ws = waxman_smits_co(Cw, Qv, phi, Sw)
    assert np.all(np.diff(co_ws) > 0), "Co should rise monotonically with Cw"
    co_dw = dual_water_co(Cw, Cwb=8.0, Swb=0.15, phi=phi, Sw=Sw)
    assert np.all(co_dw > 0)
    # At Qv=0 WS should reduce to Archie
    co_arch = waxman_smits_co(Cw, 0.0, phi, Sw)
    np.testing.assert_allclose(co_arch, (phi ** 2) * Cw, rtol=1e-9)
    print("article1 OK | WS sample:", co_ws[::5].round(3))


if __name__ == "__main__":
    test_all()
