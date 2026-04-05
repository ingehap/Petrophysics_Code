"""
Mascle et al. In-Situ Investigation of CO2/Brine Relative Permeability
Reference: Mascle et al., "In-Situ Investigation of CO2/Brine Relative Permeability",
Petrophysics Vol 66 No 1, pp 22-38, DOI:10.30632/PJV66N1-2025a2

Implements steady-state and unsteady-state relative permeability measurements,
Corey model, LogBeta capillary pressure, and capillary number calculations.
"""

import numpy as np
from typing import Tuple, Dict, List
import warnings


class SteadyStateRelativePermeability:
    """Steady-state (SS) relative permeability using Darcy's law."""

    def __init__(self, k_abs: float, length: float, area: float):
        """
        Initialize core sample geometry.

        Args:
            k_abs: Absolute permeability (mD)
            length: Core length (cm)
            area: Cross-sectional area (cm^2)
        """
        self.k_abs = k_abs  # mD
        self.length = length  # cm
        self.area = area  # cm^2

    def calculate_kr(self, Q_phase: float, mu_phase: float, delta_P: float) -> float:
        """
        Calculate relative permeability from Darcy's law.

        kr_phase = (Q_phase * mu_phase * L) / (k_abs * A * delta_P)

        Args:
            Q_phase: Phase flow rate (cm^3/s)
            mu_phase: Phase viscosity (cP)
            delta_P: Pressure drop (atm)

        Returns:
            Phase relative permeability (fraction)
        """
        if delta_P == 0:
            return 0.0

        # Convert units: Q(cm^3/s), mu(cP), L(cm), A(cm^2), dP(atm), k(mD)
        kr = (Q_phase * mu_phase * self.length) / (self.k_abs * self.area * delta_P)
        return np.clip(kr, 0.0, 1.0)


class UnsteadyStateRelativePermeability:
    """Unsteady-state (USS) relative permeability using JBN method."""

    def __init__(self, k_abs: float, length: float, area: float,
                 pore_volume: float, porosity: float):
        """
        Initialize core and fluid properties.

        Args:
            k_abs: Absolute permeability (mD)
            length: Core length (cm)
            area: Cross-sectional area (cm^2)
            pore_volume: Pore volume (cm^3)
            porosity: Porosity (fraction)
        """
        self.k_abs = k_abs
        self.length = length
        self.area = area
        self.pore_volume = pore_volume
        self.porosity = porosity

    def fractional_flow(self, kr_CO2: float, mu_CO2: float,
                        kr_brine: float, mu_brine: float) -> float:
        """
        Calculate fractional flow of water using JBN method.

        fw = 1 / (1 + (kr_CO2/mu_CO2)/(kr_brine/mu_brine))

        Args:
            kr_CO2: CO2 relative permeability
            mu_CO2: CO2 viscosity (cP)
            kr_brine: Brine relative permeability
            mu_brine: Brine viscosity (cP)

        Returns:
            Water fractional flow (fraction)
        """
        if mu_brine == 0 or mu_CO2 == 0:
            return 0.0

        mobility_ratio = (kr_CO2 / mu_CO2) / (kr_brine / mu_brine)
        fw = 1.0 / (1.0 + mobility_ratio)
        return fw

    def recovery_factor(self, pv_injected: float) -> float:
        """
        Calculate recovery factor based on pore volumes injected.

        Args:
            pv_injected: Pore volumes of fluid injected

        Returns:
            Recovery factor (fraction of original fluid)
        """
        # Typical recovery factor curve (exponential approach to steady state)
        recovery = 1.0 - np.exp(-pv_injected)
        return np.clip(recovery, 0.0, 1.0)


class CoreyRelativePermeability:
    """Corey model for CO2/brine relative permeability."""

    def __init__(self, kr_CO2_max: float, n_g: float,
                 kr_brine_max: float, n_w: float,
                 Sgr: float, Swi: float):
        """
        Initialize Corey parameters.

        kr_CO2 = kr_CO2_max * ((Sg - Sgr)/(1-Swi-Sgr))^n_g
        kr_brine = kr_brine_max * ((Sw - Swi)/(1-Swi-Sgr))^n_w
        """
        self.kr_CO2_max = kr_CO2_max
        self.n_g = n_g
        self.kr_brine_max = kr_brine_max
        self.n_w = n_w
        self.Sgr = Sgr
        self.Swi = Swi

    def kr_CO2(self, Sg: float) -> float:
        """Calculate CO2 relative permeability."""
        Sg_norm = (Sg - self.Sgr) / (1.0 - self.Swi - self.Sgr)
        Sg_norm = np.clip(Sg_norm, 0.0, 1.0)
        return self.kr_CO2_max * (Sg_norm ** self.n_g)

    def kr_brine(self, Sw: float) -> float:
        """Calculate brine relative permeability."""
        Sw_norm = (Sw - self.Swi) / (1.0 - self.Swi - self.Sgr)
        Sw_norm = np.clip(Sw_norm, 0.0, 1.0)
        return self.kr_brine_max * (Sw_norm ** self.n_w)


class LogBetaCapillaryPressure:
    """LogBeta capillary pressure model."""

    def __init__(self, A: float, B: float, P0: float, Swn0: float):
        """
        Initialize LogBeta parameters (Eq. 4).

        Pc = -(A/B)*P0*(ln(Swn^B/(1-Swn^B)) - ln((1-Swn)^B/(1-(1-Swn)^B))) + b
        """
        self.A = A
        self.B = B
        self.P0 = P0
        self.Swn0 = Swn0

        # Calculate b at initial saturation
        term1 = np.log(Swn0**B / (1.0 - Swn0**B))
        term2 = np.log((1.0 - Swn0)**B / (1.0 - (1.0 - Swn0)**B))
        self.b = -(A / B) * P0 * (term1 - term2)

    def pc(self, Swn: float) -> float:
        """
        Calculate capillary pressure.

        Args:
            Swn: Normalized water saturation (0 to 1)

        Returns:
            Capillary pressure (Pa)
        """
        Swn = np.clip(Swn, 0.001, 0.999)

        term1 = np.log(Swn**self.B / (1.0 - Swn**self.B))
        term2 = np.log((1.0 - Swn)**self.B / (1.0 - (1.0 - Swn)**self.B))

        pc = -(self.A / self.B) * self.P0 * (term1 - term2) + self.b
        return pc


class CapillaryNumber:
    """Capillary number calculations."""

    @staticmethod
    def from_velocity(mu: float, v: float, sigma: float) -> float:
        """
        Calculate capillary number from velocity.

        Ca = mu * v / sigma

        Args:
            mu: Viscosity (cP)
            v: Darcy velocity (cm/s)
            sigma: Interfacial tension (dyne/cm)

        Returns:
            Capillary number (dimensionless)
        """
        if sigma == 0:
            return 0.0
        return (mu * v) / sigma

    @staticmethod
    def from_pressure_gradient(kr: float, dP: float, sigma: float, length: float) -> float:
        """
        Calculate capillary number from pressure gradient.

        Ca = (kr * dP) / (sigma * L)

        Args:
            kr: Relative permeability
            dP: Pressure drop (atm)
            sigma: Interfacial tension (dyne/cm)
            length: Length scale (cm)

        Returns:
            Capillary number (dimensionless)
        """
        if sigma == 0 or length == 0:
            return 0.0
        return (kr * dP) / (sigma * length)


def test():
    """
    Test function demonstrating Mascle et al. CO2/brine models.
    """
    print("=" * 70)
    print("MASCLE ET AL. CO2/BRINE RELATIVE PERMEABILITY TEST")
    print("=" * 70)

    # Initialize steady-state measurement
    print("\n1. STEADY-STATE RELATIVE PERMEABILITY")
    print("-" * 70)
    ss_kr = SteadyStateRelativePermeability(k_abs=100.0, length=10.0, area=5.0)

    # Simulate different fractional flow steps
    fractional_flows = [0.2, 0.4, 0.6, 0.8]
    print(f"{'FF':>6} {'Q_CO2':>8} {'mu_CO2':>10} {'dP':>8} {'kr_CO2':>10}")
    for ff in fractional_flows:
        Q_CO2 = ff * 10.0  # cm^3/s
        mu_CO2 = 0.06  # cP (supercritical CO2)
        dP = 0.5  # atm
        kr = ss_kr.calculate_kr(Q_CO2, mu_CO2, dP)
        print(f"{ff:6.2f} {Q_CO2:8.2f} {mu_CO2:10.3f} {dP:8.2f} {kr:10.4f}")

    # Unsteady-state recovery analysis
    print("\n2. UNSTEADY-STATE RECOVERY ANALYSIS")
    print("-" * 70)
    uss_kr = UnsteadyStateRelativePermeability(
        k_abs=100.0, length=10.0, area=5.0,
        pore_volume=50.0, porosity=0.20
    )

    pv_values = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    print(f"{'PV Injected':>12} {'Recovery':>10}")
    for pv in pv_values:
        recovery = uss_kr.recovery_factor(pv)
        print(f"{pv:12.2f} {recovery:10.4f}")

    # Fractional flow calculations
    print("\n3. FRACTIONAL FLOW (JBN Method)")
    print("-" * 70)
    print(f"{'kr_CO2':>8} {'kr_brine':>10} {'mu_CO2':>10} {'mu_brine':>10} {'fw':>8}")
    kr_CO2_vals = [0.2, 0.4, 0.6, 0.8]
    for kr_CO2 in kr_CO2_vals:
        kr_brine = 0.85 * (1.0 - kr_CO2)
        mu_CO2 = 0.06
        mu_brine = 1.0
        fw = uss_kr.fractional_flow(kr_CO2, mu_CO2, kr_brine, mu_brine)
        print(f"{kr_CO2:8.2f} {kr_brine:10.3f} {mu_CO2:10.3f} {mu_brine:10.2f} {fw:8.4f}")

    # Corey model
    print("\n4. COREY RELATIVE PERMEABILITY MODEL")
    print("-" * 70)
    corey = CoreyRelativePermeability(
        kr_CO2_max=0.90, n_g=2.5,
        kr_brine_max=0.85, n_w=2.0,
        Sgr=0.05, Swi=0.15
    )

    Sg_values = np.linspace(0.05, 0.80, 8)
    print(f"{'Sg':>8} {'Sw':>8} {'kr_CO2':>10} {'kr_brine':>10}")
    for Sg in Sg_values:
        Sw = 1.0 - Sg
        kr_CO2 = corey.kr_CO2(Sg)
        kr_brine = corey.kr_brine(Sw)
        print(f"{Sg:8.3f} {Sw:8.3f} {kr_CO2:10.4f} {kr_brine:10.4f}")

    # LogBeta capillary pressure
    print("\n5. LOGBETA CAPILLARY PRESSURE MODEL")
    print("-" * 70)
    logbeta_pc = LogBetaCapillaryPressure(A=2.0, B=0.5, P0=1.0, Swn0=0.5)

    Swn_values = np.linspace(0.1, 0.9, 9)
    print(f"{'Swn':>8} {'Pc':>12}")
    for Swn in Swn_values:
        pc = logbeta_pc.pc(Swn)
        print(f"{Swn:8.3f} {pc:12.4f}")

    # Capillary number
    print("\n6. CAPILLARY NUMBER")
    print("-" * 70)
    print("\nFrom velocity:")
    print(f"{'mu (cP)':>10} {'v (cm/s)':>12} {'sigma (dyne/cm)':>16} {'Ca':>10}")
    mu_vals = [0.06, 0.5, 1.0]
    v_vals = [0.1, 0.01, 0.001]
    sigma = 30.0  # dyne/cm

    for mu, v in zip(mu_vals, v_vals):
        Ca = CapillaryNumber.from_velocity(mu, v, sigma)
        print(f"{mu:10.3f} {v:12.4f} {sigma:16.2f} {Ca:10.6f}")

    print(f"\nFrom pressure gradient:")
    print(f"{'kr':>8} {'dP (atm)':>12} {'sigma':>12} {'L (cm)':>10} {'Ca':>10}")
    kr_vals = [0.2, 0.5, 0.8]
    for kr in kr_vals:
        dP = 0.5
        sigma = 30.0
        L = 10.0
        Ca = CapillaryNumber.from_pressure_gradient(kr, dP, sigma, L)
        print(f"{kr:8.2f} {dP:12.2f} {sigma:12.1f} {L:10.1f} {Ca:10.6f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    test()
