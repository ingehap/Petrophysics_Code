"""
Richardson et al. Supercritical CO2/Brine Relative Permeability
Reference: Richardson et al., "Supercritical CO2/Brine Relative Permeability",
Petrophysics Vol 66 No 1, pp 39-51, DOI:10.30632/PJV66N1-2025a3

Implements steady-state measurements at different pressures, Corey fitting,
pressure-dependent fluid properties (density, viscosity), interfacial tension,
and capillary number calculations for supercritical CO2.
"""

import numpy as np
from typing import Tuple, Dict, List
import warnings


class SteadyStateAtPressure:
    """Steady-state relative permeability at different pressures."""

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

    def calculate_kr_at_pressure(self, Q_phase: float, mu_phase: float,
                                  dP: float, pressure_MPa: float) -> float:
        """
        Calculate relative permeability at specific pressure.

        kr_phase = (Q_phase * mu_phase * L) / (k_abs * A * dP)

        Args:
            Q_phase: Phase flow rate (cm^3/s)
            mu_phase: Phase viscosity (cP) - already pressure-corrected if needed
            dP: Pressure drop (atm)
            pressure_MPa: Measurement pressure (MPa) for reference

        Returns:
            Phase relative permeability (fraction)
        """
        if dP == 0:
            return 0.0

        kr = (Q_phase * mu_phase * self.length) / (self.k_abs * self.area * dP)
        return np.clip(kr, 0.0, 1.0)

    def scan_fractional_flow(self, pressure_MPa: float, Q_total: float = 10.0,
                            num_steps: int = 5) -> List[Tuple[float, float, float]]:
        """
        Simulate fractional flow scan at fixed pressure.

        Args:
            pressure_MPa: Test pressure (MPa)
            Q_total: Total flow rate (cm^3/s)
            num_steps: Number of fractional flow steps

        Returns:
            List of (fractional_flow, kr_CO2, kr_brine) tuples
        """
        results = []
        fractional_flows = np.linspace(0.1, 0.9, num_steps)

        for ff in fractional_flows:
            Q_CO2 = ff * Q_total
            Q_brine = (1.0 - ff) * Q_total

            # Pressure-dependent properties
            mu_CO2 = self.scco2_viscosity(pressure_MPa)
            mu_brine = 1.0  # Brine viscosity relatively constant

            dP = 0.5  # atm

            kr_CO2 = self.calculate_kr_at_pressure(Q_CO2, mu_CO2, dP, pressure_MPa)
            kr_brine = self.calculate_kr_at_pressure(Q_brine, mu_brine, dP, pressure_MPa)

            results.append((ff, kr_CO2, kr_brine))

        return results

    @staticmethod
    def scco2_density(pressure_MPa: float, temperature_C: float = 45.0) -> float:
        """
        Simple correlation for supercritical CO2 density.

        Args:
            pressure_MPa: Pressure in MPa
            temperature_C: Temperature in Celsius

        Returns:
            Density in kg/m^3
        """
        # Simplified correlation (typical range 20-45°C, 8-25 MPa)
        # At critical point: P=7.38 MPa, T=31°C, rho=467 kg/m^3
        T_K = temperature_C + 273.15

        # Pressure effect (higher pressure -> higher density)
        density = 400.0 + 15.0 * (pressure_MPa - 10.0)

        # Temperature effect (higher T -> lower density for scCO2)
        density -= 2.0 * (temperature_C - 45.0)

        return np.clip(density, 300.0, 800.0)

    @staticmethod
    def scco2_viscosity(pressure_MPa: float, temperature_C: float = 45.0) -> float:
        """
        Simple correlation for supercritical CO2 viscosity.

        Args:
            pressure_MPa: Pressure in MPa
            temperature_C: Temperature in Celsius

        Returns:
            Viscosity in cP
        """
        # Simplified correlation
        # At critical point: mu ~= 0.06 cP
        # Viscosity increases with pressure at constant T
        T_K = temperature_C + 273.15

        # Base viscosity
        mu_base = 0.04  # cP

        # Pressure correction (higher pressure -> higher viscosity)
        mu = mu_base + 0.003 * (pressure_MPa - 10.0)

        # Temperature correction (higher T -> slightly higher viscosity for gases)
        mu += 0.0001 * (temperature_C - 45.0)

        return np.clip(mu, 0.02, 0.15)


class CoreyFitting:
    """Corey model fitting for experimental data."""

    @staticmethod
    def fit_parameters(Sg_values: np.ndarray, kr_CO2_values: np.ndarray,
                       kr_CO2_max: float, Sgr: float, Swi: float) -> float:
        """
        Fit Corey exponent to relative permeability data.

        Given kr_CO2_max, solve for n_g where:
        kr_CO2 = kr_CO2_max * ((Sg - Sgr)/(1-Swi-Sgr))^n_g

        Args:
            Sg_values: Gas saturation values
            kr_CO2_values: Measured kr_CO2 values
            kr_CO2_max: Maximum kr_CO2
            Sgr: Residual gas saturation
            Swi: Irreducible water saturation

        Returns:
            Fitted exponent n_g
        """
        valid_indices = kr_CO2_values > 0.01
        if np.sum(valid_indices) < 2:
            return 2.0  # Default

        Sg = Sg_values[valid_indices]
        kr = kr_CO2_values[valid_indices]

        # Normalize saturations
        Sg_norm = (Sg - Sgr) / (1.0 - Swi - Sgr)
        Sg_norm = np.clip(Sg_norm, 0.001, 1.0)

        # Solve: kr = kr_max * Sg_norm^n
        # log(kr/kr_max) = n * log(Sg_norm)
        log_ratio = np.log(kr / kr_CO2_max)
        log_Sg_norm = np.log(Sg_norm)

        # Linear regression: log(kr/kr_max) vs log(Sg_norm)
        n_g = np.polyfit(log_Sg_norm, log_ratio, 1)[0]
        return np.clip(n_g, 0.5, 5.0)

    @staticmethod
    def corey_model(Sg: float, kr_max: float, n: float,
                    Sgr: float, Swi: float) -> float:
        """
        Calculate Corey relative permeability.

        kr = kr_max * ((Sg - Sgr)/(1-Swi-Sgr))^n

        Args:
            Sg: Gas saturation
            kr_max: Maximum relative permeability
            n: Corey exponent
            Sgr: Residual gas saturation
            Swi: Irreducible water saturation

        Returns:
            Relative permeability
        """
        Sg_norm = (Sg - Sgr) / (1.0 - Swi - Sgr)
        Sg_norm = np.clip(Sg_norm, 0.0, 1.0)
        return kr_max * (Sg_norm ** n)


class InterfacialTension:
    """Interfacial tension (IFT) correlations for CO2/brine."""

    @staticmethod
    def ift_pressure_dependence(pressure_MPa: float,
                                temperature_C: float = 45.0,
                                salinity_wt_pct: float = 0.0) -> float:
        """
        Calculate interfacial tension as function of pressure.

        Args:
            pressure_MPa: Pressure in MPa
            temperature_C: Temperature in Celsius
            salinity_wt_pct: Salinity in weight percent NaCl

        Returns:
            Interfacial tension in mN/m (dyne/cm)
        """
        # Reference: IFT decreases with pressure (CO2 density increases)
        # At low pressure (25°C, 1 atm): ~72 mN/m
        # At high pressure (45°C, 20 MPa): ~20-30 mN/m

        # Base IFT at 1 atm (0.1 MPa)
        ift_ref = 70.0  # mN/m

        # Pressure effect (decreases with pressure)
        pressure_effect = -2.5 * (pressure_MPa - 0.1)

        # Temperature effect (decreases slightly with temperature)
        temp_effect = -0.15 * (temperature_C - 25.0)

        # Salinity effect (increases with salt content)
        salinity_effect = 0.5 * salinity_wt_pct

        ift = ift_ref + pressure_effect + temp_effect + salinity_effect
        return np.clip(ift, 10.0, 75.0)

    @staticmethod
    def ift_trend(pressure_range_MPa: np.ndarray, temperature_C: float = 45.0) -> np.ndarray:
        """
        Calculate IFT across a pressure range.

        Args:
            pressure_range_MPa: Array of pressures in MPa
            temperature_C: Temperature in Celsius

        Returns:
            Array of IFT values in mN/m
        """
        return np.array([InterfacialTension.ift_pressure_dependence(p, temperature_C)
                        for p in pressure_range_MPa])


class CapillaryNumber:
    """Capillary number for CO2/brine systems."""

    @staticmethod
    def capillary_number_velocity(mu_CO2: float, v_Darcy: float,
                                  ift: float) -> float:
        """
        Calculate capillary number from velocity.

        Ca = mu * v / sigma

        Args:
            mu_CO2: CO2 viscosity (cP)
            v_Darcy: Darcy velocity (cm/s)
            ift: Interfacial tension (mN/m = dyne/cm)

        Returns:
            Capillary number (dimensionless)
        """
        if ift == 0:
            return 0.0
        return (mu_CO2 * v_Darcy) / ift

    @staticmethod
    def capillary_number_pressure_gradient(kr: float, dP: float,
                                          ift: float, length: float) -> float:
        """
        Calculate capillary number from pressure gradient.

        Ca = (kr * dP) / (sigma * L)

        Args:
            kr: Relative permeability
            dP: Pressure drop (atm, convert to consistent units)
            ift: Interfacial tension (mN/m)
            length: Length scale (cm)

        Returns:
            Capillary number (dimensionless)
        """
        if ift == 0 or length == 0:
            return 0.0

        # dP in atm to Pa: 1 atm = 101325 Pa
        # This is approximate; actual conversion depends on units used in dP
        return (kr * dP) / (ift * length)


def test():
    """
    Test function demonstrating Richardson et al. supercritical CO2 models.
    """
    print("=" * 70)
    print("RICHARDSON ET AL. SUPERCRITICAL CO2/BRINE RELATIVE PERMEABILITY TEST")
    print("=" * 70)

    # Initialize steady-state at pressure
    print("\n1. STEADY-STATE MEASUREMENTS AT DIFFERENT PRESSURES")
    print("-" * 70)
    ss_kr = SteadyStateAtPressure(k_abs=100.0, length=10.0, area=5.0)

    pressures = [10.0, 15.0, 20.0, 25.0]
    for P in pressures:
        results = ss_kr.scan_fractional_flow(pressure_MPa=P, Q_total=10.0, num_steps=5)
        print(f"\nPressure: {P:.1f} MPa")
        print(f"  {'FF':>6} {'kr_CO2':>10} {'kr_brine':>10}")
        for ff, kr_CO2, kr_brine in results:
            print(f"  {ff:6.2f} {kr_CO2:10.4f} {kr_brine:10.4f}")

    # Pressure-dependent properties
    print("\n2. PRESSURE-DEPENDENT SCCO2 PROPERTIES")
    print("-" * 70)
    pressures = np.array([8, 10, 15, 20, 25])
    print(f"{'P (MPa)':>10} {'rho (kg/m^3)':>15} {'mu (cP)':>12}")
    for P in pressures:
        rho = SteadyStateAtPressure.scco2_density(P)
        mu = SteadyStateAtPressure.scco2_viscosity(P)
        print(f"{P:10.1f} {rho:15.1f} {mu:12.4f}")

    # Corey fitting
    print("\n3. COREY MODEL FITTING")
    print("-" * 70)
    # Simulate experimental data
    Sg_exp = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
    kr_CO2_exp = np.array([0.001, 0.015, 0.050, 0.12, 0.22, 0.38, 0.62, 0.88])

    Sgr = 0.05
    Swi = 0.15

    n_g_fitted = CoreyFitting.fit_parameters(Sg_exp, kr_CO2_exp, kr_CO2_max=0.90,
                                            Sgr=Sgr, Swi=Swi)
    print(f"Fitted Corey exponent (n_g): {n_g_fitted:.3f}")

    print(f"\n{'Sg':>8} {'Measured kr':>15} {'Corey fit':>12} {'Error':>10}")
    for Sg, kr_measured in zip(Sg_exp, kr_CO2_exp):
        kr_corey = CoreyFitting.corey_model(Sg, kr_max=0.90, n=n_g_fitted,
                                           Sgr=Sgr, Swi=Swi)
        error = abs(kr_corey - kr_measured) / kr_measured * 100 if kr_measured > 0 else 0
        print(f"{Sg:8.3f} {kr_measured:15.4f} {kr_corey:12.4f} {error:10.2f}%")

    # Interfacial tension
    print("\n4. INTERFACIAL TENSION (IFT) PRESSURE DEPENDENCE")
    print("-" * 70)
    pressures = np.linspace(8, 25, 10)
    print(f"{'P (MPa)':>10} {'IFT (mN/m)':>15}")
    for P in pressures:
        ift = InterfacialTension.ift_pressure_dependence(P)
        print(f"{P:10.2f} {ift:15.2f}")

    # Capillary number analysis
    print("\n5. CAPILLARY NUMBER CALCULATIONS")
    print("-" * 70)

    # From velocity
    print("\nFrom velocity (Ca = mu*v/sigma):")
    print(f"{'P (MPa)':>10} {'mu (cP)':>12} {'v (cm/s)':>12} {'sigma (mN/m)':>14} {'Ca':>12}")
    for P in [10, 15, 20]:
        mu = SteadyStateAtPressure.scco2_viscosity(P)
        ift = InterfacialTension.ift_pressure_dependence(P)
        v = 0.01  # cm/s
        Ca = CapillaryNumber.capillary_number_velocity(mu, v, ift)
        print(f"{P:10.1f} {mu:12.4f} {v:12.3f} {ift:14.2f} {Ca:12.6f}")

    # From pressure gradient
    print("\nFrom pressure gradient (Ca = (kr*dP)/(sigma*L)):")
    print(f"{'P (MPa)':>10} {'kr':>8} {'dP (atm)':>12} {'sigma (mN/m)':>14} {'L (cm)':>8} {'Ca':>12}")
    for P in [10, 15, 20]:
        ift = InterfacialTension.ift_pressure_dependence(P)
        kr = 0.5
        dP = 0.5
        L = 10.0
        Ca = CapillaryNumber.capillary_number_pressure_gradient(kr, dP, ift, L)
        print(f"{P:10.1f} {kr:8.2f} {dP:12.2f} {ift:14.2f} {L:8.1f} {Ca:12.6f}")

    # Summary statistics
    print("\n6. PRESSURE EFFECTS SUMMARY")
    print("-" * 70)
    P_range = np.array([8, 12, 16, 20, 24])
    rho_range = np.array([SteadyStateAtPressure.scco2_density(P) for P in P_range])
    mu_range = np.array([SteadyStateAtPressure.scco2_viscosity(P) for P in P_range])
    ift_range = InterfacialTension.ift_trend(P_range)

    print(f"\nDensity increase: {rho_range[0]:.0f} -> {rho_range[-1]:.0f} kg/m^3")
    print(f"Viscosity increase: {mu_range[0]:.4f} -> {mu_range[-1]:.4f} cP")
    print(f"IFT decrease: {ift_range[0]:.2f} -> {ift_range[-1]:.2f} mN/m")

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    test()
