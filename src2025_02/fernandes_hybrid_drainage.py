"""
MODULE 7: Fernandes Hybrid Drainage Technique
Reference: Fernandes et al., "Hybrid Drainage Technique Application on Bimodal
Limestone", Petrophysics Vol 66 No 1, pp 95-109, DOI:10.30632/PJV66N1-2025a7

Implements Corey and LET relative permeability models, LogBeta capillary pressure,
capillary number calculations, and HDT workflow simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class CoreyParams:
    """Parameters for Corey relative permeability model."""
    No: float  # Oil exponent
    Nw: float  # Water exponent
    Swi: float  # Initial water saturation
    Sor: float  # Residual oil saturation


@dataclass
class LETParams:
    """Parameters for LET relative permeability model."""
    kr_oil_max: float  # Maximum oil relative permeability
    kr_water_max: float  # Maximum water relative permeability
    L: float  # L exponent (shape parameter)
    E: float  # E parameter (transition)
    T: float  # T exponent (transition)
    Swi: float  # Initial water saturation
    Sor: float  # Residual oil saturation


@dataclass
class LogBetaParams:
    """Parameters for LogBeta capillary pressure model."""
    A: float  # Amplitude parameter
    B: float  # Beta parameter (shape)
    P0: float  # Reference pressure (Pa)
    Pt: float  # Threshold pressure (Pa) for imbibition
    Swn0: float  # Reference normalized saturation
    b: float = None  # Constant shift (calculated from Swn0)


class FernandesHybridDrainage:
    """Hybrid drainage technique implementation for bimodal limestone."""

    def __init__(self):
        self.corey_params = None
        self.let_params = None
        self.logbeta_params = None

    def set_corey_params(self, No: float, Nw: float, Swi: float, Sor: float):
        """Set Corey model parameters."""
        self.corey_params = CoreyParams(No=No, Nw=Nw, Swi=Swi, Sor=Sor)

    def set_let_params(self, kr_oil_max: float, kr_water_max: float,
                       L: float, E: float, T: float, Swi: float, Sor: float):
        """Set LET model parameters."""
        self.let_params = LETParams(kr_oil_max=kr_oil_max, kr_water_max=kr_water_max,
                                     L=L, E=E, T=T, Swi=Swi, Sor=Sor)

    def set_logbeta_params(self, A: float, B: float, P0: float, Pt: float, Swn0: float):
        """Set LogBeta capillary pressure parameters."""
        # Calculate constant shift b from reference saturation
        ln_ratio_num = np.log(Swn0**B / (1 - Swn0**B))
        ln_ratio_denom = np.log((1 - Swn0)**B / (1 - (1 - Swn0)**B))
        b = (A/B) * P0 * (ln_ratio_num - ln_ratio_denom)

        self.logbeta_params = LogBetaParams(A=A, B=B, P0=P0, Pt=Pt,
                                             Swn0=Swn0, b=b)

    def normalize_saturation(self, Sw: float) -> float:
        """
        Normalize saturation: Swn = (Sw - Swi) / (1 - Swi - Sor)
        Eq 1-2 (implicitly) and Eq 3-4
        """
        if self.corey_params is None:
            raise ValueError("Corey parameters not set")

        Swi = self.corey_params.Swi
        Sor = self.corey_params.Sor
        Swn = (Sw - Swi) / (1 - Swi - Sor)
        return np.clip(Swn, 0, 1)

    def corey_kr_oil(self, Sw: float) -> float:
        """
        Corey oil relative permeability (Eq 1):
        kr_oil = Son^No
        where Son = (1 - Sw - Sor) / (1 - Swi - Sor)
        """
        if self.corey_params is None:
            raise ValueError("Corey parameters not set")

        p = self.corey_params
        Son = (1 - Sw - p.Sor) / (1 - p.Swi - p.Sor)
        Son = np.clip(Son, 0, 1)
        return Son ** p.No

    def corey_kr_water(self, Sw: float) -> float:
        """
        Corey water relative permeability (Eq 3):
        kr_water = Swn^Nw
        where Swn = (Sw - Swi) / (1 - Swi - Sor)
        """
        if self.corey_params is None:
            raise ValueError("Corey parameters not set")

        Swn = self.normalize_saturation(Sw)
        return Swn ** self.corey_params.Nw

    def let_kr_oil(self, Sw: float) -> float:
        """
        LET oil relative permeability (Eq 11):
        kr_oil = kr_oil_max * (1-Swn)^L / ((1-Swn)^L + E*(Swn)^T)
        """
        if self.let_params is None:
            raise ValueError("LET parameters not set")

        p = self.let_params
        Swn = self.normalize_saturation(Sw)

        numerator = (1 - Swn) ** p.L
        denominator = (1 - Swn) ** p.L + p.E * Swn ** p.T

        if denominator == 0:
            return 0
        return p.kr_oil_max * numerator / denominator

    def let_kr_water(self, Sw: float) -> float:
        """
        LET water relative permeability (Eq 12):
        kr_water = kr_water_max * (Swn)^L / ((Swn)^L + E*(1-Swn)^T)
        """
        if self.let_params is None:
            raise ValueError("LET parameters not set")

        p = self.let_params
        Swn = self.normalize_saturation(Sw)

        numerator = Swn ** p.L
        denominator = Swn ** p.L + p.E * (1 - Swn) ** p.T

        if denominator == 0:
            return 0
        return p.kr_water_max * numerator / denominator

    def logbeta_capillary_pressure(self, Sw: float, is_imbibition: bool = False) -> float:
        """
        LogBeta capillary pressure model (Eqs 5-7):
        Pc = -(A/B)*P0*(ln(Swn^B/(1-Swn^B)) - ln((1-Swn)^B/(1-(1-Swn)^B))) + b

        For imbibition with threshold (Eq 7):
        If Pc < Pt: Pc = -P0 / (1 - Swn)
        """
        if self.logbeta_params is None:
            raise ValueError("LogBeta parameters not set")

        p = self.logbeta_params
        Swn = self.normalize_saturation(Sw)
        Swn = np.clip(Swn, 1e-6, 1 - 1e-6)  # Avoid log(0)

        # Main LogBeta equation
        ln_ratio_num = np.log(Swn**p.B / (1 - Swn**p.B))
        ln_ratio_denom = np.log((1 - Swn)**p.B / (1 - (1 - Swn)**p.B))
        Pc = -(p.A/p.B) * p.P0 * (ln_ratio_num - ln_ratio_denom) + p.b

        # Apply threshold for imbibition
        if is_imbibition and p.Pt is not None:
            Pc_threshold = -p.P0 / (1 - Swn)
            if Pc < p.Pt:
                Pc = Pc_threshold

        return Pc

    def capillary_number(self, mu: float, v: float, sigma: float) -> float:
        """
        Capillary number (Eq 9):
        Ca = mu * v / sigma

        Parameters:
            mu: viscosity (Pa.s)
            v: velocity (m/s)
            sigma: interfacial tension (N/m)
        """
        if sigma == 0:
            raise ValueError("Interfacial tension cannot be zero")
        return mu * v / sigma

    def ct_saturation_normalization(self, Px: float, Pb: float, Po: float) -> float:
        """
        Saturation normalization for sub-resolved porosity from CT imaging (Eq 8):
        Sw = (Px - Pb) / (Pb - Po)

        Parameters:
            Px: X-ray attenuation at position x
            Pb: X-ray attenuation of brine
            Po: X-ray attenuation of oil
        """
        if Pb == Po:
            raise ValueError("Brine and oil attenuations cannot be equal")
        return (Px - Pb) / (Pb - Po)

    def hdt_workflow_simulation(self, Sw_array: np.ndarray,
                                 Ca_viscous: float, Ca_capillary: float) -> dict:
        """
        HDT workflow simulation: Two-step process
        1. Viscous oilflooding at high Ca
        2. Porous plate at low Ca (capillary-controlled)

        Returns dictionary with results for both steps.
        """
        results = {}

        # Step 1: Viscous flooding (high Ca)
        results['viscous_flooding'] = {
            'Ca': Ca_viscous,
            'kr_oil': [self.let_kr_oil(Sw) if self.let_params else self.corey_kr_oil(Sw)
                       for Sw in Sw_array],
            'kr_water': [self.let_kr_water(Sw) if self.let_params else self.corey_kr_water(Sw)
                         for Sw in Sw_array],
            'description': 'High Ca viscous displacement'
        }

        # Step 2: Porous plate (low Ca, capillary-controlled)
        results['porous_plate'] = {
            'Ca': Ca_capillary,
            'kr_oil': [self.let_kr_oil(Sw) if self.let_params else self.corey_kr_oil(Sw)
                       for Sw in Sw_array],
            'Pc': [self.logbeta_capillary_pressure(Sw, is_imbibition=False)
                   for Sw in Sw_array],
            'description': 'Low Ca capillary-controlled drainage'
        }

        return results


def test():
    """Test function for Fernandes Hybrid Drainage module."""
    print("=" * 70)
    print("MODULE 7: Fernandes Hybrid Drainage Technique Test")
    print("=" * 70)

    hdt = FernandesHybridDrainage()

    # Test Corey model
    print("\n1. COREY RELATIVE PERMEABILITY MODEL")
    print("-" * 70)
    hdt.set_corey_params(No=2.5, Nw=2.0, Swi=0.2, Sor=0.25)

    Sw_test = np.array([0.2, 0.4, 0.55, 0.7, 0.8])
    print(f"Sw values: {Sw_test}")
    print(f"kr_oil (Corey): {[f'{hdt.corey_kr_oil(Sw):.4f}' for Sw in Sw_test]}")
    print(f"kr_water (Corey): {[f'{hdt.corey_kr_water(Sw):.4f}' for Sw in Sw_test]}")

    # Test LET model
    print("\n2. LET RELATIVE PERMEABILITY MODEL")
    print("-" * 70)
    hdt.set_let_params(kr_oil_max=1.0, kr_water_max=0.5,
                       L=2.5, E=0.5, T=1.5, Swi=0.2, Sor=0.25)

    print(f"Sw values: {Sw_test}")
    print(f"kr_oil (LET): {[f'{hdt.let_kr_oil(Sw):.4f}' for Sw in Sw_test]}")
    print(f"kr_water (LET): {[f'{hdt.let_kr_water(Sw):.4f}' for Sw in Sw_test]}")

    # Test LogBeta capillary pressure
    print("\n3. LOGBETA CAPILLARY PRESSURE MODEL")
    print("-" * 70)
    hdt.set_logbeta_params(A=5.0, B=1.5, P0=20000, Pt=5000, Swn0=0.5)

    print(f"Sw values: {Sw_test}")
    Pc_vals = [hdt.logbeta_capillary_pressure(Sw) for Sw in Sw_test]
    print(f"Pc (Pa): {[f'{Pc:.2f}' for Pc in Pc_vals]}")

    # Test capillary number
    print("\n4. CAPILLARY NUMBER CALCULATION")
    print("-" * 70)
    mu = 1e-3  # Pa.s
    v = 1e-5  # m/s
    sigma = 0.03  # N/m
    Ca = hdt.capillary_number(mu, v, sigma)
    print(f"mu = {mu} Pa.s, v = {v} m/s, sigma = {sigma} N/m")
    print(f"Capillary number Ca = {Ca:.6e}")

    # Test saturation normalization
    print("\n5. CT SATURATION NORMALIZATION")
    print("-" * 70)
    Pb, Po = 100, 50  # X-ray attenuations
    Px_vals = np.array([50, 70, 90, 100])
    print(f"Pb (brine attenuation) = {Pb}, Po (oil attenuation) = {Po}")
    print(f"Px values: {Px_vals}")
    Sw_norm = [hdt.ct_saturation_normalization(Px, Pb, Po) for Px in Px_vals]
    print(f"Sw (normalized): {[f'{Sw:.4f}' for Sw in Sw_norm]}")

    # Test HDT workflow
    print("\n6. HDT WORKFLOW SIMULATION")
    print("-" * 70)
    Sw_workflow = np.linspace(0.2, 0.8, 5)
    Ca_viscous = 1e-4
    Ca_capillary = 1e-7

    workflow_results = hdt.hdt_workflow_simulation(Sw_workflow, Ca_viscous, Ca_capillary)

    print(f"Viscous Flooding (Ca = {Ca_viscous}):")
    print(f"  kr_oil: {[f'{kr:.4f}' for kr in workflow_results['viscous_flooding']['kr_oil']]}")

    print(f"Porous Plate (Ca = {Ca_capillary}):")
    print(f"  kr_oil: {[f'{kr:.4f}' for kr in workflow_results['porous_plate']['kr_oil']]}")
    Pc_workflow = workflow_results['porous_plate']['Pc']
    print(f"  Pc (Pa): {[f'{Pc:.2f}' for Pc in Pc_workflow]}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test()
