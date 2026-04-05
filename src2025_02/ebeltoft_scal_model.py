"""
Ebeltoft et al. SCAL Model for CO2 Storage Simulation
Reference: Ebeltoft et al., "SCAL Model for Simulation of CO2 Storage",
Petrophysics Vol 66 No 1, pp 7-21, DOI:10.30632/PJV66N1-2025a1

Implements LET and Corey relative permeability models, capillary pressure,
Land trapping, and Swr correlations for multiphase flow in porous media.
"""

import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt


class LETRelativePermeability:
    """LET (Leverett-Emond-Thomson) relative permeability model."""

    def __init__(self, krg_max: float, Lgas: float, Egas: float, Tgas: float,
                 krw_max: float, Lw: float, Ew: float, Tw: float,
                 Sgr: float, Swi: float, Sor: float):
        """
        Initialize LET relative permeability parameters.

        Args:
            krg_max: Maximum gas relative permeability
            Lgas, Egas, Tgas: L, E, T exponents for gas phase
            krw_max: Maximum water relative permeability
            Lw, Ew, Tw: L, E, T exponents for water phase
            Sgr: Residual gas saturation
            Swi: Irreducible water saturation
            Sor: Residual oil/gas saturation
        """
        self.krg_max = krg_max
        self.Lgas = Lgas
        self.Egas = Egas
        self.Tgas = Tgas

        self.krw_max = krw_max
        self.Lw = Lw
        self.Ew = Ew
        self.Tw = Tw

        self.Sgr = Sgr
        self.Swi = Swi
        self.Sor = Sor

    def kr_gas(self, Sg: float) -> float:
        """
        Calculate gas relative permeability using LET equation (Eq. 1).

        kr_gas(Sg) = krg_max * (Sg_norm)^L / ((Sg_norm)^L + E*(1-Sg_norm)^T)
        """
        Sg_norm = (Sg - self.Sgr) / (1.0 - self.Swi - self.Sgr)
        Sg_norm = np.clip(Sg_norm, 0.0, 1.0)

        numerator = self.krg_max * (Sg_norm ** self.Lgas)
        denominator = (Sg_norm ** self.Lgas) + self.Egas * ((1.0 - Sg_norm) ** self.Tgas)

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def kr_water(self, Sw: float) -> float:
        """
        Calculate water relative permeability using LET equation (Eq. 2).

        kr_water(Sw) = krw_max * (Sw_norm)^Lw / ((Sw_norm)^Lw + Ew*(1-Sw_norm)^Tw)
        """
        Sw_norm = (Sw - self.Swi) / (1.0 - self.Swi - self.Sor)
        Sw_norm = np.clip(Sw_norm, 0.0, 1.0)

        numerator = self.krw_max * (Sw_norm ** self.Lw)
        denominator = (Sw_norm ** self.Lw) + self.Ew * ((1.0 - Sw_norm) ** self.Tw)

        if denominator == 0:
            return 0.0
        return numerator / denominator


class LETCapillaryPressure:
    """LET capillary pressure model."""

    def __init__(self, A: float, B: float, C: float, D: float,
                 E_pc: float, F: float, Swi: float, Sor: float):
        """
        Initialize LET capillary pressure parameters (Eq. 3).

        Pc(Sw) = (A*(1-Sw_norm)^B - C*Sw_norm^D) / (1 + E_pc*(1-Sw_norm)^F)
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E_pc = E_pc
        self.F = F
        self.Swi = Swi
        self.Sor = Sor

    def pc(self, Sw: float) -> float:
        """Calculate capillary pressure (Pa)."""
        Sw_norm = (Sw - self.Swi) / (1.0 - self.Swi - self.Sor)
        Sw_norm = np.clip(Sw_norm, 0.0, 1.0)

        numerator = self.A * ((1.0 - Sw_norm) ** self.B) - self.C * (Sw_norm ** self.D)
        denominator = 1.0 + self.E_pc * ((1.0 - Sw_norm) ** self.F)

        return numerator / denominator


class LandTrapping:
    """Land model for residual saturation trapping (Eq. 6)."""

    def __init__(self, Sgi_max: float, Sgt_max: float):
        """
        Initialize Land trapping parameters.

        Sgt = Sgi / (1 + C*Sgi)  where C = 1/Sgt_max - 1/Sgi_max
        """
        self.Sgi_max = Sgi_max
        self.Sgt_max = Sgt_max
        self.C = 1.0 / Sgt_max - 1.0 / Sgi_max

    def residual_saturation(self, Sgi: float) -> float:
        """
        Calculate trapped residual saturation given initial saturation.
        """
        Sgt = Sgi / (1.0 + self.C * Sgi)
        return Sgt


class SwrCorrelation:
    """Residual water saturation correlations (Eqs. 4-5)."""

    @staticmethod
    def swr_from_permeability(kw: float, lithology: str = "sandstone") -> float:
        """
        Calculate residual water saturation from water permeability.

        Swr = a + b * log10(kw) for different lithologies

        Args:
            kw: Water permeability (mD)
            lithology: "sandstone", "carbonate", or "shale"

        Returns:
            Residual water saturation (fraction)
        """
        # Typical correlations (example values)
        correlations = {
            "sandstone": {"a": 0.15, "b": -0.08},
            "carbonate": {"a": 0.20, "b": -0.10},
            "shale": {"a": 0.25, "b": -0.12}
        }

        if lithology not in correlations:
            lithology = "sandstone"

        coef = correlations[lithology]
        swr = coef["a"] + coef["b"] * np.log10(kw)
        return np.clip(swr, 0.0, 0.5)


class CoreyRelativePermeability:
    """Corey power-law relative permeability model."""

    def __init__(self, kr_max: float, n: float, Sgr: float, Swi: float, Sor: float):
        """
        Initialize Corey model: kr = kr_max * (S_norm)^n
        """
        self.kr_max = kr_max
        self.n = n
        self.Sgr = Sgr
        self.Swi = Swi
        self.Sor = Sor

    def kr(self, S: float, phase: str = "gas") -> float:
        """
        Calculate relative permeability using Corey model.

        Args:
            S: Phase saturation
            phase: "gas" or "water"
        """
        if phase == "gas":
            S_norm = (S - self.Sgr) / (1.0 - self.Swi - self.Sgr)
        else:  # water
            S_norm = (S - self.Swi) / (1.0 - self.Swi - self.Sor)

        S_norm = np.clip(S_norm, 0.0, 1.0)
        return self.kr_max * (S_norm ** self.n)


def test():
    """
    Test function demonstrating all Ebeltoft et al. SCAL models.
    """
    print("=" * 70)
    print("EBELTOFT ET AL. SCAL MODEL TEST")
    print("=" * 70)

    # Initialize LET relative permeability model
    print("\n1. LET RELATIVE PERMEABILITY MODEL")
    print("-" * 70)
    let_kr = LETRelativePermeability(
        krg_max=0.90, Lgas=2.5, Egas=0.8, Tgas=1.5,
        krw_max=0.85, Lw=2.0, Ew=0.5, Tw=2.0,
        Sgr=0.05, Swi=0.15, Sor=0.25
    )

    # Test at different gas saturations
    Sg_values = np.linspace(0.05, 0.80, 8)
    print(f"{'Sg':>8} {'Sw':>8} {'kr_gas':>10} {'kr_water':>10}")
    for Sg in Sg_values:
        Sw = 1.0 - Sg
        kr_g = let_kr.kr_gas(Sg)
        kr_w = let_kr.kr_water(Sw)
        print(f"{Sg:8.3f} {Sw:8.3f} {kr_g:10.4f} {kr_w:10.4f}")

    # LET Capillary Pressure
    print("\n2. LET CAPILLARY PRESSURE MODEL")
    print("-" * 70)
    let_pc = LETCapillaryPressure(
        A=10.0, B=2.0, C=0.5, D=1.5,
        E_pc=0.3, F=1.0, Swi=0.15, Sor=0.25
    )

    Sw_values = np.linspace(0.15, 0.85, 8)
    print(f"{'Sw':>8} {'Pc (Pa)':>15}")
    for Sw in Sw_values:
        pc = let_pc.pc(Sw)
        print(f"{Sw:8.3f} {pc:15.2f}")

    # Land Trapping
    print("\n3. LAND TRAPPING MODEL")
    print("-" * 70)
    land = LandTrapping(Sgi_max=0.80, Sgt_max=0.30)

    Sgi_values = np.linspace(0.10, 0.80, 8)
    print(f"{'Sgi':>8} {'Sgt':>10}")
    for Sgi in Sgi_values:
        Sgt = land.residual_saturation(Sgi)
        print(f"{Sgi:8.3f} {Sgt:10.4f}")

    # Swr Correlations
    print("\n4. RESIDUAL WATER SATURATION CORRELATIONS")
    print("-" * 70)
    kw_values = [10, 50, 100, 500, 1000]
    print(f"{'kw (mD)':>10} {'Swr Sand':>12} {'Swr Carb':>12} {'Swr Shale':>12}")
    for kw in kw_values:
        swr_sand = SwrCorrelation.swr_from_permeability(kw, "sandstone")
        swr_carb = SwrCorrelation.swr_from_permeability(kw, "carbonate")
        swr_shale = SwrCorrelation.swr_from_permeability(kw, "shale")
        print(f"{kw:10d} {swr_sand:12.4f} {swr_carb:12.4f} {swr_shale:12.4f}")

    # Corey Model
    print("\n5. COREY RELATIVE PERMEABILITY MODEL")
    print("-" * 70)
    corey_gas = CoreyRelativePermeability(kr_max=0.90, n=2.0, Sgr=0.05, Swi=0.15, Sor=0.25)
    corey_water = CoreyRelativePermeability(kr_max=0.85, n=1.8, Sgr=0.05, Swi=0.15, Sor=0.25)

    Sg_values = np.linspace(0.05, 0.80, 8)
    print(f"{'Sg':>8} {'Sw':>8} {'kr_gas':>10} {'kr_water':>10}")
    for Sg in Sg_values:
        Sw = 1.0 - Sg
        kr_g = corey_gas.kr(Sg, phase="gas")
        kr_w = corey_water.kr(Sw, phase="water")
        print(f"{Sg:8.3f} {Sw:8.3f} {kr_g:10.4f} {kr_w:10.4f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    test()
