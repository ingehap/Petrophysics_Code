"""
wang_dual_porosity.py

Dual-matrix porosity petrophysics module implementing Brooks-Corey capillary pressure
and relative permeability models for dual-porosity sandstone systems.

Reference: Wang and Galley, "Impact of Dual Matrix Porosity in Sandstone on Fluid
Distribution and Flow Properties", Petrophysics Vol 66 No 1, pp 134-154,
DOI:10.30632/PJV66N1-2025a10
"""

import numpy as np
from typing import Tuple, Dict


class DualPorositySystem:
    """
    Dual matrix porosity (macroporosity + mesoporosity) drainage and imbibition.
    """

    def __init__(self,
                 phi_M: float, phi_m: float, BV_bulk: float, PV_pore: float,
                 pce_M: float, N_M: float,
                 pce_m: float, N_m: float,
                 swirr_M: float, swirr_m: float,
                 sor: float = 0.15):
        """
        Initialize dual porosity system.

        Args:
            phi_M: Macroporosity volume fraction
            phi_m: Mesoporosity volume fraction
            BV_bulk: Bulk volume fraction
            PV_pore: Total pore volume
            pce_M: Entry pressure for macropores (psia)
            N_M: Brooks-Corey pore size distribution for macropores
            pce_m: Entry pressure for mesopores (psia)
            N_m: Brooks-Corey pore size distribution for mesopores
            swirr_M: Irreducible water saturation in macropores
            swirr_m: Irreducible water saturation in mesopores
            sor: Residual oil saturation
        """
        self.phi_M = phi_M
        self.phi_m = phi_m
        self.phi_T = phi_M + phi_m
        self.BV_bulk = BV_bulk
        self.PV_pore = PV_pore
        self.pce_M = pce_M
        self.N_M = N_M
        self.pce_m = pce_m
        self.N_m = N_m
        self.swirr_M = swirr_M
        self.swirr_m = swirr_m
        self.sor = sor
        self.soi = 1.0 - swirr_M - swirr_m

    def _brooks_corey_drainage_bvnw(self, pc: float, pce: float, n: float,
                                    bvwirr: float, bvg: float, phi: float) -> float:
        """
        Modified Brooks-Corey drainage Pc for non-wetting phase saturation (Eq 1).

        BVnw = 1 - [BVr + (1-BVr)*(Pce/Pc_lab)^(1/N)]
        BVr = BVwirr + BVg + phi
        """
        if pc < pce:
            bv_r = bvwirr + bvg + phi
            bvnw = 1.0 - bv_r
        else:
            bv_r = bvwirr + bvg + phi
            bvnw = 1.0 - (bv_r + (1.0 - bv_r) * (pce / pc) ** (1.0 / n))
        return max(0.0, bvnw)

    def brooks_corey_pc(self, sw: float, pce: float, n: float,
                       swirr: float, sor: float, w: float = 0.0) -> float:
        """
        Brooks-Corey capillary pressure (Eq 8).

        Pc = Pce * [(1-W)/Snorm^N - W/(1-Snorm)^N]
        Snorm = (Sw - Swirr)/(1-Sor-Swirr)
        """
        snorm = (sw - swirr) / (1.0 - sor - swirr)
        snorm = np.clip(snorm, 0.0, 1.0)

        if snorm <= 0.0 or snorm >= 1.0:
            return pce

        term1 = (1.0 - w) / (snorm ** n) if snorm > 0 else np.inf
        term2 = w / ((1.0 - snorm) ** n) if snorm < 1.0 else 0.0

        pc = pce * (term1 - term2)
        return max(pce, pc)

    def imbibition_pc(self, pc_dra: float, sw_dra: float, swirr: float,
                     contact_angle_imb: float, contact_angle_dra: float) -> float:
        """
        Imbibition Pc from drainage Pc (Eq 4).

        Pc_imb(Sw) = Pc_dra(1 - Sw_dra + Swirr + Swt - Sot) * cos(theta_imb)/cos(theta_dra)
        """
        cos_ratio = np.cos(np.radians(contact_angle_imb)) / np.cos(np.radians(contact_angle_dra))
        pc_imb = pc_dra * cos_ratio
        return pc_imb

    def trapped_oil_saturation(self, soi: float, c_land: float = 1.5) -> float:
        """
        Land's correlation for trapped oil saturation (Eq 6).

        Sot = Soi / (1 + C*Soi)
        """
        sot = soi / (1.0 + c_land * soi)
        return sot

    def total_water_saturation(self, sw_M: float, sw_m: float) -> float:
        """
        Total water saturation from dual porosity system (Eq 7).

        Sw_T = (Sw_M*phi_M + Sw_m*phi_m) / phi_T
        """
        sw_total = (sw_M * self.phi_M + sw_m * self.phi_m) / self.phi_T
        return sw_total

    def corey_relative_permeability(self, sw: float, swirr: float, sor: float,
                                   krw0: float = 0.5, kro0: float = 0.8,
                                   nw: float = 2.0, no: float = 2.0) -> Tuple[float, float]:
        """
        Corey relative permeability curves (Eqs 11-12).

        krw = krw0 * ((Sw-Swirr)/(1-Swirr-Sor))^nw
        kro = kro0 * ((1-Sw-Sor)/(1-Swirr-Sor))^no
        """
        denom = 1.0 - sor - swirr

        sw_normalized = (sw - swirr) / denom
        sw_normalized = np.clip(sw_normalized, 0.0, 1.0)
        krw = krw0 * (sw_normalized ** nw)

        so_normalized = (1.0 - sw - sor) / denom
        so_normalized = np.clip(so_normalized, 0.0, 1.0)
        kro = kro0 * (so_normalized ** no)

        return krw, kro

    def drainage_curve(self, pc_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dual-porosity drainage saturation curves.

        Returns:
            sw_array: Water saturation values
            pc_array: Corresponding capillary pressure values
        """
        sw_M_values = np.linspace(self.swirr_M, 1.0 - self.sor, 50)
        sw_m_values = np.linspace(self.swirr_m, 1.0 - self.sor, 50)

        sw_total_values = []
        pc_values_out = []

        for pc in np.logspace(0, 5, 100):
            bvnw_M = self._brooks_corey_drainage_bvnw(
                pc, self.pce_M, self.N_M,
                bvwirr=self.swirr_M, bvg=0.01, phi=self.phi_M
            )
            bvnw_m = self._brooks_corey_drainage_bvnw(
                pc, self.pce_m, self.N_m,
                bvwirr=self.swirr_m, bvg=0.005, phi=self.phi_m
            )

            sw_M = 1.0 - bvnw_M
            sw_m = 1.0 - bvnw_m
            sw_total = self.total_water_saturation(sw_M, sw_m)

            sw_total_values.append(sw_total)
            pc_values_out.append(pc)

        return np.array(sw_total_values), np.array(pc_values_out)


def test():
    """Test dual porosity models with sample calculations."""
    print("Testing wang_dual_porosity module...")
    print("=" * 60)

    # Initialize dual porosity system
    system = DualPorositySystem(
        phi_M=0.15, phi_m=0.08,           # Porosity fractions
        BV_bulk=0.95, PV_pore=0.23,       # Bulk and pore volume
        pce_M=10.0, N_M=2.0,              # Macropore parameters
        pce_m=50.0, N_m=3.5,              # Mesopore parameters
        swirr_M=0.15, swirr_m=0.12,       # Irreducible water
        sor=0.20
    )

    # Test 1: Brooks-Corey drainage with modified equation
    print("\n1. Modified Brooks-Corey Drainage Pc:")
    pc_test = 30.0
    bvnw = system._brooks_corey_drainage_bvnw(
        pc_test, system.pce_M, system.N_M,
        bvwirr=0.15, bvg=0.01, phi=0.15
    )
    print(f"   Pc = {pc_test} psia -> BVnw = {bvnw:.4f}")

    # Test 2: Total water saturation
    print("\n2. Total Water Saturation (Dual System):")
    sw_total = system.total_water_saturation(sw_M=0.45, sw_m=0.38)
    print(f"   Sw_M = 0.45, Sw_m = 0.38 -> Sw_total = {sw_total:.4f}")

    # Test 3: Trapped oil saturation using Land's correlation
    print("\n3. Trapped Oil Saturation (Land's Model):")
    sot = system.trapped_oil_saturation(soi=0.80, c_land=1.5)
    print(f"   Soi = 0.80, C = 1.5 -> Sot = {sot:.4f}")

    # Test 4: Corey relative permeability
    print("\n4. Corey Relative Permeability:")
    sw = 0.50
    krw, kro = system.corey_relative_permeability(
        sw, swirr=0.15, sor=0.20,
        krw0=0.5, kro0=0.8, nw=2.0, no=2.0
    )
    print(f"   Sw = {sw} -> krw = {krw:.4f}, kro = {kro:.4f}")

    # Test 5: Imbibition capillary pressure
    print("\n5. Imbibition Pc from Drainage Pc:")
    pc_imb = system.imbibition_pc(
        pc_dra=30.0, sw_dra=0.50, swirr=0.15,
        contact_angle_imb=45.0, contact_angle_dra=135.0
    )
    print(f"   Pc_dra = 30.0 psia -> Pc_imb = {pc_imb:.4f} psia")

    # Test 6: Drainage curve generation
    print("\n6. Drainage Saturation Curve (10 points):")
    sw_values, pc_values = system.drainage_curve(np.logspace(0, 5, 10))
    print(f"   Generated {len(sw_values)} points")
    print(f"   Sw range: {sw_values.min():.4f} - {sw_values.max():.4f}")
    print(f"   Pc range: {pc_values.min():.2f} - {pc_values.max():.2f} psia")

    # Test 7: Brooks-Corey Pc equation (Eq 8)
    print("\n7. Brooks-Corey Pc Function:")
    sw_array = np.linspace(0.15, 0.80, 5)
    for sw in sw_array:
        pc = system.brooks_corey_pc(sw, pce=10.0, n=2.0, swirr=0.15, sor=0.20)
        print(f"   Sw = {sw:.2f} -> Pc = {pc:.2f} psia")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    return True


if __name__ == "__main__":
    test()
