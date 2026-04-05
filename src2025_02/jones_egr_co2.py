"""
jones_egr_co2.py

Enhanced Gas Recovery by CO2 Injection
Jones et al., Petrophysics Vol 66 No 1, pp 52-63, DOI:10.30632/PJV66N1-2025a4

Implements Land trapping correlation, Corey relative permeability, 1D EGR displacement,
mixing zone analysis, and recovery factor calculation.
"""

import numpy as np


class LandTrapping:
    """Land trapping correlation for residual gas saturation."""

    def __init__(self, sgt_max=0.15):
        """
        Initialize Land trapping correlation.

        Args:
            sgt_max: Maximum trapped gas saturation (dimensionless)
        """
        self.sgt_max = sgt_max
        self.C = 1.0 / sgt_max - 1.0

    def trapped_gas(self, sgi):
        """
        Calculate trapped gas saturation from injected gas saturation.

        Sgt = Sgi / (1 + C*Sgi), where C = 1/Sgt_max - 1

        Args:
            sgi: Injected gas saturation (0 to 1)

        Returns:
            Trapped gas saturation
        """
        return sgi / (1.0 + self.C * sgi)


class CoreyRelPerm:
    """Corey relative permeability model for gas/brine system."""

    def __init__(self, kr_gas_end=0.8, kr_water_end=1.0, ng=2.0, nw=2.0,
                 swi=0.15, sgr=0.05):
        """
        Initialize Corey relative permeability model.

        Args:
            kr_gas_end: End-point relative permeability of gas
            kr_water_end: End-point relative permeability of water (brine)
            ng: Corey exponent for gas
            nw: Corey exponent for water
            swi: Irreducible water saturation
            sgr: Residual gas saturation
        """
        self.kr_gas_end = kr_gas_end
        self.kr_water_end = kr_water_end
        self.ng = ng
        self.nw = nw
        self.swi = swi
        self.sgr = sgr

    def kr_gas(self, sg):
        """
        Gas relative permeability: kr_gas = kr_gas_end * ((Sg-Sgr)/(1-Swi-Sgr))^ng

        Args:
            sg: Gas saturation

        Returns:
            Gas relative permeability
        """
        if sg <= self.sgr:
            return 0.0
        norm = (sg - self.sgr) / (1.0 - self.swi - self.sgr)
        norm = np.clip(norm, 0.0, 1.0)
        return self.kr_gas_end * (norm ** self.ng)

    def kr_water(self, sw):
        """
        Water relative permeability: kr_water = ((Sw-Swi)/(1-Swi))^nw

        Args:
            sw: Water saturation

        Returns:
            Water relative permeability
        """
        if sw <= self.swi:
            return 0.0
        norm = (sw - self.swi) / (1.0 - self.swi)
        norm = np.clip(norm, 0.0, 1.0)
        return self.kr_water_end * (norm ** self.nw)

    def kr_both(self, sg):
        """
        Return kr_gas and kr_water for gas saturation sg.

        Args:
            sg: Gas saturation

        Returns:
            (kr_gas, kr_water)
        """
        sw = 1.0 - sg
        return self.kr_gas(sg), self.kr_water(sw)


class EGRSimulation:
    """1D Enhanced Gas Recovery simulation with CO2 injection."""

    def __init__(self, length=0.1, porosity=0.2, permeability=100e-12,
                 mu_ch4=1.2e-5, mu_co2=1.5e-5, initial_ch4_sat=0.8,
                 core_relperm=None, land_trapping=None):
        """
        Initialize EGR simulation.

        Args:
            length: Core length (m)
            porosity: Porosity (fraction)
            permeability: Absolute permeability (m^2)
            mu_ch4: CH4 viscosity (Pa.s)
            mu_co2: CO2 viscosity (Pa.s)
            initial_ch4_sat: Initial CH4 saturation
            core_relperm: CoreyRelPerm object
            land_trapping: LandTrapping object
        """
        self.length = length
        self.porosity = porosity
        self.permeability = permeability
        self.mu_ch4 = mu_ch4
        self.mu_co2 = mu_co2
        self.sg_initial = initial_ch4_sat
        self.sw_initial = 1.0 - initial_ch4_sat

        if core_relperm is None:
            self.core_relperm = CoreyRelPerm()
        else:
            self.core_relperm = core_relperm

        if land_trapping is None:
            self.land_trapping = LandTrapping()
        else:
            self.land_trapping = land_trapping

    def fractional_flow_co2(self, sg):
        """
        CO2 fractional flow using Corey model.

        f_CO2 = 1 / (1 + (kr_CH4*mu_CO2)/(kr_CO2*mu_CH4))

        Args:
            sg: Gas (CO2) saturation

        Returns:
            CO2 fractional flow
        """
        kr_gas, kr_water = self.core_relperm.kr_both(sg)

        # kr_gas represents CO2 after injection
        # Initially we had CH4, now it's being displaced by CO2
        # Use gas mobility for CO2, water mobility for brine
        denominator = 1.0 + (kr_water * self.mu_co2) / (max(kr_gas, 1e-8) * self.mu_ch4)
        return 1.0 / denominator

    def simulate_displacement(self, num_pv=2.0, num_cells=50, num_steps=200):
        """
        1D Buckingham-Leverett style displacement.

        Args:
            num_pv: Number of pore volumes of CO2 injected
            num_cells: Number of grid cells
            num_steps: Number of time steps

        Returns:
            (x_positions, sg_profile, pv_injected_array, recovery_array)
        """
        x = np.linspace(0, self.length, num_cells)
        dx = self.length / (num_cells - 1)

        # Initialize: CH4 everywhere
        sg = np.full(num_cells, self.sg_initial)
        sg[0] = 0.99  # Injector boundary condition (CO2)

        pv_injected_list = []
        recovery_list = []
        time_array = np.linspace(0, num_pv, num_steps)

        # Total OGIP (original gas in place)
        ogip = self.porosity * self.length * self.sg_initial

        for step, pv in enumerate(time_array):
            pv_injected_list.append(pv)

            # Cumulative gas produced = OGIP - current gas in place
            current_gip = np.trapz(sg * self.porosity, x)
            gas_produced = ogip - current_gip
            rf = gas_produced / ogip if ogip > 0 else 0.0
            recovery_list.append(rf)

            # Simple upwind transport for next step
            if step < num_steps - 1:
                sg_new = sg.copy()
                dt_pv = (time_array[step + 1] - pv) * num_pv if step < num_steps - 1 else 0

                for i in range(1, num_cells):
                    # Fractional flow gradient
                    f_i = self.fractional_flow_co2(sg[i])
                    f_i_minus = self.fractional_flow_co2(sg[i - 1])

                    # Simple upwind: CO2 moves from left (injector)
                    flux = f_i_minus
                    sg_new[i] = sg[i] + (flux - f_i) * dt_pv * 0.1  # Scaled for stability
                    sg_new[i] = np.clip(sg_new[i], 0.0, 1.0)

                sg_new[0] = 0.99  # Maintain injector BC
                sg = sg_new

        return x, sg, np.array(pv_injected_list), np.array(recovery_list)

    def mixing_zone_analysis(self, sg_profile, x_positions):
        """
        Analyze CO2/CH4 mixing zone.

        Args:
            sg_profile: Gas saturation profile
            x_positions: Spatial positions

        Returns:
            Dictionary with mixing zone metrics
        """
        # Find mixing zone: where 0.1 < sg < 0.9
        mixing_mask = (sg_profile > 0.1) & (sg_profile < 0.9)

        if np.any(mixing_mask):
            mixing_indices = np.where(mixing_mask)[0]
            x_start = x_positions[mixing_indices[0]]
            x_end = x_positions[mixing_indices[-1]]
            mixing_length = x_end - x_start
        else:
            mixing_length = 0.0

        return {
            'mixing_length': mixing_length,
            'max_sg': np.max(sg_profile),
            'min_sg': np.min(sg_profile),
            'mean_sg': np.mean(sg_profile)
        }


def test():
    """Test function for EGR simulation."""
    print("=" * 70)
    print("Jones et al. Enhanced Gas Recovery (EGR) by CO2 Injection")
    print("=" * 70)

    # Test 1: Land trapping correlation
    print("\n1. Land Trapping Correlation")
    print("-" * 70)
    land = LandTrapping(sgt_max=0.15)
    sgi_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    print(f"Sgt_max = {land.sgt_max:.3f}, C = {land.C:.3f}")
    print("Sgi     Sgt")
    for sgi in sgi_values:
        sgt = land.trapped_gas(sgi)
        print(f"{sgi:.2f}    {sgt:.4f}")

    # Test 2: Corey relative permeability
    print("\n2. Corey Relative Permeability")
    print("-" * 70)
    corey = CoreyRelPerm(kr_gas_end=0.8, ng=2.0, nw=2.0, swi=0.15, sgr=0.05)
    sg_test = np.linspace(0.05, 1.0, 10)
    print("Sg      kr_gas  kr_water")
    for sg in sg_test:
        kr_g, kr_w = corey.kr_both(sg)
        print(f"{sg:.3f}  {kr_g:.4f}  {kr_w:.4f}")

    # Test 3: EGR displacement simulation
    print("\n3. EGR Displacement Simulation")
    print("-" * 70)
    egr = EGRSimulation(length=0.1, porosity=0.2, permeability=100e-12,
                        mu_ch4=1.2e-5, mu_co2=1.5e-5, initial_ch4_sat=0.8,
                        core_relperm=corey, land_trapping=land)

    x, sg_final, pv_inj, rf_array = egr.simulate_displacement(num_pv=2.0,
                                                                num_cells=30,
                                                                num_steps=100)

    print(f"Core length: {egr.length*1000:.1f} mm")
    print(f"Porosity: {egr.porosity:.1%}")
    print(f"Permeability: {egr.permeability*1e12:.1f} mD")
    print(f"\nSimulation results after {pv_inj[-1]:.2f} PV injected:")
    print(f"  Final recovery factor: {rf_array[-1]:.3f}")
    print(f"  Final avg gas sat: {np.mean(sg_final):.3f}")
    print(f"  Final min gas sat: {np.min(sg_final):.3f}")
    print(f"  Final max gas sat: {np.max(sg_final):.3f}")

    # Test 4: Mixing zone analysis
    print("\n4. Mixing Zone Analysis")
    print("-" * 70)
    mixing = egr.mixing_zone_analysis(sg_final, x)
    print(f"Mixing zone length: {mixing['mixing_length']*1000:.2f} mm")
    print(f"Max gas saturation: {mixing['max_sg']:.3f}")
    print(f"Min gas saturation: {mixing['min_sg']:.3f}")
    print(f"Mean gas saturation: {mixing['mean_sg']:.3f}")

    # Test 5: Recovery factor vs pore volumes
    print("\n5. Recovery Factor vs Pore Volumes Injected")
    print("-" * 70)
    print("PV_inj  RF")
    for pv, rf in zip(pv_inj[::10], rf_array[::10]):
        print(f"{pv:.2f}   {rf:.4f}")

    print("\n" + "=" * 70)
    print("Test completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    test()
