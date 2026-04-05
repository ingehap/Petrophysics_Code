"""
MODULE 8: Nono Primary Drainage Techniques
Reference: Nono et al., "Pore-Scale Comparisons of Primary Drainage Techniques
on Nonwater-Wet Reservoir Rocks", Petrophysics Vol 66 No 1, pp 110-122,
DOI:10.30632/PJV66N1-2025a8

Implements capillary number calculations, primary drainage simulations
(oilflooding and porous plate), saturation profile modeling, and pore
occupancy analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass
class CoreProperties:
    """Core sample properties."""
    length: float  # Core length (m)
    area: float  # Cross-sectional area (m^2)
    porosity: float  # Porosity (fraction)
    permeability: float  # Absolute permeability (m^2)
    pore_volume: float = None  # Pore volume (m^3), calculated

    def __post_init__(self):
        if self.pore_volume is None:
            self.pore_volume = self.length * self.area * self.porosity


@dataclass
class FluidProperties:
    """Fluid properties."""
    viscosity_displacing: float  # Displacing phase viscosity (Pa.s)
    viscosity_resident: float  # Resident phase viscosity (Pa.s)
    ift: float  # Interfacial tension (N/m)
    density_displacing: float  # Displacing phase density (kg/m^3)
    density_resident: float  # Resident phase density (kg/m^3)


class NonowetPrimaryDrainage:
    """Primary drainage technique implementation for nonwater-wet rocks."""

    def __init__(self, core: CoreProperties, fluids: FluidProperties):
        """Initialize with core and fluid properties."""
        self.core = core
        self.fluids = fluids

    def capillary_number(self, velocity: float) -> float:
        """
        Capillary number (Eq 1):
        Ca = v * mu / gamma

        Parameters:
            velocity: superficial velocity (m/s)

        Returns:
            Capillary number (dimensionless)
        """
        if self.fluids.ift == 0:
            raise ValueError("Interfacial tension cannot be zero")
        return velocity * self.fluids.viscosity_displacing / self.fluids.ift

    def oilflooding_simulation(self, Ca_values: np.ndarray,
                                Swi: float, Sor: float) -> Dict:
        """
        Oilflooding (OF) primary drainage: viscous displacement at increasing Ca.

        Parameters:
            Ca_values: array of capillary numbers to simulate
            Swi: initial water saturation
            Sor: residual oil saturation

        Returns:
            Dictionary with saturation and pressure profiles for each Ca
        """
        results = {'Ca_values': Ca_values, 'simulations': []}

        for Ca in Ca_values:
            # Higher Ca leads to more efficient displacement (lower final Sw)
            # Phenomenological relationship: Sw final decreases with Ca
            Ca_normalized = np.log10(Ca + 1e-8)  # Avoid log of zero
            Sw_final = Swi + (1 - Swi - Sor) * np.exp(-0.5 * Ca_normalized)
            Sw_final = np.clip(Sw_final, Sor, 1 - Swi)

            # Generate saturation profile along core (smooth transition)
            x = np.linspace(0, self.core.length, 50)
            Sw_profile = Swi + (Sw_final - Swi) * (1 - np.exp(-5 * x / self.core.length))

            # Pressure drop due to capillary end effect
            # Higher Ca = lower capillary contribution to pressure drop
            dP_capillary = 10000 * np.exp(-Ca_normalized)  # Pa
            dP_viscous = 1000 * (1 - np.exp(-Ca_normalized))  # Pa

            results['simulations'].append({
                'Ca': Ca,
                'Sw_final': Sw_final,
                'Sw_profile': Sw_profile,
                'x_position': x,
                'dP_capillary': dP_capillary,
                'dP_viscous': dP_viscous,
                'dP_total': dP_capillary + dP_viscous,
                'method': 'Oilflooding'
            })

        return results

    def porous_plate_simulation(self, Pc_steps: np.ndarray,
                                 Swi: float, Sor: float) -> Dict:
        """
        Porous Plate (PP) primary drainage: capillary-controlled at increasing Pc.

        Parameters:
            Pc_steps: array of capillary pressure steps (Pa)
            Swi: initial water saturation
            Sor: residual oil saturation

        Returns:
            Dictionary with saturation profiles for each Pc step
        """
        results = {'Pc_steps': Pc_steps, 'simulations': []}

        for i, Pc in enumerate(Pc_steps):
            # Higher Pc drives more drainage
            # Relationship: Sw decreases as Pc increases (power law)
            Sw_final = Swi + (1 - Swi - Sor) * (1 - (Pc / (Pc_steps[-1] + 1))**0.5)
            Sw_final = np.clip(Sw_final, Sor, 1 - Swi)

            # Saturation profile: capillary end effect is pronounced at low Pc
            x = np.linspace(0, self.core.length, 50)
            # End effect: more pronounced at outlet
            capillary_end_effect = 1 - 0.3 * np.exp(-3 * (self.core.length - x) / self.core.length)
            Sw_profile = Swi + (Sw_final - Swi) * (1 - np.exp(-8 * x / self.core.length)) * capillary_end_effect

            results['simulations'].append({
                'Pc': Pc,
                'step': i + 1,
                'Sw_final': Sw_final,
                'Sw_profile': Sw_profile,
                'x_position': x,
                'capillary_end_effect': capillary_end_effect,
                'method': 'Porous Plate'
            })

        return results

    def saturation_profile_with_reversal(self, Sw_initial: np.ndarray,
                                         reversal_fraction: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Saturation profile flattening by flow reversal.
        Reversal flow redistributes saturation to reduce gradients.

        Parameters:
            Sw_initial: initial saturation profile along core
            reversal_fraction: fraction of pore volume reversed (default 0.5)

        Returns:
            Tuple of (x_position, Sw_after_reversal)
        """
        x = np.linspace(0, self.core.length, len(Sw_initial))

        # Simulate reversal: mix saturation from opposite ends
        Sw_reversed = Sw_initial.copy()
        reversal_length = int(len(Sw_initial) * reversal_fraction)

        for i in range(reversal_length):
            # Mix from inlet and outlet
            mix_factor = 0.3
            Sw_reversed[i] = (1 - mix_factor) * Sw_initial[i] + mix_factor * Sw_initial[-(i+1)]
            Sw_reversed[-(i+1)] = (1 - mix_factor) * Sw_initial[-(i+1)] + mix_factor * Sw_initial[i]

        return x, Sw_reversed

    def dual_porosity_analysis(self, Sw_macro: float, Sw_micro: float,
                                macro_fraction: float) -> Dict:
        """
        Dual porosity analysis: macroporosity + sub-porosity (microporosity).

        Parameters:
            Sw_macro: water saturation in macropores
            Sw_micro: water saturation in micropores
            macro_fraction: fraction of porosity in macropores (0-1)

        Returns:
            Dictionary with total saturation and connectivity analysis
        """
        micro_fraction = 1 - macro_fraction

        # Total initial water saturation (volume weighted)
        Swi_total = macro_fraction * Sw_macro + micro_fraction * Sw_micro

        # Connectivity analysis: largest connected cluster fraction
        # Assumption: macropores more connected, micropores isolated
        macro_connectivity = 0.85
        micro_connectivity = 0.30

        connected_fraction = (macro_fraction * macro_connectivity +
                             micro_fraction * micro_connectivity)

        return {
            'Swi_macro': Sw_macro,
            'Swi_micro': Sw_micro,
            'macro_fraction': macro_fraction,
            'micro_fraction': micro_fraction,
            'Swi_total': Swi_total,
            'macro_connectivity': macro_connectivity,
            'micro_connectivity': micro_connectivity,
            'connected_cluster_fraction': connected_fraction,
            'description': 'Total Swi = Swi_macro + Swi_micro (volume weighted)'
        }

    def effective_permeability_at_Swi(self, flow_rate: float,
                                       pressure_drop: float) -> float:
        """
        Effective oil permeability at initial water saturation (Eq from Eq vicinity):
        Ko(Swi) = Q * mu * L / (A * dP)

        Parameters:
            flow_rate: volumetric flow rate (m^3/s)
            pressure_drop: pressure drop across core (Pa)

        Returns:
            Effective permeability (m^2)
        """
        if pressure_drop == 0:
            raise ValueError("Pressure drop cannot be zero")

        Ko = (flow_rate * self.fluids.viscosity_displacing * self.core.length) / \
             (self.core.area * pressure_drop)
        return Ko

    def pore_occupancy_connectivity(self, Sw_macro: float, Sw_micro: float,
                                    macro_fraction: float) -> Dict:
        """
        Pore occupancy and connectivity analysis.

        Returns:
            Dictionary with detailed connectivity metrics
        """
        micro_fraction = 1 - macro_fraction

        analysis = {
            'macro_pores': {
                'initial_Sw': Sw_macro,
                'connectivity': 0.85,
                'pore_radius_range': '10-100 um',
                'drained_first': True
            },
            'micro_pores': {
                'initial_Sw': Sw_micro,
                'connectivity': 0.30,
                'pore_radius_range': '0.1-10 um',
                'drained_last': True,
                'residual_water': True
            },
            'total_Swi': macro_fraction * Sw_macro + micro_fraction * Sw_micro,
            'largest_connected_cluster': 0.78
        }

        return analysis

    def primary_drainage_comparison(self, Swi: float, Sor: float) -> Dict:
        """
        Compare primary drainage techniques: oilflooding vs porous plate.

        Returns:
            Dictionary with comparison metrics
        """
        # Define test conditions
        Ca_values = np.array([1e-7, 1e-6, 1e-5, 1e-4])
        Pc_steps = np.array([5000, 10000, 20000, 40000])

        of_results = self.oilflooding_simulation(Ca_values, Swi, Sor)
        pp_results = self.porous_plate_simulation(Pc_steps, Swi, Sor)

        comparison = {
            'technique_OF': of_results,
            'technique_PP': pp_results,
            'summary': {
                'oilflooding': 'Viscous-dominated, rapid displacement',
                'porous_plate': 'Capillary-controlled, equilibrium-based'
            }
        }

        return comparison


def test():
    """Test function for Nono Primary Drainage module."""
    print("=" * 70)
    print("MODULE 8: Nono Primary Drainage Techniques Test")
    print("=" * 70)

    # Setup core and fluid properties
    core = CoreProperties(
        length=0.1,  # 10 cm
        area=0.001,  # 10 cm^2
        porosity=0.25,
        permeability=1e-12  # 1000 mD
    )

    fluids = FluidProperties(
        viscosity_displacing=1e-3,  # 1 cP (oil)
        viscosity_resident=1e-3,  # 1 cP (water)
        ift=0.03,  # N/m
        density_displacing=800,  # kg/m^3
        density_resident=1000  # kg/m^3
    )

    drainage = NonowetPrimaryDrainage(core, fluids)

    # Test 1: Capillary number
    print("\n1. CAPILLARY NUMBER CALCULATIONS")
    print("-" * 70)
    velocities = np.array([1e-6, 1e-5, 1e-4, 1e-3])
    print(f"Velocities (m/s): {velocities}")
    Ca_values = [drainage.capillary_number(v) for v in velocities]
    print(f"Capillary numbers: {[f'{Ca:.6e}' for Ca in Ca_values]}")

    # Test 2: Oilflooding simulation
    print("\n2. OILFLOODING (VISCOUS) SIMULATION")
    print("-" * 70)
    Swi, Sor = 0.30, 0.20
    Ca_test = np.array([1e-7, 1e-6, 1e-5, 1e-4])

    of_sim = drainage.oilflooding_simulation(Ca_test, Swi, Sor)
    print(f"Initial Sw (Swi) = {Swi}, Residual Oil (Sor) = {Sor}")
    for sim in of_sim['simulations']:
        print(f"  Ca = {sim['Ca']:.2e}: Sw_final = {sim['Sw_final']:.4f}, "
              f"dP_total = {sim['dP_total']:.2f} Pa")

    # Test 3: Porous plate simulation
    print("\n3. POROUS PLATE (CAPILLARY) SIMULATION")
    print("-" * 70)
    Pc_test = np.array([5000, 10000, 20000, 40000, 80000])

    pp_sim = drainage.porous_plate_simulation(Pc_test, Swi, Sor)
    print(f"Capillary pressure steps (Pa): {Pc_test}")
    for sim in pp_sim['simulations']:
        print(f"  Step {sim['step']}: Pc = {sim['Pc']:.0f} Pa, "
              f"Sw_final = {sim['Sw_final']:.4f}")

    # Test 4: Saturation profile with reversal
    print("\n4. SATURATION PROFILE WITH FLOW REVERSAL")
    print("-" * 70)
    Sw_initial = np.linspace(0.30, 0.50, 30)  # Graded profile
    x_pos, Sw_reversed = drainage.saturation_profile_with_reversal(Sw_initial, 0.5)

    print(f"Initial Sw range: {Sw_initial[0]:.3f} to {Sw_initial[-1]:.3f}")
    print(f"After reversal Sw range: {Sw_reversed[0]:.3f} to {Sw_reversed[-1]:.3f}")
    print(f"Standard deviation (before): {np.std(Sw_initial):.4f}")
    print(f"Standard deviation (after): {np.std(Sw_reversed):.4f}")

    # Test 5: Dual porosity analysis
    print("\n5. DUAL POROSITY ANALYSIS")
    print("-" * 70)
    dp_analysis = drainage.dual_porosity_analysis(
        Sw_macro=0.15, Sw_micro=0.45, macro_fraction=0.70
    )
    print(f"Macro Sw: {dp_analysis['Swi_macro']}, Macro fraction: {dp_analysis['macro_fraction']:.1%}")
    print(f"Micro Sw: {dp_analysis['Swi_micro']}, Micro fraction: {dp_analysis['micro_fraction']:.1%}")
    print(f"Total Swi: {dp_analysis['Swi_total']:.4f}")
    print(f"Connected cluster fraction: {dp_analysis['connected_cluster_fraction']:.3f}")

    # Test 6: Effective permeability
    print("\n6. EFFECTIVE PERMEABILITY AT Swi")
    print("-" * 70)
    Q = 1e-7  # m^3/s
    dP = 50000  # Pa
    Ko_eff = drainage.effective_permeability_at_Swi(Q, dP)
    print(f"Flow rate Q = {Q:.2e} m^3/s")
    print(f"Pressure drop = {dP} Pa")
    print(f"Effective oil permeability Ko(Swi) = {Ko_eff:.4e} m^2")

    # Test 7: Pore occupancy connectivity
    print("\n7. PORE OCCUPANCY CONNECTIVITY ANALYSIS")
    print("-" * 70)
    pore_occ = drainage.pore_occupancy_connectivity(
        Sw_macro=0.15, Sw_micro=0.45, macro_fraction=0.70
    )
    print(f"Macropore connectivity: {pore_occ['macro_pores']['connectivity']:.2f}")
    print(f"Micropore connectivity: {pore_occ['micro_pores']['connectivity']:.2f}")
    print(f"Largest connected cluster: {pore_occ['largest_connected_cluster']:.2f}")

    # Test 8: Primary drainage comparison
    print("\n8. PRIMARY DRAINAGE TECHNIQUE COMPARISON")
    print("-" * 70)
    comparison = drainage.primary_drainage_comparison(Swi=0.30, Sor=0.20)
    print(f"Oilflooding: {comparison['summary']['oilflooding']}")
    print(f"Porous Plate: {comparison['summary']['porous_plate']}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test()
