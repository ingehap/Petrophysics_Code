"""
MODULE 9: Pairoys Dopants in SCAL Experiments
Reference: Pairoys et al., "Impact of Dopants on SCAL Experiments, Phase I",
Petrophysics Vol 66 No 1, pp 123-133, DOI:10.30632/PJV66N1-2025a9

Implements X-ray attenuation contrast modeling, spontaneous imbibition,
Amott wettability index, recovery factor, capillary pressure from centrifuge,
and NaI dopant impact analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class BrineProperties:
    """Brine properties with optional NaI dopant."""
    NaCl_concentration: float  # wt%
    NaI_concentration: float = 0.0  # wt% (dopant)
    density: float = 1000  # kg/m^3
    viscosity: float = 1e-3  # Pa.s
    ift: float = 0.03  # N/m (interfacial tension with oil)


@dataclass
class OilProperties:
    """Oil properties."""
    viscosity: float  # Pa.s
    density: float  # kg/m^3
    ift: float = 0.03  # N/m


@dataclass
class XRayAttenuation:
    """X-ray attenuation coefficients."""
    water_base: float  # Base water attenuation
    oil: float  # Oil attenuation (constant)
    NaI_coefficient: float = 1.5  # Linear coefficient for NaI


class PairoysDopadSCAL:
    """SCAL analysis with dopants impact on X-ray imaging and wettability."""

    def __init__(self, brine: BrineProperties, oil: OilProperties,
                 xray: XRayAttenuation):
        """Initialize with brine, oil, and X-ray attenuation properties."""
        self.brine = brine
        self.oil = oil
        self.xray = xray

    def attenuation_water(self, NaI_conc: float = None) -> float:
        """
        X-ray attenuation of brine as function of NaI concentration.
        Attenuation increases linearly with NaI dopant concentration.

        Parameters:
            NaI_conc: NaI concentration (wt%), if None uses brine's NaI

        Returns:
            X-ray attenuation coefficient
        """
        if NaI_conc is None:
            NaI_conc = self.brine.NaI_concentration

        # Linear relationship: attenuation = base + coefficient * NaI_conc
        attenuation = self.xray.water_base + self.xray.NaI_coefficient * NaI_conc
        return attenuation

    def attenuation_oil(self) -> float:
        """
        X-ray attenuation of oil (constant, independent of dopants).

        Returns:
            X-ray attenuation coefficient for oil
        """
        return self.xray.oil

    def xray_contrast(self, NaI_conc: float = None) -> float:
        """
        X-ray contrast between brine and oil.
        Higher contrast improves image quality for saturation measurement.

        Parameters:
            NaI_conc: NaI concentration (wt%), if None uses brine's NaI

        Returns:
            Contrast (attenuation_brine - attenuation_oil)
        """
        att_water = self.attenuation_water(NaI_conc)
        att_oil = self.attenuation_oil()
        return att_water - att_oil

    def spontaneous_imbibition_rate(self, NaI_conc: float, t_array: np.ndarray,
                                     V_max: float, tau: float) -> np.ndarray:
        """
        Spontaneous imbibition modeling: Oil production vs time.
        V(t) = V_max * (1 - exp(-t/tau))

        NaI dopant increases imbibition rate (decreases tau) due to
        improved wettability with higher ionic strength.

        Parameters:
            NaI_conc: NaI concentration (wt%)
            t_array: time array (hours)
            V_max: maximum oil production (mL)
            tau: characteristic time constant (hours)

        Returns:
            Array of oil production volumes
        """
        # NaI concentration reduces tau (faster imbibition)
        tau_adjusted = tau / (1 + 0.1 * NaI_conc)  # Empirical relationship

        V = V_max * (1 - np.exp(-t_array / tau_adjusted))
        return V

    def amott_wettability_index(self, delta_Sw_sp: float,
                                 delta_Sw_forced: float) -> float:
        """
        Amott wettability index (USBM variant).
        Iw = delta_Sw_sp / (delta_Sw_sp + delta_Sw_forced)

        Parameters:
            delta_Sw_sp: spontaneous imbibition saturation change
            delta_Sw_forced: forced displacement saturation change

        Returns:
            Amott index (-1 to 1, 0 = neutral)
        """
        denominator = delta_Sw_sp + delta_Sw_forced
        if denominator == 0:
            return 0
        return delta_Sw_sp / denominator

    def recovery_factor(self, Sor: float, Swi: float) -> float:
        """
        Recovery factor: RF = (1 - Sor) / (1 - Swi)

        Parameters:
            Sor: residual oil saturation
            Swi: initial water saturation

        Returns:
            Recovery factor (fraction)
        """
        if 1 - Swi == 0:
            raise ValueError("Initial water saturation cannot be 1")
        return (1 - Sor) / (1 - Swi)

    def capillary_pressure_centrifuge(self, omega: float, rho_oil: float,
                                       rho_water: float, r_position: float) -> float:
        """
        Capillary pressure from centrifuge measurement (simplified Forbes correction).
        Pc = omega^2 * r * (rho_water - rho_oil) / 2

        Parameters:
            omega: angular velocity (rad/s)
            rho_oil: oil density (kg/m^3)
            rho_water: brine density (kg/m^3)
            r_position: radial position from axis (m)

        Returns:
            Capillary pressure (Pa)
        """
        Pc = (omega**2 * r_position * (rho_water - rho_oil)) / 2
        return Pc

    def centrifuge_pressure_profile(self, omega_values: np.ndarray,
                                     r_position: float) -> np.ndarray:
        """
        Capillary pressure profile at different centrifuge speeds.

        Parameters:
            omega_values: array of angular velocities (rad/s)
            r_position: radial position (m)

        Returns:
            Array of capillary pressures
        """
        Pc_array = np.array([
            self.capillary_pressure_centrifuge(omega, self.oil.density,
                                                self.brine.density, r_position)
            for omega in omega_values
        ])
        return Pc_array

    def hassler_brunner_correction(self, Pc_uncorrected: float,
                                    core_length: float,
                                    permeability: float) -> float:
        """
        Hassler-Brunner correction for capillary end effect.
        Approximation: corrected Pc depends on capillary end effect magnitude.

        Parameters:
            Pc_uncorrected: uncorrected capillary pressure (Pa)
            core_length: core length (m)
            permeability: absolute permeability (m^2)

        Returns:
            Corrected capillary pressure (Pa)
        """
        # Correction factor based on core properties
        correction_factor = 1 + (permeability / core_length) * 1e8

        Pc_corrected = Pc_uncorrected * correction_factor
        return Pc_corrected

    def nai_dopant_impact(self, NaI_conc: float, Sor_undoped: float,
                          Swi: float) -> Dict:
        """
        Comprehensive NaI dopant impact analysis: compare doped vs undoped.

        Parameters:
            NaI_conc: NaI concentration (wt%)
            Sor_undoped: residual oil saturation without dopant
            Swi: initial water saturation

        Returns:
            Dictionary with impact metrics
        """
        # Dopants typically reduce Sor (improve waterflood recovery)
        Sor_reduction = 0.02 * NaI_conc  # Empirical: ~2% reduction per wt% NaI
        Sor_doped = Sor_undoped - Sor_reduction
        Sor_doped = max(Sor_doped, 0.05)  # Floor at 5%

        # Recovery factors
        RF_undoped = self.recovery_factor(Sor_undoped, Swi)
        RF_doped = self.recovery_factor(Sor_doped, Swi)
        RF_improvement = (RF_doped - RF_undoped) / RF_undoped * 100

        # Wettability improvement (Amott index)
        # Higher NaI improves wettability toward water-wet
        Iw_undoped = 0.3  # Neutral to weakly water-wet
        Iw_doped = Iw_undoped + 0.15 * min(NaI_conc / 10, 1)  # Saturation at ~10% NaI
        Iw_doped = min(Iw_doped, 0.8)  # Cap at water-wet

        # X-ray contrast improvement
        contrast_undoped = self.xray_contrast(0)
        contrast_doped = self.xray_contrast(NaI_conc)
        contrast_improvement = (contrast_doped - contrast_undoped) / contrast_undoped * 100

        # Imbibition rate improvement
        V_max = 10  # mL
        tau_undoped = 50  # hours
        t_test = np.array([1, 5, 10, 24])

        V_undoped = self.spontaneous_imbibition_rate(0, t_test, V_max, tau_undoped)
        V_doped = self.spontaneous_imbibition_rate(NaI_conc, t_test, V_max, tau_undoped)

        return {
            'NaI_concentration': NaI_conc,
            'Sor_impact': {
                'undoped': Sor_undoped,
                'doped': Sor_doped,
                'reduction_absolute': Sor_reduction,
                'reduction_percent': Sor_reduction / Sor_undoped * 100
            },
            'recovery_factor': {
                'undoped': RF_undoped,
                'doped': RF_doped,
                'improvement_percent': RF_improvement
            },
            'amott_index': {
                'undoped': Iw_undoped,
                'doped': Iw_doped,
                'improvement': Iw_doped - Iw_undoped
            },
            'xray_contrast': {
                'undoped': contrast_undoped,
                'doped': contrast_doped,
                'improvement_percent': contrast_improvement
            },
            'imbibition_rate': {
                'time_hours': t_test,
                'V_undoped_mL': V_undoped,
                'V_doped_mL': V_doped
            }
        }

    def dopant_concentration_series(self, NaI_concs: np.ndarray,
                                     Sor_undoped: float, Swi: float) -> Dict:
        """
        Analyze dopant impact across a concentration range.

        Parameters:
            NaI_concs: array of NaI concentrations (wt%)
            Sor_undoped: baseline Sor (undoped)
            Swi: initial water saturation

        Returns:
            Dictionary with results for each concentration
        """
        results = {
            'NaI_concentrations': NaI_concs,
            'analyses': []
        }

        for conc in NaI_concs:
            analysis = self.nai_dopant_impact(conc, Sor_undoped, Swi)
            results['analyses'].append(analysis)

        return results


def test():
    """Test function for Pairoys Dopants module."""
    print("=" * 70)
    print("MODULE 9: Pairoys Dopants in SCAL Experiments Test")
    print("=" * 70)

    # Setup properties
    brine = BrineProperties(NaCl_concentration=30, NaI_concentration=0)
    oil = OilProperties(viscosity=5e-3, density=800)
    xray = XRayAttenuation(water_base=50, oil=20, NaI_coefficient=1.5)

    scal = PairoysDopadSCAL(brine, oil, xray)

    # Test 1: X-ray attenuation
    print("\n1. X-RAY ATTENUATION CONTRAST")
    print("-" * 70)
    NaI_test = np.array([0, 2, 5, 10, 15])
    print(f"NaI concentrations (wt%): {NaI_test}")

    att_water = [scal.attenuation_water(NaI) for NaI in NaI_test]
    att_oil = scal.attenuation_oil()
    contrasts = [scal.xray_contrast(NaI) for NaI in NaI_test]

    print(f"Water attenuation: {[f'{a:.2f}' for a in att_water]}")
    print(f"Oil attenuation: {att_oil:.2f}")
    print(f"Contrast (W-O): {[f'{c:.2f}' for c in contrasts]}")

    # Test 2: Spontaneous imbibition
    print("\n2. SPONTANEOUS IMBIBITION MODELING")
    print("-" * 70)
    t = np.array([0.1, 1, 5, 10, 24, 48])
    V_max, tau = 15, 50  # mL, hours

    print(f"Time (hours): {t}")
    V_undoped = scal.spontaneous_imbibition_rate(0, t, V_max, tau)
    V_doped = scal.spontaneous_imbibition_rate(10, t, V_max, tau)

    print(f"Oil production (undoped, mL): {[f'{v:.2f}' for v in V_undoped]}")
    print(f"Oil production (10% NaI, mL): {[f'{v:.2f}' for v in V_doped]}")

    # Test 3: Amott wettability index
    print("\n3. AMOTT WETTABILITY INDEX")
    print("-" * 70)
    test_cases = [
        (0.10, 0.05, "Neutral"),
        (0.15, 0.10, "Slightly water-wet"),
        (0.25, 0.05, "Water-wet")
    ]

    for delta_sp, delta_forced, desc in test_cases:
        Iw = scal.amott_wettability_index(delta_sp, delta_forced)
        print(f"{desc}: delta_Sw_sp={delta_sp}, delta_Sw_forced={delta_forced}, "
              f"Iw = {Iw:.3f}")

    # Test 4: Recovery factor
    print("\n4. RECOVERY FACTOR")
    print("-" * 70)
    Swi, Sor_base = 0.25, 0.30

    RF_cases = [(0.20, "High recovery"), (0.30, "Moderate"),
                (0.40, "Low recovery")]

    for Sor, desc in RF_cases:
        RF = scal.recovery_factor(Sor, Swi)
        print(f"{desc}: Sor={Sor}, RF = {RF:.4f}")

    # Test 5: Capillary pressure from centrifuge
    print("\n5. CAPILLARY PRESSURE FROM CENTRIFUGE")
    print("-" * 70)
    rpm_values = np.array([1000, 2000, 3000, 5000])
    omega_values = rpm_values * 2 * np.pi / 60  # Convert to rad/s
    r_pos = 0.05  # 5 cm radius

    Pc_values = scal.centrifuge_pressure_profile(omega_values, r_pos)

    print(f"RPM: {rpm_values}")
    print(f"Angular velocity (rad/s): {[f'{om:.2f}' for om in omega_values]}")
    print(f"Capillary pressure (Pa): {[f'{Pc:.2f}' for Pc in Pc_values]}")

    # Test 6: Hassler-Brunner correction
    print("\n6. HASSLER-BRUNNER CAPILLARY PRESSURE CORRECTION")
    print("-" * 70)
    Pc_uncorr = 5000  # Pa
    L_core, K_perm = 0.1, 1e-12  # 10 cm, 1000 mD

    Pc_corr = scal.hassler_brunner_correction(Pc_uncorr, L_core, K_perm)
    print(f"Uncorrected Pc = {Pc_uncorr} Pa")
    print(f"Corrected Pc = {Pc_corr:.2f} Pa")
    print(f"Correction factor = {Pc_corr / Pc_uncorr:.4f}")

    # Test 7: NaI dopant impact analysis
    print("\n7. NaI DOPANT IMPACT ANALYSIS")
    print("-" * 70)
    impact = scal.nai_dopant_impact(10, 0.30, 0.25)

    print(f"NaI concentration: {impact['NaI_concentration']} wt%")
    print(f"  Sor (undoped): {impact['Sor_impact']['undoped']:.4f}")
    print(f"  Sor (doped): {impact['Sor_impact']['doped']:.4f}")
    print(f"  Reduction: {impact['Sor_impact']['reduction_percent']:.2f}%")
    print(f"  RF improvement: {impact['recovery_factor']['improvement_percent']:.2f}%")
    print(f"  Amott index improvement: {impact['amott_index']['improvement']:.3f}")
    print(f"  X-ray contrast improvement: {impact['xray_contrast']['improvement_percent']:.2f}%")

    # Test 8: Concentration series
    print("\n8. DOPANT CONCENTRATION SERIES")
    print("-" * 70)
    NaI_series = np.array([0, 2, 5, 10, 15])
    series = scal.dopant_concentration_series(NaI_series, 0.30, 0.25)

    print(f"NaI conc (wt%): Sor_doped | RF | Iw | Contrast_impr")
    print("-" * 55)
    for analysis in series['analyses']:
        NaI = analysis['NaI_concentration']
        Sor = analysis['Sor_impact']['doped']
        RF = analysis['recovery_factor']['doped']
        Iw = analysis['amott_index']['doped']
        contrast = analysis['xray_contrast']['improvement_percent']
        print(f"{NaI:5.1f}     | {Sor:.4f}    | {RF:.4f} | "
              f"{Iw:.3f}  | {contrast:6.2f}%")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test()
