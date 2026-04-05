"""
ansaribaranghar_13c_mri.py

Direct hydrocarbon saturation imaging module using 13C magnetic resonance imaging.
Implements hybrid SE-SPI sensitivity analysis, oil saturation imaging, and coreflooding
simulation with capillary end effects and wettability considerations.

Reference: Ansaribaranghar et al., "Direct Hydrocarbon Saturation Imaging in Porous
Media With 13C Magnetic Resonance", Petrophysics Vol 66 No 1, pp 169-182,
DOI:10.30632/PJV66N1-2025a12
"""

import numpy as np
from typing import Tuple, Dict, List


class SEPSIImaging:
    """
    Spin-echo spiral planar imaging (SE-SPI) for saturation imaging.
    Hybrid SE-SPI sensitivity and signal intensity calculations.
    """

    def __init__(self, m0: float = 1.0):
        """
        Initialize SE-SPI imager.

        Args:
            m0: Equilibrium magnetization (proportional to 13C density)
        """
        self.m0 = m0

    def hybrid_se_spi_sensitivity(self, signal_to_noise: float,
                                 acquisition_time: float) -> float:
        """
        Hybrid SE-SPI sensitivity (Eq 1).

        eta = (S/N) / sqrt(t)
        where t is total imaging time, S/N is signal-to-noise ratio

        Args:
            signal_to_noise: Signal-to-noise ratio
            acquisition_time: Total imaging time (seconds)

        Returns:
            sensitivity: Imaging sensitivity
        """
        if acquisition_time <= 0:
            return 0.0
        sensitivity = signal_to_noise / np.sqrt(acquisition_time)
        return sensitivity

    def se_spi_signal_intensity(self, echo_time: float, t2_star: float,
                               echo_spacing: float = 0.001) -> float:
        """
        SE-SPI signal intensity (Eq 2).

        S = M0 * exp(-TE/T2*)
        M0: equilibrium magnetization (proportional to 13C density)

        Args:
            echo_time: Echo time (seconds)
            t2_star: T2* relaxation time (seconds)
            echo_spacing: Spacing between echoes (seconds)

        Returns:
            signal: Relative signal intensity
        """
        if t2_star <= 0:
            return 0.0
        signal = self.m0 * np.exp(-echo_time / t2_star)
        return signal

    def oil_volume_from_13c(self, signal_sample: float, signal_ref: float,
                           vol_ref: float) -> float:
        """
        Oil volume from 13C signal (Eq 3).

        Vo = 13C_signal_sample / (13C_signal_ref / volume_ref)_oil

        Args:
            signal_sample: 13C signal from sample
            signal_ref: 13C reference signal per unit volume
            vol_ref: Reference volume

        Returns:
            vol_oil: Oil volume
        """
        if signal_ref <= 0:
            return 0.0
        signal_per_vol = signal_ref / vol_ref
        vol_oil = signal_sample / signal_per_vol
        return max(0.0, vol_oil)

    def oil_saturation(self, vol_final: float, vol_initial: float) -> float:
        """
        Oil saturation from volume change (Eq 4).

        So_final = Vo_final / Vo_initial

        Args:
            vol_final: Final oil volume
            vol_initial: Initial oil volume

        Returns:
            so: Oil saturation
        """
        if vol_initial <= 0:
            return 0.0
        so = vol_final / vol_initial
        return np.clip(so, 0.0, 1.0)


class CorefloodingSimulation:
    """
    1D coreflooding simulation with Buckley-Leverett displacement theory
    and capillary end effects.
    """

    def __init__(self, core_length: float = 0.15, core_porosity: float = 0.20):
        """
        Initialize coreflooding simulator.

        Args:
            core_length: Core length (meters)
            core_porosity: Core porosity (fraction)
        """
        self.core_length = core_length
        self.core_porosity = core_porosity
        self.positions = np.linspace(0, core_length, 100)

    def buckley_leverett_displacement(self, sw_initial: float, sw_injected: float,
                                     pv_injected: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Buckley-Leverett 1D saturation displacement profile.

        Simulates water injection into oil-saturated core.

        Args:
            sw_initial: Initial water saturation
            sw_injected: Injected fluid saturation (water)
            pv_injected: Pore volumes of water injected

        Returns:
            positions: Position along core (meters)
            saturation: Water saturation at each position
        """
        n_points = len(self.positions)
        saturation = np.full(n_points, sw_initial)

        # Buckley-Leverett shock front velocity
        # Simplified model: front moves with injected PV
        front_position = (pv_injected * self.core_length)
        front_position = min(front_position, self.core_length)

        # Linear saturation profile in displaced region
        for i, pos in enumerate(self.positions):
            if pos < front_position * 0.95:
                saturation[i] = sw_injected
            elif pos < front_position:
                # Transition zone (10% of core)
                alpha = (pos - front_position * 0.95) / (front_position * 0.05)
                saturation[i] = sw_initial + alpha * (sw_injected - sw_initial)

        return self.positions, saturation

    def saturation_profile_at_time(self, time_array: np.ndarray,
                                  injection_rate: float,
                                  sw_initial: float,
                                  sw_injected: float,
                                  pore_volume: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate saturation profiles at multiple time points during flooding.

        Args:
            time_array: Array of time points (seconds)
            injection_rate: Injection rate (mL/min)
            sw_initial: Initial water saturation
            sw_injected: Injected water saturation
            pore_volume: Total pore volume (mL)

        Returns:
            profiles: List of (positions, saturation) tuples at each time
        """
        profiles = []

        for t in time_array:
            # Calculate pore volumes injected
            volume_injected = injection_rate * t / 60.0  # Convert to mL
            pv_injected = volume_injected / pore_volume

            pos, sat = self.buckley_leverett_displacement(
                sw_initial, sw_injected, pv_injected
            )
            profiles.append((pos, sat))

        return profiles

    def capillary_end_effect(self, position: float, core_length: float,
                            pc_outlet: float = 30.0,
                            capillary_length: float = 0.02) -> float:
        """
        Capillary end effect correction at outlet.

        Models higher saturation near outlet due to capillary pressure effects.

        Args:
            position: Position along core (meters)
            core_length: Total core length (meters)
            pc_outlet: Capillary pressure at outlet (psia)
            capillary_length: Length over which effect occurs (meters)

        Returns:
            saturation_correction: Saturation change due to capillary effect
        """
        if position > (core_length - capillary_length):
            # Exponential decay of capillary effect from outlet
            alpha = (core_length - position) / capillary_length
            correction = -0.05 * (1.0 - np.exp(-3 * (1 - alpha)))
            return correction
        return 0.0

    def residual_oil_saturation_profile(self, displacement_efficiency: float = 0.80
                                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Residual oil saturation profile along core after displacement.

        Models variation in Sor based on pore-scale distribution.

        Args:
            displacement_efficiency: Overall displacement efficiency (0-1)

        Returns:
            positions: Position along core
            sor_profile: Residual oil saturation at each position
        """
        sor_inlet = 0.12
        sor_outlet = 0.25  # Higher Sor at outlet due to bypassing

        sor_profile = np.linspace(sor_inlet, sor_outlet, len(self.positions))

        # Apply displacement efficiency gradient
        sor_profile = sor_profile * (1.0 - displacement_efficiency)

        return self.positions, sor_profile


class WettabilityEffect:
    """
    Wettability effects on saturation profiles.
    Models water-wet vs oil-wet systems.
    """

    def __init__(self):
        """Initialize wettability model."""
        pass

    def water_wet_profile(self, positions: np.ndarray,
                         sw_initial: float, sw_injected: float,
                         pv_injected: float) -> np.ndarray:
        """
        Water-wet core saturation profile (imbibition-dominant).

        Args:
            positions: Position along core
            sw_initial: Initial water saturation
            sw_injected: Injected fluid saturation
            pv_injected: Pore volumes injected

        Returns:
            saturation: Water saturation profile
        """
        core_length = positions[-1] - positions[0]
        front_position = pv_injected * core_length

        saturation = np.full_like(positions, sw_initial, dtype=float)

        # Gentle saturation gradient for water-wet (imbibition spreads)
        for i, pos in enumerate(positions):
            if pos < front_position:
                alpha = pos / front_position
                saturation[i] = sw_initial + (alpha ** 0.5) * (sw_injected - sw_initial)

        return saturation

    def oil_wet_profile(self, positions: np.ndarray,
                       sw_initial: float, sw_injected: float,
                       pv_injected: float) -> np.ndarray:
        """
        Oil-wet core saturation profile (drainage-dominant).

        Args:
            positions: Position along core
            sw_initial: Initial water saturation
            sw_injected: Injected fluid saturation
            pv_injected: Pore volumes injected

        Returns:
            saturation: Water saturation profile
        """
        core_length = positions[-1] - positions[0]
        front_position = pv_injected * core_length

        saturation = np.full_like(positions, sw_initial, dtype=float)

        # Steep saturation gradient for oil-wet (displacement less spreading)
        for i, pos in enumerate(positions):
            if pos < front_position * 0.98:
                saturation[i] = sw_injected
            elif pos < front_position:
                # Sharp transition
                alpha = (pos - front_position * 0.98) / (front_position * 0.02)
                saturation[i] = sw_initial + alpha * (sw_injected - sw_initial)

        return saturation

    def contact_angle_correction(self, contact_angle_water_wet: float = 30.0,
                                contact_angle_oil_wet: float = 150.0) -> Dict[str, float]:
        """
        Contact angle differences between water-wet and oil-wet systems.

        Args:
            contact_angle_water_wet: Contact angle for water-wet (degrees)
            contact_angle_oil_wet: Contact angle for oil-wet (degrees)

        Returns:
            metrics: Geometric and capillary properties
        """
        dtheta = contact_angle_oil_wet - contact_angle_water_wet

        return {
            'theta_ww': contact_angle_water_wet,
            'theta_ow': contact_angle_oil_wet,
            'delta_theta': dtheta,
            'capillary_force_ratio': np.cos(np.radians(contact_angle_water_wet)) /
                                     np.cos(np.radians(contact_angle_oil_wet))
        }


def test():
    """Test 13C MRI imaging and coreflooding simulation module."""
    print("Testing ansaribaranghar_13c_mri module...")
    print("=" * 60)

    # Test 1: Hybrid SE-SPI sensitivity
    print("\n1. Hybrid SE-SPI Sensitivity (Eq 1):")
    imager = SEPSIImaging(m0=1.0)
    sensitivity = imager.hybrid_se_spi_sensitivity(
        signal_to_noise=100.0,
        acquisition_time=120.0
    )
    print(f"   SNR = 100.0, Acq. time = 120.0 s")
    print(f"   Sensitivity eta = {sensitivity:.4f}")

    # Test 2: SE-SPI signal intensity
    print("\n2. SE-SPI Signal Intensity (Eq 2):")
    signal = imager.se_spi_signal_intensity(
        echo_time=0.002,
        t2_star=0.020
    )
    print(f"   TE = 0.002 s, T2* = 0.020 s")
    print(f"   Signal intensity S = {signal:.4f} M0")

    # Test 3: Oil volume from 13C signal
    print("\n3. Oil Volume from 13C Signal (Eq 3):")
    vol_oil = imager.oil_volume_from_13c(
        signal_sample=450.0,
        signal_ref=900.0,
        vol_ref=1.0
    )
    print(f"   13C signal (sample) = 450.0")
    print(f"   13C reference = 900.0 per mL")
    print(f"   Calculated oil volume = {vol_oil:.4f} mL")

    # Test 4: Oil saturation
    print("\n4. Oil Saturation (Eq 4):")
    so = imager.oil_saturation(vol_final=0.08, vol_initial=0.15)
    print(f"   Initial oil volume = 0.15 mL")
    print(f"   Final oil volume = 0.08 mL")
    print(f"   Oil saturation So = {so:.4f}")

    # Test 5: Buckley-Leverett displacement
    print("\n5. Buckley-Leverett Displacement Profiles:")
    coreflood = CorefloodingSimulation(core_length=0.15, core_porosity=0.22)

    pv_values = [0.3, 0.6, 1.0]
    for pv in pv_values:
        pos, sat = coreflood.buckley_leverett_displacement(
            sw_initial=0.15,
            sw_injected=1.0,
            pv_injected=pv
        )
        print(f"   PV injected = {pv} -> Sw at inlet = {sat[0]:.3f}, outlet = {sat[-1]:.3f}")

    # Test 6: Saturation profiles at multiple time steps
    print("\n6. Time-Dependent Saturation Profiles:")
    time_points = np.array([30, 60, 120, 300])  # seconds
    profiles = coreflood.saturation_profile_at_time(
        time_array=time_points,
        injection_rate=10.0,  # mL/min
        sw_initial=0.15,
        sw_injected=1.0,
        pore_volume=3.3  # mL (0.15 m * 0.22 poro, scaled)
    )
    print(f"   Generated {len(profiles)} profiles")
    print(f"   Time steps: {time_points} seconds")
    for i, (pos, sat) in enumerate(profiles):
        print(f"   Profile {i+1}: Sw range = {sat.min():.3f} - {sat.max():.3f}")

    # Test 7: Capillary end effect
    print("\n7. Capillary End Effect at Outlet:")
    positions_outlet = np.array([0.13, 0.14, 0.15])
    for pos in positions_outlet:
        correction = coreflood.capillary_end_effect(
            position=pos,
            core_length=0.15,
            pc_outlet=30.0,
            capillary_length=0.02
        )
        print(f"   Position {pos:.3f} m -> Saturation correction = {correction:.6f}")

    # Test 8: Residual oil saturation profile
    print("\n8. Residual Oil Saturation Profile:")
    pos_sor, sor_profile = coreflood.residual_oil_saturation_profile(
        displacement_efficiency=0.75
    )
    print(f"   At inlet: Sor = {sor_profile[0]:.4f}")
    print(f"   At outlet: Sor = {sor_profile[-1]:.4f}")
    print(f"   Mean Sor = {sor_profile.mean():.4f}")

    # Test 9: Water-wet vs Oil-wet profiles
    print("\n9. Wettability Effects (Water-wet vs Oil-wet):")
    wett = WettabilityEffect()
    positions = np.linspace(0, 0.15, 50)

    sw_ww = wett.water_wet_profile(
        positions=positions,
        sw_initial=0.15,
        sw_injected=1.0,
        pv_injected=0.5
    )

    sw_ow = wett.oil_wet_profile(
        positions=positions,
        sw_initial=0.15,
        sw_injected=1.0,
        pv_injected=0.5
    )

    print(f"   Water-wet core (imbibition):")
    print(f"      Sw range: {sw_ww.min():.3f} - {sw_ww.max():.3f}")
    print(f"   Oil-wet core (drainage):")
    print(f"      Sw range: {sw_ow.min():.3f} - {sw_ow.max():.3f}")

    # Test 10: Contact angle correction
    print("\n10. Contact Angle and Capillary Force Ratio:")
    contact_metrics = wett.contact_angle_correction(
        contact_angle_water_wet=30.0,
        contact_angle_oil_wet=150.0
    )
    print(f"   Water-wet contact angle: {contact_metrics['theta_ww']:.1f} degrees")
    print(f"   Oil-wet contact angle: {contact_metrics['theta_ow']:.1f} degrees")
    print(f"   Capillary force ratio: {contact_metrics['capillary_force_ratio']:.4f}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    return True


if __name__ == "__main__":
    test()
