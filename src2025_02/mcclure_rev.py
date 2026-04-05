"""
mcclure_rev.py

Representative Elementary Volume (REV) Analysis for Two-Phase Flow
McClure et al., Petrophysics Vol 66 No 1, pp 64-79, DOI:10.30632/PJV66N1-2025a5

Implements REV determination via time-space averaging framework, Darcy-scale property
calculation from pore-scale fields, and convergence analysis.
"""

import numpy as np


class PoreScaleField:
    """Synthetic pore-scale field generation for testing."""

    def __init__(self, domain_size=100, resolution=50):
        """
        Initialize pore-scale field.

        Args:
            domain_size: Physical domain size (um)
            resolution: Grid resolution (points per side)
        """
        self.domain_size = domain_size
        self.resolution = resolution
        self.dx = domain_size / resolution

        # Generate synthetic pore structure
        np.random.seed(42)
        self.porosity_field = self._generate_porosity_field()
        self.saturation_field = self._generate_saturation_field()
        self.pressure_field = self._generate_pressure_field()
        self.velocity_field = self._generate_velocity_field()

    def _generate_porosity_field(self):
        """Generate synthetic porosity field."""
        # Simple random porosity field
        raw = np.random.rand(self.resolution, self.resolution)
        pore_field = np.where(raw > 0.4, 1.0, 0.0)  # Binary pore/solid
        return pore_field

    def _generate_saturation_field(self):
        """Generate synthetic water saturation field."""
        base = np.ones((self.resolution, self.resolution)) * 0.5
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        X, Y = np.meshgrid(x, y)
        sat = base + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        sat = np.clip(sat, 0.2, 0.8)
        return sat * self.porosity_field

    def _generate_pressure_field(self):
        """Generate synthetic pressure field."""
        # Linear trend with small perturbations
        x = np.linspace(1.0, 0.0, self.resolution)
        y = np.linspace(0.5, 0.5, self.resolution)
        X, Y = np.meshgrid(x, y)
        pressure = 100000 * X + 1000 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        return pressure

    def _generate_velocity_field(self):
        """Generate synthetic velocity field."""
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        X, Y = np.meshgrid(x, y)

        # Divergence-free velocity field
        vx = -np.sin(np.pi * Y) * np.cos(2 * np.pi * X)
        vy = 0.5 * np.cos(np.pi * Y) * np.sin(2 * np.pi * X)

        # Scale by porosity
        vx = vx * self.porosity_field
        vy = vy * self.porosity_field

        return np.stack([vx, vy], axis=2)


class GaussianWeighting:
    """Gaussian weighting functions for spatial/temporal averaging."""

    @staticmethod
    def gaussian_weight_1d(r, sigma):
        """
        1D Gaussian weight: w(r) = exp(-r^2/(2*sigma^2)) / normalization

        Args:
            r: Distance (or array of distances)
            sigma: Standard deviation

        Returns:
            Weight value(s)
        """
        if sigma <= 0:
            return 0.0
        w = np.exp(-r**2 / (2.0 * sigma**2))
        # Normalization for 1D
        norm = np.sqrt(2 * np.pi) * sigma
        return w / norm

    @staticmethod
    def gaussian_weight_3d(r_mag, sigma):
        """
        3D Gaussian weight: w(r) = exp(-r^2/(2*sigma^2)) / normalization

        Args:
            r_mag: Distance magnitude
            sigma: Standard deviation

        Returns:
            Weight value
        """
        if sigma <= 0:
            return 0.0
        w = np.exp(-r_mag**2 / (2.0 * sigma**2))
        # Normalization for 3D
        norm = (2.0 * np.pi * sigma**2) ** 1.5
        return w / norm


class REVAnalysis:
    """Representative Elementary Volume analysis."""

    def __init__(self, pore_field):
        """
        Initialize REV analysis.

        Args:
            pore_field: PoreScaleField object
        """
        self.pore_field = pore_field
        self.resolution = pore_field.resolution
        self.domain_size = pore_field.domain_size

    def spatial_average(self, field, center_idx, sigma):
        """
        Spatial averaging with Gaussian weighting (vectorized).

        <phi>(x) = integral over domain of phi(r) * w(r - x) dr

        Args:
            field: 2D field array
            center_idx: (i, j) center indices
            sigma: Gaussian sigma in grid points

        Returns:
            Spatially averaged value
        """
        i_c, j_c = center_idx

        # Create coordinate grids
        i_grid, j_grid = np.meshgrid(np.arange(self.resolution),
                                     np.arange(self.resolution), indexing='ij')

        # Compute distances
        di = i_grid - i_c
        dj = j_grid - j_c
        r = np.sqrt(di**2 + dj**2)

        # Vectorized Gaussian weighting
        if sigma <= 0:
            return field[i_c, j_c]

        weights = np.exp(-r**2 / (2.0 * sigma**2))
        norm = np.sum(weights)

        if norm > 1e-12:
            return np.sum(field * weights) / norm
        else:
            return field[i_c, j_c]

    def temporal_average(self, time_series, sigma_t):
        """
        Temporal averaging with Gaussian weighting.

        <<phi>>(t) = integral over time of phi(tau) * w(tau - t) dtau

        Args:
            time_series: Array of values at different times
            sigma_t: Temporal Gaussian sigma (in time units)

        Returns:
            Temporally averaged value
        """
        n_time = len(time_series)
        t_c = n_time // 2

        weighted_sum = 0.0
        weight_sum = 0.0

        for t in range(n_time):
            dt = t - t_c
            weight = GaussianWeighting.gaussian_weight_1d(dt, sigma_t)
            weighted_sum += time_series[t] * weight
            weight_sum += weight

        if weight_sum > 1e-12:
            return weighted_sum / weight_sum
        else:
            return time_series[t_c]

    def calculate_porosity(self, roi_size):
        """
        Calculate porosity from pore-scale indicator function.

        phi = <indicator_function_pore>

        Args:
            roi_size: Region of interest size (grid points)

        Returns:
            Averaged porosity
        """
        i_c = self.resolution // 2
        j_c = self.resolution // 2
        return self.spatial_average(self.pore_field.porosity_field,
                                   (i_c, j_c), sigma=roi_size/2)

    def calculate_saturation(self, roi_size):
        """
        Calculate water saturation from pore-scale indicator.

        Sw = <indicator_water> / phi

        Args:
            roi_size: Region of interest size (grid points)

        Returns:
            Averaged water saturation
        """
        phi = self.calculate_porosity(roi_size)
        if phi < 1e-6:
            return 0.0

        sat_field = self.pore_field.saturation_field
        i_c = self.resolution // 2
        j_c = self.resolution // 2
        avg_sat = self.spatial_average(sat_field, (i_c, j_c), sigma=roi_size/2)

        return avg_sat / phi

    def calculate_velocity(self, roi_size, phase='water'):
        """
        Calculate phase velocity from averaged pore-scale velocity.

        v_phase = <v * indicator> / <indicator>

        Args:
            roi_size: Region of interest size (grid points)
            phase: 'water' or 'gas'

        Returns:
            Magnitude of phase velocity (m/s)
        """
        i_c = self.resolution // 2
        j_c = self.resolution // 2

        if phase == 'water':
            indicator = self.pore_field.saturation_field
        else:  # gas
            indicator = 1.0 - self.pore_field.saturation_field

        vel_field = self.pore_field.velocity_field
        vx_weighted = vel_field[:, :, 0] * indicator
        vy_weighted = vel_field[:, :, 1] * indicator

        avg_vx = self.spatial_average(vx_weighted, (i_c, j_c), sigma=roi_size/2)
        avg_vy = self.spatial_average(vy_weighted, (i_c, j_c), sigma=roi_size/2)
        avg_indicator = self.spatial_average(indicator, (i_c, j_c), sigma=roi_size/2)

        if avg_indicator > 1e-6:
            avg_vx /= avg_indicator
            avg_vy /= avg_indicator

        v_mag = np.sqrt(avg_vx**2 + avg_vy**2)
        return v_mag

    def calculate_pressure(self, roi_size):
        """
        Calculate averaged pressure from pore-scale field.

        p_phase = <p * indicator> / <indicator>

        Args:
            roi_size: Region of interest size (grid points)

        Returns:
            Averaged pressure (Pa)
        """
        i_c = self.resolution // 2
        j_c = self.resolution // 2
        return self.spatial_average(self.pore_field.pressure_field,
                                   (i_c, j_c), sigma=roi_size/2)

    def convergence_analysis(self, max_roi_size=None, num_points=15):
        """
        Analyze convergence of averaged properties with increasing averaging volume.

        Args:
            max_roi_size: Maximum ROI size (grid points)
            num_points: Number of sampling points

        Returns:
            Dictionary with convergence data
        """
        if max_roi_size is None:
            max_roi_size = self.resolution // 2

        roi_sizes = np.logspace(1, np.log10(max_roi_size), num_points).astype(int)
        roi_sizes = np.unique(roi_sizes)

        porosity_values = []
        saturation_values = []
        velocity_values = []
        pressure_values = []

        for roi_size in roi_sizes:
            porosity_values.append(self.calculate_porosity(roi_size))
            saturation_values.append(self.calculate_saturation(roi_size))
            velocity_values.append(self.calculate_velocity(roi_size))
            pressure_values.append(self.calculate_pressure(roi_size))

        return {
            'roi_sizes': roi_sizes,
            'porosity': np.array(porosity_values),
            'saturation': np.array(saturation_values),
            'velocity': np.array(velocity_values),
            'pressure': np.array(pressure_values)
        }

    def estimate_rev_size(self, property_name, convergence_threshold=0.05):
        """
        Estimate REV size based on convergence criterion.

        REV is where the property variation drops below threshold.

        Args:
            property_name: 'porosity', 'saturation', 'velocity', or 'pressure'
            convergence_threshold: Relative threshold for convergence (0.05 = 5%)

        Returns:
            Estimated REV size in micrometers
        """
        conv = self.convergence_analysis()
        property_data = conv[property_name]
        roi_sizes = conv['roi_sizes']

        # Check relative change
        for i in range(1, len(property_data)):
            relative_change = np.abs(property_data[i] - property_data[i-1]) / (
                np.abs(property_data[i-1]) + 1e-12)
            if relative_change < convergence_threshold:
                return roi_sizes[i] * self.pore_field.dx

        # If no convergence found, return last value
        return roi_sizes[-1] * self.pore_field.dx


def test():
    """Test function for REV analysis."""
    print("=" * 70)
    print("McClure et al. Representative Elementary Volume (REV) Analysis")
    print("=" * 70)

    # Test 1: Generate pore-scale field
    print("\n1. Pore-Scale Field Generation")
    print("-" * 70)
    pore_field = PoreScaleField(domain_size=100, resolution=256)
    print(f"Domain size: {pore_field.domain_size} um")
    print(f"Resolution: {pore_field.resolution} x {pore_field.resolution}")
    print(f"Grid spacing: {pore_field.dx:.3f} um")

    poro_bulk = np.mean(pore_field.porosity_field)
    sat_bulk = np.mean(pore_field.saturation_field)
    print(f"Bulk porosity: {poro_bulk:.3f}")
    print(f"Bulk water sat: {sat_bulk:.3f}")

    # Test 2: Gaussian weighting function
    print("\n2. Gaussian Weighting Function")
    print("-" * 70)
    r_values = np.array([0, 1, 2, 3, 5])
    sigma = 2.0
    print(f"Sigma: {sigma:.2f}")
    print("r       w(r)")
    for r in r_values:
        w = GaussianWeighting.gaussian_weight_1d(r, sigma)
        print(f"{r:2d}      {w:.6f}")

    # Test 3: REV analysis initialization
    print("\n3. REV Analysis Setup")
    print("-" * 70)
    rev = REVAnalysis(pore_field)
    print(f"REV analyzer initialized")
    print(f"Domain: {rev.domain_size} um x {rev.domain_size} um")
    print(f"Resolution: {rev.resolution} x {rev.resolution}")

    # Test 4: Spatial averaging at different scales
    print("\n4. Spatial Averaging at Different ROI Sizes")
    print("-" * 70)
    roi_sizes = [5, 10, 20, 40]
    print("ROI_size (pts)  Porosity  Saturation  Velocity (m/s)  Pressure (Pa)")
    for roi_size in roi_sizes:
        poro = rev.calculate_porosity(roi_size)
        sat = rev.calculate_saturation(roi_size)
        vel = rev.calculate_velocity(roi_size)
        pres = rev.calculate_pressure(roi_size)
        print(f"{roi_size:6d}          {poro:.4f}    {sat:.4f}        {vel:.6e}      {pres:.2e}")

    # Test 5: Convergence analysis
    print("\n5. Convergence Analysis")
    print("-" * 70)
    conv = rev.convergence_analysis(num_points=12)
    print("ROI_size (um)  Porosity  Saturation  Velocity (m/s)  Pressure (Pa)")
    for i, roi_size_pts in enumerate(conv['roi_sizes'][::2]):
        roi_size_um = roi_size_pts * pore_field.dx
        poro = conv['porosity'][i*2]
        sat = conv['saturation'][i*2]
        vel = conv['velocity'][i*2]
        pres = conv['pressure'][i*2]
        print(f"{roi_size_um:6.2f}          {poro:.4f}    {sat:.4f}        {vel:.6e}      {pres:.2e}")

    # Test 6: REV size estimation
    print("\n6. REV Size Estimation (5% convergence threshold)")
    print("-" * 70)
    rev_poro = rev.estimate_rev_size('porosity', convergence_threshold=0.05)
    rev_sat = rev.estimate_rev_size('saturation', convergence_threshold=0.05)
    rev_vel = rev.estimate_rev_size('velocity', convergence_threshold=0.05)
    rev_pres = rev.estimate_rev_size('pressure', convergence_threshold=0.05)

    print(f"REV (porosity):  {rev_poro:.2f} um")
    print(f"REV (saturation): {rev_sat:.2f} um")
    print(f"REV (velocity):   {rev_vel:.2f} um")
    print(f"REV (pressure):   {rev_pres:.2f} um")

    print("\n" + "=" * 70)
    print("Test completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    test()
