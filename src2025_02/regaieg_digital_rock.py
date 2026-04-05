"""
regaieg_digital_rock.py

Digital Rock Physics for Relative Permeability Prediction
Regaieg et al., Petrophysics Vol 66 No 1, pp 80-94, DOI:10.30632/PJV66N1-2025a6

Implements Pore Network Model (PNM), invasion percolation drainage, Corey model
fitting, Buckley-Leverett fractional flow, and Monte Carlo uncertainty analysis.
"""

import numpy as np


class PoreNetworkModel:
    """Simple pore network model with conductance-based pressure solution."""

    def __init__(self, num_pores=100, network_type='random'):
        """
        Initialize pore network model.

        Args:
            num_pores: Number of pores in network
            network_type: 'random', 'grid', or 'bethe'
        """
        self.num_pores = num_pores
        self.network_type = network_type

        # Generate network connectivity
        if network_type == 'random':
            self.connectivity = self._random_network()
        elif network_type == 'grid':
            self.connectivity = self._grid_network()
        else:
            self.connectivity = self._bethe_network()

        # Pore properties
        self.pore_radius = np.random.uniform(10, 50, num_pores)  # micrometers
        self.pore_volume = (4/3) * np.pi * self.pore_radius**3

        # Throat properties
        self.throat_radius = {}
        self.throat_length = {}
        self._initialize_throats()

    def _random_network(self):
        """Random network connectivity."""
        # Random pore pairs with ~2 connections per pore on average
        connectivity = []
        for i in range(self.num_pores):
            degree = np.random.poisson(2)
            for _ in range(degree):
                j = np.random.randint(0, self.num_pores)
                if i != j:
                    connectivity.append((min(i, j), max(i, j)))

        # Remove duplicates
        connectivity = list(set(connectivity))
        return connectivity

    def _grid_network(self):
        """2D grid network."""
        grid_side = int(np.sqrt(self.num_pores))
        connectivity = []

        for i in range(grid_side):
            for j in range(grid_side):
                pore_idx = i * grid_side + j
                if pore_idx >= self.num_pores:
                    break

                # Right neighbor
                if j < grid_side - 1:
                    connectivity.append((pore_idx, pore_idx + 1))
                # Bottom neighbor
                if i < grid_side - 1:
                    connectivity.append((pore_idx, pore_idx + grid_side))

        return connectivity

    def _bethe_network(self):
        """Bethe lattice-like network."""
        connectivity = []
        coordination = 4  # Coordination number

        for i in range(self.num_pores):
            for _ in range(coordination // 2):
                j = (i + np.random.randint(1, self.num_pores)) % self.num_pores
                connectivity.append((min(i, j), max(i, j)))

        connectivity = list(set(connectivity))
        return connectivity

    def _initialize_throats(self):
        """Initialize throat radii and lengths."""
        for pore_i, pore_j in self.connectivity:
            throat_key = (pore_i, pore_j)
            # Throat radius is minimum of connected pores
            r_throat = min(self.pore_radius[pore_i], self.pore_radius[pore_j])
            self.throat_radius[throat_key] = r_throat * np.random.uniform(0.5, 1.0)
            self.throat_length[throat_key] = np.random.uniform(50, 150)

    def conductance(self, pore_i, pore_j, viscosity=1e-3):
        """
        Hagen-Poiseuille conductance: g = pi*r^4 / (8*mu*L)

        Args:
            pore_i, pore_j: Pore indices
            viscosity: Fluid viscosity (Pa.s)

        Returns:
            Conductance (m^3/Pa.s)
        """
        throat_key = (min(pore_i, pore_j), max(pore_i, pore_j))

        if throat_key not in self.throat_radius:
            return 0.0

        r = self.throat_radius[throat_key] * 1e-6  # Convert to meters
        L = self.throat_length[throat_key] * 1e-6
        g = np.pi * r**4 / (8.0 * viscosity * L)

        return g

    def solve_pressure(self, flow_rate=1e-10, inlet_pore=0, outlet_pore=None):
        """
        Solve pressure field using conductance matrix with simple iterative solver.

        Sum of flows at each node = 0 (mass conservation)
        Q_ij = g_ij * (p_i - p_j)

        Args:
            flow_rate: Injection flow rate (m^3/s)
            inlet_pore: Inlet pore index
            outlet_pore: Outlet pore index

        Returns:
            Pressure array (Pa)
        """
        if outlet_pore is None:
            outlet_pore = self.num_pores - 1

        # Build conductance matrix (dense for simplicity)
        G = np.zeros((self.num_pores, self.num_pores))

        for pore_i, pore_j in self.connectivity:
            g = self.conductance(pore_i, pore_j)
            G[pore_i, pore_j] = -g
            G[pore_j, pore_i] = -g
            G[pore_i, pore_i] += g
            G[pore_j, pore_j] += g

        # Boundary conditions
        G[inlet_pore, :] = 0
        G[inlet_pore, inlet_pore] = 1.0

        G[outlet_pore, :] = 0
        G[outlet_pore, outlet_pore] = 1.0

        # Right-hand side
        b = np.zeros(self.num_pores)
        b[inlet_pore] = 100000.0  # Inlet pressure 100 kPa
        b[outlet_pore] = 0.0      # Outlet at 0 Pa

        # Solve using simple Gaussian elimination
        try:
            pressure = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            # If singular, use least squares
            pressure = np.linalg.lstsq(G, b, rcond=None)[0]

        return np.asarray(pressure).flatten()


class InvasionPercolation:
    """Invasion percolation for drainage (CO2 displaces water)."""

    def __init__(self, pnm, interfacial_tension=0.05, contact_angle=0):
        """
        Initialize invasion percolation.

        Args:
            pnm: PoreNetworkModel object
            interfacial_tension: IFT between phases (N/m)
            contact_angle: Contact angle (degrees)
        """
        self.pnm = pnm
        self.interfacial_tension = interfacial_tension
        self.contact_angle = np.radians(contact_angle)

    def capillary_entry_pressure(self, throat_idx):
        """
        Capillary entry pressure: Pc = 2*sigma*cos(theta) / r

        Args:
            throat_idx: (pore_i, pore_j) tuple

        Returns:
            Entry pressure (Pa)
        """
        pore_i, pore_j = throat_idx
        r_throat = self.pnm.throat_radius.get(throat_idx, 20e-6)
        if r_throat == 0:
            r_throat = 20e-6

        r_throat_m = r_throat * 1e-6
        pc = 2.0 * self.interfacial_tension * np.cos(self.contact_angle) / r_throat_m

        return pc

    def invasion_sequence(self):
        """
        Sequential pore filling by invading phase during drainage.

        Returns:
            List of invaded pores in order
        """
        invaded = set()
        invasion_order = []

        # Sort throats by entry pressure
        throat_pressures = []
        for pore_i, pore_j in self.pnm.connectivity:
            pc = self.capillary_entry_pressure((pore_i, pore_j))
            throat_pressures.append(((pore_i, pore_j), pc))

        throat_pressures.sort(key=lambda x: x[1])

        # Start invasion from pore 0
        invaded.add(0)
        invasion_order.append(0)

        # Invade pores connected to invaded region
        for (pore_i, pore_j), pc in throat_pressures:
            if pore_i in invaded and pore_j not in invaded:
                invaded.add(pore_j)
                invasion_order.append(pore_j)
            elif pore_j in invaded and pore_i not in invaded:
                invaded.add(pore_i)
                invasion_order.append(pore_i)

        return invasion_order


class RelativePermeability:
    """Relative permeability from pore-scale dynamics."""

    @staticmethod
    def corey_kr(so, sor, swi, kr_o_end=1.0, kr_w_end=1.0, no=2.0, nw=2.0):
        """
        Corey relative permeability model.

        kr_oil = kr_oil_end * ((So-Sor)/(1-Swi-Sor))^no
        kr_water = kr_water_end * ((Sw-Swi)/(1-Swi-Sor))^nw

        Args:
            so: Oil saturation
            sor: Residual oil saturation
            swi: Irreducible water saturation
            kr_o_end: End-point kr for oil
            kr_w_end: End-point kr for water
            no, nw: Corey exponents

        Returns:
            (kr_oil, kr_water)
        """
        sw = 1.0 - so

        # Oil relative permeability
        if so <= sor:
            kr_o = 0.0
        else:
            norm_o = (so - sor) / (1.0 - swi - sor)
            kr_o = kr_o_end * (np.clip(norm_o, 0, 1) ** no)

        # Water relative permeability
        if sw <= swi:
            kr_w = 0.0
        else:
            norm_w = (sw - swi) / (1.0 - swi - sor)
            kr_w = kr_w_end * (np.clip(norm_w, 0, 1) ** nw)

        return kr_o, kr_w

    @staticmethod
    def fractional_flow(so, sor, swi, kr_o_end=1.0, kr_w_end=1.0,
                       no=2.0, nw=2.0, mu_o=1e-3, mu_w=1e-3):
        """
        Buckley-Leverett fractional flow.

        fw = 1 / (1 + (kr_o*mu_w)/(kr_w*mu_o))

        Args:
            so: Oil saturation
            sor, swi: Residual saturations
            kr_o_end, kr_w_end: End-point relative permeabilities
            no, nw: Corey exponents
            mu_o, mu_w: Viscosities (Pa.s)

        Returns:
            Water fractional flow
        """
        kr_o, kr_w = RelativePermeability.corey_kr(
            so, sor, swi, kr_o_end, kr_w_end, no, nw)

        if kr_w < 1e-10:
            return 0.0

        denominator = 1.0 + (kr_o * mu_w) / (kr_w * mu_o)
        fw = 1.0 / denominator

        return np.clip(fw, 0, 1)

    @staticmethod
    def dfw_dso(so, sor, swi, kr_o_end=1.0, kr_w_end=1.0,
               no=2.0, nw=2.0, mu_o=1e-3, mu_w=1e-3):
        """
        Derivative of water fractional flow.

        dfw/dSo = derivative of fractional flow

        Args:
            so: Oil saturation
            Others: Same as fractional_flow

        Returns:
            dfw/dSo
        """
        dso = 1e-4
        f_plus = RelativePermeability.fractional_flow(
            so + dso, sor, swi, kr_o_end, kr_w_end, no, nw, mu_o, mu_w)
        f_minus = RelativePermeability.fractional_flow(
            so - dso, sor, swi, kr_o_end, kr_w_end, no, nw, mu_o, mu_w)

        return (f_plus - f_minus) / (2.0 * dso)

    @staticmethod
    def welge_tangent(so_array, kr_o_end=1.0, kr_w_end=1.0, no=2.0, nw=2.0,
                     mu_o=1e-3, mu_w=1e-3):
        """
        Welge tangent construction for shock front position.

        Finds saturation where tangent to fw curve passes through (Swi, 0).

        Args:
            so_array: Array of oil saturations
            Others: Corey parameters and viscosities

        Returns:
            Shock front saturation
        """
        max_slope = -1.0
        shock_so = so_array[0]

        sor = 0.05
        swi = 0.15

        for so in so_array:
            fw = RelativePermeability.fractional_flow(
                so, sor, swi, kr_o_end, kr_w_end, no, nw, mu_o, mu_w)
            dfw = RelativePermeability.dfw_dso(
                so, sor, swi, kr_o_end, kr_w_end, no, nw, mu_o, mu_w)

            # Welge: tangent intercepts Sw=Swi at fw=0
            if dfw > 1e-6:
                slope = fw / (so - swi) if (so - swi) > 1e-6 else 0
                if slope > max_slope:
                    max_slope = slope
                    shock_so = so

        return shock_so


class MonteCarloUncertainty:
    """Monte Carlo analysis of kr uncertainty."""

    @staticmethod
    def sample_pore_sizes(num_samples=100, mean_radius=30, std_radius=10):
        """
        Sample pore size distribution.

        Args:
            num_samples: Number of Monte Carlo samples
            mean_radius: Mean pore radius (micrometers)
            std_radius: Std dev of pore radius

        Returns:
            Array of pore radius realizations
        """
        samples = np.random.normal(mean_radius, std_radius, (num_samples, 50))
        samples = np.clip(samples, 5, 100)
        return samples

    @staticmethod
    def compute_kr_realizations(pore_size_samples, sor=0.05, swi=0.15):
        """
        Generate kr curves from pore size samples.

        Args:
            pore_size_samples: (num_samples, num_pores) array
            sor, swi: Residual saturations

        Returns:
            (so_array, kr_array) where kr_array is (num_samples, len(so_array))
        """
        num_samples = pore_size_samples.shape[0]
        so_array = np.linspace(sor, 1.0 - swi, 20)
        kr_samples = np.zeros((num_samples, len(so_array)))

        for i in range(num_samples):
            # Vary exponents based on pore size distribution
            std_norm = np.std(pore_size_samples[i])
            no = 2.0 + std_norm / 30.0  # Higher std -> lower exponent
            nw = 2.0 + std_norm / 30.0

            for j, so in enumerate(so_array):
                kr_o, kr_w = RelativePermeability.corey_kr(
                    so, sor, swi, kr_o_end=1.0, kr_w_end=1.0,
                    no=no, nw=nw)
                kr_samples[i, j] = kr_w

        return so_array, kr_samples

    @staticmethod
    def percentiles(kr_array, percentiles_list=[10, 50, 90]):
        """
        Compute percentiles of kr realizations.

        Args:
            kr_array: (num_samples, num_saturations) array
            percentiles_list: List of percentiles [P10, P50, P90]

        Returns:
            Dictionary with percentile curves
        """
        result = {}
        for p in percentiles_list:
            result[f'P{p}'] = np.percentile(kr_array, p, axis=0)

        return result


def test():
    """Test function for digital rock analysis."""
    print("=" * 70)
    print("Regaieg et al. Digital Rock Physics: Relative Permeability")
    print("=" * 70)

    # Test 1: Pore Network Model
    print("\n1. Pore Network Model Generation")
    print("-" * 70)
    pnm = PoreNetworkModel(num_pores=50, network_type='random')
    print(f"Network type: random")
    print(f"Number of pores: {pnm.num_pores}")
    print(f"Number of throats: {len(pnm.connectivity)}")
    print(f"Avg pore radius: {np.mean(pnm.pore_radius):.2f} um")
    print(f"Avg throat count: {2*len(pnm.connectivity)/pnm.num_pores:.2f}")

    # Test 2: Conductance calculation
    print("\n2. Throat Conductance (Hagen-Poiseuille)")
    print("-" * 70)
    viscosity = 1e-3  # Pa.s
    print(f"Fluid viscosity: {viscosity*1e3:.1f} cP")
    print("Sample throat conductances:")
    for i, (pore_i, pore_j) in enumerate(pnm.connectivity[:5]):
        g = pnm.conductance(pore_i, pore_j, viscosity)
        print(f"  Throat {pore_i}-{pore_j}: {g:.6e} m^3/Pa.s")

    # Test 3: Pressure solution
    print("\n3. Pressure Field Solution")
    print("-" * 70)
    pressure = pnm.solve_pressure(flow_rate=1e-10, inlet_pore=0, outlet_pore=49)
    print(f"Inlet pressure: {pressure[0]/1000:.2f} kPa")
    print(f"Outlet pressure: {pressure[49]/1000:.2f} kPa")
    print(f"Pressure drop: {(pressure[0]-pressure[49])/1000:.2f} kPa")
    print(f"Mean internal pressure: {np.mean(pressure[1:-1])/1000:.2f} kPa")

    # Test 4: Invasion percolation
    print("\n4. Invasion Percolation (Drainage)")
    print("-" * 70)
    ip = InvasionPercolation(pnm, interfacial_tension=0.05, contact_angle=0)
    invasion = ip.invasion_sequence()
    print(f"Invasion sequence (first 10 pores): {invasion[:10]}")
    print(f"Total invaded pores: {len(invasion)}")

    # Test 5: Corey relative permeability
    print("\n5. Corey Relative Permeability Model")
    print("-" * 70)
    sor = 0.05
    swi = 0.15
    so_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    print(f"SOR={sor:.3f}, SWI={swi:.3f}")
    print("So      kr_oil  kr_water")
    for so in so_values:
        kr_o, kr_w = RelativePermeability.corey_kr(so, sor, swi)
        print(f"{so:.2f}   {kr_o:.4f}  {kr_w:.4f}")

    # Test 6: Fractional flow and Welge tangent
    print("\n6. Buckley-Leverett Fractional Flow")
    print("-" * 70)
    so_array = np.linspace(swi, 1.0 - sor, 30)
    print("So      fw      dfw/dSo")
    for so in so_array[::5]:
        fw = RelativePermeability.fractional_flow(so, sor, swi)
        dfw = RelativePermeability.dfw_dso(so, sor, swi)
        print(f"{so:.3f}  {fw:.4f}  {dfw:.4f}")

    # Welge tangent
    shock_so = RelativePermeability.welge_tangent(so_array)
    shock_fw = RelativePermeability.fractional_flow(shock_so, sor, swi)
    print(f"\nWelge shock front: So = {shock_so:.3f}, fw = {shock_fw:.3f}")

    # Test 7: Monte Carlo uncertainty analysis
    print("\n7. Monte Carlo Uncertainty Analysis (P10/P50/P90)")
    print("-" * 70)
    pore_samples = MonteCarloUncertainty.sample_pore_sizes(num_samples=100,
                                                            mean_radius=30,
                                                            std_radius=10)
    so_mc, kr_mc = MonteCarloUncertainty.compute_kr_realizations(pore_samples)
    percentiles = MonteCarloUncertainty.percentiles(kr_mc)

    print(f"Monte Carlo samples: {pore_samples.shape[0]}")
    print("So      P10     P50     P90")
    for i in range(0, len(so_mc), 3):
        p10 = percentiles['P10'][i]
        p50 = percentiles['P50'][i]
        p90 = percentiles['P90'][i]
        print(f"{so_mc[i]:.3f}  {p10:.4f}  {p50:.4f}  {p90:.4f}")

    print("\n" + "=" * 70)
    print("Test completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    test()
