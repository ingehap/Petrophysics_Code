"""
Article 3: Petrophysical Considerations for CO2 Capture and Storage
Kumar and Lauderdale-Smith (Petrophysics, Vol. 65, No. 1, Feb 2024, pp. 51-69)

Estimates pore-volume storage capacity of a saline aquifer for CO2 and
the residual + dissolution trapping fractions, following the standard
DOE/USGS volumetric method:
    M_CO2 = A * h * phi * (1 - Sw_irr) * rho_CO2 * E
"""
import numpy as np


def co2_storage_mass(area_m2, thickness_m, phi, Sw_irr, rho_co2=700.0, efficiency=0.05):
    """Returns CO2 stored mass in kilograms."""
    pv = area_m2 * thickness_m * phi * (1.0 - Sw_irr)
    return pv * rho_co2 * efficiency


def trapping_partition(M_total, f_residual=0.2, f_dissolution=0.15, f_mineral=0.05):
    """Split total stored mass into trapping mechanisms (rest = structural)."""
    f_struct = max(0.0, 1.0 - f_residual - f_dissolution - f_mineral)
    return {
        "structural_kg": M_total * f_struct,
        "residual_kg": M_total * f_residual,
        "dissolution_kg": M_total * f_dissolution,
        "mineral_kg": M_total * f_mineral,
    }


def test_all():
    # 10 km x 10 km, 50 m, 22% phi, 30% Swirr
    M = co2_storage_mass(1e8, 50.0, 0.22, 0.30)
    assert M > 0
    parts = trapping_partition(M)
    assert abs(sum(parts.values()) - M) < 1e-3
    # Sensitivity: doubling efficiency doubles mass
    M2 = co2_storage_mass(1e8, 50.0, 0.22, 0.30, efficiency=0.10)
    np.testing.assert_allclose(M2, 2 * M, rtol=1e-12)
    print(f"article3 OK | M = {M/1e9:.2f} Mt CO2 | structural = "
          f"{parts['structural_kg']/1e9:.2f} Mt")


if __name__ == "__main__":
    test_all()
