"""
3D Look-Ahead Electromagnetic Inversion in Near-Vertical Wells - A Bridge
Between Vertical and Horizontal Wells 3D Geology Understanding

Reference:
    El-Khamry, A., Ma, J., Clegg, N. M., Lozinsky, C., and Bikchandaev, E.
    (2026). 3D Look-Ahead Electromagnetic Inversion in Near-Vertical Wells.
    Petrophysics, 67(3), 571-580.
    DOI: 10.30632/PJV67N3-2026a7

The paper presents a 3D look-ahead EM inversion for near-vertical wells that
bridges the 1D (vertical-well) and 3D (horizontal-well) geosteering workflows,
enabling earlier detection and mapping of resistivity ahead of the bit in
laterally heterogeneous settings.

This module implements the standard building blocks of such a workflow:
    - A layered look-ahead forward model (apparent resistivity sensed by a
      directional EM tool approaching a boundary ahead of the bit).
    - Distance-to-boundary (DTB) ahead-of-bit estimation from the apparent
      resistivity trend.
    - A 3D dipping-boundary geometry update from azimuthal responses.
    - A simple Kalman-style recursive update that "bridges" the 1D estimate
      toward a 3D boundary picture as new measurements arrive.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Look-ahead forward model
# ---------------------------------------------------------------------------

def lookahead_apparent_resistivity(rho_here: float, rho_ahead: float,
                                   distance_to_boundary: float,
                                   depth_of_investigation: float) -> float:
    """
    Apparent resistivity sensed by a look-ahead tool as it approaches a planar
    boundary, modelled as an exponential blend between the current bed (rho_here)
    and the bed ahead (rho_ahead) weighted by how far the boundary sits inside
    the depth of investigation (DOI):

        w = exp(-DTB / DOI)
        Ra = rho_here ** (1-w) * rho_ahead ** w      (log-linear mixing)

    Parameters
    ----------
    rho_here  : resistivity of the bed the bit is in, ohm.m
    rho_ahead : resistivity of the approaching bed, ohm.m
    distance_to_boundary : DTB ahead of the bit, m
    depth_of_investigation : look-ahead DOI, m

    Returns
    -------
    Apparent resistivity, ohm.m.
    """
    w = math.exp(-max(distance_to_boundary, 0.0) / depth_of_investigation)
    return rho_here ** (1.0 - w) * rho_ahead ** w


def distance_to_boundary(ra: float, rho_here: float, rho_ahead: float,
                         depth_of_investigation: float) -> float:
    """
    Invert the look-ahead model for the distance-to-boundary ahead of the bit.

        w = ln(Ra/rho_here) / ln(rho_ahead/rho_here)
        DTB = -DOI * ln(w)
    """
    denom = math.log(rho_ahead / rho_here)
    if denom == 0.0:
        return float("inf")
    w = math.log(ra / rho_here) / denom
    w = min(max(w, 1e-9), 1.0)
    return -depth_of_investigation * math.log(w)


# ---------------------------------------------------------------------------
# 2. 3D dipping-boundary geometry from azimuthal responses
# ---------------------------------------------------------------------------

@dataclass
class Boundary3D:
    """A planar boundary ahead of / around the bit in 3D."""
    dtb: float          # distance to boundary along the tool axis, m
    dip_deg: float      # apparent dip, degrees
    azimuth_deg: float  # azimuth of maximum approach, degrees


def boundary_from_azimuthal(signals: np.ndarray, azimuths_deg: np.ndarray,
                            dtb: float) -> Boundary3D:
    """
    Estimate boundary dip and azimuth from the azimuthal variation of a
    directional EM signal (a sinusoid fit: a + b*cos(az - az0)).

    Parameters
    ----------
    signals      : directional response per azimuth bin.
    azimuths_deg : azimuth of each bin, degrees.
    dtb          : distance-to-boundary estimate, m.

    Returns
    -------
    Boundary3D with azimuth (of max signal) and a dip proxy from the
    modulation amplitude.
    """
    az = np.radians(np.asarray(azimuths_deg, float))
    s = np.asarray(signals, float)
    c = np.sum(s * np.cos(az))
    d = np.sum(s * np.sin(az))
    az0 = math.degrees(math.atan2(d, c)) % 360.0
    amp = math.hypot(c, d) / max(np.sum(np.abs(s)), 1e-12)
    dip = math.degrees(math.atan(amp))  # modulation -> apparent dip proxy
    return Boundary3D(dtb=dtb, dip_deg=dip, azimuth_deg=az0)


# ---------------------------------------------------------------------------
# 3. Recursive 1D -> 3D bridging update (scalar Kalman filter)
# ---------------------------------------------------------------------------

@dataclass
class DTBTracker:
    """
    Recursively refine the distance-to-boundary estimate as the bit advances,
    bridging the initial 1D estimate toward a stable 3D boundary picture.
    """
    dtb: float
    var: float = 25.0          # estimate variance, m^2
    process_var: float = 1.0   # per-step model variance, m^2

    def update(self, measured_dtb: float, meas_var: float = 4.0) -> float:
        """One predict/update cycle; returns the refined DTB."""
        # Predict (uncertainty grows as we drill).
        self.var += self.process_var
        # Update (Kalman gain).
        K = self.var / (self.var + meas_var)
        self.dtb = self.dtb + K * (measured_dtb - self.dtb)
        self.var = (1.0 - K) * self.var
        return self.dtb


# ---------------------------------------------------------------------------
# 4. Convenience: full workflow example
# ---------------------------------------------------------------------------

def example_workflow():
    """Run a complete example and print key results."""
    print("=" * 64)
    print("3D Look-Ahead EM Inversion in Near-Vertical Wells")
    print("Ref: El-Khamry et al., Petrophysics 67(3) 2026")
    print("=" * 64)

    rho_here, rho_ahead, doi = 20.0, 2.0, 15.0  # approaching a conductive bed
    print(f"\nLook-ahead apparent resistivity approaching a boundary "
          f"(Rh={rho_here}, ahead={rho_ahead}, DOI={doi} m):")
    for dtb in (30.0, 15.0, 8.0, 3.0):
        ra = lookahead_apparent_resistivity(rho_here, rho_ahead, dtb, doi)
        inv = distance_to_boundary(ra, rho_here, rho_ahead, doi)
        print(f"  DTB={dtb:>5.1f} m  Ra={ra:6.2f}  inverted DTB={inv:5.1f} m")

    # 3D geometry from a synthetic azimuthal sweep (boundary toward 120 deg).
    az = np.arange(0, 360, 30)
    sig = 1.0 + 0.4 * np.cos(np.radians(az - 120.0))
    b = boundary_from_azimuthal(sig, az, dtb=8.0)
    print(f"\n3D boundary: DTB={b.dtb:.1f} m  dip={b.dip_deg:.1f} deg  "
          f"azimuth={b.azimuth_deg:.0f} deg")

    # Recursive bridging update.
    tr = DTBTracker(dtb=30.0)
    print("\nRecursive DTB tracking as the bit advances:")
    for meas in (24.0, 17.0, 11.0, 6.0):
        print(f"  measured {meas:5.1f} m -> tracked {tr.update(meas):5.2f} m "
              f"(var {tr.var:.2f})")

    return b


if __name__ == "__main__":
    example_workflow()
