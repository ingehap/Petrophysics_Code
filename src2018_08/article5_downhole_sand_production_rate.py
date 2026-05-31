"""
Article 5: Downhole Sand-Production Evaluation for Sand-Management Applications
Swarnanto, Srihirunrusmee, Lilaprathuang, Panmamuang, Wuthicharn, Mukerji,
Duangprasert, Puttisounthorn, Millot, Saavedra, Nollet (2018)
DOI: 10.30632/PJV59V4-2018a4

An acoustically isolated piezoelectric sensor run with a production-logging tool
counts the discrete impacts of individual sand grains (up to ~1500 hits/s) at
each perforation.  Converting that grain-count rate into a sand-production mass
rate gives a zonal sand profile used to set the maximum sand-free rate (MSFR)
and the maximum allowable sand rate (MASR) for sand management.

Implements:

  - Single-grain volume (sphere)  V = (4/3)*pi*r^3
  - Vertical-resolution correction  VRC = interval/0.4572  (>= 1)
  - Volumetric sand rate  VSPR = V*VRC*count_rate
  - Mass sand rate  SPR = VSPR*sand_density

Note: this issue's PDF has a text layer, and this article's sand-rate conversion
chain survived the extraction; the relations below are transcribed from the
paper (the 0.4572 m = 18 in vertical-resolution datum is reported explicitly).
SI units (length m, density kg/m^3, rate per second).
"""

import numpy as np

PLT_RESOLUTION = 0.4572      # m (18 in) vertical resolution datum


# ---------------------------------------------- sand rate --------------

def grain_volume(radius):
    """Volume of one (spherical) sand grain  V = (4/3)*pi*r^3."""
    return (4.0 / 3.0) * np.pi * np.asarray(radius, float) ** 3


def vertical_resolution_correction(interval):
    """Vertical-resolution correction  VRC = interval/0.4572, floored at 1.

    A sand-productive interval longer than the tool's 0.4572 m (18 in) vertical
    resolution scales the count up; shorter intervals take no correction.
    """
    return np.maximum(np.asarray(interval, float) / PLT_RESOLUTION, 1.0)


def volumetric_sand_rate(grain_vol, vrc, count_rate):
    """Volumetric sand-production rate  VSPR = V*VRC*count_rate  (m^3/s)."""
    return grain_vol * vrc * np.asarray(count_rate, float)


def mass_sand_rate(vspr, sand_density=2650.0):
    """Mass sand-production rate  SPR = VSPR*sand_density  (kg/s).

    sand_density defaults to quartz grain density (2650 kg/m^3).
    """
    return vspr * sand_density


def sand_production_rate(radius, interval, count_rate, sand_density=2650.0):
    """End-to-end grain count rate -> sand mass rate (kg/s)."""
    vrc = vertical_resolution_correction(interval)
    vspr = volumetric_sand_rate(grain_volume(radius), vrc, count_rate)
    return mass_sand_rate(vspr, sand_density)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Downhole Sand-Production Rate")
    print("=" * 60)

    # Sphere volume and its cubic growth with radius
    assert np.isclose(grain_volume(1.0), 4.0 / 3.0 * np.pi)
    assert np.isclose(grain_volume(2e-4) / grain_volume(1e-4), 8.0)

    # VRC: short interval -> 1, a 0.9144 m (36 in) interval -> 2x
    assert vertical_resolution_correction(0.2) == 1.0
    assert np.isclose(vertical_resolution_correction(0.9144), 2.0)

    # Worked example: 200-micron grains, 603 counts/s, single PLT station
    spr = sand_production_rate(radius=1e-4, interval=0.3, count_rate=603.0)
    spr_gph = spr * 3600.0 * 1000.0                    # kg/s -> g/hr
    print(f"  sand rate 603 cnt/s    = {spr_gph:.2f} g/hr")
    assert spr > 0

    # Doubling the count rate doubles the mass rate (linear)
    assert np.isclose(sand_production_rate(1e-4, 0.3, 1206.0), 2.0 * spr)
    print("  PASS")
    return {"spr_kg_s": float(spr)}


if __name__ == "__main__":
    test_all()
