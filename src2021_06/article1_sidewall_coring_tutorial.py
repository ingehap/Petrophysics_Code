"""
Article 1: Tutorial - A Century of Sidewall Coring Evolution and Challenges,
           From Shallow Land to Deep Water
Jackson (2021)
DOI: 10.30632/PJV62N3-2021t1

A historical tutorial tracing ~100 years of sidewall coring, from percussion
(bullet/gun) coring to modern rotary/mechanical sidewall coring (RSWC).  The
article is descriptive; its only quantitative content is a comparison table of
the major service-company rotary coring tools (plug diameter, max plug length,
core capacity per run, and pressure/temperature ratings).

Implements:

  - Cylindrical core-plug volume  V = pi*(d/2)^2 * L
  - Per-run recovered core volume  V_run = V_plug * capacity
  - The Fig. 14 rotary-coring tool table as structured data
  - Tool selection by pressure / temperature rating

Note: this is a descriptive review (no equations in the paper); the plug-volume
geometry is the natural computation its tool-dimension data supports.  Diameter
and length in inches, volume in cubic inches, pressure in psi, temperature in F.
"""

import numpy as np

# Fig. 14 - major service-company rotary sidewall coring tools.
# fields: name, company, dia_in, len_in, cap_min, cap_max, psi, temp_F
TOOLS = [
    ("Xaminer Core",        "Halliburton",  1.50, 2.40,  80, 100, 35000, 400),
    ("Xaminer Dual-Coring", "Halliburton",  1.50, 2.40, 100, 120, 35000, 400),
    ("HRSCT-B",             "Halliburton",  1.50, 2.40,  80, 100, 25000, 400),
    ("RSCT Core",           "Halliburton",  0.94, 1.75,  80, 100, 20000, 350),
    ("MaxCOR",              "Baker Hughes", 1.50, 2.50,  60,  60, 25000, 400),
    ("PowerCOR",            "Baker Hughes", 1.00, 1.80,  60,  60, 25000, 400),
    ("XL-Rock",             "Schlumberger", 1.50, 3.50,  44,  50, 30000, 400),
    ("MSCT",                "Schlumberger", 0.92, 2.00,  20,  75, 25000, 350),
]


# ---------------------------------------------- geometry ----------------

def plug_volume(diameter_in, length_in):
    """Cylindrical core-plug volume  V = pi*(d/2)^2 * L  (cubic inches)."""
    return np.pi * (np.asarray(diameter_in, float) / 2.0) ** 2 * np.asarray(length_in, float)


def per_run_volume(diameter_in, length_in, capacity):
    """Total core volume recovered in one run = plug volume * capacity."""
    return plug_volume(diameter_in, length_in) * capacity


# ---------------------------------------------- tool selection ----------

def select_tools(min_psi=0.0, min_temp_F=0.0, min_diameter_in=0.0):
    """Return tools rated at or above the given pressure/temp/diameter."""
    out = []
    for name, co, d, L, cmin, cmax, psi, tF in TOOLS:
        if psi >= min_psi and tF >= min_temp_F and d >= min_diameter_in:
            out.append(name)
    return out


def largest_plug_tool():
    """Name of the tool with the largest single-plug volume."""
    vols = [(plug_volume(d, L), name) for name, co, d, L, *_ in TOOLS]
    return max(vols)[1]


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Tutorial - A Century of Sidewall Coring")
    print("=" * 60)

    # MaxCOR 1.5 x 2.5 in plug
    v_max = plug_volume(1.5, 2.5)
    print(f"  MaxCOR plug volume     = {v_max:.4f} in^3  (expect 4.4179)")
    assert abs(v_max - 4.4179) < 1e-3

    # XL-Rock has the largest single plug (1.5 x 3.5 in)
    v_xl = plug_volume(1.5, 3.5)
    print(f"  XL-Rock plug volume    = {v_xl:.3f} in^3  (expect 6.185)")
    assert abs(v_xl - 6.185) < 1e-2
    assert largest_plug_tool() == "XL-Rock"

    # Legacy MSCT 0.92 x 2.0 in plug
    v_msct = plug_volume(0.92, 2.0)
    print(f"  MSCT plug volume       = {v_msct:.4f} in^3  (expect 1.3296)")
    assert abs(v_msct - 1.3296) < 1e-3
    # large-diameter tools recover much more rock than legacy tools
    assert v_max > 3 * v_msct

    # Per-run volume
    print(f"  MaxCOR per-run (60)    = {per_run_volume(1.5, 2.5, 60):.1f} in^3")
    assert abs(per_run_volume(1.5, 2.5, 60) - 265.07) < 0.1

    # Tool selection: only the deepwater 35,000-psi / 400 F tools qualify
    deep = select_tools(min_psi=30000, min_temp_F=400, min_diameter_in=1.5)
    print(f"  tools >= 30k psi/400F/1.5in = {deep}")
    assert "Xaminer Core" in deep and "XL-Rock" in deep
    assert "MSCT" not in deep and "RSCT Core" not in deep
    print("  PASS")
    return {"maxcor_vol": v_max, "xlrock_vol": v_xl, "largest": largest_plug_tool()}


if __name__ == "__main__":
    test_all()
