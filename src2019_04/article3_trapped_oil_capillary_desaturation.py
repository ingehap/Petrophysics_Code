"""
Article 3: Pore-Scale Insights on Trapped Oil During Waterflooding of Sandstone
           Rocks of Varying Wettability States
Berthet, Hebert, Barbouteau, Andriamananjaona, Rivenq (2019)
DOI: 10.30632/PJV60N2-2019a1

Capillary desaturation experiments (combined with micro-CT) measure how the
residual (trapped) oil saturation falls as the capillary number rises during
waterflooding, and how the wettability state shifts the capillary desaturation
curve.  Above a critical capillary number, viscous forces mobilize the
capillary-trapped oil.

Implements:

  - Capillary number  Nc = mu*v/sigma
  - Capillary desaturation curve  Sor(Nc)
  - Wettability-state shift of the residual oil and critical Nc
  - Trapping efficiency (fraction of oil that remains trapped)

Note: this issue's PDF has a text layer but its typeset formula glyphs were
dropped in extraction, so these are faithful standard-form reconstructions of
the capillary-desaturation analysis the paper applies.  Capillary number is
dimensionless; saturations are fractions.
"""

import numpy as np


# ---------------------------------------------- capillary number -------

def capillary_number(mu, v, sigma):
    """Capillary number  Nc = mu*v/sigma (viscous / capillary forces)."""
    return mu * np.asarray(v, float) / sigma


def capillary_desaturation(Nc, sor_low=0.05, sor_high=0.35, nc_crit=1e-5, p=0.8):
    """Capillary desaturation curve  Sor(Nc).

        Sor = sor_low + (sor_high - sor_low)/(1 + (Nc/Nc_crit)^p)
    Plateau at sor_high for Nc << Nc_crit; drops toward sor_low for Nc >> Nc_crit.
    """
    Nc = np.asarray(Nc, float)
    return sor_low + (sor_high - sor_low) / (1.0 + (Nc / nc_crit) ** p)


def wettability_cdc(state):
    """Return CDC parameters (sor_high, nc_crit) for a wettability state.

    Water-wet rock traps more oil and needs a higher Nc to mobilize it than
    mixed/oil-wet rock.
    """
    table = {
        "water-wet": (0.40, 2e-5),
        "mixed-wet": (0.25, 8e-6),
        "oil-wet": (0.15, 3e-6),
    }
    return table[state]


def trapping_efficiency(sor, soi):
    """Fraction of the initial oil that remains trapped  Sor/Soi."""
    return sor / soi


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Trapped Oil & Capillary Desaturation")
    print("=" * 60)

    # Capillary number rises with velocity, falls with interfacial tension
    assert capillary_number(1e-3, 1e-5, 0.03) < capillary_number(1e-3, 1e-3, 0.03)
    assert capillary_number(1e-3, 1e-4, 0.05) < capillary_number(1e-3, 1e-4, 0.02)

    # CDC: residual oil is high and flat at low Nc, drops above the critical Nc
    sor_lo_nc = capillary_desaturation(1e-7)
    sor_hi_nc = capillary_desaturation(1e-2)
    print(f"  Sor at low/high Nc     = {sor_lo_nc:.3f} / {sor_hi_nc:.3f}")
    assert sor_lo_nc > sor_hi_nc
    assert abs(sor_lo_nc - 0.35) < 0.02           # plateau near sor_high
    # monotonic decrease
    nc = np.logspace(-8, -2, 20)
    assert np.all(np.diff(capillary_desaturation(nc)) <= 1e-12)

    # Wettability: water-wet traps the most oil and needs the highest Nc
    sh_ww, nc_ww = wettability_cdc("water-wet")
    sh_ow, nc_ow = wettability_cdc("oil-wet")
    print(f"  Sor_high water-wet / oil-wet = {sh_ww} / {sh_ow}")
    assert sh_ww > sh_ow and nc_ww > nc_ow
    # at a fixed Nc the water-wet rock retains more residual oil
    sor_ww = capillary_desaturation(5e-6, sor_high=sh_ww, nc_crit=nc_ww)
    sor_ow = capillary_desaturation(5e-6, sor_high=sh_ow, nc_crit=nc_ow)
    assert sor_ww > sor_ow

    # Trapping efficiency
    te = trapping_efficiency(0.30, 0.65)
    print(f"  trapping efficiency    = {te:.2f}")
    assert 0.0 < te < 1.0
    print("  PASS")
    return {"sor_low_nc": float(sor_lo_nc), "sor_high_nc": float(sor_hi_nc),
            "trap_eff": te}


if __name__ == "__main__":
    test_all()
