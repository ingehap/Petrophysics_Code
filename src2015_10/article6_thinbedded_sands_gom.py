"""
Article 6: Applying a Consistent Evaluation Approach to Thin-Bedded Sands in a
           Gulf of Mexico Deepwater Field
Salunke, Hamman (2015)
Reference: Petrophysics Vol. 56, No. 5 (October 2015), pp. 511-520
DOI: none assigned (this issue predates SPWLA DOI assignment)

Thin-bedded (laminated) sand-shale sequences below log resolution under-read net
pay on conventional logs.  The consistent evaluation combines the Thomas-Stieber
model (laminar/dispersed/structural shale distribution from total porosity vs.
shale volume) with anisotropic horizontal/vertical resistivity (Rh-Rv) to
recover the laminar-shale volume, the sand porosity, and the true sand water
saturation for low-resistivity, low-contrast pay.

Implements:

  - Thomas-Stieber laminated and dispersed shale porosity trends
  - Laminar-shale volume and sand porosity from the Thomas-Stieber model
  - Laminated Rh (parallel) / Rv (series) sand-shale resistivity
  - Sand resistivity from Rh-Rv and Archie sand water saturation

Note: this article's body was beyond the PDF text extraction for this issue
(the source text truncates within an earlier article), so this module is a
methodology proxy implementing the standard thin-bed (Thomas-Stieber + Rh-Rv)
analysis the title/abstract describe, consistent with how other truncated
articles are handled in this repository.  Porosity/saturation/volume as
fractions, resistivity in ohm-m.
"""

import numpy as np


# ---------------------------------------------- Thomas-Stieber --------------

def thomas_stieber_laminated(vlam, phi_sand, phi_shale):
    """Total porosity along the laminated-shale trend (Thomas & Stieber, 1975)

        phi_total = phi_sand*(1 - Vlam) + phi_shale*Vlam,

    where Vlam is the laminar-shale volume fraction.
    """
    return phi_sand * (1.0 - vlam) + phi_shale * vlam


def thomas_stieber_dispersed(vdisp, phi_sand):
    """Total porosity along the dispersed-shale trend

        phi_total = phi_sand - Vdisp*(1 - ... ) ~ phi_sand - Vdisp,

    dispersed clay fills sand pores, reducing porosity below phi_sand (for
    Vdisp <= phi_sand).
    """
    return phi_sand - vdisp


def laminar_shale_volume(phi_total, vsh_total, phi_sand, phi_shale):
    """Laminar-shale volume and sand porosity from the Thomas-Stieber model.

    For a laminated sand-shale series the shale volume is laminar, so
        Vlam = (phi_sand - phi_total)/(phi_sand - phi_shale),
    and the clean-sand porosity is recovered as
        phi_sand_eff = (phi_total - phi_shale*Vlam)/(1 - Vlam).
    Returns (Vlam, phi_sand_eff).
    """
    vlam = (phi_sand - phi_total) / (phi_sand - phi_shale)
    phi_sand_eff = (phi_total - phi_shale * vlam) / (1.0 - vlam)
    return vlam, phi_sand_eff


# ---------------------------------------------- Rh-Rv resistivity --------------

def laminated_rh(rsand, rshale, vlam):
    """Horizontal (parallel) resistivity  1/Rh = (1-Vlam)/Rsand + Vlam/Rshale."""
    return 1.0 / ((1.0 - vlam) / rsand + vlam / rshale)


def laminated_rv(rsand, rshale, vlam):
    """Vertical (series) resistivity  Rv = (1-Vlam)*Rsand + Vlam*Rshale."""
    return (1.0 - vlam) * rsand + vlam * rshale


def sand_resistivity(rh, rshale, vlam):
    """Sand resistivity from the horizontal resistivity and laminar-shale volume

        Rsand = (1 - Vlam) / (1/Rh - Vlam/Rshale),

    inverting the parallel-conduction Rh relation (the laminated low-resistivity
    pay correction).
    """
    return (1.0 - vlam) / (1.0 / rh - vlam / rshale)


def archie_sw(rsand, rw, phi_sand, m=2.0, n=2.0, a=1.0):
    """Archie sand water saturation  Sw = (a*Rw/(phi_sand^m * Rsand))^(1/n)."""
    return (a * rw / (phi_sand ** m * rsand)) ** (1.0 / n)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 6: Thin-Bedded Sands (Gulf of Mexico)")
    print("=" * 60)

    phi_sand, phi_shale = 0.30, 0.10

    # Thomas-Stieber: total porosity is a linear mix along the laminated trend
    pt = thomas_stieber_laminated(0.4, phi_sand, phi_shale)
    print(f"  laminated phi (Vlam=0.4) = {pt:.3f}")
    assert np.isclose(pt, 0.30 * 0.6 + 0.10 * 0.4)
    # dispersed clay reduces porosity below clean sand
    assert thomas_stieber_dispersed(0.05, phi_sand) < phi_sand

    # Recover laminar-shale volume and sand porosity from a measured total porosity
    vlam, phi_sand_eff = laminar_shale_volume(pt, vsh_total=0.4,
                                              phi_sand=phi_sand, phi_shale=phi_shale)
    print(f"  recovered Vlam / phi_sand = {vlam:.3f} / {phi_sand_eff:.3f}")
    assert np.isclose(vlam, 0.4) and np.isclose(phi_sand_eff, phi_sand)

    # Rh-Rv: Rv (series) >= Rh (parallel); recover Rsand from Rh and Vlam
    rsand_true, rshale = 20.0, 1.5
    rh = laminated_rh(rsand_true, rshale, vlam)
    rv = laminated_rv(rsand_true, rshale, vlam)
    print(f"  Rh / Rv                = {rh:.3f} / {rv:.3f}")
    assert rv > rh
    rsand = sand_resistivity(rh, rshale, vlam)
    assert np.isclose(rsand, rsand_true)

    # Using the high Rsand (not the suppressed Rh) lowers the sand water saturation
    sw_sand = archie_sw(rsand, rw=0.05, phi_sand=phi_sand_eff)
    sw_bulk = archie_sw(rh, rw=0.05, phi_sand=phi_sand_eff)
    print(f"  Sw (sand / bulk Rh)    = {sw_sand:.3f} / {sw_bulk:.3f}")
    assert sw_sand < sw_bulk          # avoids underestimating thin-bed pay
    print("  PASS")
    return {"Vlam": float(vlam), "Rsand": float(rsand), "Sw_sand": float(sw_sand)}


if __name__ == "__main__":
    test_all()
