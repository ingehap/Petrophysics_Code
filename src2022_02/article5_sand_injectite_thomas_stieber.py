"""
Article 5: Evaluating Petrophysical Properties and Volumetrics Uncertainties
           of Sand Injectite Reservoirs - Norwegian North Sea
Kotwicki, Baig, Johansen, Leirdal, Aftret, Sandstad, Anthonsen,
Gianotten, Hansen, Firinu (2022)
DOI: 10.30632/PJV63N1-2022a5

Sand-injectite reservoirs carry shale clasts suspended in clean sand, so a
bulk (BCA) log analysis underestimates hydrocarbon volume.  This module
compares three routes - bulk conventional analysis, a CT-scan high-
resolution reference, and a modified Thomas-Stieber laminated-sand analysis
that solves for the sand phase only.

Implements:

  - Effective porosity  PHIE = PHIT - Vsh * PHITsh                (Eq. 1)
  - Herron permeability  K = Af * exp(Sum Bi * Mi)                (Eq. 2)
  - CT-scan porosity mixing  PHIT = Fsd*PHITSand + Fsh*PHITShale  (Eq. 3)
  - Constant-BVW saturation  Sw = BVW / PHIT                      (Eq. 4)
  - Grain density mixing  RHOMA = Fsd*RHOMASand + Fsh*RHOMAShale  (Eq. 5)
  - Sand counting with Fsd cutoff                                 (Eq. 6)
  - Net thickness (bulk and Thomas-Stieber)                       (Eqs. 7-8)
  - HVOLH (bulk and Thomas-Stieber)                               (Eqs. 9-10)
  - Thomas-Stieber sand fraction / FNTG + Poupon-Archie sand Sw   (aux)

Cutoffs from the paper: Fsd >= 0.30, Swt <= 0.65, PHIT >= 0.15.
"""

import numpy as np

# Endpoints / cutoffs from the paper
PHIT_SAND = 0.38            # clean-sand total porosity
PHIT_SHALE = 0.27          # shale total porosity
RHOMA_SAND = 2.65          # clean quartz grain density (g/cc)
RHOMA_SHALE = 2.72         # shale matrix density (g/cc)
FSD_CUT = 0.30             # sand-fraction cutoff
SWT_CUT = 0.65             # water-saturation cutoff
PHIT_CUT = 0.15            # porosity cutoff


# ---------------------------------------------- Eq. 1: PHIE -------------

def effective_porosity(phit, vsh, phit_sh=PHIT_SHALE):
    """PHIE = PHIT - Vsh * PHITsh  (Eq. 1)."""
    return np.asarray(phit, float) - np.asarray(vsh, float) * phit_sh


# ---------------------------------------------- Eq. 2: Herron K ---------

def herron_permeability(mineral_fracs, B_coeffs, Af=1.0):
    """K = Af * exp(Sum_i Bi * Mi)  (Eq. 2).

    mineral_fracs, B_coeffs : equal-length weight fractions and constants.
    """
    m = np.asarray(mineral_fracs, float)
    b = np.asarray(B_coeffs, float)
    return float(Af * np.exp(np.sum(b * m)))


# ---------------------------------------------- Eqs. 3, 5: mixing -------

def porosity_mixing(fsd, fsh, phit_sand=PHIT_SAND, phit_shale=PHIT_SHALE):
    """PHIT = Fsd*PHITSand + Fsh*PHITShale  (Eq. 3)."""
    return np.asarray(fsd, float) * phit_sand + np.asarray(fsh, float) * phit_shale


def sand_porosity_from_total(phit, fsd, fsh, phit_shale=PHIT_SHALE):
    """Solve Eq. 3 for the sand porosity: (PHIT - Fsh*PHITShale)/Fsd."""
    fsd = np.asarray(fsd, float)
    return (np.asarray(phit, float) - np.asarray(fsh, float) * phit_shale) / np.where(fsd > 1e-6, fsd, np.nan)


def grain_density_mixing(fsd, fsh, rhoma_sand=RHOMA_SAND, rhoma_shale=RHOMA_SHALE):
    """RHOMA = Fsd*RHOMASand + Fsh*RHOMAShale  (Eq. 5)."""
    return np.asarray(fsd, float) * rhoma_sand + np.asarray(fsh, float) * rhoma_shale


# ---------------------------------------------- Eq. 4: constant BVW -----

def saturation_from_bvw(bvw, phit):
    """Sw = BVW / PHIT  (Eq. 4)."""
    phit = np.asarray(phit, float)
    return np.asarray(bvw, float) / np.where(phit > 1e-6, phit, np.nan)


# ---------------------------------------------- Archie / Poupon ---------

def poupon_invert_rt_sand(rt, vsh, rsh, a, m, rw, phit_sand):
    """Strip shale conductivity (Poupon) to recover the sand-phase Rt.

    1/Rt = Vsh/Rsh + (PHITSand^m)/(a*Rw) * (1/RtSand)  ->  solve RtSand.
    """
    rt = np.asarray(rt, float)
    vsh = np.asarray(vsh, float)
    cond_sand = 1.0 / rt - vsh / rsh
    cond_sand = np.where(cond_sand > 1e-9, cond_sand, 1e-9)
    return (phit_sand ** m) / (a * rw) / cond_sand


def archie_saturation(rt_sand, phit_sand, a=1.0, m=2.0, n=2.0, rw=0.03):
    """SwtSand = (a*Rw / (PHITSand^m * RtSand))^(1/n)  (Archie)."""
    sw = (a * rw / (phit_sand ** m * np.asarray(rt_sand, float))) ** (1.0 / n)
    return np.clip(sw, 0.0, 1.0)


# ---------------------------------------------- Thomas-Stieber ----------

def thomas_stieber_fntg(phit, vsh, phi_sand=PHIT_SAND, phi_shale=PHIT_SHALE):
    """Laminated-shale net-to-gross FNTG from the Thomas-Stieber crossplot.

    For laminated shale, total porosity decreases linearly from the clean
    sand point to the shale point with shale lamina fraction Vlam:
        PHIT = (1 - Vlam)*phi_sand + Vlam*phi_shale
    so  Vlam = (phi_sand - PHIT)/(phi_sand - phi_shale)  and FNTG = 1 - Vlam.
    """
    vlam = (phi_sand - np.asarray(phit, float)) / (phi_sand - phi_shale)
    vlam = np.clip(vlam, 0.0, 1.0)
    return 1.0 - vlam


# ---------------------------------------------- Eq. 6: sand counting ----

def sand_count(fsd, fsd_cut=FSD_CUT):
    """Number of samples with sand fraction at or above the cutoff (Eq. 6)."""
    return int(np.sum(np.asarray(fsd, float) >= fsd_cut))


# ---------------------------------------------- Eqs. 7-10: volumetrics --

def _net_flag(fsd, swt, phit):
    return ((np.asarray(fsd, float) >= FSD_CUT) &
            (np.asarray(swt, float) <= SWT_CUT) &
            (np.asarray(phit, float) >= PHIT_CUT))


def net_thickness_bulk(fsd, swt, phit, dz):
    """Bulk net thickness = Sum dz over samples passing cutoffs (Eq. 7)."""
    return float(np.sum(_net_flag(fsd, swt, phit)) * dz)


def net_thickness_ts(fntg, fsd, swt, phit, dz):
    """Thomas-Stieber net thickness = Sum dz * FNTG over passing samples (Eq. 8)."""
    flag = _net_flag(fsd, swt, phit)
    return float(np.sum(np.asarray(fntg, float)[flag]) * dz)


def hvolh_bulk(phit, swt, fsd, dz):
    """Bulk HVOLH = Sum dz * PHIT * (1 - Swt) over passing samples (Eq. 9)."""
    flag = _net_flag(fsd, swt, phit)
    phit = np.asarray(phit, float); swt = np.asarray(swt, float)
    return float(np.sum((phit * (1.0 - swt))[flag]) * dz)


def hvolh_ts(phit_sand, swt_sand, fntg, fsd, swt_bulk, phit_bulk, dz):
    """Thomas-Stieber HVOLH on the sand phase (Eq. 10).

    = Sum dz * FNTG * PHITSand * (1 - SwtSand) over passing samples.
    """
    flag = _net_flag(fsd, swt_bulk, phit_bulk)
    term = (np.asarray(fntg, float) * np.asarray(phit_sand, float) *
            (1.0 - np.asarray(swt_sand, float)))
    return float(np.sum(term[flag]) * dz)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Sand-Injectite Petrophysics & Volumetrics")
    print("=" * 60)

    dz = 0.1
    # three facies blocks: clean dyke, brecciated oil, brecciated gas
    fsd = np.r_[np.full(50, 0.97), np.full(50, 0.55), np.full(50, 0.50)]
    fsh = 1.0 - fsd

    phit = porosity_mixing(fsd, fsh)                          # Eq. 3
    rhoma = grain_density_mixing(fsd, fsh)                    # Eq. 5
    phit_sand = sand_porosity_from_total(phit, fsd, fsh)      # invert Eq. 3
    print(f"  PHIT  range            = {phit.min():.3f} - {phit.max():.3f}")
    print(f"  RHOMA range            = {rhoma.min():.3f} - {rhoma.max():.3f}")
    assert np.allclose(phit_sand, PHIT_SAND, atol=1e-6)

    # constant BVW saturation (Eq. 4)
    bvw_sand = PHIT_SAND * 0.06
    swt_sand = saturation_from_bvw(bvw_sand, phit_sand)
    print(f"  Swt (sand phase)       = {swt_sand[0]:.3f}")
    assert swt_sand[0] < 0.10

    # bulk Swt is higher because shale clasts carry bound water
    swt_bulk = np.r_[np.full(50, 0.08), np.full(50, 0.45), np.full(50, 0.55)]

    # Thomas-Stieber FNTG (aux) - falls from clean to brecciated
    fntg = thomas_stieber_fntg(phit, fsh * 0)  # vsh handled via phit here
    fntg = np.clip(fsd, 0, 1)                   # sand fraction is the laminar NTG
    print(f"  FNTG range             = {fntg.min():.3f} - {fntg.max():.3f}")

    # Poupon -> Archie sand saturation in the gas breccia
    rt = np.full(150, 4.0)
    vsh = fsh
    rt_sand = poupon_invert_rt_sand(rt, vsh, rsh=2.0, a=1.0, m=2.0, rw=0.03,
                                    phit_sand=PHIT_SAND)
    sw_archie = archie_saturation(rt_sand, PHIT_SAND)
    print(f"  Archie SwtSand (gas)   = {sw_archie[-1]:.3f}")
    assert 0.0 <= sw_archie[-1] <= 1.0

    # Herron permeability for clean quartz sand
    K = herron_permeability(mineral_fracs=[0.97, 0.03],
                            B_coeffs=[8.0, -3.0])
    print(f"  Herron K (clean sand)  = {K:.1f} (relative mD)")
    assert K > 0

    # PHIE (Eq. 1)
    phie = effective_porosity(phit, vsh)
    assert np.all(phie <= phit + 1e-9)

    # volumetrics: TS recovers more HVOLH than BCA in the breccia
    net_b = net_thickness_bulk(fsd, swt_bulk, phit, dz)
    net_t = net_thickness_ts(fntg, fsd, swt_bulk, phit, dz)
    hv_b = hvolh_bulk(phit, swt_bulk, fsd, dz)
    hv_t = hvolh_ts(phit_sand, swt_sand, fntg, fsd, swt_bulk, phit, dz)
    print(f"  net thickness  BCA/TS  = {net_b:.2f} / {net_t:.2f} m")
    print(f"  HVOLH          BCA/TS  = {hv_b:.3f} / {hv_t:.3f} m")
    assert hv_t > hv_b, "Thomas-Stieber should recover more HVOLH than bulk"
    print("  PASS")
    return {"hvolh_bca": hv_b, "hvolh_ts": hv_t, "net_bca": net_b, "net_ts": net_t}


if __name__ == "__main__":
    test_all()
