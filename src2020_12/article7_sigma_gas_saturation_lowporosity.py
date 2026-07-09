"""
Article 7: Multidetector Pulsed-Neutron Tool Application in a Low-Porosity
           Reservoir - A Case Study in Mutiara Field, Indonesia
Wijaya, Aulianagara, Guo, Naibaho, Asriwan, Amirudin (2020)
DOI: 10.30632/PJV61N6-2020a7

Gas saturation from the sigma (capture cross section) porosity balance in a
low-porosity (~12 p.u.) shaly sandstone, where the small porosity and the
moderate sigma-water/sigma-gas contrast make the answer sensitive to the
endpoints.  A third detector on the multidetector pulsed-neutron tool improves
the sigma precision; the sigma matrix/shale endpoints are read from the sigma
crossplot and a linear shale correction is applied.

Implements:

  - Clean sigma gas saturation                                       (Eq. 1)
      Sg = (Sigma_log - Sigma_ma*(1-phi) - Sigma_w*phi)/(phi*(Sigma_g-Sigma_w))
  - Shaly sigma gas saturation with Vsh, Sigma_sh terms              (Eq. 2)
  - Saturation sensitivity per capture unit (low-porosity caveat)

Note: this issue's PDF text layer drops the typeset glyphs; the forms here are
faithful standard-form reconstructions of the sigma porosity balance.  The
endpoints are the paper's quoted values: Sigma_matrix = 7.5, Sigma_shale = 27,
Sigma_water = 24, Sigma_gas = 3 c.u.; porosity ~12 p.u.; 5 kppm NaCl water.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

SIGMA_MA = 7.5           # c.u., matrix
SIGMA_SH = 27.0          # c.u., shale
SIGMA_W = 24.0           # c.u., formation water (5 kppm NaCl)
SIGMA_G = 3.0            # c.u., gas


# ---------------------------------------------- clean Sg ----------------

def sigma_log_clean(phi, sg, sigma_ma=SIGMA_MA, sigma_w=SIGMA_W, sigma_g=SIGMA_G):
    """Forward sigma from the clean porosity balance."""
    return petrolib.nuclear.sigma_forward(
        phi, 1.0 - sg, sigma_ma=sigma_ma, sigma_w=sigma_w, sigma_hc=sigma_g
    )


def gas_saturation_clean(sigma_log, phi, sigma_ma=SIGMA_MA, sigma_w=SIGMA_W,
                         sigma_g=SIGMA_G):
    """Clean (shale-free) gas saturation from sigma  (Eq. 1)."""
    return petrolib.nuclear.sw_from_sigma(
        sigma_log, phi, sigma_ma=sigma_ma, sigma_w=sigma_g, sigma_hc=sigma_w
    )


# ---------------------------------------------- shaly Sg ----------------

def sigma_log_shaly(phi, vsh, sg, sigma_ma=SIGMA_MA, sigma_sh=SIGMA_SH,
                    sigma_w=SIGMA_W, sigma_g=SIGMA_G):
    """Forward sigma including a shale term."""
    return petrolib.nuclear.sigma_forward(
        phi, 1.0 - sg, sigma_ma=sigma_ma, sigma_w=sigma_w, sigma_hc=sigma_g, vsh=vsh, sigma_sh=sigma_sh
    )


def gas_saturation_shaly(sigma_log, phi, vsh, sigma_ma=SIGMA_MA, sigma_sh=SIGMA_SH,
                         sigma_w=SIGMA_W, sigma_g=SIGMA_G):
    """Shaly gas saturation from sigma with a linear shale correction  (Eq. 2)."""
    return petrolib.nuclear.sw_from_sigma(
        sigma_log, phi, sigma_ma=sigma_ma, sigma_w=sigma_g, sigma_hc=sigma_w, vsh=vsh, sigma_sh=sigma_sh
    )


def saturation_sensitivity(phi, sigma_w=SIGMA_W, sigma_g=SIGMA_G):
    """|dSigma/dSg| = phi*(Sigma_w - Sigma_g): sigma swing per unit Sg.

    Small in low-porosity / low-contrast rock, so a given sigma error maps to a
    large saturation error - the paper's central caveat.
    """
    return petrolib.nuclear.sigma_sensitivity(phi, sigma_w, sigma_g)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Sigma Gas Saturation (Low-Porosity, Indonesia)")
    print("=" * 60)

    phi = 0.12
    # Clean: plant Sg = 0.6, forward to sigma, invert and recover it
    sg_true = 0.60
    sig = sigma_log_clean(phi, sg_true)
    sg = gas_saturation_clean(sig, phi)
    print(f"  clean  sigma_log {sig:.3f} c.u. -> Sg = {sg:.3f}  (true {sg_true})")
    assert abs(sg - sg_true) < 1e-9
    # gas lowers sigma below the wet value
    assert sigma_log_clean(phi, 0.0) > sigma_log_clean(phi, 1.0)

    # Shaly: 20% shale, plant Sg = 0.5; ignoring shale would bias the answer
    vsh = 0.20
    sig_sh = sigma_log_shaly(phi, vsh, 0.50)
    sg_sh = gas_saturation_shaly(sig_sh, phi, vsh)
    print(f"  shaly  sigma_log {sig_sh:.3f} c.u. -> Sg = {sg_sh:.3f}  (true 0.50)")
    assert abs(sg_sh - 0.50) < 1e-9
    sg_ignore_shale = gas_saturation_clean(sig_sh, phi)
    print(f"  ignoring shale -> Sg   = {sg_ignore_shale:.3f}  (biased)")
    assert abs(sg_ignore_shale - 0.50) > 0.1          # shale matters

    # Low-porosity sensitivity caveat: the sigma swing per unit Sg is small
    sens_lo = saturation_sensitivity(0.12)
    sens_hi = saturation_sensitivity(0.30)
    print(f"  |dSigma/dSg| @12/30 pu = {sens_lo:.2f} / {sens_hi:.2f} c.u.")
    assert sens_lo < sens_hi                          # harder at low porosity
    print("  PASS")
    return {"sg_clean": float(sg), "sg_shaly": float(sg_sh),
            "sensitivity_12pu": float(sens_lo)}


if __name__ == "__main__":
    test_all()
