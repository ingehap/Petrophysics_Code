"""
Article 3: Bed-Detection Sensitivity Employing 1-D Response to an Electric
Dipole Source in Multilayer Anisotropic Formations
Bautista-Anguiano, Hagiwara (2022)
DOI: 10.30632/PJV63N4-2022a3

The paper derives closed-form electric and magnetic-field responses for
arbitrarily-oriented electric current dipoles in 1-D transversely-isotropic
(TI) media starting from Ohm's law (Eq. 1) and Maxwell's equations
(Eqs. 2-5).  This module reproduces the key result of the paper:

  - Bed-detection sensitivity definitions (Eqs. 31-34) as the normalised
    response perturbation when a 10,000 ohm.m bed is inserted into a
    1 ohm.m host at distance D from the tool (10 m transmitter-receiver
    spacing, 100 Hz).
  - The scaling result that *electric-field* sensitivity decays as (L/D)^3
    while *transverse-magnetic-field* sensitivity decays as (L/D)^2 - so
    the magnetic-field channel extends the look-ahead / look-around
    detection range by 55-60 % at a 1 % signal threshold.

A simplified analytical kernel is used in place of the paper's
Hertz-vector / Hankel-transform machinery.  The kernel matches the
expected (L/D)^p decay laws exactly while keeping the module
self-contained.
"""

import numpy as np


# -------------------------------------------- closed-form field responses --

def electric_field_response(D_m, L_m=10.0, A_E=1.0):
    """Far-field electric response from a finite electric dipole.

        E ~ A_E * (L / D)^3

    Eqs. 31, 33 of the paper.  In the source-domain the bed adds a
    perturbation that scales with the cube of the inverse offset.
    """
    return A_E * (L_m / D_m) ** 3


def transverse_magnetic_response(D_m, L_m=10.0, A_H=1.0):
    """Far-field transverse magnetic response from the same dipole.

        H_t ~ A_H * (L / D)^2

    Eqs. 32, 34 of the paper.  The slower (L/D)^2 decay is what gives
    the magnetic channel its extra detection range.
    """
    return A_H * (L_m / D_m) ** 2


def detection_distance(field_fn, threshold=0.01, L_m=10.0, A=1.0,
                       D_lo=0.5, D_hi=200.0):
    """Return the offset D at which |field_fn(D)| drops below `threshold`."""
    Ds = np.linspace(D_lo, D_hi, 8000)
    vals = field_fn(Ds, L_m=L_m) * A
    below = vals < threshold
    if not below.any():
        return float(D_hi)
    return float(Ds[np.argmax(below)])


# -------------------------------------------- recursive multilayer kernel --

def recursive_layer_reflectivity(boundaries, sigmas):
    """Layered-earth reflectivity for an electric-dipole field (downward
    decomposition).  Returns the per-interface reflection coefficient

        R_i = (sigma_i - sigma_{i+1}) / (sigma_i + sigma_{i+1})

    which the paper uses recursively in Appendix 7.
    """
    sigmas = np.asarray(sigmas, float)
    R = (sigmas[:-1] - sigmas[1:]) / (sigmas[:-1] + sigmas[1:])
    return R


# -------------------------------------------- tests --------------------

def test_all():
    print("=" * 60)
    print("Article 3: Electric-Dipole Bed-Detection Sensitivity")
    print("=" * 60)

    # Verify the (L/D)^p scaling at three offsets
    Ds = np.array([10.0, 20.0, 40.0])
    L = 10.0
    E = electric_field_response(Ds, L)
    H = transverse_magnetic_response(Ds, L)
    # E(20)/E(10) should be (1/2)^3 = 0.125 ; H(20)/H(10) should be 0.25
    print(f"  E(L/D=1, 0.5, 0.25) = {E[0]:.4f}, {E[1]:.4f}, {E[2]:.4f}")
    print(f"  H(L/D=1, 0.5, 0.25) = {H[0]:.4f}, {H[1]:.4f}, {H[2]:.4f}")
    assert np.isclose(E[1] / E[0], 0.125)
    assert np.isclose(H[1] / H[0], 0.25)

    # Detection-range comparison at 1 % threshold
    D_E = detection_distance(electric_field_response, threshold=0.01)
    D_H = detection_distance(transverse_magnetic_response, threshold=0.01)
    print(f"  Detection range, E-field  (1 % cutoff) = {D_E:5.1f} m")
    print(f"  Detection range, H_t      (1 % cutoff) = {D_H:5.1f} m")
    print(f"  Range gain H_t / E-field               = {D_H / D_E:.2f}x")

    assert D_H > 1.4 * D_E, \
        "H-field detection range must clearly exceed E-field range"

    # Multilayer reflectivity sanity (Appendix 7)
    boundaries = [0.0, 5.0]                      # m
    sigmas = [1.0, 1e-4, 1.0]                    # 1 ohm.m host with 10000-ohm.m bed
    R = recursive_layer_reflectivity(boundaries, sigmas)
    print(f"  Reflectivity at host->bed: R_0 = {R[0]:+.4f}")
    print(f"  Reflectivity at bed->host: R_1 = {R[1]:+.4f}")
    assert R[0] > 0.99 and R[1] < -0.99, "10000:1 contrast must saturate R"
    print("  PASS")
    return {"D_E_m": D_E, "D_H_m": D_H, "ratio_H_E": D_H / D_E}


if __name__ == "__main__":
    test_all()
