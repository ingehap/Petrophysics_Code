"""
Article 8: The Effect of Joint Roughness on Shear Behavior of 3D-Printed
           Samples Containing a Non-Persistent Joint
Fereshtenejad, Kim, Song (2021)
DOI: 10.30632/PJV62N5-2021a8

Powder-based (binder-jetting) 3D printing is evaluated for reproducing the
shear behavior of rock joints at three controlled roughness levels
(JRC 6.5 / 11.5 / 17.5).  Direct-shear tests under constant normal load are
run on pure-joint and non-persistent-joint samples and compared to plaster.

Implements:

  - Z2 root-mean-square slope of a joint profile
  - Tse & Cruden (1979) correlation  JRC = 32.2 + 32.47*log10(Z2)
  - Barton-Bandis peak shear strength
        tau = sigma_n * tan(phi_b + JRC*log10(JCS/sigma_n))
  - Mohr-Coulomb shear strength  tau = c + sigma_n*tan(phi)
  - Secant shear stiffness  k_s = tau_peak / u_peak

Note: this paper transcribes no equations; the relations here are standard
rock-mechanics forms (Tse & Cruden is the one the paper explicitly uses),
flagged as reconstructions.  Stresses in MPa, angles in degrees, JRC 0-20.
"""

import numpy as np


# ---------------------------------------------- Z2 / JRC ---------------

def z2_roughness(profile, dx):
    """Root-mean-square first derivative (slope) Z2 of a height profile."""
    y = np.asarray(profile, float)
    slopes = np.diff(y) / dx
    return float(np.sqrt(np.mean(slopes ** 2)))


def jrc_from_z2(z2):
    """Tse & Cruden (1979)  JRC = 32.2 + 32.47*log10(Z2), clipped to [0, 20]."""
    jrc = 32.2 + 32.47 * np.log10(z2)
    return float(np.clip(jrc, 0.0, 20.0))


# ---------------------------------------------- Barton-Bandis ----------

def barton_bandis_strength(sigma_n, jrc, jcs, phi_b):
    """Barton-Bandis peak shear strength (MPa).

    tau = sigma_n * tan(phi_b + JRC*log10(JCS/sigma_n)).
    """
    i = jrc * np.log10(jcs / sigma_n)             # mobilized roughness angle
    return sigma_n * np.tan(np.radians(phi_b + i))


def dilation_angle(jrc, jcs, sigma_n):
    """Mobilized dilation (roughness) angle  i = JRC*log10(JCS/sigma_n)."""
    return jrc * np.log10(jcs / sigma_n)


# ---------------------------------------------- Mohr-Coulomb -----------

def mohr_coulomb_strength(cohesion, sigma_n, phi):
    """Mohr-Coulomb shear strength  tau = c + sigma_n*tan(phi)."""
    return cohesion + sigma_n * np.tan(np.radians(phi))


# ---------------------------------------------- stiffness --------------

def shear_stiffness(tau_peak, u_peak):
    """Secant shear stiffness  k_s = tau_peak / u_peak  (MPa/mm)."""
    return tau_peak / u_peak


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Joint Roughness & Shear of 3D-Printed Samples")
    print("=" * 60)

    # A rougher synthetic profile yields a larger Z2 and JRC
    x = np.arange(0, 100.0, 0.5)
    smooth = 0.5 * np.sin(2 * np.pi * x / 40.0)
    rough = smooth + 1.5 * np.sin(2 * np.pi * x / 5.0)
    z2_s = z2_roughness(smooth, 0.5)
    z2_r = z2_roughness(rough, 0.5)
    jrc_s = jrc_from_z2(z2_s)
    jrc_r = jrc_from_z2(z2_r)
    print(f"  Z2 smooth / rough      = {z2_s:.3f} / {z2_r:.3f}")
    print(f"  JRC smooth / rough     = {jrc_s:.1f} / {jrc_r:.1f}")
    assert z2_r > z2_s and jrc_r > jrc_s
    assert 0.0 <= jrc_r <= 20.0

    # Tse-Cruden anchor: Z2 ~ 0.205 -> JRC ~ 9.8
    assert abs(jrc_from_z2(0.205) - 9.86) < 0.3

    # Barton-Bandis: shear strength rises with roughness and normal stress
    tau = barton_bandis_strength(sigma_n=1.0, jrc=11.5, jcs=20.0, phi_b=30.0)
    print(f"  Barton-Bandis tau      = {tau:.3f} MPa (JRC=11.5, sn=1)")
    assert abs(tau - 1.0) < 0.05      # ~ tan(45 deg)

    taus = [barton_bandis_strength(1.0, j, 20.0, 30.0) for j in (6.5, 11.5, 17.5)]
    assert taus[0] < taus[1] < taus[2]
    tau_lo = barton_bandis_strength(0.5, 11.5, 20.0, 30.0)
    tau_hi = barton_bandis_strength(1.5, 11.5, 20.0, 30.0)
    assert tau_hi > tau_lo

    # Mohr-Coulomb and stiffness
    mc = mohr_coulomb_strength(cohesion=0.2, sigma_n=1.0, phi=35.0)
    ks = shear_stiffness(tau_peak=1.0, u_peak=0.8)
    print(f"  Mohr-Coulomb tau       = {mc:.3f} MPa;  k_s = {ks:.2f} MPa/mm")
    assert mc > 0 and ks > 0
    print("  PASS")
    return {"JRC_rough": jrc_r, "tau_BB": tau, "tau_by_JRC": taus}


if __name__ == "__main__":
    test_all()
