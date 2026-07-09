"""
Article 5: 'Log-Soak-Log' Experiment in Tengiz Field: Novel Technology for
           In-Situ Imbibition Measurements to Support an Improved Oil Recovery
           Project
Seth, Villegas, Iskakov, Playton, Lindsell, Cordova, Turmanbekova, Wang (2019)
DOI: 10.30632/PJV60N5-2019a5

A high-salinity brine is communicated into the matrix and time-lapse
pulsed-neutron Sigma (thermal-neutron capture cross section) logging before,
during and after a soak tracks the water-saturation change from spontaneous
imbibition, quantifying in-situ imbibition to support an improved-oil-recovery
project.  The brine salinity is boosted so the Sigma contrast between brine and
oil makes a small saturation change detectable.

Implements:

  - Sigma water saturation from the porosity balance
  - Saturation change from time-lapse Sigma (log - soak - log)
  - Sigma sensitivity per unit Sw and detectability vs noise
  - Salinity -> brine Sigma (capture units)

Note: this issue's PDF has a text layer but the paper is a field-experiment /
design study with no numbered equations; these are the standard Sigma-saturation
relations the method relies on.  Paper anchors: brine ~220 c.u.
(~450,000 ppm), design target to detect a >5% Sw change in a 2-p.u. rock.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- sigma saturation --------

def sigma_water_saturation(sigma_log, phi, sigma_ma, sigma_w, sigma_hc):
    """Water saturation from the Sigma porosity balance (clipped to [0,1])."""
    return petrolib.nuclear.sw_from_sigma(
        sigma_log, phi, sigma_ma=sigma_ma, sigma_w=sigma_w, sigma_hc=sigma_hc
    )


def saturation_change(sigma_before, sigma_after, phi, sigma_ma, sigma_w, sigma_hc):
    """Water-saturation change between two time-lapse Sigma logs (after - before)."""
    sw_b = sigma_water_saturation(sigma_before, phi, sigma_ma, sigma_w, sigma_hc)
    sw_a = sigma_water_saturation(sigma_after, phi, sigma_ma, sigma_w, sigma_hc)
    return sw_a - sw_b


def sigma_sensitivity(phi, sigma_w, sigma_hc):
    """Sigma swing per unit Sw  dSigma/dSw = phi*(Sigma_w - Sigma_hc)  (c.u.)."""
    return petrolib.nuclear.sigma_sensitivity(phi, sigma_w, sigma_hc)


def detectable(dsw, phi, sigma_w, sigma_hc, sigma_noise=1.0):
    """True if a saturation change dsw produces a Sigma shift above the noise."""
    return abs(dsw) * sigma_sensitivity(phi, sigma_w, sigma_hc) > sigma_noise


def brine_sigma(salinity_ppm):
    """Approximate brine Sigma (c.u.) from NaCl salinity (ppm).

    ~22 c.u. fresh water rising ~ linearly; ~450,000 ppm -> ~220 c.u.
    """
    return petrolib.nuclear.sigma_w_from_salinity(salinity_ppm, model="linear450k")


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Log-Soak-Log In-Situ Imbibition (Tengiz)")
    print("=" * 60)

    # High-salinity brine gives a high Sigma (large contrast with oil)
    sig_w = brine_sigma(450000.0)
    print(f"  brine Sigma @450 kppm  = {sig_w:.0f} c.u.")
    assert abs(sig_w - 220.0) < 1.0
    assert brine_sigma(450000.0) > brine_sigma(50000.0)

    # Imbibition raises water saturation -> Sigma increases over the soak
    phi, sig_ma, sig_hc = 0.02, 9.0, 22.0          # 2-p.u. rock
    # plant Sw 0.30 -> 0.45 (a 15% change); forward to Sigma before/after
    sw0, sw1 = 0.30, 0.45
    sig_before = sig_ma * (1 - phi) + phi * (sw0 * sig_w + (1 - sw0) * sig_hc)
    sig_after = sig_ma * (1 - phi) + phi * (sw1 * sig_w + (1 - sw1) * sig_hc)
    dsw = saturation_change(sig_before, sig_after, phi, sig_ma, sig_w, sig_hc)
    print(f"  recovered dSw          = {dsw:.3f}  (true {sw1 - sw0:.2f})")
    assert abs(dsw - (sw1 - sw0)) < 1e-6

    # Detectability: a 5% change in a 2-p.u. rock with 220-c.u. brine produces a
    # ~0.2 c.u. shift, above the ~0.15 c.u. Sigma repeatability the high-salinity
    # brine design achieves (the experiment's design criterion)
    print(f"  Sigma sensitivity      = {sigma_sensitivity(phi, sig_w, sig_hc):.2f} c.u./Sw")
    assert detectable(0.05, phi, sig_w, sig_hc, sigma_noise=0.15)
    # with fresh water (zero brine-oil contrast) the same change is undetectable
    assert not detectable(0.05, phi, brine_sigma(0.0), sig_hc, sigma_noise=0.15)
    print("  PASS")
    return {"brine_sigma": float(sig_w), "dSw": float(dsw),
            "sensitivity": float(sigma_sensitivity(phi, sig_w, sig_hc))}


if __name__ == "__main__":
    test_all()
