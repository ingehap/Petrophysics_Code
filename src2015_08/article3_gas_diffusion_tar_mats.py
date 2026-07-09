"""
Article 3: Gas Diffusion into Oil, Reservoir Baffling and Tar Mats Analyzed by
           Downhole Fluid Analysis, Pressure Transients, Core Extracts and DSTs
Achourov, Pfeiffer, Kollien, Betancourt, Zuo, di Primio, Mullins (2015)
Reference: Petrophysics Vol. 56, No. 4 (August 2015), pp. 346-357
DOI: none assigned (this issue predates SPWLA DOI assignment)

Two adjacent fault blocks with the same charge history are at different stages
of the same reservoir-fluid-geodynamic (RFG) process: a late gas charge diffuses
down into the oil column, expelling asphaltenes toward the base and forming a
tar mat.  The diffusion of gas into a semi-infinite oil column follows Fick's
law (complementary-error-function profile); asphaltene gradients are modeled
with the Flory-Huggins-Zuo (FHZ) gravity term (Yen-Mullins cluster sizes for
heavy oil), and a tar mat forms where the asphaltene content exceeds its
solubility limit.

Implements:

  - Diffusion length  L = sqrt(D*t)  and baffle equilibration time  tau = L^2/D
  - Fickian gas-concentration profile  C(x,t) = C0*erfc(x/(2*sqrt(D*t)))
  - FHZ asphaltene gravity gradient (cluster particle volume)
  - Tar-mat onset where asphaltene content exceeds the solubility limit

Note: this is a reservoir-fluid-geodynamics paper; the relations below are the
standard diffusion / FHZ physics it relies on.  The typeset glyphs were dropped
in extraction, so they are standard-form reconstructions.  SI units: D in m^2/s,
length in m, time in s, temperature in K.
"""

import math

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

KB = 1.380649e-23             # Boltzmann constant, J/K
G_EARTH = 9.81                # m/s^2


def _erfc(x):
    return np.array([math.erfc(v) for v in np.atleast_1d(x)])


# ---------------------------------------------- diffusion --------------

def diffusion_length(diffusion_coeff, time):
    """Characteristic diffusion length  L = sqrt(D*t)."""
    return petrolib.flow_transport.diffusion_length(diffusion_coeff, time)


def equilibration_time(length, diffusion_coeff):
    """Diffusive equilibration time across a length (e.g. a reservoir baffle)

        tau = L^2/D.
    """
    return petrolib.flow_transport.diffusion_time(length, diffusion_coeff)


def gas_concentration_profile(c0, x, diffusion_coeff, time):
    """Fickian gas-concentration profile into a semi-infinite oil column

        C(x,t) = C0 * erfc( x / (2*sqrt(D*t)) ),

    with x the distance from the gas-oil contact; the gas front advances as
    sqrt(D*t).
    """
    out = np.atleast_1d(petrolib.flow_transport.erfc_profile(c0, x, D=diffusion_coeff, t=time))
    return out if out.size > 1 else float(out[0])


# ---------------------------------------------- asphaltene / tar mat --------------

def fhz_asphaltene_gradient(od_ref, h_ref, h, particle_volume, delta_rho, temperature):
    """FHZ asphaltene optical-density gravity gradient (dominant term)

        OD(h) = OD_ref * exp[ Va*delta_rho*g*(h_ref - h)/(kB*T) ],

    with Va the asphaltene-particle volume (clusters for heavy oil); asphaltene
    content increases downward (h < h_ref).
    """
    exponent = particle_volume * delta_rho * G_EARTH * (h_ref - h) / (KB * temperature)
    return od_ref * np.exp(exponent)


def tar_mat_onset(asphaltene_content, solubility_limit):
    """Tar-mat onset where asphaltene content exceeds the solubility limit

        excess = asphaltene_content - solubility_limit  (tar mat forms if > 0).

    Returns (is_tar_mat, excess).
    """
    excess = asphaltene_content - solubility_limit
    return bool(excess > 0.0), excess


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Gas Diffusion into Oil & Tar Mats")
    print("=" * 60)

    # Diffusion length grows as sqrt(D*t); equilibration time as L^2/D
    d = 1.0e-9
    L = diffusion_length(d, 50e6 * 3.156e7)         # ~50 Myr
    print(f"  diffusion length 50Myr = {L:.1f} m")
    assert L > 0 and np.isclose(equilibration_time(L, d), (50e6 * 3.156e7), rtol=1e-6)

    # Fickian profile: full concentration at the GOC, decaying into the column
    assert np.isclose(gas_concentration_profile(1.0, 0.0, d, 1e15), 1.0)
    c_near = gas_concentration_profile(1.0, 5.0, d, 1e15)
    c_far = gas_concentration_profile(1.0, 50.0, d, 1e15)
    print(f"  gas conc near/far      = {c_near:.3f} / {c_far:.3f}")
    assert 1.0 > c_near > c_far > 0.0

    # FHZ: heavy-oil clusters give a strong downward asphaltene increase
    va_cluster = (4.0 / 3.0) * np.pi * (3.0e-9) ** 3
    od_base = fhz_asphaltene_gradient(0.5, h_ref=0.0, h=-50.0, particle_volume=va_cluster,
                                      delta_rho=80.0, temperature=360.0)
    assert od_base > 0.5

    # Tar mat forms only when asphaltene content exceeds the solubility limit
    is_tar, excess = tar_mat_onset(0.40, 0.35)
    no_tar, deficit = tar_mat_onset(0.20, 0.35)
    print(f"  tar mat (0.40/0.20)    = {is_tar} / {no_tar}")
    assert is_tar and not no_tar and excess > 0 and deficit < 0
    print("  PASS")
    return {"L_50Myr": float(L), "c_near": float(c_near), "OD_base": float(od_base)}


if __name__ == "__main__":
    test_all()
