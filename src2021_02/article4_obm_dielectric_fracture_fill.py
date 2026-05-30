"""
Article 4: Identifying Fracture-Filling Material in Oil-Based Mud With
           Dielectric Borehole Imaging
Schlicht, Zhang, Luling, Graham, Cournot, Sadownyk (2021)
DOI: 10.30632/PJV62N1-2021a3

In oil-based mud a galvanic resistivity imager cannot inject current across the
insulating mud, so a high-frequency dielectric (displacement-current) imager is
used.  The button response is governed by the complex permittivity of the
material in front of the pad; cemented (calcite), open oil/mud-filled, and
conductive (clay/brine) fracture fills separate cleanly in the apparent
permittivity / loss-tangent plane, which is the basis for classifying the
fracture-filling material from the image.

Implements:

  - CRIM permittivity mixing  sqrt(eps_eff) = sum_i phi_i sqrt(eps_i)
  - Complex permittivity  eps* = eps' - j*( eps'' + sigma/(w eps0) )
  - Loss tangent  tan(d) = eps_imag / eps_real
  - Button-electrode admittance of a thin fracture gap
  - Fracture-fill classifier (open / calcite-cemented / conductive)

Note: this issue's source PDF has no usable text layer, so the CRIM mixing law
and the complex-permittivity relations are faithful standard-form
reconstructions of the dielectric-imaging physics the paper applies.  Frequency
in Hz; permittivities relative (to vacuum); conductivity in S/m.
"""

import numpy as np

EPS0 = 8.854e-12         # F/m

# Reference relative permittivities of candidate fracture fills
EPS_OIL = 2.2            # oil / oil-based mud
EPS_CALCITE = 7.5        # calcite cement
EPS_WATER = 78.0         # brine / water-bearing clay
EPS_ROCK = 6.0           # tight carbonate/shale matrix


# ---------------------------------------------- CRIM mixing -------------

def crim_permittivity(volumes, perms):
    """Complex-refractive-index-method effective permittivity.

        sqrt(eps_eff) = sum_i phi_i sqrt(eps_i)
    volumes are volume fractions (sum to 1); perms the component permittivities.
    """
    v = np.asarray(volumes, float)
    e = np.asarray(perms, float)
    return float((np.sum(v * np.sqrt(e))) ** 2)


# ---------------------------------------------- complex permittivity ----

def complex_permittivity(eps_real, sigma, freq_hz, eps_imag=0.0):
    """Complex relative permittivity  eps* = eps' - j(eps'' + sigma/(w eps0))."""
    w = 2.0 * np.pi * freq_hz
    return eps_real - 1j * (eps_imag + sigma / (w * EPS0))


def loss_tangent(eps_real, sigma, freq_hz, eps_imag=0.0):
    """Loss tangent  tan(d) = Im(eps*) / Re(eps*)  (separates conductive fill)."""
    e = complex_permittivity(eps_real, sigma, freq_hz, eps_imag)
    return float(-e.imag / e.real)


# ---------------------------------------------- button admittance -------

def gap_admittance(eps_real, sigma, freq_hz, area_m2, gap_m):
    """Complex admittance of a parallel-plate fracture gap in front of a button.

        Y = (sigma + j w eps0 eps') * A / d
    The real part is conductive, the imaginary part capacitive (displacement).
    """
    w = 2.0 * np.pi * freq_hz
    return (sigma + 1j * w * EPS0 * eps_real) * area_m2 / gap_m


# ---------------------------------------------- classifier --------------

def classify_fill(eps_app, tan_d, eps_open=3.5, tan_cond=0.5):
    """Classify the fracture fill from apparent permittivity and loss tangent.

      tan(d) > tan_cond            -> conductive (clay / brine) fill
      eps_app < eps_open           -> open (oil / mud-filled) fracture
      otherwise                    -> mineral-cemented (calcite) fill
    """
    if tan_d > tan_cond:
        return "conductive (clay/brine)"
    if eps_app < eps_open:
        return "open (oil/mud)"
    return "calcite-cemented"


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Fracture Fill From Dielectric Imaging (OBM)")
    print("=" * 60)

    freq = 1e9            # 1 GHz dielectric imager

    # CRIM: a calcite-cemented fracture reads higher permittivity than an
    # oil-filled one (matrix mixed with a 30% fracture aperture)
    eps_open = crim_permittivity([0.7, 0.3], [EPS_ROCK, EPS_OIL])
    eps_cal = crim_permittivity([0.7, 0.3], [EPS_ROCK, EPS_CALCITE])
    eps_wet = crim_permittivity([0.7, 0.3], [EPS_ROCK, EPS_WATER])
    print(f"  eps  open / calcite / wet = {eps_open:.2f} / {eps_cal:.2f} / {eps_wet:.2f}")
    assert eps_open < eps_cal < eps_wet

    # Loss tangent flags the conductive (brine) fill; oil/calcite are low-loss.
    # Classify on the high-resolution button reading over the fracture trace
    # itself (the fill permittivity), not the matrix-diluted bulk mixture.
    td_oil = loss_tangent(EPS_OIL, sigma=1e-9, freq_hz=freq)
    td_cal = loss_tangent(EPS_CALCITE, sigma=1e-4, freq_hz=freq)
    td_wet = loss_tangent(EPS_WATER, sigma=4.0, freq_hz=freq)
    print(f"  loss tangent oil / calcite / wet = {td_oil:.3f} / {td_cal:.3f} / {td_wet:.3f}")
    assert td_wet > td_cal and td_wet > td_oil

    # Button admittance: conductive gap is dominated by its real (conductive)
    # part, the oil gap by its imaginary (capacitive) part
    Y_oil = gap_admittance(EPS_OIL, 1e-9, freq, area_m2=1e-4, gap_m=2e-3)
    Y_brine = gap_admittance(EPS_WATER, 4.0, freq, area_m2=1e-4, gap_m=2e-3)
    assert abs(Y_oil.imag) > abs(Y_oil.real)            # oil: capacitive
    assert Y_brine.real > abs(Y_oil.real)               # brine: conductive

    # Classifier separates the three fracture-fill types
    c_open = classify_fill(EPS_OIL, td_oil)
    c_cal = classify_fill(EPS_CALCITE, td_cal)
    c_wet = classify_fill(EPS_WATER, td_wet)
    print(f"  classes: {c_open} | {c_cal} | {c_wet}")
    assert c_open == "open (oil/mud)"
    assert c_cal == "calcite-cemented"
    assert c_wet.startswith("conductive")
    print("  PASS")
    return {"eps_open": eps_open, "eps_cal": eps_cal, "eps_wet": eps_wet}


if __name__ == "__main__":
    test_all()
