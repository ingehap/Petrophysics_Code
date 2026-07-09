"""
Article 3: Self-Compensated Pulsed-Neutron Spectroscopy Measurements
Zhou, Rose, Miles, Gendur, Wang, Sullivan (2020)
DOI: 10.30632/PJV61N6-2020a3

The yields-to-weights normalization FY2W (which turns relative elemental yields
into dry-weight fractions) is classically derived from oxide closure or
inelastic-capture closure, both noisy in cased hole.  The self-compensated
algorithm instead predicts the capture and inelastic FY2W directly from
high-precision raw measurements (apparent sigma, count-rate ratios, hole size)
for each detector, and forms differential near-over-far dry-weight elements for
Fe, Ca, Si so that borehole/completion contributions (cement Ca/Si, OBM carbon,
casing Fe) cancel.

Implements:

  - Yields-to-weights  W_i = FY2W * S_i * Y_i                 (Eq. 1)
  - Predicted capture / inelastic FY2W from raw measurements  (Eqs. 2-5 proxy)
  - Apparent dry-weight elements for near/far x capture/inelastic
  - Differential (near - far) element that cancels borehole signal

Note: this issue's PDF text layer dropped the typeset formula bodies of
Eqs. 2-5, so the FY2W predictor here is a faithful standard-form proxy of the
described method (FY2W rising with hole size, far-detector inelastic FY2W below
near because of its deeper depth of investigation).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- yields-to-weights -------

def yield_to_weight(fy2w, S, Y):
    """Dry-weight element from yield  W = FY2W * S * Y  (Eq. 1)."""
    return petrolib.nuclear.yields_to_weights(fy2w, S, Y)


# ---------------------------------------------- predicted FY2W ----------

def predict_fy2w(sigma_app, hole_in, mode="capture", detector="near"):
    """Predict the yields-to-weights factor from raw measurements (Eqs. 2-5).

    A smooth surrogate: FY2W rises with borehole size (more environmental
    dilution to undo) and with apparent sigma; the far detector's larger depth
    of investigation gives a smaller inelastic FY2W than the near detector.
    """
    base = 1.0 + 0.03 * (hole_in - 6.0) + 0.004 * (sigma_app - 20.0)
    if mode == "inelastic":
        base *= 1.15 if detector == "near" else 1.02   # far DOI deeper -> lower
    return base


# ---------------------------------------------- differential element ----

def apparent_element(formation_w, borehole_w, depth_weight):
    """Apparent dry-weight element = formation*sensitivity + borehole (additive).

    depth_weight is the detector's relative sensitivity to the formation; the
    borehole contribution is common-mode (adds the same to each detector).
    """
    return formation_w * depth_weight + borehole_w


def differential_dry_weight(w_near, w_far):
    """Near - far differential dry-weight element (cancels common borehole)."""
    return np.asarray(w_near, float) - np.asarray(w_far, float)


def recover_formation(w_near, w_far, k_near, k_far):
    """Recover the formation element from the near/far differential.

        W_form = (w_near - w_far) / (k_near - k_far)
    The common borehole term cancels in the numerator.
    """
    return differential_dry_weight(w_near, w_far) / (k_near - k_far)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: Self-Compensated Pulsed-Neutron Spectroscopy")
    print("=" * 60)

    # FY2W prediction: rises with hole size; far inelastic below near inelastic
    f_small = predict_fy2w(22.0, hole_in=6.0)
    f_big = predict_fy2w(22.0, hole_in=12.0)
    print(f"  capture FY2W 6in / 12in = {f_small:.3f} / {f_big:.3f}")
    assert f_big > f_small
    f_in_near = predict_fy2w(22.0, 8.0, mode="inelastic", detector="near")
    f_in_far = predict_fy2w(22.0, 8.0, mode="inelastic", detector="far")
    print(f"  inelastic FY2W near/far = {f_in_near:.3f} / {f_in_far:.3f}")
    assert f_in_far < f_in_near

    # Differential near-far cancels a borehole Ca (cement) contribution: vary the
    # cement Ca and the recovered formation Ca must not change.
    ca_form = 0.12                     # true formation calcium dry weight
    k_near, k_far = 0.55, 0.85         # far detector sees more formation
    for cement_ca in (0.0, 0.10, 0.30):
        w_near = apparent_element(ca_form, cement_ca, k_near)
        w_far = apparent_element(ca_form, cement_ca, k_far)
        ca_rec = recover_formation(w_near, w_far, k_near, k_far)
        print(f"  cement Ca {cement_ca:.2f} -> recovered formation Ca = {ca_rec:.4f}")
        assert abs(ca_rec - ca_form) < 1e-9

    # Yields-to-weights basic check
    W = yield_to_weight(f_big, 0.9, 0.2)
    assert W > 0
    print("  PASS")
    return {"fy2w_12in": f_big, "fy2w_inel_near": f_in_near,
            "fy2w_inel_far": f_in_far, "ca_form": ca_form}


if __name__ == "__main__":
    test_all()
