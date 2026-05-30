"""
Article 10: Through-Casing Formation Conductivity Measurement Based on Transient
            Electromagnetic Logging Data
Sheng, Shen, Shen, Zhu, Zang (2019)
DOI: 10.30632/PJV60N5-2019a10

A transient-electromagnetic (TEM) method measures formation conductivity through
casing.  A step current is shut off and the decaying induced voltage is
recorded; the casing signal dominates early time, but the late-time decay is
governed by the formation conductivity, which is recovered from the late-time
response.

Implements:

  - Late-time TEM decay  V(t) ~ C * sigma^(3/2) * t^(-5/2)
  - Diffusion depth  d = sqrt(2*t/(mu*sigma))
  - Formation conductivity from the late-time response

Note: this issue's source-PDF text extract ended before this article (present
only as a table-of-contents entry), so this module is a faithful methodology
proxy implementing the standard late-time TEM relations the paper's title
describes.
"""

import numpy as np

MU0 = 4e-7 * np.pi


# ---------------------------------------------- TEM ---------------------

def late_time_voltage(t, sigma, C=1.0):
    """Late-time TEM induced voltage  V(t) = C*sigma^(3/2)*t^(-5/2)."""
    return C * sigma ** 1.5 * np.asarray(t, float) ** (-2.5)


def diffusion_depth(t, sigma, mu=MU0):
    """Electromagnetic diffusion depth  d = sqrt(2*t/(mu*sigma))  (m)."""
    return np.sqrt(2.0 * np.asarray(t, float) / (mu * sigma))


def conductivity_from_late_time(t, voltage, C=1.0):
    """Recover formation conductivity from a late-time decay sample.

        sigma = (V*t^(5/2)/C)^(2/3)
    """
    return (np.asarray(voltage, float) * np.asarray(t, float) ** 2.5 / C) ** (2.0 / 3.0)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 10: Through-Casing Conductivity via TEM")
    print("=" * 60)

    # Late-time voltage decays as t^(-5/2) and rises with conductivity
    t = np.array([1e-3, 4e-3, 16e-3])
    v = late_time_voltage(t, sigma=0.1)
    print(f"  V(t) decay ratio       = {v[0]/v[1]:.1f}  (expect 4^2.5 = 32)")
    assert abs(v[0] / v[1] - 4 ** 2.5) < 1e-6
    assert late_time_voltage(1e-3, 0.2) > late_time_voltage(1e-3, 0.1)

    # Diffusion depth grows with time and with resistivity (lower sigma)
    assert diffusion_depth(1e-2, 0.1) > diffusion_depth(1e-3, 0.1)
    assert diffusion_depth(1e-2, 0.01) > diffusion_depth(1e-2, 0.1)

    # Recover a planted formation conductivity from the late-time response
    sigma_true = 0.25                          # S/m (~4 ohm-m)
    t_late = 5e-3
    v_meas = late_time_voltage(t_late, sigma_true, C=2.0)
    sigma_hat = conductivity_from_late_time(t_late, v_meas, C=2.0)
    print(f"  recovered sigma        = {sigma_hat:.4f}  (true {sigma_true})")
    assert abs(sigma_hat - sigma_true) < 1e-9
    print("  PASS")
    return {"sigma_recovered": float(sigma_hat),
            "diff_depth": float(diffusion_depth(1e-2, 0.1))}


if __name__ == "__main__":
    test_all()
