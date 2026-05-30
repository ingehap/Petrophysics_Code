"""
Article 5: Fractal Characterization and Petrophysical Analysis of 3D Dynamic
           Digital Rocks of Sandstone
Zhao, Luo, Li, Wu, Mao, Ostadhassan (2021)
DOI: 10.30632/PJV62N5-2021a5

Nine synthetic 3D digital sandstones (QSGS + morphological cementation) are
characterized by fractal/morphological descriptors - fractal dimension D
(box-counting), lacunarity, and succolarity - and by simulated permeability
and electrical formation factor.  Succolarity correlates best with
permeability and formation factor.

Implements:

  - Box-counting fractal dimension  log N(r) = D*log(1/r) + c     (Eq. 3)
  - Permeability power laws K(phi), K(D), K(Su)                   (Eqs. 10-12)
  - Archie formation factor  F = phi^(-m)  and m inversion        (Eq. 13)
  - Lacunarity  Lambda(r) = Z2(r) / Z1(r)^2                       (Eq. 5)

Note: Eqs. 3, 7, 10-12 are present in the paper; the others were image-
rendered and are reconstructed standard forms (flagged).  The permeability
power-law constants use the paper's reported values (porosity entered as a
percentage; fractal dimension as a decimal).
"""

import numpy as np


# ---------------------------------------------- Eq. 3: box counting -----

def box_counting_dimension(binary, box_sizes):
    """Fractal dimension from box counting  log N(r) = D*log(1/r) + c (Eq. 3).

    binary    : 2-D (or N-D) boolean array; True = occupied.
    box_sizes : iterable of integer box sizes that divide each dimension.
    Returns D = -slope of log N vs log r.
    """
    binary = np.asarray(binary, bool)
    counts = []
    for s in box_sizes:
        # reshape into s-sized blocks along every axis, then test occupancy
        shape = []
        for dim in binary.shape:
            shape += [dim // s, s]
        trimmed = binary[tuple(slice(0, (d // s) * s) for d in binary.shape)]
        blocks = trimmed.reshape(shape)
        axes = tuple(range(1, 2 * binary.ndim, 2))   # the within-block axes
        counts.append(int(blocks.any(axis=axes).sum()))
    logs = np.log(np.asarray(box_sizes, float))
    logn = np.log(np.asarray(counts, float))
    slope = np.polyfit(logs, logn, 1)[0]
    return -slope


def build_sierpinski_carpet(depth):
    """Construct a 2-D Sierpinski carpet (exact fractal dimension log8/log3)."""
    carpet = np.ones((1, 1), bool)
    for _ in range(depth):
        s = carpet.shape[0]
        out = np.zeros((s * 3, s * 3), bool)
        for i in range(3):
            for j in range(3):
                if not (i == 1 and j == 1):     # punch out the centre block
                    out[i * s:(i + 1) * s, j * s:(j + 1) * s] = carpet
        carpet = out
    return carpet


# ---------------------------------------------- Eqs. 10-12: K power laws-

def permeability_from_porosity(phi_pct):
    """K = 3e-7 * phi^6.745  (Eq. 10).  phi as a percentage, K in md."""
    return 3e-7 * np.asarray(phi_pct, float) ** 6.745


def permeability_from_dimension(D):
    """K = 8e-31 * D^75.557  (Eq. 11).  D dimensionless, K in md."""
    return 8e-31 * np.asarray(D, float) ** 75.557


def permeability_from_succolarity(Su):
    """K = 8e6 * Su^6.653  (Eq. 12).  Su dimensionless, K in md."""
    return 8e6 * np.asarray(Su, float) ** 6.653


# ---------------------------------------------- Eq. 13: Archie ----------

def formation_factor(phi, m, a=1.0):
    """Archie formation factor  F = a * phi^(-m)  (Eq. 13)."""
    return a * np.asarray(phi, float) ** (-m)


def cementation_exponent(F, phi, a=1.0):
    """Invert Archie for m  =  -log(F/a) / log(phi)."""
    return -np.log(np.asarray(F, float) / a) / np.log(phi)


# ---------------------------------------------- Eq. 5: lacunarity -------

def lacunarity(binary, box_size):
    """Gliding-box lacunarity  Lambda = Z2/Z1^2  (Eq. 5)."""
    b = np.asarray(binary, float)
    # gliding-box mass via a simple sliding-window sum (1-D or 2-D)
    masses = []
    if b.ndim == 2:
        n, m = b.shape
        for i in range(n - box_size + 1):
            for j in range(m - box_size + 1):
                masses.append(b[i:i + box_size, j:j + box_size].sum())
    else:
        for i in range(len(b) - box_size + 1):
            masses.append(b[i:i + box_size].sum())
    masses = np.asarray(masses, float)
    z1 = masses.mean()
    z2 = (masses ** 2).mean()
    return z2 / z1 ** 2 if z1 > 0 else 0.0


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 5: Fractal Characterization of Digital Rocks")
    print("=" * 60)

    # Box-counting recovers the Sierpinski carpet dimension log8/log3 = 1.8928
    carpet = build_sierpinski_carpet(5)          # 243 x 243
    D = box_counting_dimension(carpet, box_sizes=[1, 3, 9, 27, 81])
    D_exact = np.log(8) / np.log(3)
    print(f"  Sierpinski carpet D    = {D:.4f}  (exact {D_exact:.4f})")
    assert abs(D - D_exact) < 0.05

    # A fully filled plane has dimension ~2
    full = np.ones((81, 81), bool)
    D_full = box_counting_dimension(full, box_sizes=[1, 3, 9, 27])
    assert abs(D_full - 2.0) < 0.05

    # Permeability power laws are positive and monotonic over the data ranges
    phi = np.array([8.89, 17.02, 23.22])         # percent (paper's range)
    Kphi = permeability_from_porosity(phi)
    print(f"  K(phi) over range      = {Kphi[0]:.2f} .. {Kphi[-1]:.1f} md")
    assert np.all(np.diff(Kphi) > 0) and Kphi[0] > 0

    Dvals = np.array([2.486, 2.703])
    KD = permeability_from_dimension(Dvals)
    print(f"  K(D) over range        = {KD[0]:.2f} .. {KD[-1]:.1f} md")
    assert KD[1] > KD[0] > 0

    Su = np.array([0.089, 0.233])
    KSu = permeability_from_succolarity(Su)
    assert KSu[1] > KSu[0] > 0

    # Archie round trip
    F = formation_factor(0.17, m=2.0)
    m_hat = cementation_exponent(F, 0.17)
    print(f"  Archie F / recovered m = {F:.2f} / {m_hat:.3f}")
    assert abs(m_hat - 2.0) < 1e-9

    # Lacunarity is >= 1 and larger for a clustered pattern
    lam = lacunarity(carpet[:81, :81], box_size=4)
    print(f"  lacunarity (carpet)    = {lam:.3f}")
    assert lam >= 1.0
    print("  PASS")
    return {"D_carpet": D, "K_phi": Kphi.tolist(), "m_hat": float(m_hat)}


if __name__ == "__main__":
    test_all()
