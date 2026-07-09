"""
Article 8: Mapping Complex Injectite Bodies With Multiwell Electromagnetic
           3D Inversion Data
Clegg, Eriksen, Best, Tollefsen, Kowicki, Marchant (2021)
DOI: 10.30632/PJV62N1-2021a7

Sand injectites are resistive bodies whose complex geometry is poorly resolved
by a single well.  A multiwell (cross-well / surface-to-borehole) electromagnetic
survey illuminates the interwell volume from many source-receiver paths; a
regularized 3D inversion then reconstructs the interwell conductivity, revealing
the injectite.  The modules implement the EM diffusion scale, a straight-path
cross-well sensitivity (attenuation-tomography) forward operator, and a
Tikhonov (smoothness-regularized) least-squares inversion that recovers a
resistive injectite anomaly between wells.

Implements:

  - EM skin depth  delta = sqrt(2/(w mu sigma)) = 503 sqrt(rho/f)
  - Cross-well path forward operator  d = G @ sigma  (path-integrated response)
  - Tikhonov inversion  m = (G^T G + lambda L^T L)^-1 G^T d
  - Resistive-body recovery on an interwell grid

Note: this issue's source PDF has no usable text layer, so the skin-depth
relation and the regularized-inversion scheme are faithful standard-form
reconstructions of the EM imaging the paper applies (full 3D Maxwell modelling
is replaced by a straight-path tomographic operator demonstrating the same
ill-posed inverse and its regularization).  Resistivity in ohm-m.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

MU0 = 4.0e-7 * np.pi      # H/m


# ---------------------------------------------- skin depth --------------

def skin_depth(rho_ohm_m, freq_hz):
    """EM skin depth  delta = sqrt(2 rho / (w mu0)) = 503 sqrt(rho/f)  (m)."""
    return petrolib.em_dielectric.skin_depth(rho_ohm_m, freq_hz)


# ---------------------------------------------- forward operator --------

def build_path_operator(nx, ny, sources, receivers):
    """Straight-path cross-well sensitivity matrix G (n_paths x n_cells).

    Each source-receiver path crosses the (nx, ny) interwell grid; G[i, j] is
    the length of path i inside cell j (a simple line-integral kernel), so the
    path datum is the path integral of the cell property.  sources/receivers
    are (x, y) coordinates in grid units.
    """
    rows = []
    for (sx, sy) in sources:
        for (rx, ry) in receivers:
            kernel = np.zeros(nx * ny)
            npts = 200
            xs = np.linspace(sx, rx, npts)
            ys = np.linspace(sy, ry, npts)
            seglen = np.hypot(rx - sx, ry - sy) / npts
            for x, y in zip(xs, ys):
                ix = min(max(int(x), 0), nx - 1)
                iy = min(max(int(y), 0), ny - 1)
                kernel[iy * nx + ix] += seglen
            rows.append(kernel)
    return np.array(rows)


def laplacian_operator(nx, ny):
    """First-difference smoothness operator L for Tikhonov regularization."""
    rows = []
    for iy in range(ny):
        for ix in range(nx):
            if ix < nx - 1:
                r = np.zeros(nx * ny)
                r[iy * nx + ix] = -1.0
                r[iy * nx + ix + 1] = 1.0
                rows.append(r)
            if iy < ny - 1:
                r = np.zeros(nx * ny)
                r[iy * nx + ix] = -1.0
                r[(iy + 1) * nx + ix] = 1.0
                rows.append(r)
    return np.array(rows)


# ---------------------------------------------- inversion ---------------

def tikhonov_inverse(G, d, L, lam):
    """Smoothness-regularized least squares  m = (G^T G + lam L^T L)^-1 G^T d."""
    GtG = G.T @ G
    LtL = L.T @ L
    return np.linalg.solve(GtG + lam * LtL, G.T @ d)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: Multiwell EM 3D Inversion (Injectites)")
    print("=" * 60)

    # Skin depth: 503*sqrt(rho/f); resistive injectite is probed deeper
    d_shale = skin_depth(2.0, 1000.0)
    d_sand = skin_depth(50.0, 1000.0)
    print(f"  skin depth shale / sand = {d_shale:.1f} / {d_sand:.1f} m")
    assert abs(skin_depth(1.0, 1.0) - 503.3) < 1.0
    assert d_sand > d_shale

    # Build a 12x12 interwell grid with a resistive injectite blob
    nx = ny = 12
    true = np.full((ny, nx), 0.05)            # background conductivity (S/m)
    true[5:9, 5:9] = 0.01                      # resistive injectite (lower sigma)
    m_true = true.ravel()

    # Multiwell illumination: a horizontal fan (left well -> right well) plus a
    # crossing vertical fan (top -> bottom) so the interwell body is resolved in
    # both directions rather than smeared along a single ray azimuth.
    left = [(0.0, y) for y in np.linspace(0, ny - 1, 12)]
    right = [(nx - 1.0, y) for y in np.linspace(0, ny - 1, 12)]
    top = [(x, 0.0) for x in np.linspace(0, nx - 1, 12)]
    bottom = [(x, ny - 1.0) for x in np.linspace(0, nx - 1, 12)]
    G = np.vstack([build_path_operator(nx, ny, left, right),
                   build_path_operator(nx, ny, top, bottom)])
    print(f"  G shape (paths x cells) = {G.shape}")

    # Forward data + a little noise; invert with smoothness regularization
    rng = np.random.default_rng(8)
    d = G @ m_true
    d += 0.01 * np.std(d) * rng.standard_normal(d.shape)
    L = laplacian_operator(nx, ny)
    m_inv = tikhonov_inverse(G, d, L, lam=2.0)

    img = m_inv.reshape(ny, nx)
    body = img[5:9, 5:9].mean()
    background = np.r_[img[:4, :4].ravel(), img[:4, -4:].ravel()].mean()
    print(f"  recovered body / background sigma = {body:.3f} / {background:.3f}")
    # the injectite cell block is recovered as more resistive (lower sigma)
    assert body < background
    # the recovered model correlates with the truth
    corr = np.corrcoef(m_inv, m_true)[0, 1]
    print(f"  model-truth correlation = {corr:.3f}")
    assert corr > 0.5
    print("  PASS")
    return {"corr": float(corr), "body": float(body),
            "background": float(background)}


if __name__ == "__main__":
    test_all()
