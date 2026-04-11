"""
Garcia et al. (2023), Petrophysics 64(6): 879-889.
Tracking fluid components from 2D NMR (T1-T2 or D-T2) maps by approximating the
map as a superposition of 2D Gaussian distributions; pore volume of each fluid
is the integral of its Gaussian.
"""
import numpy as np
from scipy.optimize import least_squares


def gaussian2d(X, Y, A, mx, my, sx, sy):
    return A * np.exp(-0.5 * (((X - mx) / sx) ** 2 + ((Y - my) / sy) ** 2))


def synth_map(grid_x, grid_y, components):
    X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")
    M = np.zeros_like(X)
    for c in components:
        M += gaussian2d(X, Y, *c)
    return M


def fit_gaussians(M, grid_x, grid_y, n_components, init=None):
    X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")

    def unpack(p): return p.reshape(n_components, 5)

    def resid(p):
        comps = unpack(p)
        pred = np.zeros_like(M)
        for c in comps:
            pred += gaussian2d(X, Y, *c)
        return (pred - M).ravel()

    if init is None:
        init = []
        for i in range(n_components):
            init += [M.max() / n_components,
                     grid_x[len(grid_x) * (i + 1) // (n_components + 1)],
                     grid_y[len(grid_y) * (i + 1) // (n_components + 1)],
                     (grid_x[-1] - grid_x[0]) / 8,
                     (grid_y[-1] - grid_y[0]) / 8]
    res = least_squares(resid, x0=np.array(init))
    return unpack(res.x)


def pore_volume(component, dx, dy):
    A, _, _, sx, sy = component
    return A * 2 * np.pi * sx * sy  # analytic gaussian integral


def test_all():
    gx = np.linspace(-3, 3, 60); gy = np.linspace(-3, 3, 60)
    truth = [(1.0, -1.0, -1.0, 0.4, 0.4),  # oil
             (0.7,  1.2,  0.5, 0.5, 0.3)]  # water
    M = synth_map(gx, gy, truth)
    fitted = fit_gaussians(M, gx, gy, n_components=2)
    pv = [pore_volume(c, gx[1] - gx[0], gy[1] - gy[0]) for c in fitted]
    print("Garcia et al. 2D NMR Gaussian decomposition:")
    print(f"  fitted means: {[(round(c[1],2), round(c[2],2)) for c in fitted]}")
    print(f"  pore volumes: {[round(v,3) for v in pv]}")
    assert len(fitted) == 2
    print("  PASS")


if __name__ == "__main__":
    test_all()
