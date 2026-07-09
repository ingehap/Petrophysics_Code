"""
Article 4: Finite-Volume Computations of Shale Tortuosity and Permeability From
           3D Pore Networks Extracted From Scanning Electron Tomographic Images
Almasoodi, Reza (2019)
DOI: 10.30632/PJV60N3-2019a3

Tortuosity and permeability are computed directly on a segmented pore network by
solving a conservation (Laplace) equation with the finite-volume method.  The
effective conductivity of the connected pore space gives the tortuosity
(tau = phi * sigma_fluid / sigma_eff), and a Kozeny-Carman relation converts the
pore geometry and tortuosity into permeability.

Implements:

  - Finite-volume Laplace solver on a 2D pore grid (harmonic face conductivity)
  - Effective conductivity from the steady-state flux
  - Tortuosity  tau = phi * sigma_fluid / sigma_eff
  - Kozeny-Carman permeability  k = phi^3 / (c * tau^2 * Sv^2)

Note: this issue's source PDF has no usable text layer (scanned issue), so the
titles/authors/DOIs are taken from the journal metadata and these are faithful
standard-form reconstructions of the finite-volume tortuosity/permeability
method the paper applies (a 2D FVM stands in for the 3D solve).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# ---------------------------------------------- FVM Laplace -------------

def solve_laplace(pore, sigma_pore=1.0, sigma_solid=1e-9):
    """Solve the FVM Laplace equation for potential across a 2D grid.

    `pore` is a binary (ny, nx) mask (True = pore).  A unit potential drop is
    applied left->right; returns (potential, effective_conductivity).
    Harmonic face conductivities couple neighbouring cells; Dirichlet on the
    left/right faces, no-flow top/bottom.
    """
    pore = np.asarray(pore, bool)
    ny, nx = pore.shape
    cond = np.where(pore, sigma_pore, sigma_solid)
    N = ny * nx
    idx = lambda r, c: r * nx + c
    A = np.zeros((N, N)); b = np.zeros(N)

    for r in range(ny):
        for c in range(nx):
            k = idx(r, c)
            if c == 0:
                A[k, k] = 1.0; b[k] = 1.0           # inlet potential = 1
                continue
            if c == nx - 1:
                A[k, k] = 1.0; b[k] = 0.0           # outlet potential = 0
                continue
            diag = 0.0
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx:
                    g = 2.0 * cond[r, c] * cond[rr, cc] / (cond[r, c] + cond[rr, cc])
                    A[k, idx(rr, cc)] -= g
                    diag += g
            A[k, k] = diag
    phi = np.linalg.solve(A, b).reshape(ny, nx)

    # total current across the first internal face, then normalize by the slab
    # geometry: sigma_eff = I*L/(A*dV), with L = nx-1, A = ny, dV = 1
    g_face = 2.0 * cond[:, 0] * cond[:, 1] / (cond[:, 0] + cond[:, 1])
    current = np.sum(g_face * (phi[:, 0] - phi[:, 1]))
    sigma_eff = current * (nx - 1) / ny
    return phi, float(sigma_eff)


# ---------------------------------------------- tortuosity / k ----------

def tortuosity(porosity, sigma_eff, sigma_fluid=1.0):
    """Electrical/diffusive tortuosity  tau = phi*sigma_fluid/sigma_eff."""
    return porosity * sigma_fluid / sigma_eff


def kozeny_carman(porosity, tau, surface_area, c=2.0):
    """Kozeny-Carman permeability  k = phi^3 / (c*tau^2*Sv^2)."""
    return petrolib.flow_transport.kozeny_carman(
        porosity, specific_surface=surface_area, tau=tau, c=c, grain_term=False)


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Shale Tortuosity & Permeability (FVM)")
    print("=" * 60)

    # A fully open slab has tortuosity ~ 1 (straight paths)
    open_grid = np.ones((10, 10), bool)
    _, sig_open = solve_laplace(open_grid)
    tau_open = tortuosity(1.0, sig_open)
    print(f"  open-slab sigma / tau  = {sig_open:.3f} / {tau_open:.3f}")
    assert abs(sig_open - 1.0) < 1e-6 and abs(tau_open - 1.0) < 1e-3

    # A tortuous path (offset barriers forcing the flow to wind) has tau > 1 and
    # lower effective conductivity
    grid = np.ones((11, 12), bool)
    grid[0:8, 4] = False                          # barrier from the top
    grid[3:11, 8] = False                         # barrier from the bottom
    phi_field, sig_t = solve_laplace(grid)
    poro = grid.mean()
    tau_t = tortuosity(poro, sig_t)
    print(f"  tortuous sigma / tau   = {sig_t:.3f} / {tau_t:.2f}")
    assert sig_t < sig_open and tau_t > tau_open

    # Kozeny-Carman: higher tortuosity -> lower permeability
    k_low_tau = kozeny_carman(0.2, 1.5, 1e6)
    k_high_tau = kozeny_carman(0.2, 4.0, 1e6)
    print(f"  k (tau 1.5 / 4.0)      = {k_low_tau:.2e} / {k_high_tau:.2e}")
    assert k_high_tau < k_low_tau
    print("  PASS")
    return {"tau_open": float(tau_open), "tau_tortuous": float(tau_t),
            "sigma_tortuous": sig_t}


if __name__ == "__main__":
    test_all()
