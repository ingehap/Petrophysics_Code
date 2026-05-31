"""
Article 4: Pore-Scale Evaluation of Dielectric Measurements in Formations with
           Complex Pore and Grain Structures
Huangye Chen and Zoya Heidari (2014)
Reference: Petrophysics Vol. 55, No. 6 (December 2014), pp. 587-597
DOI: none assigned (this issue predates SPWLA DOI assignment)

Pore-scale dielectric simulation on CT-scan voxel grids shows that the standard
CRIM mixing law mis-estimates water-filled porosity when the pore network is
tortuous or pyrite is present.  A new directional mixing model replaces the
porosity weight on the water term with a tortuosity-dependent coefficient, which
brings the water-filled-porosity error below 10%.

Implements:

  - Directional mean-square displacement and random-walk tortuosity (Eqs. 1-9)
  - Complex relative permittivity  eps* = eps' - j*sigma/(w*eps0)  (Eq. 10)
  - General and reservoir-rock CRIM mixing (Eqs. 14, 15) and the CRIM water
    saturation (Eq. 16)
  - New directional, tortuosity-weighted mixing model (Eqs. 17, 25) with the
    tortuosity coefficient  f = alpha*tau^gamma + beta  (Eq. 18)

Note: this issue's PDF has a text layer; the CRIM forms (Eqs. 14-16), the
complex-permittivity definition (Eq. 10), the random-walk tortuosity (Eqs. 1-9)
and the directional model (Eqs. 17-18, 25) are transcribed from the body, while
the dropped glyphs and the per-sample calibration constants (Eqs. 26-27) are
reconstructed / left as inputs.  Permittivities relative, sigma in S/m.
"""

import numpy as np

EPS0 = 8.854e-12  # vacuum permittivity, F/m


# ---------------------------------------------- random-walk tortuosity --------------

def mean_square_displacement(positions, t_index):
    """Directional and overall mean-square displacement of n random walkers
    (Eqs. 1-4)

        <x^2> = (1/n) sum_i x_i^2,   <r^2> = <x^2> + <y^2> + <z^2>,

    with ``positions`` an (n, 3, T) array of walker coordinates.  Returns
    (msd_xyz, msd_total) for the requested time index.
    """
    disp = positions[:, :, t_index] - positions[:, :, 0]
    msd_xyz = np.mean(disp ** 2, axis=0)
    return msd_xyz, float(msd_xyz.sum())


def tortuosity(msd_free, msd_porous):
    """Directional tortuosity factor (Eqs. 6-9)

        tau = (free-space MSD rate)/(actual porous-medium MSD rate),

    so a freely diffusing void cube has tau = 1 and a constricted network has
    tau > 1.
    """
    return np.asarray(msd_free, float) / np.asarray(msd_porous, float)


# ---------------------------------------------- complex permittivity --------------

def complex_permittivity(eps_real, sigma, freq):
    """Complex relative permittivity (Eq. 10)

        eps* = eps' - j*sigma/(w*eps0),   w = 2*pi*freq.
    """
    w = 2.0 * np.pi * freq
    return eps_real - 1j * sigma / (w * EPS0)


# ---------------------------------------------- CRIM --------------

def crim_general(eps_components, concentrations):
    """General CRIM mixing law (Eq. 14)

        sqrt(eps) = sum_i C_i*sqrt(eps_i),

    with volumetric concentrations C_i summing to one.
    """
    eps = np.asarray(eps_components, complex)
    c = np.asarray(concentrations, float)
    return (np.sum(c * np.sqrt(eps))) ** 2


def crim_reservoir(sw, phi, eps_w, eps_hc, eps_matrix):
    """Reservoir-rock CRIM (Eq. 15)

        sqrt(eps) = Sw*phi*sqrt(eps_w) + (1-Sw)*phi*sqrt(eps_hc)
                    + (1-phi)*sqrt(eps_matrix).
    """
    root = (sw * phi * np.sqrt(eps_w) + (1 - sw) * phi * np.sqrt(eps_hc)
            + (1 - phi) * np.sqrt(eps_matrix))
    return root ** 2


def crim_water_saturation(eps, phi, eps_w, eps_hc, eps_matrix):
    """Water saturation from reservoir CRIM (Eq. 16)."""
    num = np.sqrt(eps) - (1 - phi) * np.sqrt(eps_matrix) - phi * np.sqrt(eps_hc)
    den = phi * (np.sqrt(eps_w) - np.sqrt(eps_hc))
    return num / den


# ---------------------------------------------- directional model --------------

def tortuosity_coefficient(tau, alpha, gamma, beta):
    """Tortuosity-dependent mixing coefficient (Eq. 18)

        f = alpha*tau^gamma + beta,

    the power-law calibration that replaces the plain porosity weight on the
    water term.  alpha, gamma, beta depend on pore structure and grain shape.
    """
    return alpha * np.asarray(tau, float) ** gamma + beta


def directional_permittivity(fw, phi_w, eps_w, eps_matrix):
    """New directional mixing model, simplified for sand/carbonate (Eq. 25)

        sqrt(eps_x) = fw*phi_w*sqrt(eps_w) + (1-phi_w)*sqrt(eps_matrix),

    with the tortuosity coefficient fw on the water term and matrix/HC
    coefficients set to one.
    """
    root = fw * phi_w * np.sqrt(eps_w) + (1 - phi_w) * np.sqrt(eps_matrix)
    return root ** 2


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 4: Pore-Scale Dielectric Measurements")
    print("=" * 60)

    freq = 1e9  # 1 GHz

    # A constricted network diffuses slower than free space -> tau > 1
    n, T = 200, 50
    rng = np.random.default_rng(0)
    free = np.cumsum(rng.normal(0, 1.0, (n, 3, T)), axis=2)
    porous = np.cumsum(rng.normal(0, 1.0, (n, 3, T)), axis=2) * 0.6
    _, msd_free = mean_square_displacement(free, T - 1)
    _, msd_por = mean_square_displacement(porous, T - 1)
    tau = tortuosity(msd_free, msd_por)
    print(f"  tortuosity tau = {tau:.3f}")
    assert tau > 1.0

    # Complex permittivity: conductivity loads the imaginary part
    eps_star = complex_permittivity(15.0, sigma=0.1, freq=freq)
    assert np.isclose(eps_star.real, 15.0) and eps_star.imag < 0

    # CRIM round-trips Sw, and the general form matches the reservoir form
    sw_true, phi = 0.35, 0.20
    eps = crim_reservoir(sw_true, phi, eps_w=76.0, eps_hc=2.2, eps_matrix=7.5)
    eps_gen = crim_general([76.0, 2.2, 7.5],
                           [sw_true * phi, (1 - sw_true) * phi, 1 - phi])
    assert np.isclose(eps, eps_gen)
    sw_rec = crim_water_saturation(eps, phi, 76.0, 2.2, 7.5)
    print(f"  CRIM eps={eps:.2f}  Sw_recovered={sw_rec:.3f}")
    assert np.isclose(sw_rec, sw_true)

    # Directional model: the tortuosity coefficient grows with tortuosity, and
    # for fw=1 it collapses onto the plain CRIM water-filled form
    fw = tortuosity_coefficient(tau, alpha=0.05, gamma=1.5, beta=0.8)
    fw2 = tortuosity_coefficient(2 * tau, alpha=0.05, gamma=1.5, beta=0.8)
    assert fw2 > fw
    phi_w = sw_true * phi
    eps_dir = directional_permittivity(1.0, phi_w, 76.0, 7.5)
    eps_crim_dry = crim_reservoir(1.0, phi_w, 76.0, 2.2, 7.5)  # Sw=1 -> all water
    print(f"  directional eps (fw=1) = {eps_dir:.2f}")
    assert np.isclose(eps_dir, eps_crim_dry)
    print("  PASS")
    return {"tau": float(tau), "Sw": float(sw_rec), "fw": float(fw)}


if __name__ == "__main__":
    test_all()
