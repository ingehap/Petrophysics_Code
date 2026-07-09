"""
Article 7: Using Digital Rock Physics to Evaluate Novel Percussion Core Quality
Lakshtanov, Zapata, Saucier, Cook, Eve, Lancaster, Lane, Gettemy, Sincock,
Liu, Geetan, Draper, Gill (2022)
DOI: 10.30632/PJV63N6-2022a7

Implements the digital-rock-physics QC workflow the paper applies to
pre- and post-test micro-CT scans of percussion-sidewall-core (PSWC)
plugs:

  - Synthetic 3-D voxel sand-pack with overlapping spherical "grains"
    (the analogue of a binary-segmented micro-CT cube).
  - "Pre-test" geometry and "post-test" geometry: the latter includes
    a percussion-damage zone (grain crushing -> porosity reduction and
    smaller effective grain size).
  - Porosity from pore-voxel count (binary segmentation).
  - Kozeny-Carman absolute permeability from porosity and the
    surface-area-to-volume ratio measured directly on the cube:

        k = phi^3 / (c * Sv^2 * (1 - phi)^2)            (Eq. KC)

    where Sv is the specific surface area (1/length) estimated from the
    pore/grain interface count and c is a tortuosity-shape constant.
  - Per-slice porosity and per-slice permeability profiles - the
    damage map the paper uses to spot crushed / compacted layers.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# --------------------------------------------- synthetic sand-pack -----

def make_sand_pack(shape=(64, 64, 64), grain_radius_voxels=4,
                   n_grains=350, seed=0):
    """Generate a binary voxel cube: 0 = pore, 1 = grain."""
    rng = np.random.default_rng(seed)
    cube = np.zeros(shape, dtype=np.uint8)
    nx, ny, nz = shape
    centres_x = rng.integers(0, nx, n_grains)
    centres_y = rng.integers(0, ny, n_grains)
    centres_z = rng.integers(0, nz, n_grains)
    x = np.arange(nx)[:, None, None]
    y = np.arange(ny)[None, :, None]
    z = np.arange(nz)[None, None, :]
    for cx, cy, cz in zip(centres_x, centres_y, centres_z):
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        cube[d2 <= grain_radius_voxels ** 2] = 1
    return cube


def apply_percussion_damage(cube, damage_layer=slice(20, 36),
                            fines_radius=2, n_fines=900, seed=1):
    """Crush grains in a depth band - add many small grains (fines) and
    keep large grains untouched -> the band's porosity drops + Sv rises.
    """
    rng = np.random.default_rng(seed)
    damaged = cube.copy()
    sub = damaged[:, :, damage_layer].copy()
    nx, ny = sub.shape[0], sub.shape[1]
    nz_d = sub.shape[2]
    cx = rng.integers(0, nx, n_fines)
    cy = rng.integers(0, ny, n_fines)
    cz = rng.integers(0, nz_d, n_fines)
    x = np.arange(nx)[:, None, None]
    y = np.arange(ny)[None, :, None]
    z = np.arange(nz_d)[None, None, :]
    for cxi, cyi, czi in zip(cx, cy, cz):
        d2 = (x - cxi) ** 2 + (y - cyi) ** 2 + (z - czi) ** 2
        sub[d2 <= fines_radius ** 2] = 1
    damaged[:, :, damage_layer] = sub
    return damaged


# --------------------------------------------- porosity ----------------

def porosity(cube):
    """Pore-voxel fraction."""
    return float((cube == 0).sum() / cube.size)


def porosity_profile(cube, axis=2):
    """Per-slice porosity along the requested axis."""
    other = tuple(i for i in range(cube.ndim) if i != axis)
    return (cube == 0).mean(axis=other)


# --------------------------------------------- specific surface area ---

def specific_surface_area_voxels(cube):
    """Count of grain-pore voxel interfaces divided by the total voxel
    count - a direct discrete proxy for Sv (1 / voxel).
    """
    faces = 0
    grain = cube == 1
    pore = cube == 0
    for axis in range(3):
        a = np.swapaxes(grain, 0, axis)
        b = np.swapaxes(pore, 0, axis)
        faces += int(((a[:-1] & b[1:]) | (b[:-1] & a[1:])).sum())
    return faces / cube.size


def specific_surface_area_profile(cube, axis=2):
    """Per-slice Sv along the requested axis."""
    n = cube.shape[axis]
    out = np.zeros(n)
    for k in range(n):
        if axis == 0:
            sub = cube[k:k + 1]
        elif axis == 1:
            sub = cube[:, k:k + 1]
        else:
            sub = cube[:, :, k:k + 1]
        out[k] = specific_surface_area_voxels(sub)
    return out


# --------------------------------------------- Kozeny-Carman ----------

def kozeny_carman_permeability(phi, Sv, c=5.0, voxel_size_um=2.0):
    """k = phi^3 / (c * Sv^2 * (1 - phi)^2), Sv in 1/voxel.

    Returned in millidarcy (1 D = 0.9869e-12 m^2).
    """
    Sv_per_m = Sv / (voxel_size_um * 1e-6)
    k_m2 = petrolib.flow_transport.kozeny_carman(
        phi, specific_surface=Sv_per_m, c=c, grain_term=True)
    return k_m2 / 0.9869e-15  # m^2 -> mD


def permeability_profile(cube, axis=2, voxel_size_um=2.0):
    """Per-slice Kozeny-Carman permeability profile (mD)."""
    phi = porosity_profile(cube, axis)
    Sv = specific_surface_area_profile(cube, axis)
    return kozeny_carman_permeability(phi, Sv, voxel_size_um=voxel_size_um)


# --------------------------------------------- tests --------------------

def test_all():
    print("=" * 60)
    print("Article 7: PSWC Quality via Digital Rock Physics")
    print("=" * 60)

    pre = make_sand_pack(shape=(48, 48, 48), seed=0)
    post = apply_percussion_damage(pre, damage_layer=slice(16, 30),
                                   fines_radius=2, n_fines=700, seed=1)

    phi_pre = porosity(pre)
    phi_post = porosity(post)
    Sv_pre = specific_surface_area_voxels(pre)
    Sv_post = specific_surface_area_voxels(post)
    k_pre = kozeny_carman_permeability(phi_pre, Sv_pre)
    k_post = kozeny_carman_permeability(phi_post, Sv_post)

    print(f"  Bulk  phi   pre  = {phi_pre:.3f}   post = {phi_post:.3f}")
    print(f"  Bulk  Sv    pre  = {Sv_pre:.4f}   post = {Sv_post:.4f}  (1/vox)")
    print(f"  Bulk  k     pre  = {k_pre:8.2f} mD")
    print(f"  Bulk  k     post = {k_post:8.2f} mD")

    phi_z_pre = porosity_profile(pre, axis=2)
    phi_z_post = porosity_profile(post, axis=2)
    k_z_pre = permeability_profile(pre, axis=2)
    k_z_post = permeability_profile(post, axis=2)
    damage_zone = slice(16, 30)
    inside = np.zeros(post.shape[2], dtype=bool); inside[damage_zone] = True

    print(f"  Damage-zone slices: phi drop  = "
          f"{phi_z_pre[inside].mean() - phi_z_post[inside].mean():.3f}")
    print(f"  Damage-zone slices: k drop    = "
          f"{k_z_pre[inside].mean() - k_z_post[inside].mean():.2f} mD")
    print(f"  Undamaged slices:   phi drop  = "
          f"{phi_z_pre[~inside].mean() - phi_z_post[~inside].mean():.3f}")

    # Sanity:
    #  - bulk porosity / permeability must DROP after percussion damage
    #  - damage-zone porosity drop > undamaged porosity drop
    assert phi_post < phi_pre
    assert k_post < k_pre
    phi_drop_in = phi_z_pre[inside].mean() - phi_z_post[inside].mean()
    phi_drop_out = phi_z_pre[~inside].mean() - phi_z_post[~inside].mean()
    assert phi_drop_in > phi_drop_out + 0.02, "Damage band must dominate phi drop"
    print("  PASS")
    return {"phi_pre": phi_pre, "phi_post": phi_post,
            "k_pre_mD": k_pre, "k_post_mD": k_post,
            "phi_drop_damaged": float(phi_drop_in),
            "phi_drop_undamaged": float(phi_drop_out)}


if __name__ == "__main__":
    test_all()
