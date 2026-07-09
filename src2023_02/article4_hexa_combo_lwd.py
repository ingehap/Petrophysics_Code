"""
Article 4: First Hexa-Combo Logging-While-Drilling Run in Kuwait: A Case Study
Saleh, Al-Khudari, Al-Azmi, Al-Otaibi, Patnaik, Joshi, Abdulkarim, Aki, Fahri,
Sanyal, Sainuddin (2023)
DOI: 10.30632/PJV64N1-2023a4

Generates a synthetic six-tool LWD log suite (GR, multi-DOI resistivity,
neutron porosity, azimuthal bulk density, compressional + shear slowness,
NMR T2 distribution) through a Marrat-style tight fractured carbonate, then
runs the case-study interpretation workflow:

  - Triple-combo porosity & lithology (RHOB-NPHI density porosity, GR Vsh).
  - Archie water saturation Sw.
  - NMR T2 distribution -> Bound-Volume Irreducible (BVI) and Free-Fluid
    Index (FFI) using a 33 ms cutoff.
  - Dynamic geomechanical moduli from RHOB, Vp = 1e6/DTC, Vs = 1e6/DTS:
      K, G, Poisson's ratio, Young's modulus.
  - Brittleness index BI = (E_norm + Vfrac_quartz_norm)/2.
  - Perforation-interval picking on combined criteria (phi_e > 0.06,
    Sw < 0.40, BI > 0.55).
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


# --------------------------------------------- synthetic log suite ---------

def generate_marrat_logs(depth_top=8000.0, depth_bot=8400.0, step=0.5,
                         seed=0):
    """Return a dict of LWD curves over a Marrat-style carbonate interval (ft)."""
    rng = np.random.default_rng(seed)
    z = np.arange(depth_top, depth_bot, step)
    n = len(z)

    # Three sub-intervals: tight matrix / fractured pay / mudstone
    layer = np.zeros(n, dtype=int)
    layer[(z >= 8080) & (z < 8230)] = 1
    layer[(z >= 8230) & (z < 8260)] = 2

    gr = np.where(layer == 0, 18.0,
         np.where(layer == 1, 22.0, 80.0)) + rng.normal(0, 4.0, n)
    rhob = np.where(layer == 0, 2.74,
           np.where(layer == 1, 2.56, 2.55)) + rng.normal(0, 0.012, n)
    nphi = np.where(layer == 0, 0.04,
           np.where(layer == 1, 0.14, 0.28)) + rng.normal(0, 0.010, n)
    dt_comp = np.where(layer == 0, 55.0,
              np.where(layer == 1, 65.0, 95.0)) + rng.normal(0, 1.0, n)
    dt_shear = dt_comp * np.where(layer == 0, 1.80,
                                  np.where(layer == 1, 1.86, 2.10))
    rt_deep = np.where(layer == 0, 800.0,
              np.where(layer == 1, 250.0, 2.5)) \
              * np.exp(rng.normal(0, 0.25, n))

    # NMR T2 distribution (40 bins, log-spaced 0.3 ms .. 3000 ms)
    t2_bins = np.logspace(-0.5, 3.5, 40)
    nmr = np.zeros((n, len(t2_bins)))
    for i in range(n):
        if layer[i] == 0:                # tight: BVI dominated
            mu = np.log(8.0); sig = 0.6
        elif layer[i] == 1:              # fractured pay: bimodal
            mu = np.log(80.0); sig = 0.7
            nmr[i] += 0.4 * np.exp(-((np.log(t2_bins) - np.log(5.0)) / 0.5) ** 2)
        else:                             # mudstone: clay-bound
            mu = np.log(2.0); sig = 0.4
        nmr[i] += np.exp(-((np.log(t2_bins) - mu) / sig) ** 2)
        nmr[i] /= nmr[i].sum()
    return dict(depth=z, gr=gr, rhob=rhob, nphi=nphi,
                dt_comp=dt_comp, dt_shear=dt_shear, rt=rt_deep,
                nmr=nmr, t2_bins=t2_bins, layer=layer)


# --------------------------------------------- standard log interpretation --

def gr_to_vsh(gr, gr_clean=15.0, gr_shale=120.0):
    return petrolib.porosity_lithology.gamma_ray_index(gr, gr_clean, gr_shale)


def density_porosity(rhob, rho_ma=2.71, rho_f=1.0):
    return petrolib.porosity_lithology.density_porosity(rhob, rho_ma, rho_f, clip=(0.0, 1.0))


def effective_porosity(phi_d, nphi, vsh, phi_sh=0.30):
    phi_t = petrolib.porosity_lithology.neutron_density_porosity(nphi, phi_d, method="mean")
    return petrolib.porosity_lithology.effective_porosity(phi_t, vsh, phi_sh, clip=(0.0, 1.0))


def archie_sw(phi, rt, Rw=0.04, m=2.0, n=2.0, a=1.0):
    # HAZARD (LIBRARY_MERGE_PLAN.md section 9): this article's argument order
    # is (phi, rt) with a baked field default Rw=0.04 — the canonical order is
    # (rt, rw, phi=).  Mapped explicitly; the phi floor is historical.
    return petrolib.saturation_resistivity.archie_sw(rt, Rw, phi=np.maximum(phi, 1e-3), a=a, m=m, n=n,
                           clip=(0.0, 1.0))


# --------------------------------------------- NMR partitioning ----------

def nmr_partition(nmr, t2_bins, t2_cutoff=33.0):
    cutoff_idx = np.searchsorted(t2_bins, t2_cutoff)
    bvi = nmr[:, :cutoff_idx].sum(axis=1)
    ffi = nmr[:, cutoff_idx:].sum(axis=1)
    return bvi, ffi


# --------------------------------------------- geomechanics --------------

def geomechanics(rhob, dt_comp, dt_shear):
    """Return Vp, Vs (ft/s), K, G, nu, E.  RHOB in g/cc, DT in us/ft."""
    rhob_si = rhob * 1000.0
    vp = 1.0e6 / dt_comp * 0.3048           # m/s
    vs = 1.0e6 / dt_shear * 0.3048
    G = rhob_si * vs ** 2 / 1e9             # GPa
    K = rhob_si * (vp ** 2 - 4.0 / 3.0 * vs ** 2) / 1e9
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    E = 9 * K * G / (3 * K + G)
    return vp, vs, K, G, nu, E


def brittleness_index(E, vfrac_qtz_like=0.5):
    """Wang-Gale-style BI: half from normalised E, half from brittle mineral
    volume (proxied by 1 - Vsh)."""
    E_n = (E - 20.0) / 50.0
    return 0.5 * np.clip(E_n, 0.0, 1.0) + 0.5 * np.clip(vfrac_qtz_like, 0.0, 1.0)


# --------------------------------------------- perforation picker --------

def pick_perfs(depth, phi_e, sw, bi, min_thickness_ft=4.0):
    flag = (phi_e > 0.06) & (sw < 0.40) & (bi > 0.55)
    intervals = []
    i = 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j < len(flag) and flag[j]:
                j += 1
            top, bot = depth[i], depth[min(j, len(depth) - 1)]
            if bot - top >= min_thickness_ft:
                intervals.append((float(top), float(bot)))
            i = j
        else:
            i += 1
    return intervals


# --------------------------------------------- tests --------------------

def test_all():
    print("=" * 60)
    print("Article 4: Hexa-Combo LWD Case Study (Marrat-style)")
    print("=" * 60)

    logs = generate_marrat_logs()
    z = logs["depth"]

    vsh = gr_to_vsh(logs["gr"])
    phi_d = density_porosity(logs["rhob"])
    phi_e = effective_porosity(phi_d, logs["nphi"], vsh)
    sw = archie_sw(phi_e, logs["rt"])
    bvi, ffi = nmr_partition(logs["nmr"], logs["t2_bins"])
    vp, vs, K, G, nu, E = geomechanics(logs["rhob"], logs["dt_comp"],
                                       logs["dt_shear"])
    bi = brittleness_index(E, vfrac_qtz_like=(1.0 - vsh))

    pay = (logs["layer"] == 1)
    print(f"  Interval summary  (n samples = {len(z)})")
    print(f"    GR        mean      pay = {logs['gr'][pay].mean():5.1f} API")
    print(f"    Phi_e     mean      pay = {phi_e[pay].mean():.3f}")
    print(f"    Sw        mean      pay = {sw[pay].mean():.3f}")
    print(f"    FFI/Phi_e mean      pay = {(ffi[pay] / np.maximum(phi_e[pay], 1e-3)).mean():.3f}")
    print(f"    Brittleness mean    pay = {bi[pay].mean():.3f}")
    print(f"    Young's  E mean     pay = {E[pay].mean():5.1f} GPa")
    print(f"    Poisson  nu mean    pay = {nu[pay].mean():.3f}")

    perfs = pick_perfs(z, phi_e, sw, bi)
    print(f"  Picked perforation intervals (>= 4 ft, phi_e>0.06, Sw<0.40, BI>0.55):")
    for top, bot in perfs:
        print(f"     {top:7.1f} - {bot:7.1f}  ({bot - top:4.1f} ft)")

    # Sanity: at least one pay interval, perfs land inside pay zone
    assert len(perfs) >= 1, "Should pick at least one perforation interval"
    for top, bot in perfs:
        mid = 0.5 * (top + bot)
        assert 8080.0 <= mid <= 8230.0, f"Perf {top}-{bot} outside pay zone"
    print("  PASS")
    return {"n_perfs": len(perfs),
            "mean_phi_pay": float(phi_e[pay].mean()),
            "mean_sw_pay": float(sw[pay].mean()),
            "mean_bi_pay": float(bi[pay].mean())}


if __name__ == "__main__":
    test_all()
