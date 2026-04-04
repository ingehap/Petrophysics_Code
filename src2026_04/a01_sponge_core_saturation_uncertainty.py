"""
Uncertainty Quantification of Lab-Computed Saturation Data
From Sponge Cores Using Monte Carlo Simulation

Reference:
    Alghazal, M. and Krinis, D. (2026). Uncertainty Quantification of
    Lab-Computed Saturation Data From Sponge Cores Using Monte Carlo
    Simulation. Petrophysics, 67(2), 248–262.
    DOI: 10.30632/PJV67N2-2026a1

Implements:
  - Dean-Stark saturation calculations (Eqs. 1–7)
  - Mass-balance water salinity estimation (Eq. 11)
  - Water-evaporation iterative correction (Fig. 11 algorithm)
  - Monte Carlo uncertainty propagation with triangular/normal/uniform
    input distributions
  - Deterministic one-factor-at-a-time sensitivity analysis
  - Spearman rank-correlation tornado chart
"""

import numpy as np
import scipy.stats as stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Core equations (Eqs. 1–7 of the paper)
# ---------------------------------------------------------------------------

def salt_correction_factor(std_ppm: float) -> float:
    """
    Compute the salt correction factor Csf (Eq. 1 / Eq. 2).

    Parameters
    ----------
    std_ppm : float
        Total dissolved solids in ppm (parts per million).

    Returns
    -------
    float
        Dimensionless salt correction factor.

    Notes
    -----
    Csf = 1 / (1 - Std / 1e6)  (standard brine-density correction)
    """
    return 1.0 / (1.0 - std_ppm / 1e6)


def formation_water_volume(Vd: float, std_ppm: float, rho_w: float = 1.0) -> float:
    """
    Formation water volume corrected for dissolved salts (Eq. 1 / Eq. 2).

    Parameters
    ----------
    Vd      : Distilled water volume measured in Dean-Stark tube, cm³
    std_ppm : Total dissolved solids, ppm
    rho_w   : Water density at reservoir conditions, g/cm³

    Returns
    -------
    Vw : Formation water volume, cm³
    """
    Csf = salt_correction_factor(std_ppm)
    Vw = Vd * Csf
    return Vw


def oil_volume(mwet: float, mdry: float, Vw: float, rho_w: float,
               rho_o: float, V_sponge_oil: float = 0.0) -> float:
    """
    Oil volume from gravimetric weight difference (Eq. 3).

    Parameters
    ----------
    mwet          : Wet mass of sample before Dean-Stark, g
    mdry          : Dry mass after fluid extraction, g
    Vw            : Formation water volume, cm³
    rho_w         : Water density, g/cm³
    rho_o         : Dead oil density, g/cm³
    V_sponge_oil  : Oil volume from spectrometric sponge analysis, cm³

    Returns
    -------
    Vo : Total oil volume, cm³
    """
    mass_diff = mwet - mdry - Vw * rho_w
    Vo = mass_diff / rho_o + V_sponge_oil
    return max(Vo, 0.0)


def surface_saturations(Vo: float, Vw: float, Vp: float) -> Tuple[float, float]:
    """
    Fluid saturations at surface conditions (Eqs. 4–5).

    Returns
    -------
    (Sos, Sws) : Oil and water saturations as fractions [0, 1]
    """
    Sos = Vo / Vp
    Sws = Vw / Vp
    return Sos, Sws


def reservoir_saturations(Sos: float, Sws: float,
                           phi_a: float, phi_s: float,
                           Bo: float, Bw: float) -> Tuple[float, float]:
    """
    Saturations corrected to reservoir conditions (Eqs. 6–7).

    Parameters
    ----------
    Sos, Sws : Surface oil and water saturations (fractions)
    phi_a    : Lab-measured porosity in ambient conditions (fraction)
    phi_s    : Stress-corrected porosity at net-overburden pressure (fraction)
    Bo       : Oil formation volume factor (reservoir bbl / stock-tank bbl)
    Bw       : Water formation volume factor

    Returns
    -------
    (Sor, Swr) : Reservoir-condition oil and water saturations
    """
    Sor = (Sos * phi_a * Bo) / phi_s
    Swr = (Sws * phi_a * Bw) / phi_s
    return Sor, Swr


# ---------------------------------------------------------------------------
# 2. Water salinity from mass balance (Eq. 11)
# ---------------------------------------------------------------------------

def estimate_salinity_mass_balance(mwet: float, mtol: float,
                                    mmeth: float, Vd: float,
                                    rho_d: float = 1.0) -> float:
    """
    Derive formation-water salinity from sample masses before and after
    solvent extraction (Eq. 11 of the paper).

    Parameters
    ----------
    mwet  : Sample wet mass before Dean-Stark, g
    mtol  : Sample mass after toluene cleaning, g
    mmeth : Sample mass after methanol leaching, g
    Vd    : Distilled water volume from Dean-Stark, cm³
    rho_d : Distilled water density ≈ 1.0 g/cm³

    Returns
    -------
    std_ppm : Estimated water salinity, ppm
    """
    m_salt = mtol - mmeth           # salt mass leached by methanol, g
    m_water = Vd * rho_d            # distilled water mass, g
    if m_water + m_salt <= 0:
        return 0.0
    # salinity = salt_mass / (salt_mass + water_mass)  → ppm
    std_ppm = (m_salt / (m_water + m_salt)) * 1e6
    return std_ppm


# ---------------------------------------------------------------------------
# 3. Iterative water-evaporation correction (Fig. 11 algorithm)
# ---------------------------------------------------------------------------

def correct_for_water_evaporation(Sos_surf: float, Sws_surf: float,
                                   phi_a: float, phi_s: float,
                                   Bo: float, Bw: float,
                                   mwet: float, mdry: float,
                                   rho_w: float, rho_o: float,
                                   Vp: float, Vd: float,
                                   std_ppm_init: float,
                                   max_iter: int = 200,
                                   tol: float = 1e-4) -> Dict:
    """
    Iteratively adjust water salinity to achieve Sor + Swr = 1.0 at
    reservoir conditions, accounting for water evaporation losses.

    The algorithm (Fig. 11) starts from the mass-balance salinity estimate,
    then incrementally adds evaporated brine volume until material balance
    is satisfied.

    Returns
    -------
    dict with keys: Sor, Swr, std_ppm_final, evap_fraction
    """
    std_ppm = std_ppm_init
    for _ in range(max_iter):
        Vw = formation_water_volume(Vd, std_ppm, rho_w)
        Vo = oil_volume(mwet, mdry, Vw, rho_w, rho_o)
        Sos, Sws = surface_saturations(Vo, Vw, Vp)
        Sor, Swr = reservoir_saturations(Sos, Sws, phi_a, phi_s, Bo, Bw)
        total = Sor + Swr
        if abs(total - 1.0) < tol:
            break
        # Increase salinity → larger Vw → smaller Vo → total moves toward 1
        std_ppm *= (1.0 + 0.01 * (1.0 - total))
        std_ppm = min(std_ppm, 350_000)  # physical upper bound

    evap_frac = (formation_water_volume(Vd, std_ppm) - Vd) / Vd if Vd > 0 else 0.0
    return {"Sor": Sor, "Swr": Swr,
            "std_ppm_final": std_ppm, "evap_fraction": evap_frac}


# ---------------------------------------------------------------------------
# 4. Monte Carlo simulation
# ---------------------------------------------------------------------------

@dataclass
class InputDistributions:
    """
    Probability distributions for nine key variables (Table 1 of paper).
    Each entry is (distribution_type, params) where params vary by type:
      'triangular'  → (low, mode, high)   [cm³ or g]
      'normal'      → (mean, std)
      'uniform'     → (low, high)
    """
    # Lab-measured (triangular)
    Vd_cm3:          Tuple = ('triangular', (49.0, 50.0, 51.0))   # distilled water vol
    diameter_cm:     Tuple = ('triangular', (3.76, 3.80, 3.84))   # plug diameter
    length_cm:       Tuple = ('triangular', (5.95, 6.00, 6.05))   # plug length
    mwet_g:          Tuple = ('triangular', (398.0, 400.0, 402.0))
    weight_loss_g:   Tuple = ('triangular', (9.8, 10.0, 10.2))    # mwet - mdry
    grain_vol_cm3:   Tuple = ('triangular', (118.5, 120.0, 121.5))
    # Fluid properties (normal)
    rho_o_gcc:       Tuple = ('normal', (0.85, 0.02))
    Bo_res_stk:      Tuple = ('normal', (1.15, 0.03))
    # Salinity (uniform or triangular)
    std_ppm:         Tuple = ('uniform', (20_000, 150_000))

    # Fixed context
    phi_s: float = 0.22
    phi_a: float = 0.23
    Bw: float    = 1.02
    rho_w: float = 1.07
    V_sponge_oil: float = 0.0


def _draw(dist_spec: Tuple, n: int, rng: np.random.Generator) -> np.ndarray:
    dtype, params = dist_spec
    if dtype == 'triangular':
        lo, mode, hi = params
        c = (mode - lo) / (hi - lo)
        return rng.triangular(lo, mode, hi, size=n)
    elif dtype == 'normal':
        mu, sigma = params
        return rng.normal(mu, sigma, size=n)
    elif dtype == 'uniform':
        lo, hi = params
        return rng.uniform(lo, hi, size=n)
    else:
        raise ValueError(f"Unknown distribution type: {dtype}")


def run_monte_carlo(dists: InputDistributions,
                    n_simulations: int = 10_000,
                    seed: int = 42) -> Dict:
    """
    Propagate uncertainty through the Dean-Stark saturation workflow.

    Returns
    -------
    dict with keys:
        Sor_array  : ndarray of reservoir-condition oil saturations
        Swr_array  : ndarray of reservoir-condition water saturations
        p10, p50, p90, mean, std : scalar statistics for Sor
        inputs     : dict of sampled input arrays (for Spearman analysis)
    """
    rng = np.random.default_rng(seed)
    n = n_simulations

    # Sample all inputs
    Vd    = _draw(dists.Vd_cm3,       n, rng)
    diam  = _draw(dists.diameter_cm,  n, rng)
    leng  = _draw(dists.length_cm,    n, rng)
    mwet  = _draw(dists.mwet_g,       n, rng)
    wloss = _draw(dists.weight_loss_g, n, rng)  # = mwet - mdry
    gvol  = _draw(dists.grain_vol_cm3, n, rng)
    rho_o = _draw(dists.rho_o_gcc,    n, rng)
    Bo    = _draw(dists.Bo_res_stk,   n, rng)
    salinity = _draw(dists.std_ppm,   n, rng)

    # Derived quantities
    mdry     = mwet - wloss
    bulk_vol = np.pi / 4.0 * diam**2 * leng     # cm³
    Vp       = bulk_vol - gvol                   # pore volume
    Vp       = np.where(Vp > 0, Vp, 1e-3)        # guard

    # Saturation equations
    Vw  = formation_water_volume(Vd, salinity, dists.rho_w)
    # Vectorised oil volume (no sponge contribution for simplicity)
    mass_diff = mwet - mdry - Vw * dists.rho_w
    Vo  = np.where(mass_diff / rho_o > 0, mass_diff / rho_o, 0.0)

    Sos = Vo / Vp
    Sws = Vw / Vp
    Sor, Swr = reservoir_saturations(Sos, Sws,
                                      dists.phi_a, dists.phi_s,
                                      Bo, dists.Bw)

    inputs = {"Vd": Vd, "diameter": diam, "length": leng,
              "weight_loss": wloss, "grain_vol": gvol,
              "rho_o": rho_o, "Bo": Bo, "salinity": salinity}

    return {
        "Sor_array": Sor,
        "Swr_array": Swr,
        "p10":  float(np.percentile(Sor * 100, 10)),
        "p50":  float(np.percentile(Sor * 100, 50)),
        "p90":  float(np.percentile(Sor * 100, 90)),
        "mean": float(np.mean(Sor * 100)),
        "std":  float(np.std(Sor * 100)),
        "inputs": inputs,
    }


# ---------------------------------------------------------------------------
# 5. Spearman rank-correlation sensitivity (tornado chart data)
# ---------------------------------------------------------------------------

def spearman_sensitivity(mc_result: Dict) -> Dict[str, float]:
    """
    Compute Spearman rank correlation coefficients between each input
    variable and the simulated oil saturation (Fig. 7 of paper).

    Parameters
    ----------
    mc_result : output of run_monte_carlo()

    Returns
    -------
    dict mapping variable name → Spearman rho (sorted by |rho|)
    """
    Sor = mc_result["Sor_array"]
    coeffs = {}
    for name, arr in mc_result["inputs"].items():
        rho, _ = stats.spearmanr(arr, Sor)
        coeffs[name] = float(rho)
    return dict(sorted(coeffs.items(), key=lambda kv: abs(kv[1]), reverse=True))


# ---------------------------------------------------------------------------
# 6. Deterministic one-factor-at-a-time sensitivity
# ---------------------------------------------------------------------------

def deterministic_sensitivity(base: Dict,
                               param_ranges: Optional[Dict] = None) -> Dict[str, Tuple[float, float]]:
    """
    Vary each parameter ±1 std (or a supplied range) while holding others
    fixed and report the resulting change in Sor (percentage points).

    Parameters
    ----------
    base          : dict with keys matching InputDistributions fields (scalar)
    param_ranges  : {param_name: (low, high)} override; defaults provided.

    Returns
    -------
    dict mapping param_name → (delta_Sor_low, delta_Sor_high)  [% s.u.]
    """
    defaults = {
        "std_ppm":     (20_000, 150_000),
        "Vd_cm3":      (49.0, 51.0),
        "rho_o_gcc":   (0.80, 0.90),
        "Bo_res_stk":  (1.10, 1.25),
        "diameter_cm": (3.76, 3.84),
        "length_cm":   (5.95, 6.05),
    }
    ranges = {**defaults, **(param_ranges or {})}

    def calc_Sor(params):
        Vw = formation_water_volume(params["Vd_cm3"], params["std_ppm"], params["rho_w"])
        mass_diff = params["mwet_g"] - params["mdry_g"] - Vw * params["rho_w"]
        Vo = max(mass_diff / params["rho_o_gcc"], 0.0)
        Vp = params["Vp_cm3"]
        Sos, Sws = surface_saturations(Vo, Vw, Vp)
        Sor, _ = reservoir_saturations(Sos, Sws,
                                        params["phi_a"], params["phi_s"],
                                        params["Bo_res_stk"], params["Bw"])
        return Sor * 100.0

    Sor_base = calc_Sor(base)
    result = {}
    for param, (lo, hi) in ranges.items():
        if param not in base:
            continue
        low_params  = {**base, param: lo}
        high_params = {**base, param: hi}
        dSor_low  = calc_Sor(low_params)  - Sor_base
        dSor_high = calc_Sor(high_params) - Sor_base
        result[param] = (dSor_low, dSor_high)
    return result


# ---------------------------------------------------------------------------
# 7. Convenience: full workflow example
# ---------------------------------------------------------------------------

def example_workflow():
    """Run a complete example and print key results."""
    print("=" * 60)
    print("Sponge Core Saturation Uncertainty – Monte Carlo")
    print("Ref: Alghazal & Krinis, Petrophysics 67(2) 2026")
    print("=" * 60)

    dists = InputDistributions()
    result = run_monte_carlo(dists, n_simulations=10_000)

    print(f"\nMonte Carlo Results (n = 10,000):")
    print(f"  P10  Oil Saturation : {result['p10']:.1f} % s.u.")
    print(f"  P50  Oil Saturation : {result['p50']:.1f} % s.u.")
    print(f"  P90  Oil Saturation : {result['p90']:.1f} % s.u.")
    print(f"  Mean               : {result['mean']:.1f} % s.u.")
    print(f"  Std Dev            : {result['std']:.1f} % s.u.")

    coeffs = spearman_sensitivity(result)
    print("\nSpearman Rank Correlations (Tornado):")
    for var, rho in coeffs.items():
        bar = "█" * int(abs(rho) * 20) if not np.isnan(rho) else ""
        sign = "+" if rho >= 0 else "-"
        print(f"  {var:<20s} {sign}{bar} {rho:+.3f}")

    # Mass-balance salinity example
    std_est = estimate_salinity_mass_balance(
        mwet=400.0, mtol=385.0, mmeth=383.5, Vd=50.0)
    print(f"\nMass-balance salinity estimate: {std_est:.0f} ppm")
    if np.isnan(std_est):
        std_est = 66_000.0   # fallback for display

    return result


if __name__ == "__main__":
    example_workflow()
