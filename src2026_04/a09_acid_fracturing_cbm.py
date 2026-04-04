"""
Acid Fracturing-Induced Fracture Propagation in Deep Coalbed
Methane Wells: A Case Study in Daning-Jixian Block

Reference:
    Zhao, H., Jin, B., Zhen, H., and Li, S. (2026). Acid Fracturing-
    Induced Fracture Propagation in Deep Coalbed Methane Wells: A Case
    Study in Daning-Jixian Block. Petrophysics, 67(2), 386–403.
    DOI: 10.30632/PJV67N2-2026a9

Implements:
  - Acid dissolution and fracture-pressure reduction model
  - True-triaxial fracture propagation physics (simplified 2D)
  - Acid concentration effect on fracture complexity
  - Perforation-location effect (top-of-seam preferred)
  - Fracture network complexity index
  - Field parameter optimisation (acid concentration = 10% sulfamic acid)
  - Acid etching conductivity estimation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Coal seam and acid parameters
# ---------------------------------------------------------------------------

@dataclass
class CoalSeamParams:
    """Deep No. 8 coal seam, Daning-Jixian Block, Ordos Basin."""
    depth_m:          float = 2200.0   # burial depth
    sigma_H_MPa:      float = 55.0    # max horizontal stress
    sigma_h_MPa:      float = 38.0    # min horizontal stress
    sigma_v_MPa:      float = 60.0    # vertical stress
    Pp_MPa:           float = 20.0    # pore pressure
    E_GPa:            float = 2.5     # Young's modulus (coal is soft)
    nu:               float = 0.30    # Poisson's ratio
    T_MPa:            float = 1.5     # tensile strength
    permeability_mD:  float = 0.05    # matrix permeability
    cleat_perm_mD:    float = 5.0     # cleat permeability

    @property
    def fracture_pressure_MPa(self) -> float:
        """Breakdown pressure = sigma_h + T (Hubbert-Willis)."""
        return self.sigma_h_MPa + self.T_MPa


@dataclass
class AcidParams:
    """Sulfamic acid fracturing fluid properties."""
    acid_type:         str   = "sulfamic"
    concentration_pct: float = 10.0    # % by weight
    density_gcc:       float = 1.06    # g/cm³
    viscosity_mPas:    float = 1.2     # mPa·s (low viscosity)
    react_rate_const:  float = 0.012   # dissolution rate constant, 1/(s·wt%)


# ---------------------------------------------------------------------------
# 2. Acid dissolution model
# ---------------------------------------------------------------------------

def acid_dissolution_depth(C_pct: float, t_contact_s: float,
                             react_rate: float, coal_density: float = 1.4
                             ) -> float:
    """
    Effective dissolution depth of acid into coal cleat faces.

        d_etch = k_r * C * t   (simplified first-order kinetics)

    Parameters
    ----------
    C_pct        : Acid concentration, wt%
    t_contact_s  : Contact time, s
    react_rate   : Reaction rate constant, 1/(s·wt%)
    coal_density : Coal bulk density, g/cm³

    Returns
    -------
    d_etch : Etching depth, mm
    """
    d_etch_m = react_rate * C_pct * t_contact_s / coal_density
    return d_etch_m * 1000.0   # convert to mm


def fracture_pressure_reduction(seam: CoalSeamParams,
                                 acid: AcidParams,
                                 t_contact_s: float = 300.0) -> float:
    """
    Reduction in fracture initiation pressure due to acid treatment.
    Acid dissolves cleat-face cementation, reducing tensile strength
    and increasing effective cleat permeability.

    ΔP_f = T0 - T_acid   where T_acid = T0 * exp(-alpha * C * t)

    Returns
    -------
    delta_Pf : Pressure reduction, MPa (positive = beneficial reduction)
    """
    alpha   = 0.002   # softening coefficient, 1/(wt%·s)
    T_acid  = seam.T_MPa * np.exp(-alpha * acid.concentration_pct * t_contact_s)
    return seam.T_MPa - T_acid


def breakdown_pressure_with_acid(seam: CoalSeamParams,
                                  acid: AcidParams,
                                  t_contact_s: float = 300.0) -> float:
    """Effective breakdown pressure after acid treatment, MPa."""
    delta_Pf = fracture_pressure_reduction(seam, acid, t_contact_s)
    return seam.fracture_pressure_MPa - delta_Pf


# ---------------------------------------------------------------------------
# 3. Fracture propagation geometry (simplified 2D)
# ---------------------------------------------------------------------------

def fracture_half_length_vs_volume(Q_m3: float,
                                    h_m: float,
                                    E_prime: float,
                                    eta: float = 0.6) -> float:
    """
    PKN-type fracture half-length scaling:

        xf = [E' * Q * eta / (h^2 * p_net)]^(1/2)   (simplified)

    Parameters
    ----------
    Q_m3    : Total injected volume, m³
    h_m     : Fracture height, m
    E_prime : Plane-strain modulus = E / (1 - nu²), MPa
    eta     : Fluid efficiency (fraction)

    Returns
    -------
    xf : Half-length, m
    """
    # Simplified proportionality (excludes p_net dependence for brevity)
    xf = np.sqrt(E_prime * Q_m3 * eta) / h_m * 0.5
    return max(xf, 1.0)


def fracture_width_acid(xf: float, p_net: float,
                         E_prime: float) -> float:
    """
    Maximum fracture width at wellbore (KGD model):

        w0 = 2 * p_net * xf / E'

    Parameters
    ----------
    xf      : Half-length, m
    p_net   : Net pressure, MPa
    E_prime : Plane-strain modulus, MPa

    Returns
    -------
    w0 : Max width, mm
    """
    return 2.0 * p_net * xf / E_prime * 1000.0  # → mm


# ---------------------------------------------------------------------------
# 4. Fracture complexity index
# ---------------------------------------------------------------------------

def fracture_complexity_index(sigma_H: float, sigma_h: float,
                               p_net: float, T_coal: float,
                               acid_concentration: float,
                               perf_at_top: bool = True) -> float:
    """
    Empirical fracture complexity index (FCI) for acid fracturing in coal.

    Higher FCI → more complex fracture network with natural fracture connections.

    FCI = (p_net / (sigma_H - sigma_h)) * acid_factor * perf_factor

    Where:
      acid_factor : increases with concentration (diminishing returns > 15%)
      perf_factor : 1.2 if perforations at seam top, 1.0 otherwise

    Parameters
    ----------
    sigma_H, sigma_h   : Horizontal principal stresses, MPa
    p_net              : Net fracturing pressure, MPa
    T_coal             : Coal tensile strength, MPa
    acid_concentration : Acid wt%
    perf_at_top        : True if perforations at top of coal seam

    Returns
    -------
    FCI : Complexity index (dimensionless; > 1.0 → complex network likely)
    """
    stress_diff = max(sigma_H - sigma_h, 0.1)

    # Acid factor: diminishing returns beyond ~12%
    acid_factor = 1.0 + 0.15 * np.log1p(acid_concentration / 5.0)

    # Perforation-location factor
    perf_factor = 1.2 if perf_at_top else 1.0

    FCI = (p_net / stress_diff) * acid_factor * perf_factor
    return FCI


# ---------------------------------------------------------------------------
# 5. Acid etching conductivity
# ---------------------------------------------------------------------------

def acid_etched_conductivity(w_mm: float,
                              C_pct: float,
                              t_contact_s: float = 300.0,
                              beta: float = 2.0) -> float:
    """
    Acid-etched fracture conductivity (mD·m).

        Fc = beta * w_etch^3 / 12   (cubic law for etched aperture)
    where
        w_etch = w_mm + d_etch (acid dissolution adds to mechanical width)

    Parameters
    ----------
    w_mm      : Mechanical fracture width, mm
    C_pct     : Acid concentration, wt%
    t_contact_s: Contact time, s
    beta      : Conductivity factor (accounts for surface roughness)

    Returns
    -------
    Fc : Fracture conductivity, mD·m
    """
    d_etch = acid_dissolution_depth(C_pct, t_contact_s, 0.012)   # mm
    w_tot  = (w_mm + d_etch) * 1e-3  # convert mm → m
    Fc     = beta * w_tot**3 / 12.0   # m² * m = m³ → convert to mD·m
    return Fc * 1e12  # 1 m² = 1e12 mD


# ---------------------------------------------------------------------------
# 6. Concentration and perforation optimisation
# ---------------------------------------------------------------------------

def optimise_acid_parameters(seam: CoalSeamParams,
                              candidate_concentrations: np.ndarray,
                              Q_m3: float = 500.0,
                              Q_rate: float = 8.0) -> Dict:
    """
    Evaluate FCI and breakdown pressure for a range of acid concentrations,
    with perforations at top vs. bottom of seam.

    Paper's optimum: 10% sulfamic acid, perforations at seam top.

    Returns
    -------
    dict with 'concentrations', 'FCI_top', 'FCI_bottom',
              'Pf_reduction', 'recommended_conc'
    """
    E_prime = seam.E_GPa * 1e3 / (1.0 - seam.nu**2)   # MPa
    fci_top = []
    fci_bot = []
    pf_red  = []

    for C in candidate_concentrations:
        acid = AcidParams(concentration_pct=C)
        p_net = Q_rate * 0.05 + 5.0  # proxy: net pressure from rate + viscosity
        fci_top.append(fracture_complexity_index(
            seam.sigma_H_MPa, seam.sigma_h_MPa,
            p_net, seam.T_MPa, C, perf_at_top=True))
        fci_bot.append(fracture_complexity_index(
            seam.sigma_H_MPa, seam.sigma_h_MPa,
            p_net, seam.T_MPa, C, perf_at_top=False))
        pf_red.append(fracture_pressure_reduction(seam, acid))

    fci_top  = np.array(fci_top)
    fci_bot  = np.array(fci_bot)
    pf_red   = np.array(pf_red)
    best_idx = int(np.argmax(fci_top))

    return {
        "concentrations":   candidate_concentrations,
        "FCI_top":          fci_top,
        "FCI_bottom":       fci_bot,
        "Pf_reduction_MPa": pf_red,
        "recommended_conc": float(candidate_concentrations[best_idx]),
        "best_FCI_top":     float(fci_top[best_idx]),
    }


# ---------------------------------------------------------------------------
# 7. Example workflow
# ---------------------------------------------------------------------------

def example_workflow():
    print("=" * 60)
    print("Acid Fracturing in Deep Coalbed Methane Wells")
    print("Ref: Zhao et al., Petrophysics 67(2) 2026")
    print("=" * 60)

    seam = CoalSeamParams()
    acid = AcidParams(concentration_pct=10.0)

    print(f"\nCoal seam breakdown pressure (no acid): "
          f"{seam.fracture_pressure_MPa:.1f} MPa")
    pf_red = fracture_pressure_reduction(seam, acid)
    print(f"Pressure reduction with 10% sulfamic acid: {pf_red:.2f} MPa")
    print(f"Effective breakdown pressure: "
          f"{breakdown_pressure_with_acid(seam, acid):.1f} MPa")

    # Dissolution etching
    d = acid_dissolution_depth(10.0, t_contact_s=300, react_rate=0.012)
    print(f"\nAcid etching depth (10%, 300 s contact): {d:.3f} mm")

    # Fracture geometry
    E_prime = seam.E_GPa * 1e3 / (1.0 - seam.nu**2)
    xf  = fracture_half_length_vs_volume(Q_m3=500.0, h_m=15.0, E_prime=E_prime)
    w0  = fracture_width_acid(xf, p_net=8.0, E_prime=E_prime)
    Fc  = acid_etched_conductivity(w0, C_pct=10.0)
    print(f"\nFracture geometry (500 m³ injection):")
    print(f"  Half-length : {xf:.1f} m")
    print(f"  Max width   : {w0:.2f} mm")
    print(f"  Conductivity: {Fc:.1f} mD·m")

    # FCI comparison: top vs. bottom perforations
    FCI_top = fracture_complexity_index(seam.sigma_H_MPa, seam.sigma_h_MPa,
                                         p_net=8.0, T_coal=seam.T_MPa,
                                         acid_concentration=10.0,
                                         perf_at_top=True)
    FCI_bot = fracture_complexity_index(seam.sigma_H_MPa, seam.sigma_h_MPa,
                                         p_net=8.0, T_coal=seam.T_MPa,
                                         acid_concentration=10.0,
                                         perf_at_top=False)
    print(f"\nFCI (perforations at top): {FCI_top:.3f}")
    print(f"FCI (perforations at bottom): {FCI_bot:.3f}")
    print(f"→ Top perforation is {'preferred' if FCI_top > FCI_bot else 'not preferred'}")

    # Acid concentration sweep
    conc_arr = np.arange(5, 21, 2.5)
    opt      = optimise_acid_parameters(seam, conc_arr)
    print(f"\nConcentration optimisation:")
    for C, ft, fb, pr in zip(opt["concentrations"], opt["FCI_top"],
                               opt["FCI_bottom"], opt["Pf_reduction_MPa"]):
        print(f"  {C:.1f}%  FCI_top={ft:.3f}  FCI_bot={fb:.3f}  "
              f"ΔPf={pr:.2f} MPa")
    print(f"\nRecommended acid concentration: {opt['recommended_conc']:.1f}%  "
          f"(FCI_top = {opt['best_FCI_top']:.3f})")

    return opt


if __name__ == "__main__":
    example_workflow()
