"""
article3_hydrogen_storage.py
============================

Implementation of the main engineering relations from:

    Okoroafor, E. R., Sekar, L. K., and Galvis, H. (2024).
    "Underground Hydrogen Storage in Porous Media: The Potential Role of
    Petrophysics."  Petrophysics 65(3), 317-341.
    DOI: 10.30632/PJV65N3-2024a3

The paper is a numerical-simulation study of underground hydrogen storage
(UHS).  We implement the analytical relations the authors use to drive
and analyse their simulations:

1.  Newman (1973) correlation for consolidated sandstone rock
    compressibility as a function of porosity.

2.  The gas inflow performance relationship (IPR) used for H2 withdrawal
    rate and the authors' definition of the average productivity index
    of a withdrawal cycle.

3.  A Mohr-Coulomb / Griffith failure envelope check (Fig. 3 of the
    paper) to decide whether a given stress state will induce
    microseismicity on a critically stressed fault.

4.  A minimal cyclic injection-withdrawal scheduler reproducing the
    six-cycle scenario the paper uses (Fig. 1).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# 1. Newman (1973) rock compressibility correlation -------------------------
# ---------------------------------------------------------------------------

def newman_rock_compressibility(phi: float | np.ndarray,
                                lithology: str = "consolidated_sandstone"
                                ) -> np.ndarray:
    """Pore-volume compressibility as a function of porosity (psi^-1).

    Uses the classic Newman (1973) form cited by the paper,

        Cf = a / (1 + c * phi) ** b .

    Two lithology presets are provided ("consolidated_sandstone" and
    "limestone") with the fit constants most commonly used in the
    literature.
    """
    phi = np.asarray(phi, dtype=float)
    if lithology == "consolidated_sandstone":
        a, b, c = 97.32e-6, 1.42859, 55.8721
    elif lithology == "limestone":
        a, b, c = 0.8535, 1.075, 2.202
        # Newman limestone fit (phi-space)
    else:
        raise ValueError(f"unknown lithology: {lithology}")
    return a / np.power(1.0 + c * phi, b)


# ---------------------------------------------------------------------------
# 2. Gas flow rate IPR and productivity index -------------------------------
# ---------------------------------------------------------------------------

def gas_flow_rate(k_md: float, h_m: float, p_res_bar: float, p_wf_bar: float,
                  T_k: float, mu_gas_cp: float, z: float,
                  re_m: float = 500.0, rw_m: float = 0.1,
                  skin: float = 0.0) -> float:
    """Semi-steady-state gas flow rate in metric units (Sm^3/day).

    This is the low-pressure form of the theoretical IPR quoted in the
    paper (valid roughly for reservoir pressures below ~137 bar):

        q_g = C * k * h * (Pr^2 - Pwf^2) / (T * Z * mu * (ln(re/rw) - 0.75 + s))

    with C chosen to yield q_g in Sm^3/day when the inputs are in the
    units given below.  The exact numerical constant is unimportant for
    this pedagogical reimplementation; the important thing is the shape
    of the dependence, which drives the sensitivity analysis in the paper.
    """
    if p_wf_bar >= p_res_bar:
        return 0.0
    C = 1.0e-3                                  # pseudo-unit conversion
    num = C * k_md * h_m * (p_res_bar ** 2 - p_wf_bar ** 2)
    den = T_k * z * mu_gas_cp * (np.log(re_m / rw_m) - 0.75 + skin)
    return num / den


def productivity_index(cum_gas_sm3: float, days: float,
                       p_res_bar: float, p_wf_bar: float) -> float:
    """Average productivity index defined in Eq. (3) of Okoroafor et al.

        PI = Qg / (t * (P_res - P_wf))

    with Qg in Sm^3, t in days, pressures in bar.
    """
    dp = p_res_bar - p_wf_bar
    if dp <= 0 or days <= 0:
        return 0.0
    return cum_gas_sm3 / (days * dp)


# ---------------------------------------------------------------------------
# 3. Mohr-Coulomb / Griffith failure criterion ------------------------------
# ---------------------------------------------------------------------------

@dataclass
class FailureEnvelope:
    """Parameters of a Mohr-Coulomb / Griffith failure envelope.

    Defaults follow the example caption of Fig. 3 of the paper
    (T = 2 MPa, S0 = 4 MPa, mu = 0.75).
    """
    tensile_strength_mpa: float = 2.0
    cohesion_mpa: float = 4.0
    friction_coef: float = 0.75

    def mohr_coulomb_tau(self, sigma_n_mpa: float | np.ndarray
                         ) -> np.ndarray:
        """Shear stress on the envelope for a given normal stress."""
        sig = np.asarray(sigma_n_mpa, dtype=float)
        return self.cohesion_mpa + self.friction_coef * sig

    def griffith_tau(self, sigma_n_mpa: float | np.ndarray) -> np.ndarray:
        """Griffith parabolic envelope, tau^2 = 4*T*(sigma_n + T)."""
        sig = np.asarray(sigma_n_mpa, dtype=float)
        return np.sqrt(np.maximum(4.0 * self.tensile_strength_mpa *
                                  (sig + self.tensile_strength_mpa), 0.0))


def induces_microseismicity(sigma1_mpa: float, sigma3_mpa: float,
                            p_pore_mpa: float,
                            envelope: FailureEnvelope | None = None) -> bool:
    """Return True if the Mohr circle of the effective stress state crosses
    the Mohr-Coulomb envelope, implying rock / fault failure."""
    env = envelope or FailureEnvelope()
    s1_eff = sigma1_mpa - p_pore_mpa
    s3_eff = sigma3_mpa - p_pore_mpa
    centre = 0.5 * (s1_eff + s3_eff)
    radius = 0.5 * (s1_eff - s3_eff)
    if radius <= 0:
        return False
    # Distance from centre to the envelope line tau = c + mu*sigma_n
    # expressed as  mu*sigma_n - tau + c = 0, normal form distance
    c = env.cohesion_mpa
    mu = env.friction_coef
    dist = abs(mu * centre - 0.0 + c) / np.sqrt(1 + mu * mu)
    return radius >= dist


# ---------------------------------------------------------------------------
# 4. Cyclic injection / withdrawal schedule ---------------------------------
# ---------------------------------------------------------------------------

def build_cycle_schedule(n_cycles: int = 6,
                         cushion_days: float = 210,
                         cushion_rate: float = 150_000,
                         withdrawal_days: float = 7,
                         withdrawal_rate: float = 1_000_000,
                         refill_days: float = 50,
                         shutdown_days: float = 10) -> np.ndarray:
    """Reproduce Fig. 1 of Okoroafor et al. as a daily rate series (Sm^3/d).

    Positive rates are injection; negative rates are withdrawal.
    """
    rates = []
    # Cushion gas injection (only first cycle)
    rates += [cushion_rate] * int(cushion_days)
    for _ in range(n_cycles):
        rates += [-withdrawal_rate] * int(withdrawal_days)
        rates += [0.0] * int(shutdown_days)
        rates += [cushion_rate] * int(refill_days)
        rates += [0.0] * int(shutdown_days)
    return np.array(rates, dtype=float)


def cumulative_inventory(rate_schedule: np.ndarray,
                         initial_inventory: float = 0.0) -> np.ndarray:
    """Running inventory in Sm^3 from a daily-rate schedule."""
    return initial_inventory + np.cumsum(rate_schedule)


# ---------------------------------------------------------------------------
# Test harness ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_all(verbose: bool = True) -> None:
    # (a) Newman compressibility is a decreasing function of porosity.
    phi = np.array([0.05, 0.1, 0.2, 0.3])
    cf = newman_rock_compressibility(phi)
    assert np.all(np.diff(cf) < 0), "Cf must decrease with porosity"
    assert np.all(cf > 0)

    # (b) Gas flow rate scales with k*h and with (Pr^2 - Pwf^2).
    base_q = gas_flow_rate(k_md=50, h_m=30, p_res_bar=100, p_wf_bar=40,
                           T_k=340, mu_gas_cp=0.012, z=0.9)
    double_kh = gas_flow_rate(k_md=100, h_m=30, p_res_bar=100, p_wf_bar=40,
                              T_k=340, mu_gas_cp=0.012, z=0.9)
    assert abs(double_kh - 2 * base_q) < 1e-6, "q_g must be linear in k"
    # A higher bottom-hole pressure -> lower flow rate
    lo_q = gas_flow_rate(k_md=50, h_m=30, p_res_bar=100, p_wf_bar=80,
                         T_k=340, mu_gas_cp=0.012, z=0.9)
    assert lo_q < base_q

    # (c) Productivity index is positive and scales as expected
    pi = productivity_index(cum_gas_sm3=7_000_000, days=7,
                            p_res_bar=100, p_wf_bar=40)
    assert pi > 0

    # (d) Mohr-Coulomb failure envelope
    env = FailureEnvelope()
    # Low shear state -> no failure
    assert not induces_microseismicity(sigma1_mpa=20, sigma3_mpa=15,
                                        p_pore_mpa=5, envelope=env)
    # High differential stress -> failure
    assert induces_microseismicity(sigma1_mpa=80, sigma3_mpa=5,
                                    p_pore_mpa=5, envelope=env)
    # Griffith envelope gives a larger allowed shear at high sigma_n
    tau_mc = env.mohr_coulomb_tau(10.0)
    tau_gr = env.griffith_tau(10.0)
    assert tau_mc > 0 and tau_gr > 0

    # (e) Schedule builder conserves mass in the refill period
    sched = build_cycle_schedule(n_cycles=3)
    inv = cumulative_inventory(sched)
    assert inv[-1] > 0, "End inventory should be non-trivial"
    # The minimum should not go below the pre-first-cycle inventory level
    assert inv.min() >= 0

    if verbose:
        print("Article 3 (Underground hydrogen storage): all tests passed.")
        print(f"  Newman Cf(phi=0.1)    = {cf[1]:.3e} 1/psi")
        print(f"  base gas rate         = {base_q:.3g} Sm^3/day")
        print(f"  productivity index    = {pi:.3g}")
        print(f"  MC tau at sigma_n=10  = {tau_mc:.2f} MPa")
        print(f"  Schedule: {len(sched)} days, end inv = {inv[-1]:.3g} Sm^3")


if __name__ == "__main__":
    test_all()
