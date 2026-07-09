"""
Article 7: Case Studies on Multistring Isolation Evaluation in P&A Operations
Zhang, Mueller, Bryce, Brockway, Iskander (2021)
DOI: 10.30632/PJV62N6-2021a7

Field case studies of an acoustic Multistring Isolation Evaluation (MSIE)
technology that assesses cement / zonal isolation behind an OUTER casing while
logging from inside production tubing or an inner casing - through two metal
strings - so tubing need not be pulled during plug-and-abandonment.  A monopole
transmitter and 3-/5-ft receivers record waveforms transformed to the
frequency domain; resonance amplitudes change between isolated and free-pipe
states.

Implements:

  - Acoustic impedance  Z = rho * v                            (R1)
  - Reflection coefficient  R = (Z2 - Z1)/(Z2 + Z1)            (R2)
  - Transmitted energy  1 - R^2  (and two-interface loss)      (R3)
  - Casing thickness resonance  f_n = n*v/(2*d)                (R4)
  - Impedance classification (cement / liquid / gas)
  - Isolation qualification logic (continuous + cumulative)

Note: this is a case-study paper with no equations and a proprietary
inversion; the relations here are standard acoustics (flagged as
reconstructions) implementing a physically-plausible forward + interpretation
demonstrator.  Impedance in MRayl, density in kg/m^3, velocity in m/s.
"""

import numpy as np

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib

# Standard material properties (reconstructed defaults; not from the paper)
RHO_STEEL, V_STEEL = 7850.0, 5900.0      # kg/m^3, m/s
Z_STEEL = RHO_STEEL * V_STEEL / 1e6      # MRayl (~46)
Z_CEMENT = 5.4
Z_BRINE = 1.5
Z_GAS = 0.05

# Pulse-echo impedance cutoffs (MRayl)
Z_LIQUID_MAX = 2.6
Z_CEMENT_MIN = 3.0


# ---------------------------------------------- R1-R3: acoustics --------

def acoustic_impedance(rho, v):
    """Acoustic impedance  Z = rho * v  (R1).  Returns MRayl for SI inputs."""
    return petrolib.acoustic_geomech.acoustic_impedance(rho, v, out="mrayl")


def reflection_coefficient(z1, z2):
    """Normal-incidence reflection coefficient  R = (Z2 - Z1)/(Z2 + Z1)  (R2)."""
    return petrolib.acoustic_geomech.reflection_coefficient(z1, z2)


def transmitted_energy(z1, z2):
    """Energy transmission coefficient  1 - R^2  (R3)."""
    return petrolib.acoustic_geomech.transmission_energy(z1, z2)


def two_interface_energy_fraction(z_metal, z_fluid):
    """Energy fraction surviving a round trip through a tubing wall.

    Two fluid<->metal interfaces in, two out (4 crossings); used to check the
    paper's ~95% energy-loss-through-one-tubing-layer statement.
    """
    t = transmitted_energy(z_fluid, z_metal)
    return t ** 4


# ---------------------------------------------- R4: resonance -----------

def thickness_resonance(d_m, v=V_STEEL, n=1):
    """Plate thickness resonance  f_n = n * v / (2 d)  (R4).  Hz."""
    return n * v / (2.0 * d_m)


# ---------------------------------------------- classification ----------

def classify_behind_casing(z_behind):
    """Map the impedance behind casing to gas / liquid / cement."""
    if z_behind < Z_LIQUID_MAX * 0.1:
        return "gas"
    if z_behind < Z_LIQUID_MAX:
        return "liquid"
    if z_behind >= Z_CEMENT_MIN:
        return "cement"
    return "transition"


# ---------------------------------------------- qualification -----------

def isolation_qualifies(isolated_flag, dz_ft,
                        min_continuous_ft=15.0, n_continuous=2,
                        min_cumulative_ft=200.0):
    """Operational P&A isolation criterion.

    Requires at least `n_continuous` continuous isolated runs each >=
    `min_continuous_ft`, AND total isolated footage >= `min_cumulative_ft`.
    isolated_flag : boolean array per depth step of length-`dz_ft` samples.
    """
    flag = np.asarray(isolated_flag, bool)
    cumulative = flag.sum() * dz_ft
    # find continuous runs
    runs, run = [], 0
    for v in flag:
        if v:
            run += 1
        else:
            if run:
                runs.append(run)
            run = 0
    if run:
        runs.append(run)
    long_runs = [r for r in runs if r * dz_ft >= min_continuous_ft]
    return bool((len(long_runs) >= n_continuous) and (cumulative >= min_cumulative_ft))


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 7: Multistring Isolation Evaluation (Acoustic)")
    print("=" * 60)

    print(f"  Z steel / cement / brine = {Z_STEEL:.1f} / {Z_CEMENT} / {Z_BRINE} MRayl")
    assert 45 < Z_STEEL < 47

    # Strong steel/brine mismatch -> most energy reflected
    R = reflection_coefficient(Z_BRINE, Z_STEEL)
    print(f"  R (brine->steel)       = {R:.3f}")
    assert R > 0.9

    # The paper's ~95% energy loss through a single tubing layer
    surviving = two_interface_energy_fraction(Z_STEEL, Z_BRINE)
    loss = 1.0 - surviving
    print(f"  energy lost thru tubing= {100*loss:.1f}%")
    assert loss > 0.95

    # Casing wall resonance (9.625-in casing, ~10 mm wall)
    f1 = thickness_resonance(0.010)
    print(f"  thickness resonance f1 = {f1/1e3:.0f} kHz")
    assert 250e3 < f1 < 350e3

    # Behind-casing classification
    assert classify_behind_casing(Z_CEMENT) == "cement"
    assert classify_behind_casing(Z_BRINE) == "liquid"
    assert classify_behind_casing(Z_GAS) == "gas"

    # Isolation qualification: build a synthetic isolated-flag log (0.5-ft step)
    dz = 0.5
    n = 1200                      # 600 ft logged
    flag = np.zeros(n, bool)
    flag[100:160] = True          # 30-ft continuous isolated run
    flag[300:340] = True          # 20-ft continuous isolated run
    flag[500:820] = True          # 160-ft continuous isolated run
    qual = isolation_qualifies(flag, dz)
    iso_ft = flag.sum() * dz
    print(f"  isolated footage       = {iso_ft:.0f} ft  -> qualifies={qual}")
    assert iso_ft >= 200.0 and qual is True
    # too little isolation should fail
    assert isolation_qualifies(flag[:200], dz) is False
    assert iso_ft >= 200.0
    print("  PASS")
    return {"Z_steel": Z_STEEL, "tubing_loss": loss, "f1": f1, "qualifies": qual}


if __name__ == "__main__":
    test_all()
