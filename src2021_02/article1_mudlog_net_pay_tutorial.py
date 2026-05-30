"""
Article 1 (Tutorial): Maximizing Value From Mudlogs - Integrated Approach to
                      Determine Net Pay
Malik, Hanson, Clinch (2021)
DOI: 10.30632/PJV62N1-2021t1

A tutorial on extracting quantitative net-pay information from mudlogs.  Total
gas is first normalized to remove the drilling-parameter (ROP / flow / hole
size) dilution that otherwise masks the true formation gas content; the
chromatographic C1-C5 components are then reduced to the classic Haworth
wetness / balance / character ratios (and Pixler component ratios) that type
the hydrocarbon; finally an integrated cutoff scheme (gas show + porosity +
shale-volume + water-saturation) flags reservoir and pay and sums net pay and
net-to-gross over an interval.

Implements:

  - Gas normalization  GN = G * Q / (ROP * A_bit)   (gas per unit rock volume)
  - Haworth ratios  Wh (wetness), Bh (balance), Ch (character)
  - Haworth interpretation rules (dry gas / gas / oil / residual)
  - Pixler component ratios  C1/C2 ... C1/C5
  - Integrated net-pay / net-to-gross cutoff accounting

Note: this issue's source PDF has no usable text layer (image-rendered
typesetting), so the numbered formulas are faithful standard-form
reconstructions of the mudlog gas-ratio and net-pay methods the tutorial
describes.  Gas-ratio cutoffs follow the published Haworth/Pixler conventions.
"""

import numpy as np

# Classic Haworth (1985) wetness-ratio productivity bands (per cent)
WH_DRYGAS_MAX = 0.5
WH_GAS_MAX = 17.5
WH_OIL_MAX = 40.0


# ---------------------------------------------- gas normalization -------

def normalized_gas(total_gas, rop_m_hr, flow_lpm, bit_diam_in):
    """Gas normalized to rock volume  GN = G * Q / (ROP * A_bit).

    Higher ROP liberates more gas per unit time and higher mud flow dilutes it,
    so the formation gas content is proportional to  G * flow / (ROP * area).
    total_gas in gas units (or %), ROP in m/hr, flow in L/min, bit in inches.
    Returns a normalized gas index (same units as total_gas, drilling removed).
    """
    area = np.pi * (np.asarray(bit_diam_in, float) * 0.0254 / 2.0) ** 2   # m^2
    rop = np.maximum(np.asarray(rop_m_hr, float), 1e-6)
    return np.asarray(total_gas, float) * np.asarray(flow_lpm, float) / (rop * area)


# ---------------------------------------------- Haworth ratios ----------

def haworth_ratios(c1, c2, c3, c4, c5):
    """Haworth wetness / balance / character ratios from C1-C5 (gas units).

      Wh = 100 * (C2+C3+C4+C5) / (C1+C2+C3+C4+C5)   (wetness, %)
      Bh = (C1+C2+C3+C4+C5) / (C3+C4+C5)            (balance)
      Ch = (C4+C5) / C3                             (character)
    """
    c1, c2, c3, c4, c5 = (float(x) for x in (c1, c2, c3, c4, c5))
    total = c1 + c2 + c3 + c4 + c5
    heavy = c3 + c4 + c5
    wh = 100.0 * (c2 + c3 + c4 + c5) / total
    bh = total / heavy if heavy > 0 else np.inf
    ch = (c4 + c5) / c3 if c3 > 0 else 0.0
    return wh, bh, ch


def haworth_interpretation(wh, bh):
    """Classify the hydrocarbon from the wetness Wh and balance Bh ratios.

    Follows the Haworth productivity bands:
      Wh < 0.5            -> very dry gas
      0.5 <= Wh < 17.5    -> gas (Bh > Wh) or gas-condensate
      17.5 <= Wh < 40     -> oil
      Wh >= 40            -> residual oil / very low productivity
    Bh < Wh signals increasing density / heavier product.
    """
    if wh < WH_DRYGAS_MAX:
        return "dry gas"
    if wh < WH_GAS_MAX:
        return "gas" if bh > wh else "gas-condensate"
    if wh < WH_OIL_MAX:
        return "oil"
    return "residual oil"


def pixler_ratios(c1, c2, c3, c4, c5):
    """Pixler light-component ratios  C1/C2, C1/C3, C1/C4, C1/C5.

    A productive zone shows ratios that decrease from C1/C2 to C1/C5 and lie
    within the Pixler band (C1/C2 roughly 2-35); C1/C5 > ~200 with a flat
    profile indicates non-productive gas.
    """
    c = [float(x) for x in (c2, c3, c4, c5)]
    return [float(c1) / x if x > 0 else np.inf for x in c]


# ---------------------------------------------- net pay -----------------

def pay_flags(gas_show, phi, vsh, sw,
              gas_cut=1.0, phi_cut=0.08, vsh_cut=0.40, sw_cut=0.60):
    """Per-sample reservoir and pay boolean flags from integrated cutoffs.

    Reservoir = porosity and shale-volume cutoffs met.
    Pay       = reservoir AND a gas show AND water-saturation cutoff met.
    Returns (reservoir_flag, pay_flag) boolean arrays.
    """
    phi = np.asarray(phi, float)
    vsh = np.asarray(vsh, float)
    sw = np.asarray(sw, float)
    gas = np.asarray(gas_show, float)
    reservoir = (phi >= phi_cut) & (vsh <= vsh_cut)
    pay = reservoir & (gas >= gas_cut) & (sw <= sw_cut)
    return reservoir, pay


def net_pay(depth, pay_flag):
    """Sum the thickness of flagged pay samples (depth uniform or irregular)."""
    depth = np.asarray(depth, float)
    dz = np.gradient(depth)
    return float(np.sum(np.abs(dz)[np.asarray(pay_flag, bool)]))


def net_to_gross(depth, reservoir_flag, pay_flag):
    """Net pay / gross reservoir thickness (fraction)."""
    depth = np.asarray(depth, float)
    dz = np.abs(np.gradient(depth))
    gross = float(np.sum(dz[np.asarray(reservoir_flag, bool)]))
    pay = float(np.sum(dz[np.asarray(pay_flag, bool)]))
    return pay / gross if gross > 0 else 0.0


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 1: Maximizing Value From Mudlogs (Net Pay)")
    print("=" * 60)

    # Gas normalization removes the drilling-rate effect: the same formation
    # drilled twice as fast liberates (and reads) twice the raw gas, but the
    # normalized gas is identical.
    gn_slow = normalized_gas(50.0, rop_m_hr=10.0, flow_lpm=2000.0, bit_diam_in=8.5)
    gn_fast = normalized_gas(100.0, rop_m_hr=20.0, flow_lpm=2000.0, bit_diam_in=8.5)
    print(f"  GN slow / fast         = {gn_slow:.1f} / {gn_fast:.1f}")
    assert abs(gn_slow - gn_fast) / gn_slow < 1e-9   # identical after normalizing

    # Haworth ratios on a gas-bearing chromatograph
    wh, bh, ch = haworth_ratios(80.0, 8.0, 4.0, 2.0, 1.0)
    interp = haworth_interpretation(wh, bh)
    print(f"  Wh = {wh:.2f}%  Bh = {bh:.2f}  Ch = {ch:.2f} -> {interp}")
    assert 0.5 <= wh < WH_GAS_MAX and interp in ("gas", "gas-condensate")

    # A very dry gas (almost all C1) classifies as dry gas
    wh_dry, bh_dry, _ = haworth_ratios(99.0, 0.3, 0.1, 0.05, 0.02)
    assert haworth_interpretation(wh_dry, bh_dry) == "dry gas"

    # An oil-prone chromatograph (wetness in the 17.5-40% band) classifies as oil
    wh_oil, bh_oil, _ = haworth_ratios(75.0, 10.0, 7.0, 5.0, 3.0)
    print(f"  oil-case Wh            = {wh_oil:.2f}% -> {haworth_interpretation(wh_oil, bh_oil)}")
    assert haworth_interpretation(wh_oil, bh_oil) == "oil"

    # Pixler ratios decrease for a productive light-oil/gas profile
    pix = pixler_ratios(80.0, 8.0, 4.0, 2.0, 1.0)
    print(f"  Pixler C1/Cn           = {np.array2string(np.array(pix), precision=1)}")
    assert pix[0] < pix[1] < pix[2] < pix[3]

    # Integrated net pay over a 20-sample (10 m) synthetic mud/log interval
    depth = np.linspace(2000.0, 2019.0, 20)        # 1-m samples
    rng = np.random.default_rng(1)
    phi = np.full(20, 0.05); phi[5:15] = 0.18      # a 10-m porous sand
    vsh = np.full(20, 0.6);  vsh[5:15] = 0.15
    sw = np.full(20, 0.9);   sw[5:13] = 0.35       # oil down to sample 12
    gas = np.full(20, 0.2);  gas[5:13] = 12.0      # gas shows over the pay
    res, pay = pay_flags(gas, phi, vsh, sw)
    npay = net_pay(depth, pay)
    ntg = net_to_gross(depth, res, pay)
    print(f"  reservoir samples      = {int(res.sum())}  pay samples = {int(pay.sum())}")
    print(f"  net pay                = {npay:.1f} m   N/G = {ntg:.2f}")
    assert int(pay.sum()) == 8 and abs(npay - 8.0) < 1.0
    assert 0.0 < ntg <= 1.0
    print("  PASS")
    return {"Wh": wh, "Bh": bh, "interp": interp, "net_pay_m": npay, "NTG": ntg}


if __name__ == "__main__":
    test_all()
