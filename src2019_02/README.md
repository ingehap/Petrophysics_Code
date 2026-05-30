# Petrophysics February 2019 - Vol. 60, No. 1

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 60, No. 1 (February 2019) - the volume-opening issue: an organic-mudstone
storage-capacity tutorial, four "Best of the 2018 Symposium, Part 2" papers, a
three-piece Depth Control section, and three regular submissions.

## Quick start

```bash
pip install numpy

# Run all 11 module tests
python test_all.py

# Or run a single article
python article9_azimuthal_gr_geosteering.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_organic_mudstone_storage_part1_tutorial.py` | *Tutorial:* Organic-Mudstone Petrophysics: Workflow to Estimate Storage Capacity (Part 1) | Newsham, Comisky, Chemali | 10.30632/PJV60N1Y2019t1 |
| `article2_carbonate_netpay_cutoffs.py` | Defining Net-Pay Cutoffs in Carbonates Using Advanced Petrophysical Methods | Skalinski, Mallan, Edwards, Sun, Toumelin, Kelly, Wushur, Sullivan | 10.30632/PJV60N1Y2019a1 |
| `article3_2d_nmr_t1t2_shale.py` | Frequency and Temperature Dependence of 2D NMR T1-T2 Maps of Shale | Kausik, Freed, Fellah, Feng, Ling, Simpson | 10.30632/PJV60N1Y2019a2 |
| `article4_insitu_saturation_core_comparison.py` | Maintaining and Reconstructing In-Situ Saturations: Whole Core, Sidewall Core, and Pressurized Sidewall Core in the Permian Basin | Blount, McMullen, Durand, Driskill | 10.30632/PJV60N1Y2019a3 |
| `article5_composite_cement_well_integrity.py` | Novel Composite Cement for Improved Well Integrity Evaluation | Elshahawi, Huang, Pollock, Veedu | 10.30632/PJV60N1Y2019a4 |
| `article6_depth_love_hate_essay.py` | Depth: A Love and Hate Story | Theys | 10.30632/PJV60N1Y2019a5 |
| `article7_groningen_depth_control.py` | *Technical Note:* Connecting the Dots—Proper Depth Control in the Discovery of the Groningen Field | Fokkema, Visser | 10.30632/PJV60N1Y2019a6 |
| `article8_drillers_depth_correction.py` | Correction of Driller's Depth: Field Example Using Driller's Way-Point Depth Correction Methodology | Bolt | 10.30632/PJV60N1Y2019a7 |
| `article9_azimuthal_gr_geosteering.py` | Modeling of Azimuthal Gamma-Ray Tools for Use in Geosteering in Unconventional Reservoirs | Wang, Stockhausen, Wyatt, Gulick | 10.30632/PJV60N1Y2019a8 |
| `article10_hydraulic_fracturing_stress_test.py` | Feasibility and Design of Hydraulic Fracturing Stress Tests Using a Quantitative Risk Assessment and Control Approach | Bérard, Chugunov, Desroches, Prioul | 10.30632/PJV60N1Y2019a9 |
| `article11_neutron_generator_vs_ambe.py` | Neutron Generators as Alternatives to Am-Be Sources in Well Logging: An Assessment of Fundamentals | Badruzzaman, Schmidt, Antolak | 10.30632/PJV60N1Y2019a10 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2019_02.pdf`)
> has a text layer, so the titles, authors, page ranges, and DOIs were read from
> the contents page and paper bodies (the DOI suffix is printed as
> `PJV60N1Y2019…`, with `Y` in place of a hyphen that year). The machine
> extraction captured the full bodies of the **tutorial and articles a1-a8** but
> **truncated after a8** (page ~112), so **articles a9-a10 were available only as
> table-of-contents entries** and are implemented as **methodology proxies**. As
> with the other issues, the typeset formula glyphs were dropped in extraction,
> so the numbered formulas are faithful standard-form reconstructions.

## Implementation notes & substitutions

- **Article 1 (Newsham et al.)** *(tutorial)*: bulk volume water / movable-fluid
  index, free + Langmuir adsorbed gas, and the Buckley-Leverett fractional flow
  / water cut tying storage to producibility.

- **Article 2 (Skalinski et al.)**: Winland R35 and a Lucia rock-fabric
  permeability transform, a permeability→porosity cutoff inversion, and net-pay
  / net-to-gross from integrated porosity / Vsh / Sw cutoffs.

- **Article 3 (Kausik et al.)**: the T1/T2 ratio and fluid typing on a 2D map,
  the temperature dependence of T2 (~ T/viscosity), and the frequency (field)
  dependence of the T1/T2 ratio.

- **Article 4 (Blount et al.)**: Dean-Stark water/oil saturation, fluid-loss
  factors by core type (pressurized > whole > sidewall), the mass-balance
  reconstruction of in-situ saturation, and a saturation-closure check.

- **Article 5 (Elshahawi et al.)**: acoustic impedance Z = ρ·v, the reflection
  coefficient, annulus classification (gas / liquid / cement) by impedance, and
  a cement bond index from attenuation.

- **Article 6 (Theys)** *(essay)*: wireline cable stretch - elastic dL = T·L/(E·A)
  and thermal dL = α·L·ΔT - the true-depth correction, and depth uncertainty.

- **Article 7 (Fokkema & Visser)** *(technical note)*: a marker depth-tie shift,
  the net-pay-thickness error from a depth mismatch, and the resulting
  gas-in-place error - plus the datum correction.

- **Article 8 (Bolt)**: driller's-depth correction - drillstring stretch under
  buoyed own-weight (∝ L²) and hook load, thermal elongation, and a way-point
  interpolation of the correction between calibration depths.

- **Article 9 (Wang et al.)**: azimuthal sector averaging, the up-down GR
  contrast as a boundary-proximity indicator, the distance-to-boundary from the
  contrast decay, and the apparent bedding dip from the sinusoidal image.

- **Article 10 (Bérard et al.)** *(methodology proxy)*: a mini-frac G-function
  pressure decline, the closure-pressure pick, the minimum-stress gradient, and
  a quantitative-risk probability of a successful test.

- **Article 11 (Badruzzaman et al.)** *(methodology proxy)*: the Am-Be vs D-T
  source comparison (energy / output / pulsed), the energy- and
  porosity-dependent neutron slowing-down length, the porosity sensitivity, and
  the counting-statistics precision.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2019)
DOI: <doi>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
