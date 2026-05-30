# Petrophysics April 2020 - Vol. 61, No. 2

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 61, No. 2 (April 2020) - a hybrid issue with a **"Best of the Society of
Core Analysts (SCA) 2019 International Symposium"** special section (articles
1-6) followed by regular submissions (articles 7-9).  Topics span critical gas
saturation by micro-CT, coupled NMR/ultrasonic core measurement, crushed-rock
Klinkenberg permeability, a dielectric CEC proxy, multiscale wettability
upscaling, Lattice-Boltzmann gas-condensate relative permeability, shale
imbibition relative permeability and capillary pressure, spontaneous mixed-wet
imbibition, and chemically induced formation damage.

## Quick start

```bash
pip install numpy scipy

# Run all 9 module tests
python test_all.py

# Or run a single article
python article3_crushed_rock_klinkenberg.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_critical_gas_saturation_microct.py` | Determination of Critical Gas Saturation by Micro-CT | Berg, Gao, Georgiadis, Brussee, Coorn, van der Linde, Dietderich, Alpak, Eriksen, Mooijer-van den Heuvel, Southwick, Appel, Wilson | 10.30632/PJV61N2-2020a1 |
| `article2_coupled_nmr_ultrasonic.py` | A New Apparatus for Coupled Low-Field NMR and Ultrasonic Measurements in Rocks at Reservoir Conditions | Connolly, Sarout, Dautriat, May, Johns | 10.30632/PJV61N2-2020a2 |
| `article3_crushed_rock_klinkenberg.py` | Low-Permeability Measurement on Crushed Rock: Insights | Profice, Lenormand | 10.30632/PJV61N2-2020a3 |
| `article4_cec_dielectric_proxy.py` | A New CEC-Measurement Proxy Using High-Frequency Dielectric Analysis of Crushed Rock | Stokes, Yang, Ezebuiro, Fischer | 10.30632/PJV61N2-2020a4 |
| `article5_wettability_upscaling.py` | Workflow for Upscaling Wettability From the Nanoscale to Core Scale | Rücker, Bartels, Bultreys, Boone, Singh, Garfi, Scanziani, Spurin, Yesufu-Rufai, Krevor, Blunt, Wilson, Mahani, Cnudde, Luckham, Georgiadis, Berg | 10.30632/PJV61N2-2020a5 |
| `article6_gas_condensate_lbm_relperm.py` | Estimation of Gas-Condensate Relative Permeability Using a Lattice Boltzmann Modeling Approach | Schembre-McCabe, Kamath, Fager, Crouse | 10.30632/PJV61N2-2020a6 |
| `article7_shale_imbibition_relperm_pc.py` | Effect of Injection Pressure on the Imbibition Relative Permeability and Capillary Pressure Curves of Shale Gas Matrix | Al-Ameri, Mazeel | 10.30632/PJV61N2-2020a7 |
| `article8_spontaneous_imbibition_mixedwet.py` | Spontaneous Gas-Water Imbibition in Mixed-Wet Pores | Wang, He, Xiao, Wang, Ma | 10.30632/PJV61N2-2020a8 |
| `article9_chemical_formation_damage_shale.py` | Chemically Induced Formation Damage in Shale | Wick, Taneja, Gupta, Sondergeld, Rai | 10.30632/PJV61N2-2020a9 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2020_04.pdf`)
> has a text layer, so the article titles, authors, page ranges, and DOIs were
> read directly from the contents page and paper bodies. The machine text
> extraction captured the full bodies of **articles 1-5** (and the first page of
> article 6) but **truncated at page 206**, so **articles 6 (partly) and 7-9
> were available only as table-of-contents entries**. Articles 7-9 are therefore
> **methodology proxies** that implement the standard, well-established
> quantitative relations their titles describe; and, as with the other issues,
> the typeset formula glyphs were dropped in extraction, so the numbered formulas
> are faithful standard-form reconstructions built from the preserved variable
> definitions and the textbook closed-forms each paper cites.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions. A few practical notes:

- **Article 1 (Berg et al.)** *(no equations):* critical gas saturation as the
  3D percolation threshold - a random field is thresholded by gas fraction and
  the spanning-cluster onset gives Sgc - plus a connectivity-based relative-
  permeability proxy (scipy.ndimage labeling). Sgc lands in the physical range
  (paper: 0.20-0.25).

- **Article 2 (Connolly et al.)**: the NMR T1 recovery / T2 decay (Eqs. 1-2),
  the Brownstein-Tarr surface relaxation 1/T2 = 1/T2bulk + ρs(S/V) + diffusion
  (Eq. 3), the multiexponential T2 sum (Eq. 4), and the elastic velocities
  Vp = √[(K+4µ/3)/ρ], Vs = √(µ/ρ) (Eqs. 5-6).

- **Article 3 (Profice & Lenormand)**: the Klinkenberg apparent permeability
  k_app = k_l(1 + b/Pm) (Eq. 3) with the 1/Pm extrapolation to liquid
  permeability, the mean pressure (Eq. 4), and the theoretical He/N₂ slip-factor
  ratio (Eqs. 1-2) - reproduces the paper's value of 2.9.

- **Article 4 (Stokes et al.)**: the RH-dependent piecewise-linear CEC
  calibration CEC = S_RH·(ε′−2.5) + C (Eqs. 2-5), anchored at the pure-quartz
  point (ε′ = 2.5, CEC = 0) with three RH regimes and a constant correction
  C ≈ 4 meq/100 g; the linear fit scores R² > 0.98.

- **Article 5 (Rücker et al.)** *(no closed-form equations):* the Young-Laplace
  drainage threshold radius, the Wenzel roughness-corrected contact angle, and
  volume-weighted contact-angle upscaling - the nanoscale-to-core workflow.

- **Article 6 (Schembre-McCabe et al.)** *(only first page in the extract):* the
  capillary number N_c = µv/σ (Eq. 1) plus the standard capillary-desaturation
  (rate-effect) gas relative permeability vs N_c and base Corey krg/kro curves
  the LBM study parameterizes.

- **Article 7 (Al-Ameri & Mazeel)** *(methodology proxy):* Brooks-Corey
  capillary pressure and wetting/gas relative permeability, with the
  injection-pressure effect modeled as a rising imbibed-water saturation that
  suppresses gas relative permeability.

- **Article 8 (Wang et al.)** *(methodology proxy):* the Lucas-Washburn √t
  imbibition length, the Young-Laplace capillary driving pressure (positive for
  water-wet, negative for oil-wet), and the net mixed-wet capillary force from
  the water-wet / oil-wet pore fractions.

- **Article 9 (Wick et al.)** *(methodology proxy):* the retained-permeability
  (damage) ratio, a clay-swelling permeability reduction k = k0(1−ε)ⁿ, the
  Kozeny-Carman porosity-permeability sensitivity, and the fracture cubic law.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2020)
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
