# Petrophysics October 2019 - Vol. 60, No. 5

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 60, No. 5 (October 2019) - the **"Best of the 2019 SPWLA Symposium"** issue
(articles 1-7) plus regular submissions (articles 8-10).  Topics span
thermal-maturity-adjusted log interpretation, free/adsorbed gas quantification,
machine-learning depth matching, net-sand estimation from borehole images, an
in-situ "log-soak-log" imbibition experiment, time-lapse micro-CT of mud
invasion, a through-casing dual-source acoustic tool, unconventional rock typing,
ANN bulk-density prediction, and a through-casing transient-EM conductivity
measurement.

## Quick start

```bash
pip install numpy

# Run all 10 module tests
python test_all.py

# Or run a single article
python article7_through_casing_acoustic_dualsource.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_tmali_organic_shales.py` | Thermal Maturity-Adjusted Log Interpretation (TMALI) in Organic Shales | Craddock, Miles, Lewis, Pomerantz | 10.30632/PJV60N5-2019a1 |
| `article2_free_adsorbed_gas_shale.py` | More Accurate Quantification of Free and Adsorbed Gas in Shale Reservoirs | Ansari, Merletti, Gramin, Armitage | 10.30632/PJV60N5-2019a2 |
| `article3_ml_depth_matching.py` | A Machine-Learning Framework for Automating Well-Log Depth Matching | Le, Liang, Zimmermann, Zeroug, Heliot | 10.30632/PJV60N5-2019a3 |
| `article4_netsand_borehole_image_nn.py` | Estimating Net Sand From Borehole Images in Laminated Deepwater Reservoirs With a Neural Network | Gong, Keele, Toumelin, Clinch | 10.30632/PJV60N5-2019a4 |
| `article5_log_soak_log_imbibition.py` | 'Log-Soak-Log' Experiment in Tengiz Field: Novel Technology for In-Situ Imbibition Measurements to Support an Improved Oil Recovery Project | Seth, Villegas, Iskakov, Playton, Lindsell, Cordova, Turmanbekova, Wang | 10.30632/PJV60N5-2019a5 |
| `article6_microct_invasion_mudcake.py` | Experimental Method for Time-Lapse Micro-CT Imaging of Mud-Filtrate Invasion and Mudcake Deposition | Schroeder, Torres-Verdín | 10.30632/PJV60N5-2019a6 |
| `article7_through_casing_acoustic_dualsource.py` | A Through-Casing Acoustic Logging Tool Using Dual-Source Transmitters | Tang, Su, Zhuang | 10.30632/PJV60N5-2019a7 |
| `article8_unconventional_rock_typing.py` | Presenting a Multifaceted Approach to Unconventional Rock Typing and Technical Validation—Case Study in the Permian Basin | Perry, Hayes | 10.30632/PJV60N5-2019a8 |
| `article9_ann_bulk_density_drilling.py` | Application of Artificial Neural Network to Predict Formation Bulk Density While Drilling | Gowida, Elkatatny, Abdulraheem | 10.30632/PJV60N5-2019a9 |
| `article10_through_casing_tem_conductivity.py` | Through-Casing Formation Conductivity Measurement Based on Transient Electromagnetic Logging Data | Sheng, Shen, Shen, Zhu, Zang | 10.30632/PJV60N5-2019a10 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2019_10.pdf`,
> 15 MB) has a text layer, so the article titles, authors, page ranges, and DOIs
> were read directly from the contents page and paper bodies. The machine text
> extraction captured the full bodies of **articles 1-7**, was **truncated inside
> Article 8** (mid-references at journal page 657), and contained **articles 9-10
> only as table-of-contents entries**. Articles 9-10 are therefore implemented as
> **methodology proxies** of the standard methods their titles describe, and
> Article 8 uses the standard Amaefule HFU / Winland forms it cites; as with the
> other issues, the typeset formula glyphs were dropped in extraction, so the
> numbered formulas are faithful standard-form reconstructions.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions. A few practical notes:

- **Article 1 (Craddock et al.)**: element molar fractions (Eq. 1), electron
  density ρ_e = 2ρ·ΣZ/ΣA (Eq. 3) and apparent density ρ_a = 1.0704ρ_e − 0.1883
  (Eq. 4), the kerogen hydrogen index (Eq. 6), TOC→kerogen volume (Eq. 10), the
  bulk-density/density-porosity pair (Eqs. 11-12), and a maturity-adjusted
  kerogen density vs Ro - shows a wrong (immature) endpoint biases kerogen
  volume.

- **Article 2 (Ansari et al.)** *(body was font-garbled in extraction):* the free
  gas G_free = φ(1−Sw)/Bg, the Langmuir adsorbed gas Gc = ρ_b·VL·P/(PL+P), the
  Gibbs adsorbed-phase-density correction, the adsorbed-monolayer porosity
  correction, and total gas-in-place.

- **Article 3 (Le et al.)** *(no numbered equations):* the cross-correlation
  alignment lag, dynamic time warping, and Pearson correlation that the ML
  depth-matcher is built on - recovers a planted depth shift.

- **Article 4 (Gong et al.)** *(no numbered equations):* a neural network
  regressing the sand fraction from borehole-image brightness-histogram features,
  beating a fixed cutoff that the OBM nonlinearity defeats (NN RMSE < cutoff
  RMSE, R > 0.9).

- **Article 5 (Seth et al.)** *(field experiment):* Sigma water saturation, the
  time-lapse saturation change, the Sigma sensitivity per unit Sw, and the
  detectability of a 5% change in a 2-p.u. rock with a 220-c.u. brine - the
  experiment's design criterion.

- **Article 6 (Schroeder & Torres-Verdín)** *(imaging method):* the Beer-Lambert
  attenuation law, CT porosity/saturation from voxel attenuation, and
  sqrt-of-time mudcake growth with an advancing invasion front.

- **Article 7 (Tang et al.)**: the dual-source casing-cancellation delay
  τ = L/v_casing, the destructive-interference removal of the casing wave, and
  slowness-time-coherence (semblance) picking - recovers a ~3700 m/s formation
  slowness that the casing wave otherwise masks.

- **Article 8 (Perry & Hayes)** *(extract truncated mid-references):* the Amaefule
  RQI = 0.0314√(k/φ), normalized porosity, FZI = RQI/φ_z, and the Winland R35
  relation, plus HFU assignment by clustering log(FZI) - FZI is constant within a
  flow unit.

- **Article 9 (Gowida et al.)** *(methodology proxy):* a feed-forward tanh ANN
  predicting formation bulk density from six drilling parameters, scored by R /
  RMSE / AAPE.

- **Article 10 (Sheng et al.)** *(methodology proxy):* the late-time TEM decay
  V(t) ~ σ^{3/2}·t^{−5/2}, the EM diffusion depth, and recovery of the formation
  conductivity from the late-time response.

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
