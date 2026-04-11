# Petrophysics_Code

Unofficial Python implementations of articles published in
[*Petrophysics*](https://www.spwla.org/petrophysics-journal) — the journal of the
Society of Petrophysicists and Well Log Analysts (SPWLA).

Each module translates the key algorithms and equations from a single journal
article into self-contained Python code with synthetic-data demonstrations.
The implementations are meant for learning and experimentation, not as a
replacement for the original papers.

## Requirements

* Python 3.9+
* NumPy ≥ 1.24
* SciPy ≥ 1.10
* scikit-learn ≥ 1.2 (required by `src2024_08`)

## Repository layout

```
Petrophysics_Code/
├── src2023_12/   Vol. 64 No. 6 (Dec 2023)  —  8 modules + test suite
├── src2024_02/   Vol. 65 No. 1 (Feb 2024)  —  7 modules + test suite
├── src2024_04/   Vol. 65 No. 2 (Apr 2024)  —  6 modules + test suite
├── src2024_06/   Vol. 65 No. 3 (Jun 2024)  —  8 modules + test suite
├── src2024_08/   Vol. 65 No. 4 (Aug 2024)  — 14 modules + test suite
├── src2024_10/   Vol. 65 No. 5 (Oct 2024)  — 10 modules + test suite
├── src2024_12/   Vol. 65 No. 6 (Dec 2024)  — 13 modules + test suite
├── src2025_02/   Vol. 66 No. 1 (Feb 2025)  — 12 modules + test suite
├── src2025_04/   Vol. 66 No. 2 (Apr 2025)  —  9 modules + test suite
├── src2025_06/   Vol. 66 No. 3 (Jun 2025)  —  8 modules + test suite
├── src2025_08/   Vol. 66 No. 4 (Aug 2025)  — 11 modules + test suite
├── src2025_10/   Vol. 66 No. 5 (Oct 2025)  — 11 modules + test suite
├── src2025_12/   Vol. 66 No. 6 (Dec 2025)  — 13 modules + test suite
├── src2026_02/   Vol. 67 No. 1 (Feb 2026)  — 11 modules + test suite
└── src2026_04/   Vol. 67 No. 2 (Apr 2026)  — 12 modules + test suite
```

---

## src2023_12 — Vol. 64, No. 6 (December 2023)

"Best Papers of the 2023 Symposium" issue covering deeply-invaded saturation
inversion, a proposed universal wellbore data format, mud-gas viscosity
estimation, 2D NMR fluid component decomposition, salt-cavern creep damage for
underground storage, a new pulsed-neutron C/O instrument, GAN super-resolution
of borehole image logs, and CO₂ solubility in saline brine.

| Module | Topic | Reference |
| --- | --- | --- |
| `bennis_invasion_sw` | Radial water-saturation inversion in deeply-invaded tight-gas sandstone: tanh-transition Sw(r) profile between invaded and virgin zones, Archie forward model, multi-DOI apparent-resistivity volume averaging, and least-squares recovery of (r_invaded, Sw_invaded, Sw_virgin) | Bennis et al., pp. 931–953 |
| `bradley_wellbore_format` | Proposed universal wellbore data format: JSON-backed hierarchical container with metadata, units, named axes, and arbitrary-dimensional channels supporting both simple 1D logs (GR) and complex multidimensional measurements such as ultradeep azimuthal resistivity (depth × azimuth × DOI) | Bradley et al., pp. 823–836 |
| `cely_mudgas_viscosity` | Reservoir-oil viscosity estimation in the Breidablikk Field from advanced mud-gas data: Pixler/Haworth gas ratios (wetness, balance, character) from C1–nC5 fractions, plus a multivariate linear regressor for log10(viscosity) calibrated against PVT measurements | Cely et al., pp. 919–930 |
| `garcia_nmr_gaussian` | 2D NMR fluid-component tracking via Gaussian decomposition: synthetic 2D map generator, multi-component 2D Gaussian least-squares fit on a (T1, T2)-style grid, and analytic per-component pore volume from the Gaussian integral 2π·A·σx·σy | Garcia et al., pp. 879–889 |
| `khan_salt_creep` | Nonlinear creep-damage model for solution-mined salt caverns used for H₂/CO₂ storage: Norton power-law steady-state creep ε̇ = A·σⁿ coupled to a Kachanov damage variable D with effective stress σ/(1−D), time-marched to predict cavern strain, damage, and fractional volumetric closure | Khan et al., pp. 954–969 |
| `mcglynn_pulsed_neutron` | Pulsed-neutron spectroscopy forward + inverse model: simultaneous inelastic C/O ratio, capture sigma (c.u.), and gas ratio response for three-phase saturation, with a constrained least-squares solver recovering (S_oil, S_gas, S_water) under the Σ S = 1 closure | McGlynn et al., pp. 900–918 |
| `trevizan_gan_image_log` | Generative adversarial network super-resolution for real-time borehole image logs: tiny PyTorch generator (Conv-ReLU-Upsample) and discriminator with a BCE + L1 training step, plus a NumPy bilinear-upsampling fallback when torch is unavailable | Trevizan & Menezes de Jesus, pp. 890–899 |
| `wang_co2_solubility` | CO₂ solubility in saline brine for CCS trapping: Henry's-law constant H(T), Setschenow salting-out activity coefficient γ(m_NaCl, T), CH₄-competition correction, and a reservoir-scale dissolved-CO₂ trapping capacity (kg CO₂ per m³ rock) from porosity, water saturation, and brine density | Wang & Ehlig-Economides, pp. 970–977 |

DOI pattern: `10.30632/PJV64N6-2023aNN`

---

## src2024_02 — Vol. 65, No. 1 (February 2024)

Mixed-topic issue covering shaly-sand conductivity theory, formation-tester
fluid sampling, CO₂ storage, regression methodology, thermally-cycled granite
permeability, and two machine-learning contributions (a contest summary and
a DTW-based analog approach for rock mechanics).

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_waxman_smits_dual_water` | Shaly-sand conductivity: Waxman-Smits Co = φ^m*·Sw^n*·(Cw + B·Qv/Sw) with Waxman-Thomas temperature/salinity-dependent counter-ion conductance B(Cw,T), and Dual Water Co = φ^m·Sw^n·[(1−Swb/Sw)·Cw + (Swb/Sw)·Cwb]; Archie reduction at Qv = 0 as a built-in cross-check | Rasmus, Kennedy & Homan, pp. 5–31 |
| `article2_contamination_transient` | Formation-tester cleanup transient analysis: power-law contamination decay η(V) = η∞ + A·V^(−b) fitted with non-linear least squares to (volume, contamination) pairs, and analytical inversion to predict the pumped volume required to reach a target contamination threshold (e.g., 5 % OBM filtrate) | Gelvez & Torres-Verdín, pp. 32–50 |
| `article3_co2_storage` | Volumetric CO₂ storage capacity for saline aquifers: M = A·h·φ·(1−Sw,irr)·ρ_CO2·E (DOE/USGS method), plus a four-way trapping partition (structural, residual, dissolution, mineral) with user-supplied fractions and a sensitivity check that capacity scales linearly with storage efficiency | Kumar & Lauderdale-Smith, pp. 51–69 |
| `article4_least_squares` | OLS vs. reverse OLS vs. reduced major axis (RMA / geometric-mean) regression for petrophysical crossplots: synthetic-error demonstration of OLS slope attenuation toward zero when the predictor is noisy, and the bracketing property OLS ≤ RMA ≤ reverse-OLS | Etnyre, pp. 70–94 |
| `article5_granite_thermal` | Permeability of granite under thermal cycling: empirical model k(T,N) = k₀·exp(α·(T−T₀))·(1 + β·ln(1+N)) capturing microcrack-driven permeability growth with both peak temperature T and cycle count N, plus a linearized least-squares fit recovering (k₀, α) from laboratory data | Yu, Li, Wu, Wang, Zhang & Zhao, pp. 95–107 |
| `article6_ml_contest` | SPWLA PDDA 2023 contest baseline: gradient-boosted regression (with closed-form ridge fallback if scikit-learn is absent) trained on standard well logs (GR, RHOB, NPHI, DT, log RT) to predict porosity and water saturation, scored with the contest's RMSE metric on a held-out tail of a synthetic well | Fu, Yu, Xu, Ashby, McDonald, Pan, Deng, Szabó, Hanzelik, Kalmár, Alatwah & Lee, pp. 108–127 |
| `article7_dtw_rockmech` | Analog-well rock mechanics prediction: dynamic time warping (DTW) distance between target and library log curves, k-nearest-analog selection, and inverse-distance-weighted regression of a target property (e.g., UCS, Young's modulus) from the matched analogs | Cai, Ding, Li, Yin & Feng, pp. 128+ |

DOI pattern: `10.30632/PJV65N1-2024aNN` (NN = 1 … 7)

---

## src2024_04 — Vol. 65, No. 2 (April 2024)

Mixed-topic issue covering machine learning, core analysis, formation evaluation, reservoir characterization, and integration. Digital-rock relative permeability for chalk, microscopic ionic capacitor models, NMR core analysis procedures, quantitative productivity-controlling factor evaluation for ultradeep gas wells, ML prediction of triple-combo logs from drilling dynamics with physics-based joint inversion, and deep-learning semantic segmentation of shale SEM pore images.

| Module | Topic | Reference |
| --- | --- | --- |
| `grader_digital_rock` | Digital-rock relative permeability for high-porosity / low-permeability Valhall chalk: Brooks-Corey two-phase relperms (krw, kro) with Corey exponents, endpoint saturations (Swi, Sor) derived from a digital pore-size distribution (smallest pores → irreducible water, largest pores → residual oil), wettability switching (water-wet vs oil-wet), Buckley-Leverett fractional flow fw(Sw) | Grader et al., pp. 149–157 |
| `liu_ionic_capacitor` | Three microscopic ionic capacitor models for petrophysics: (I) intergranular pore parallel-plate capacitor C = ε·A/d, (II) particle-with-isolated-pore spherical capacitor C = 4πε·rR/(R−r), (III) pyrite/graphite/organic conductive-particle capacitor with charge-multiplication factor; time-varying double-layer charge q(t) = CV₀(1−e^(−t/τ)), salinity-dependent effective capacitance | Liu et al., pp. 158–172 |
| `zhang_nmr_core` | NMR core analysis procedures: synthetic CPMG echo-train forward model S(t) = ΣAᵢ·exp(−t/T2ᵢ), Tikhonov-regularised non-negative least-squares (NNLS) T2 inversion on a log-spaced grid, bound/free-fluid partitioning by T2 cutoff (default 33 ms sandstone), surface-relaxivity pore-radius conversion r = G·ρ₂·T2, simple D-T2 (diffusion-relaxation) correlation map for fluid identification | Zhang, Song, Luo, Lin & Liu, pp. 173–193, DOI 10.30632/PJV65N2-2024a3 |
| `xiong_productivity_factors` | Quantitative evaluation of high-productivity controlling factors for ultradeep gas wells (Qixia Formation): min-max normalization of geological/petrophysical indicators (degree of dolomitization, high-energy shoal-mound complex distribution, fracture development, porosity, permeability), grey relational analysis (GRA) grades against productivity reference series, AHP eigenvector-method weights from a pairwise comparison matrix, composite weighted productivity score per well, factor ranking | Xiong et al., pp. 194–214 |
| `lee_mwd_triple_combo` | Two-stage MWD workflow: (1) Random Forest regression mapping drilling dynamics (WOB, RPM, ROP, torque, mechanical specific energy MSE) to triple-combo logs (gamma ray, bulk density, neutron porosity, deep resistivity); (2) physics-based joint inversion for density porosity φd = (ρma−ρb)/(ρma−ρf), Vsh from linear gamma-ray, shale-corrected average porosity, and Archie water saturation Sw = (Rw/(φᵐ·Rt))^(1/n) | Lee et al., pp. 215–232 |
| `chen_sem_pore_segmentation` | Deep-learning "pore-net" semantic segmentation of shale SEM images: synthetic SEM image generator with random circular pores and Gaussian noise, lightweight thresholding-plus-morphological-opening/closing baseline segmenter, optional small U-Net architecture (PyTorch, two encoder/decoder stages with skip connections), porosity from pixel fraction, pore-size distribution via connected-component labelling, IoU evaluation against ground truth | Chen et al., pp. 233–245 |

DOI pattern: `10.30632/PJV65N2-2024aNN` (NN = 1 … 6)

---

## src2024_06 — Vol. 65, No. 3 (June 2024)

Special Issue on Petrophysics for the Energy Transition and Fundamental Rock Physics. Nuclear Logging for CCS and Low-Carbon Applications, Claystone Nuclear Repository Characterisation, Underground Hydrogen Storage, Automatic Facies Analysis in the Crust-Mantle Transition Zone, Deep-Learning LWD Image Interpretation, 2D T1–T2 NMR Source-Rock Saturation, Shale Hole-Fracture Damage Mechanics, and Joint R35 / Fractal MICP Rock Typing.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_nuclear_logging_ccs` | Nuclear logging for CCS, nuclear repositories, and geothermal systems: pulsed-neutron capture (PNC) Sigma from thermal-neutron time decay N(t) = N₀·exp(−Σvt) (Eq. 1), gas-phase diffusion correction Σ_D (Appendix 1), carbon/oxygen (C/O) ratio CO₂ vs hydrocarbon discrimination, capture-unit (c.u.) conversion, plume tracking | Badruzzaman, pp. 274–301 |
| `article2_claystone_repository` | Petrophysical analyses for claystone-hosted nuclear waste repository search (BGE, Germany) from legacy oilfield logs: vertical variogram analysis of gamma-ray for layer-thickness detection (Fig. 4), Lag1 enhanced variance with P10 threshold, short/long median-filter residual GR curve, Archie-type effective-diffusivity model for clay porosity and tortuosity | Strobel, pp. 302–316 |
| `article3_hydrogen_storage` | Underground hydrogen storage (UHS) in porous media: Newman (1973) rock-compressibility correlation for consolidated sandstone, gas inflow performance relationship (IPR) for H₂ withdrawal, average cycle productivity index, Mohr-Coulomb / Griffith failure envelope for induced-seismicity risk on critically stressed faults (Fig. 3), six-cycle injection-withdrawal scheduler (Fig. 1) | Okoroafor, Sekar & Galvis, pp. 317–341 |
| `article4_facies_classification` | Automatic facies analysis in the crust-mantle transition zone (Oman Drilling Project CM2A / CM2B, dunite / gabbro / harzburgite): FaciesSpect (PCA + hierarchical agglomerative clustering), CBML (PCA + Gaussian mixture model + HMM depth regulariser), HRA (K-means on log attributes), borehole-image per-depth statistics (mean, contrast) as features | Morelli, Yang, Maehara, Cai, Moe, Yamada & Matter, pp. 342–363 |
| `article5_lwd_image_deeplearning` | Deep-learning LWD azimuthal density image interpretation: U-Net "PickNet" edge segmentation on 20×16 images, fully-connected "FitNet" sinusoid fitter for amplitude / phase / mean depth, synthetic image generator per Appendix 1 (random sinusoidal density contrasts + Gaussian noise), deterministic gradient-based edge picker and least-squares sinusoid fit as CPU analogues | Molossi, Roncoroni & Pipan, pp. 365–387 |
| `article6_nmr_t1t2_saturation` | 2D T1–T2 NMR oil and water saturation in preserved source rocks: inversion-recovery CPMG forward model S(t1,t2) = Σ Mᵢ(1−2e^(−t1/T1ᵢ))e^(−t2/T2ᵢ), Tikhonov-regularised non-negative least-squares 2D inversion on log-spaced (T1,T2) grid (MUPen2D analogue), user-defined oil/water region integration, fluid-filled porosity conversion (Eq. 2) | Althaus, Chen, Sun & Broyles, pp. 388–396 |
| `article7_shale_fracture_damage` | Damage and failure of prefabricated hole-fracture defects in shale under uniaxial compression with DIC: Inglis (1913) elliptical fracture-tip stress σ_tip = σ_applied·(1 + 2a/b) with angle projection, Kirsch (1898) circular-pore 3σ concentration factor, empirical relative peak-strength reduction vs. fracture-bedding angle, combined hole-plus-fracture interaction | Jiang, Qu & Liu, pp. 397–410 |
| `article8_r35_fractal_rock_typing` | Joint R35 / fractal MICP rock typing (Middle East Iraq carbonates): Washburn equation r = 2σ·|cosθ|/P with σ = 480 dyn/cm, θ = 140° (Eq. 1), Winland/Pittman R35 pore-throat radius at 35 % mercury saturation with 1.6 / 2.5 µm thresholds, whole-curve fractal dimension Dₙ from log-log N_r vs r slope (Eq. 7, N_r ~ r^(−Dₙ)), three-class rock typing | Duan, Zhong, Fu, Xu, Deng, Ling & Li, pp. 411–424 |

DOI pattern: `10.30632/PJV65N3-2024aNN` (NN = 1 … 8)

---

## src2024_08 — Vol. 65, No. 4 (August 2024)

Special Issue on Advancements in Mud Logging. ML-Based GOR and Fluid-Property Prediction from Advanced and Standard Mud Gas, Real-Time Fluid Identification, Heavy-Oil Viscosity Mapping, PVT Comparison and GOR Prediction, New Gas Logging Instrumentation, Mud Gas Quantification, Drill-Bit Metamorphism Detection, GPC-UV Cuttings Analysis, Magnetic-Susceptibility Permeability, and Automated Lithology from Cuttings Images.

| Module | Topic | Reference |
| --- | --- | --- |
| `gor_prediction_ml` | ML GOR prediction from advanced mud gas (AMG) C1–C5 compositions: Random Forest, MLP, Gaussian Process Regression trained on PVT database, QC metrics (Wetness Wh, Balance Bh, Character Ch), log10(GOR) modelling, 5-fold cross-validation, MAPE evaluation (≈35 %) | Arief & Yang, pp. 433–454 |
| `shale_fluid_prediction` | AMG-based fluid property prediction in shale (unconventional) reservoirs: extraction efficiency correction (EEC) for C1–C5, moving-average smoothing, continuous GOR log generation, minimum total-gas QC threshold, horizontal-well fluid heterogeneity for hydraulic-fracturing optimization | Yang, Arief, Niemann & Houbiers, pp. 455–469 |
| `realtime_fluid_id` | Real-time fluid identification integrating AMG with LWD petrophysical logs: radar (star) plot similarity matching against PVT database, Random Forest for GOR, AdaBoost for fluid density, density-neutron gas flagging, six-class fluid-type classification (black oil → dry gas) | Kopal, Yerkinkyzy, Nygård, Cely, Ungar, Donnadieu & Yang, pp. 470–483 |
| `standard_mudgas_typing` | Standard mud gas fluid typing using C1/C2, C1/C3, and Bernard ratio thresholds: Type I / Type II field classification via Fisher discriminant, pseudo-EEC correction for OBM wells (background subtraction + scale factors), threshold calibration from PVT database | Yang, Uleberg, Cely, Yerkinkyzy, Donnadieu & Kristiansen, pp. 484–495 |
| `ml_fluid_typing` | ML-based oil/gas classification from standard mud gas: Random Forest classifier, 8-feature engineering (C1/C2, C1/C3, C2/C3, Bernard, wetness, normalized C1–C3), three-approach feature selection (forward / backward / manual), AUC and accuracy metrics, hyperparameter tuning | Cely, Siedlecki, Ng, Liashenko, Donnadieu & Yang, pp. 496–506 |
| `heavy_oil_viscosity` | Heavy-oil viscosity mapping from standard mud gas (Peregrino Field): C1/C2 ratio-based viscosity calibration palette from reference wells, log-linear interpolation, 5 % tolerance QC band, three-class viscosity classification, pressure-gradient density estimation | Bravo, Cely, Yerkinkyzy, Xavier, Masuti, de Souza, Donnadieu & Yang, pp. 507–518 |
| `prospect_fluid_estimation` | Prospect evaluation fluid estimation from standard mud gas: triangle and diamond composition plots for C1–C3, C2/C3–GOR linear correlation (R² ≈ 0.79), continuous GOR log prediction, compositional gradient detection across reservoir zones | Ungar, Yerkinkyzy, Bravo & Yang, pp. 519–531 |
| `pvt_gor_snorre` | PVT comparison and GOR prediction in Snorre Field: dynamic extraction efficiency correction (EEC) from ROP / mud weight / total gas, dual ML dataset approach (NCS-wide + field-specific RF), star diagram ratio comparison, injection-gas identification (GOR > 10 000), production GOR validation (< 30 % error) | Caldas, Kirkman, Ungar & Yang, pp. 532–547 |
| `membrane_gas_logging` | Semipermeable-membrane degasser with NDIR infrared spectroscopy: Beer-Lambert law, multi-component least-squares spectral inversion, Fick's-law membrane permeability model for C1–C5, extraction efficiency correction, Gaussian absorption profiles at alkane central wavelengths (3.31–3.42 µm) | Cheng, Ye, Wang, Yin, Chen, Huang, Yang & Wang, pp. 548–564 |
| `mudgas_response` | Mud gas response variation causes and two quantification techniques: gas-marker method (SCF/ton from ROP, bit area, flow rate, trap efficiency), normalization technique to reference drilling conditions, ROP / flow-rate / mud-weight sensitivity analysis, production correlation in coal-gas reservoirs | Donovan, pp. 565–584 |
| `alkene_hydrogen_dbm` | Drill-bit metamorphism (DBM) detection from real-time alkene and hydrogen: C2=/C2 (ethylene/ethane) ratio alarm, H2 co-indicator, four-level severity classification (none / mild / moderate / severe), WOB correlation (R ≈ 0.91), POOH decision-support recommendations | Qubaisi, Kharaba, Hewitt & Sanclemente, pp. 585–592 |
| `gpc_uv_cuttings` | GPC-UV method for reservoir fluid analysis from drill cuttings: gel permeation chromatography simulation, 3-D isoabsorbance envelope (retention time × wavelength × intensity), feature extraction (peak RT, signal strength, area, wavelength span), API gravity / GOR estimation, OBM contamination assessment | Yang, Cely, Moore & Michael, pp. 593–603 |
| `magnetic_permeability` | Magnetic-susceptibility-derived permeability from drill cuttings (Culzean Triassic): high-field paramagnetic clay volume estimation (Eqs. 3–4, illite k = 41 × 10⁻⁵ SI, quartz k = −1.5 × 10⁻⁵ SI), ferromagnetic contaminant removal, overburden correction, Gaussian averaging for core-scale reconciliation (R² = 0.949), XRD validation (R² = 0.909) | Banks, Tugwell & Potter, pp. 604–623 |
| `lithobia_cuttings` | LiOBIA: object-based cuttings image analysis for automated lithology: instance segmentation, color (RGB mean/std) and texture (contrast, homogeneity, entropy) feature extraction, k-NN classification in feature space, PCA manifold analysis, five-lithology library (sandstone / limestone / shale / siltstone / dolomite), depth-log generation via majority vote (> 90 % accuracy) | Yamada, Di Santo, Bondabou, Prashant, Di Daniel, Su, Francois, Ouaaba, Lockyer & Prioul, pp. 624–648 |

DOI pattern: `10.30632/PJV65N4-2024aNN` (NN = 1 … 14)

---

## src2024_10 — Vol. 65, No. 5 (October 2024)

Probe Permeameter Calibration and Application, Core-Analysis Saturation Correction, MRI-Based Relative Permeability, Digital Rock Permeability Anisotropy, Shaly-Sand Water Saturation Equations, NMR Thin-Bed and Lateral Permeability Characterisation, Machine-Learning Permeability and Lithofacies Prediction, and Core-Log Depth Matching.

| Module | Topic | Reference |
| --- | --- | --- |
| `probe_permeameter` | Probe permeameter testing: geometric factor, depth of investigation, o-ring / silicone-rubber tip calibration, surface impairment correction, grain-size–permeability relationship, CO₂ injectivity and trapping assessment | Jensen & Uroza, pp. 665–681 |
| `dean_stark_saturation` | Reconstructing in-situ saturation from Dean-Stark lab measurements: pore-volume expansion (PVE) correction, clay dehydration correction, degasification correction (logarithmic water / linear oil models), kw–bw linear constraint, coefficient estimation, normalisation to 100 % | Zhang, Xu, Lu, Qi & Lia, pp. 682–698 |
| `relative_permeability_mri` | Model-free unsteady-state relative permeability from MRI saturation profiles: capillary dispersion coefficient, fractional mobility, Corey-type Kr comparison, capillary pressure model (Eq. 13), synthetic saturation-profile generation | Zamiri, Afrough, Marica, Romero-Zerón, Nicot & Balcom, pp. 699–710 |
| `permeability_anisotropy` | Permeability anisotropy in presalt carbonates via digital rock petrophysics: reservoir quality index (RQI), flow zone indicator (FZI), hydraulic flow unit (HFU) classification, arithmetic / harmonic / geometric upscaling, Kv/Kh ratio at multiple vertical windows, facies-based statistics | Silva Junior, Victor, Surmas, Barroso & Perosi, pp. 711–738 |
| `water_saturation_equations` | Water saturation equations for unconsolidated reservoirs: Archie, Indonesian, Modified Indonesian (Woodhouse), Simandoux, Waxman-Smits, Dual Water, Suriname Clay (Eq. 8), Suriname Clay-and-Silt (Eq. 9), Suriname Laminar Clay-and-Silt (Eq. 10), BPPI heterogeneity index (Eq. 7), Swirr from NMR correlation (Eq. 11) | Acosta, Mijland & Nandlal, pp. 739–764 |
| `thin_bed_nmr` | Thin-bed NMR response in horizontal wells: LWD NMR sensitivity kernel, apparent porosity via convolution, shoulder-bed averaging, thin-bed correction factor, tool stand-off correction, bed-boundary detection | Ramadan, Allen & Allam, pp. 765–771 |
| `lateral_permeability_nmr` | Lateral permeability variations in heterogeneous carbonates: Timur-Coates NMR permeability, SDR NMR permeability, azimuthal permeability from oriented formation tests, micro-resistivity heterogeneity index, lateral (azimuthal) permeability profile construction | Fouda, Taher, Fateh & Kumar, pp. 772–788 |
| `ml_permeability` | ML vs conventional permeability estimation: Timur-Coates model (Eqs. 15–16), feature engineering (moving-window statistics), PCA / SVD / DWT / autoencoder dimensionality reduction (Eqs. 2–8), Random Forest, SVR, kNN, Ridge, Lasso, ANN, Archie Sw (Eqs. 17–20), MAE / RSE metrics (Eqs. 14, 21), group k-fold cross-validation | Raheem, Pan, Morales & Torres-Verdín, pp. 789–812 |
| `lithofacies_prediction` | High-resolution lithofacies prediction: petrophysical cutoff-based facies definition (gas sand / wet sand / shale), feature engineering from GR, LLD, RHOB, Extra Trees (ET) classifier, XGBoost (XGB) classifier, confusion matrix, F1-score evaluation, k-fold and random-subsampling cross-validation | Satti, Khan, Mahmood, Manzoor, Hussain & Malik, pp. 813–834 |
| `rddtw_depth_matching` | Core-log depth adaptive matching using RDDTW: standard DTW, constrained DTW (Sakoe-Chiba band), derivative DTW, Regularised Derivative DTW with Excessive Warping Regularised Function (EWRF), PCC baseline, Particle Swarm Optimisation (PSO) for depth-shift estimation, R² / RMSE evaluation | Fang, Zhou, Xiao & Liao, pp. 835–851 |

DOI pattern: `10.30632/PJV65N5-2024aNN` (NN = 1 … 10)

---

## src2024_12 — Vol. 65, No. 6 (December 2024)

Best Papers of the 2024 SPWLA Annual Symposium, Rio de Janeiro. Image-Based AI Applications, Well Integrity, New Technologies (sourceless density, tracer sampling, GPC fluid analysis), and Fundamental Studies (permeability, wettability, fracability, perched water).

| Module | Topic | Reference |
| --- | --- | --- |
| `m01_image_rock_properties` | Thin-section image AI for analog petrophysical properties from drill cuttings: texture-feature extraction, cosine-similarity database matching, porosity / permeability / Archie-m prediction, cutting-size sensitivity (clastic ≈ 85 %, carbonate ≈ 38 % match rate) | Britton, Cox & Ma, pp. 866–874 |
| `m02_dip_picking` | AI-driven automatic dip picking in horizontal wells: CNN zone classification (no-bedding / sinusoidal / non-sinusoidal), Hough-transform sinusoid fitting, DBSCAN clustering of partial dips, path-based non-sinusoidal merging, real-time block continuity | Perrier, He, Bize-Forest & Quesada, pp. 875–886 |
| `m03_synthetic_borehole_images` | Synthetic borehole images from outcrop photographs: strip cutting at well diameter, mirror-symmetry 3-D extrusion, cylindrical intersection and unwrapping, standard BHI colour palette, azimuthal rotation for field alignment | Fornero, Menezes de Jesus, Fernandes & Trevizan, pp. 887–894 |
| `m04_well_integrity_ccs` | Well integrity throughout the CCS project life cycle: cement bond index (CBL), ultrasonic acoustic-impedance quality scoring, casing corrosion assessment, CO₂-resistant / epoxy-resin material impact, risk scoring, phase-specific measurement strategy | Valstar, Nettleton, Borchardt, Costeno, Landry & Laronga, pp. 896–912 |
| `m05_casing_cement_inspection` | Logging two casing sizes simultaneously: pulse-echo resonance-frequency thickness estimation, dual-string corrosion evaluation, cement plug acoustic-impedance verification | Hawthorn, Ingebretson, Girneata, Delabroy, Winther, Steinsiek & Leslie, pp. 913–918 |
| `m06_noise_logging` | Advanced noise logging (ANL) from leak detection to quantitative flow profiling: noise power amplitude in frequency bands, broadband leak detection, borehole / reservoir flow separation via frequency cutoff (4 kHz), relative flow-rate allocation | Galli & Pirrone, pp. 919–927 |
| `m07_sourceless_density` | Sourceless neutron-gamma density (sNGD): inelastic / capture gamma-ray separation via time gating, hydrogen-index-based neutron-transport correction, spine-relation density computation, environmental corrections (hole size, mud weight, salinity, standoff) | Mauborgne et al., pp. 929–943 |
| `m08_tracer_aquifer_sampling` | Low-toxicity D₂O tracer for CCS aquifer sampling: contamination calculation from deuterium concentrations, salinity correction, density-porosity, Rwa-based salinity estimation (Archie Sw = 1), pressure-gradient fluid-density estimation | Taplin, Peyret, Jackson & Hitchen, pp. 944–956 |
| `m09_gpc_fluid_properties` | GPC-UV-RI spectra + machine learning for API gravity from cuttings: synthetic 3-D tensor generation (elution time × wavelength × intensity), LASSO regression, Monte Carlo data augmentation, dilution-effect correction for cutting extracts | Cely, Yang, Yerkinkyzy, Michael & Moore, pp. 957–969 |
| `m10_permeability_prediction` | Physics-based probabilistic permeability in thin-layered reservoirs: dielectric dispersion log (DDL) spectral-representation inversion (Stroud et al. ansatz), Bayesian core-to-log grain-size / CEC correlations, transport-theory permeability (Revil & Cathles), Monte Carlo uncertainty | Pirrone, Bona & Galli, pp. 971–982 |
| `m11_wettability_adsorption` | Wettability quantification via water adsorption isotherms: BET isotherm model, monolayer-ratio wettability index, contact-angle correlation, work of adhesion (Schlangen et al.), mineral-mixture linear-mixing model | Silveira de Araujo & Heidari, pp. 983–994 |
| `m12_fracability_evaluation` | Fracability evaluation for tight sandstone reservoirs: dynamic-to-static mechanical conversion (Eqs. 1–3), mineral + acoustic-modulus + comprehensive brittleness (Eq. 8), fracture generation / vertical expansion / azimuth / network complexity analysis, horizontal stress difference coefficient Kₕ (Eq. 12) | Qian, Wang & Xie, pp. 995–1009 |
| `m13_perched_water` | Perched water detection in deepwater Miocene fields: drainage capillary-pressure Sw profile, Archie resistivity Sw, perched-water flagging by Sw comparison, transition-zone estimation, volumetric impact, water-chemistry origin classification | Kostin & Sanchez-Ramirez, pp. 1010–1022 |

DOI pattern: `10.30632/PJV65N6-2024aNN` (NN = 1 … 13)

---

## src2025_02 — Vol. 66, No. 1 (February 2025)

Best Papers of the 2023 SCA International Symposium. Underground Carbon Capture, Storage, and EOR; Pore-Scale Imaging and Modeling; New SCAL Techniques and Interpretation.

| Module | Topic | Reference |
| --- | --- | --- |
| `scal_model_ccs` | LET relative-permeability and capillary-pressure correlations, Leverett J-scaling, Land trapping, CO₂ storage capacity, base / optimistic / pessimistic SCAL model for CCS | Ebeltoft et al., pp. 10–25 |
| `co2_brine_relperm` | Corey model, Buckley-Leverett fractional flow, SS analytical kr, capillary end-effect correction, JBN USS interpretation, SS + USS reconciliation | Mascle et al., pp. 26–43 |
| `ss_co2_brine_relperm` | Steady-state scCO₂-brine kr at two pore pressures, pressure-effect comparison, drainage / imbibition hysteresis, material balance, wettability indicator | Richardson et al., pp. 44–53 |
| `enhanced_gas_recovery` | Land trapping for CH₄ vs CO₂ (partial-wetting detection), Burdine Pc, LET kr, EGR displacement efficiency, ISSM saturation, gravity-stable flood criterion | Jones et al., pp. 54–66 |
| `rev_two_phase_flow` | Energy-dissipation-based relative permeability (Eqs. 7–11), temporal REV convergence analysis, ergodicity test, fluctuation analysis, SCAL duration guide | McClure et al., pp. 68–79 |
| `digital_rock_physics` | Pore-network generation, mixed-wet contact-angle anchoring, invasion-percolation drainage kr, ESRGAN resolution metrics, DRP vs SCAL comparison | Regaieg et al., pp. 80–92 |
| `hybrid_drainage` | Hybrid Drainage Technique (viscous flood + capillary steps) vs viscous oilflood on bimodal limestone, NMR T₂ bimodal distribution, profile homogeneity | Fernandes et al., pp. 94–109 |
| `pore_scale_drainage` | Porous-plate vs oilflood invasion, micro / meso / macro pore classification, pore-occupancy analysis, effective permeability, wettability artifacts | Nono et al., pp. 110–122 |
| `dopant_impact_scal` | X-ray attenuation contrast with NaI (≈7× improvement), Amott wettability index, doped vs undoped oil recovery, spontaneous imbibition rate, Sor impact | Pairoys et al., pp. 123–133 |
| `dual_porosity_sandstone` | Dual Brooks-Corey Pc, imbibition Pc from drainage Pc (contact-angle correction), Land trapped-oil, NMR Gaussian deconvolution, dual-porosity Corey kr | Wang & Galley, pp. 134–154 |
| `mr_bulk_saturation` | CPMG multi-exponential decay, ¹³C oil volume, ¹H + ¹³C water volume, ²³Na water volume, saturation workflow, Dean-Stark validation | Ansaribaranghar et al., pp. 155–168 |
| `mr_saturation_imaging` | ¹³C 1-D SE-SPI oil profiling, ¹H total-fluid profiling, water-by-subtraction, capillary end-effect detection, oil-wet CEE profiles, Dean-Stark validation (< 1 s.u.) | Ansaribaranghar et al., pp. 169–182 |

DOI pattern: `10.30632/PJV66N1-2025aNN` (NN = 1 … 12)

---

## src2025_04 — Vol. 66, No. 2 (April 2025)

UDAR / LWD Technologies, Reservoir Porosity and Pore Characterization, Overpressure Analysis, Neutron Porosity Logging, and Well Integrity / Cementing.

| Module | Topic | Reference |
| --- | --- | --- |
| `udar_look_ahead` | UDAR look-ahead-while-drilling: antenna tilt calibration, SNR estimation, model distribution analysis, multi-frequency signal combination for depth-of-detection | Cuadros et al., pp. 190–211 |
| `stochastic_inversion` | High-performance stochastic inversion for UDAR data: reversible-jump MCMC (RJMCMC), MALA proposals, parallel tempering, 1-D layer-cake Bayesian uncertainty | Sviridov et al., pp. 212–236 |
| `gip_porosity` | Improved GIP method for shale effective porosity: pressure-decay model, curve fitting for equilibrium pressure, rapid porosity without full equilibrium | Jiang et al., pp. 237–249 |
| `unconventional_porosity` | Total porosity and fluid saturations for tight rocks: CRA/GRI, retort, NMR T₂ distribution, comparison framework and volumetric modelling | Cheng et al., pp. 250–266 |
| `ultrasonic_pore_characterization` | Ultrasonic microscopy imaging of carbonate pore structure: acoustic impedance, Otsu thresholding, shape descriptors, Fourier descriptors, 3-D pore reconstruction | Chen et al., pp. 267–282 |
| `overpressure_isotope` | Overpressure genetic analysis via isotope logging: Eaton/Bowers pore pressure, NCT estimation, loading/unloading classification, δ¹³C methane diagnosis | Hu et al., pp. 283–293 |
| `neutron_porosity_sensitivity` | Neutron porosity sensitivity functions in casedhole: FSF (weight window), ISF (particle tracking), FSF↔ISF relationship, fast-forward modelling for porosity | Varignier et al., pp. 294–317 |
| `filter_cake_isolation` | Drilling fluid filter cake effect on cement zonal isolation: DFFC layer classification, second-interface shear strength & channelling pressure, curing time effects | Yang et al., pp. 318–330 |
| `microannuli_leak_rate` | Ultrasonic log analysis and microannuli leak rate quantification: impedance-to-thickness mapping, Hagen-Poiseuille flow (liquid & gas), bond index, sensitivity analysis | Machicote et al., pp. 331–347 |

DOI pattern: `10.30632/PJV66N2-2025aNN` (NN = 2 … 10)

---

## src2025_06 — Vol. 66, No. 3 (June 2025)

New Technology, Thomas-Stieber-Based Shaly-Sand Petrophysics, Basic Petrophysics Studies, and Rock Mechanics / Geomechanics.

| Module | Topic | Reference |
| --- | --- | --- |
| `core_scanner` | EM core scanner: CRIM-based resistivity / dielectric permittivity inversion and water-filled porosity at 3.8 GHz | Mirza et al., pp. 352–363 |
| `thomas_stieber_tyurin` | Thomas-Stieber-Tyurin (T-S-T) clay-volume-based thin-bed model with dispersed / structural clay and uncertainty analysis | Tyurin & Davenport, pp. 365–391 |
| `thomas_stieber_welllog` | Fit-for-purpose T-S diagram in the well-log domain (nuclear-log forward models, multi-class rock typing) | Eghbali & Torres-Verdín, pp. 392–423 |
| `toc_prediction` | TOC prediction: ΔlogR, dual-shale-content, stacking ensemble ML, sliding-window core homing, Cook's distance outlier removal | Dong et al., pp. 425–448 |
| `cross_calibrated_permeability` | Coates / Timur cross-calibrated permeability, SwXCal correlation, pore-throat classification (nano–mega) | Sifontes et al., pp. 449–466 |
| `shale_microparams` | PFC2D shale micro-parameter calibration via stacking ensemble (PBM + SJM), orthogonal design, sensitivity analysis | Jiang et al., pp. 468–488 |
| `fracturing_fluid_damage` | Fracturing-fluid damage assessment: NMR T₂ analysis, hydrolock damage, fracture conductivity, production comparison | Li et al., pp. 489–520 |
| `injection_fluid_optimization` | Injection-fluid optimization for tight-oil energy storage: imbibition modelling, shut-in time optimization, fluid ranking | Xiao et al., pp. 521–535 |

DOI pattern: `10.30632/PJV66N3-2025aNN` (NN = 1 … 8)

---

## src2025_08 — Vol. 66, No. 4 (August 2025)

Special Issue on Well Integrity — General, Corrosion Evaluation, Defect Detection, and Cement / Formation Evaluation Behind Casing.

| Module | Topic | Reference |
| --- | --- | --- |
| `pa_genai_extraction` | GenAI-based P&A data extraction: simulated OCR, semantic text chunking, TF-vector search (RAG pipeline), rule-based hole/casing/cement extraction, QC checks | Kolay et al., pp. 545–554 |
| `fiber_optics_sensing` | Distributed fiber-optic sensing: DTS temperature-anomaly leak detection, DAS waterfall acoustic-event detection, temporal stacking for SNR improvement, diagnostic-time comparison (≈85 % reduction) | Bazaid et al., pp. 555–565 |
| `seven_pipe_em_corrosion` | Multi-frequency EM eddy-current pipe inspection: forward model for up to 7 concentric pipes, cost function (magnitude + phase misfit + regularisation), gradient-descent inversion with backtracking line search, metal-loss estimation | Fouda et al., pp. 566–577 |
| `sectorial_em_scanning` | Sectorial EM scanning tool: azimuthal pipe-wall-thickness model with localised defects and ovalization, per-sector EM response, defect classification (localised / uniform / deformation / nominal), averaging-EM comparison | Jawed et al., pp. 578–593 |
| `fbe_cement_evaluation` | Cement bond evaluation for FBE-coated casings: ultrasonic pulse-echo waveform through multi-layer media, flexural-wave resonance impedance estimation, azimuthal scan with free-pipe / cemented differentiation | Bazaid et al., pp. 594–615 |
| `acoustic_imaging` | High-resolution acoustic imaging (512-sensor array): synthetic casing-surface generation, time-of-flight and amplitude imaging, 3-D point-cloud generation, flood-fill defect detection and classification (pit / corrosion / perforation) | Alatigue et al., pp. 616–630 |
| `pulsed_eddy_current` | Pulsed eddy-current (PEC) casing-break detection: time-transient signal simulation for multi-pipe completions, VDL-style log generation, break detection from late-time channel analysis, time-lapse differencing, pipe-layer identification | Jawed et al., pp. 631–646 |
| `anomaly_detection_vmd` | Automated anomaly detection via signal mode decomposition: VMD, multivariate VMD (MVMD), hierarchical multiresolution VMD (HMVMD), feature extraction, Bayesian decision tree with Markov collar-spacing prior for collar / anomaly classification, SNR enhancement | Wang et al., pp. 647–661 |
| `koopman_enkf_deformation` | Through-tubing casing deformation inspection: state parameterisation (eccentricity ratio, direction, ovality), DMD-based Koopman transition model, simplified EM observation model, ensemble Kalman filter (EnKF) sequential estimation | Manh et al., pp. 662–676 |
| `cement_snhr_emi` | Through-tubing cement evaluation: selective non-harmonic resonance (SNHR) resonance-power-loss analysis, electromechanical impedance (EMI) admittance measurement, feedforward neural-network eccentricity correction, combined Bond Index | Zeghlache et al., pp. 677–688 |
| `wave_separation_slowness` | Formation slowness estimation behind casing: STC analysis, linear moveout (LMO) correction + stacking, preliminary casing-wave subtraction, time-variant (TV) correlation weighting for constrained separation, slowness spectrum projection | Sun et al., pp. 689–700 |

DOI pattern: `10.30632/PJV66N4-2025aNN` (NN = 1 … 11)

---

## src2025_10 — Vol. 66, No. 5 (October 2025)

Log Interpretation, Rock Mechanics, Machine-Learning Petrophysics, NMR, Digital Rock, Cementing Quality, and Neutron Logging.

| Module | Topic | Reference |
| --- | --- | --- |
| `a1_log_interpretation` | Kozeny permeability, Archie m-exponent from surface area, parallel conduction model, iso-frame elastic model, Gassmann substitution, Biot coefficient | Proestakis & Fabricius, pp. 705–727 |
| `a2_damage_model` | M-integral computation, local mechanical failure driving factor, initial / microscopic / total damage, Weibull-based damage constitutive model | Liu et al., pp. 728–740 |
| `a3_youngs_modulus` | Dynamic / static Young's modulus, Mullen lithology models, Steiber Vsh, FZI/DRT rock typing, nonlinear regression model, simple BPNN | Al-Dousari et al., pp. 741–762 |
| `a4_multimodal_permeability` | LSTM for time-series logs, 1-D CNN for NMR T₂ images, DNN for text features, explicit tensor interaction (binary planes + ternary core) | Fang et al., pp. 764–784 |
| `a5_missing_log_prediction` | 1-D U-Net encoder-decoder with skip connections, LSTM depth-trend module, hybrid fusion for missing-log prediction | Oppong et al., pp. 785–806 |
| `a6_carbonate_petrophysics` | Shale volume (linear / Larionov), density-neutron porosity, water saturation (Archie / Indonesian / Simandoux), Timur permeability, net-pay flagging | Fadhil, pp. 807–838 |
| `a7_nmr_porosity_correction` | Rock magnetic susceptibility from minerals, internal gradient field, NMR T₂ relaxation (bulk + surface + diffusion), T₂ spectrum correction, porosity correction model | Zhu et al., pp. 840–857 |
| `a8_digital_core_conductivity` | Archie's first / second laws with directional anisotropy, bimodal saturation exponent, wettability / salinity effects, 3-D digital core generation, resistivity simulation | Feng & Zou, pp. 858–871 |
| `a9_cementing_quality` | Slip interface boundary conditions, coupling stiffness matrix, relative amplitude vs. shear stiffness / USA, stiffness inversion from amplitude, cement quality classification | Pan et al., pp. 872–885 |
| `a10_neutron_log_shale` | Migration / slowing-down / diffusion lengths (SNUPAR-like), neutron porosity transforms (SS / LS / DOL), effective Lm*, nonlinear shale response modelling | Rasmus, pp. 887–893 |
| `a11_fracture_identification` | Synthetic borehole image generation with sinusoidal fractures, feature extraction (gradient, variance), threshold & CNN-based detectors, F1 score with depth tolerance | Lee et al., pp. 894–914 |

DOI pattern: `10.30632/PJV66N5-2025aNN` (NN = 1 … 11)

---

## src2025_12 — Vol. 66, No. 6 (December 2025)

Best Papers of the 2024 SCA International Symposium.

| Module | Topic | Reference |
| --- | --- | --- |
| `pgs_rock_typing` | PGS rock typing and Corey-parameter relative permeability trend modelling | Akbar et al., pp. 924–938 |
| `dl_permeability` | Deep-learning permeability inference from 3-D greyscale images | Youssef et al., pp. 939–955 |
| `primary_drainage` | Review and modelling of primary drainage techniques (centrifuge, porous-plate, viscous oil flood) | Fernandes et al., pp. 957–968 |
| `analog_kr` | CO₂/brine drainage relative permeability estimation from analog two-phase data | Schembre-McCabe et al., pp. 969–981 |
| `co2_uptake` | CO₂ uptake capacity in source-rock shales via NMR | Chen et al., pp. 982–994 |
| `drp_wettability` | Digital Rock Physics pore-scale wettability and relative permeability simulation | Faisal et al., pp. 996–1012 |
| `electrokinetic` | Electrokinetic (zeta-potential / streaming-potential) wettability assessment | Halisch et al., pp. 1013–1031 |
| `dopant_wettability` | Pore-scale dopant impact on wettability alteration | Nono et al., pp. 1032–1042 |
| `low_salinity_ior` | Low-salinity brine wettability alteration / IOR for presalt carbonates | Karoussi et al., pp. 1043–1060 |
| `nanopore_adsorption` | Wettability effects on adsorption and capillary condensation in nanopores | Nguyen et al., pp. 1061–1071 |
| `carbon13_mr` | ¹³C MR relaxation-time wettability characterisation of core plugs | Ansaribaranghar et al., pp. 1073–1089 |
| `kerogen_mr` | Fluid quantification and kerogen assessment in shales (¹³C and ¹H MR) | Zamiri et al., pp. 1090–1100 |
| `mri_rel_perm` | Model-free relative permeability via rapid in-situ ²³Na MRI saturation monitoring | Zamiri et al., pp. 1101–1117 |

DOI pattern: `10.30632/PJV66N6-2025aNN` (NN = 1 … 13)

---

## src2026_02 — Vol. 67, No. 1 (February 2026)

Best Papers from the SPWLA 66th Annual Symposium, Dubai, May 17–21, 2025.

| Module | Topic | Reference |
| --- | --- | --- |
| `drill_cuttings_ai` | AI-enhanced reservoir characterization from drill-cuttings images and elemental analysis | Kriscautzky et al. |
| `dts_co2_monitoring` | Real-time CO₂ injection monitoring via fiber-optic DTS modelling | Pirrone & Mantegazza |
| `nmr_discrete_inversion` | Discrete inversion method for NMR data processing and fluid typing | Gao et al. |
| `depth_alignment` | Dynamic depth alignment of well logs using continuous optimization | Westeng et al. |
| `fluid_identification` | Integrated technique for reservoir fluid distribution in Norwegian oil fields | Bravo et al. |
| `multiphysics_inversion` | Advanced logging techniques for complex turbidite reservoir characterization | Datir et al. |
| `nmr_bitumen` | NMR characterization of secondary organic matter and hydrocarbons | Al Mershed et al. |
| `co2_sequestration` | Effect of CO₂ sequestration on carbonate formation integrity | Al-Hamad et al. |
| `tortuosity_permeability` | Tortuosity assessment for reliable permeability quantification | Arrieta et al. |
| `pgs_type_curve` | Novel type curve for sandstone rock typing | Musu et al. |
| `udar_methods` | UDAR joint inversion, multidimensional inversion, and look-ahead mapping | Wu et al.; Saputra et al.; Ma et al. |

DOI pattern: `10.30632/PJV67N1-2026a{1..15}`

---

## src2026_04 — Vol. 67, No. 2 (April 2026)

| Module | Topic | Reference |
| --- | --- | --- |
| `a01_sponge_core_saturation_uncertainty` | Monte Carlo uncertainty quantification of sponge-core saturation data | Alghazal & Krinis |
| `a02_nmr_wettability_pore_partitioning` | NMR T₂-based wettability pore partitioning and oil recovery effects | Aljishi, Chitrala, Dang & Rai |
| `a03_water_rock_mechanical_ae` | Water-rock interactions, mechanical degradation, and acoustic emission in sandstones | Zhao |
| `a04_wireline_anomaly_diagnosis` | Dual-signal (tension + vibration) fusion for wireline anomaly diagnosis | Liu, Zhang, Fan et al. |
| `a05_ail_hierarchical_correction` | Hierarchical correction of array induction logging in horizontal wells | Qiao, Wang, Deng, Xu & Yuan |
| `a06_bioclastic_limestone_classification` | Integrated geological + petrophysical classification for marine bioclastic limestones | Guo, Duan, Du et al. |
| `a07_knowledge_guided_dcdnn` | Knowledge-guided dilated convolutional DNN for reservoir parameter prediction | Yu, Pan, Guo et al. |
| `a08_shale_induced_stress_fracture` | Induced-stress-difference modelling for double fracturing in deep shale | Ci |
| `a09_acid_fracturing_cbm` | Acid fracturing propagation in deep coalbed methane wells | Zhao, Jin, Zhen & Li |
| `a10_interlayer_fracture_propagation` | Dynamic interlayer fracture propagation in coal-bearing strata (FDEM proxy) | Zhao, Jin, Guo et al. |
| `a11_awi_cement_evaluation` | Anti-water-invasion evaluation of cement slurry via conductivity-jump detection | Zhang, Zhang, Zhang et al. |
| `a12_depth_shifting_ml` | Automatic well-log depth shifting (DTW, cross-correlation, ridge regression, 1-D CNN) | Pan, Fu, Xu et al. |

DOI pattern: `10.30632/PJV67N2-2026aNN` (NN = 1 … 12)

---

## Usage

Every module can be run as a standalone script:

```
python -m src2023_12.bennis_invasion_sw
python -m src2024_02.article1_waxman_smits_dual_water
python -m src2024_04.grader_digital_rock
python -m src2024_06.article1_nuclear_logging_ccs
python -m src2024_08.gor_prediction_ml
python -m src2024_10.probe_permeameter
python -m src2024_12.m01_image_rock_properties
python -m src2025_02.scal_model_ccs
python -m src2025_04.udar_look_ahead
python -m src2025_06.core_scanner
python -m src2025_08.pa_genai_extraction
python -m src2025_10.a1_log_interpretation
python -m src2025_12.pgs_rock_typing
python -m src2026_02.depth_alignment
python -m src2026_04.a01_sponge_core_saturation_uncertainty
```

Each package includes a master test runner:

```
python -m src2023_12.run_all
python -m src2024_02.run_all_tests
python -m src2024_04.test_all
python -m src2024_06.test_all
python -m src2024_08.test_all
python -m src2024_10.test_all
python -m src2024_12.test_all_modules
python -m src2025_02.test_all
python -m src2025_04.test_all
python -m src2025_06.run_all_tests
python -m src2025_08.test_all
python -m src2025_10.test_all
python -m src2025_12.test_all
python -m src2026_02.test_all
python -m src2026_04.test_all
```

The `src2026_04` modules each export an `example_workflow()` function that
demonstrates the key algorithms with synthetic data:

```
from src2026_04 import a12_depth_shifting_ml

a12_depth_shifting_ml.example_workflow()
```

## Disclaimer

These implementations are *unofficial* and are not affiliated with or endorsed
by the SPWLA or any of the original authors. They are intended as educational
companions to the published articles. Please refer to the original papers for
authoritative descriptions of the methods.
