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

## Repository layout

```
Petrophysics_Code/
├── src2025_06/   Vol. 66 No. 3 (Jun 2025)  —  8 modules + test suite
├── src2025_08/   Vol. 66 No. 4 (Aug 2025)  — 11 modules + test suite
├── src2025_10/   Vol. 66 No. 5 (Oct 2025)  — 11 modules + test suite
├── src2025_12/   Vol. 66 No. 6 (Dec 2025)  — 13 modules + test suite
├── src2026_02/   Vol. 67 No. 1 (Feb 2026)  — 11 modules + test suite
└── src2026_04/   Vol. 67 No. 2 (Apr 2026)  — 12 modules + test suite
```

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
python -m src2025_06.core_scanner
python -m src2025_08.pa_genai_extraction
python -m src2025_10.a1_log_interpretation
python -m src2025_12.pgs_rock_typing
python -m src2026_02.depth_alignment
python -m src2026_04.a01_sponge_core_saturation_uncertainty
```

Each package includes a master test runner:

```
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
