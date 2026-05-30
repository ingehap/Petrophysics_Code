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
* scikit-image ≥ 0.21 (required by `src2023_04`)
* xgboost ≥ 1.7 (required by `src2023_04`)

## Repository layout

```
Petrophysics_Code/
├── src2022_10/   Vol. 63 No. 5 (Oct 2022)  —  5 modules + test suite
├── src2022_12/   Vol. 63 No. 6 (Dec 2022)  —  7 modules + test suite
├── src2023_02/   Vol. 64 No. 1 (Feb 2023)  —  9 modules + test suite
├── src2023_04/   Vol. 64 No. 2 (Apr 2023)  — 11 modules + test suite
├── src2023_06/   Vol. 64 No. 3 (Jun 2023)  —  9 modules + test suite
├── src2023_08/   Vol. 64 No. 4 (Aug 2023)  —  6 modules + test suite
├── src2023_10/   Vol. 64 No. 5 (Oct 2023)  — 11 modules + test suite
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

---

## src2022_10 — Vol. 63, No. 5 (October 2022)

Regular (non-themed) issue spanning rock mechanics, capillary pressure modelling, tight-rock permeability methodology, in-situ CT visualisation of mud-filtrate invasion, and acid-gas cement degradation. Five papers implemented; a sixth (Gao et al., "Coring Method for Dolomite Rocks With Well-Developed Joint Fissures Based on Permeability Reinforcement", DOI suffix presumed `a6`) is listed in the issue TOC but its body was not present in the source-PDF extract, so no module is included for it.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_nanoindentation` | Tutorial / review of nanoindentation for shale mechanics: Oliver-Pharr framework with hardness H = P_max / A (Eq. 1); unloading stiffness S = dP/dh (Eq. 2); ideal Berkovich tip-area function A = 24.5·h_c² (Eq. 3); reduced-to-Young's-modulus composite compliance 1/E_r = (1−ν_s²)/E_s + (1−ν_i²)/E_i with diamond E_i = 1141 GPa, ν_i = 0.07 (Eq. 4); Gupta et al. (2018) shear-modulus estimator G = 95.3·slope − 0.35 GPa (Eq. 5); log-creep fit h(t) − h₀ = b·log10(t/t₀) (Eqs. 9–10); mixed-mode fracture toughness K_c = α·√(E/H)·P_max / c^(3/2) (Eq. 11); 100-indent synthetic array reproduces the paper's Woodford-shale statistic E_s ≈ 31 ± 3.4 GPa | Sondergeld & Rai, pp. 576–590 |
| `article2_shale_capillary_pressure` | Three-parameter Pc(Sw) model for shale that admits a non-zero entry pressure (unlike van Genuchten) and a non-plateau trend (unlike Brooks-Corey): Young-Laplace Pc = 4·γ·cos θ / d (Eq. 1); normalised saturation Sw* = (Sw − Swirr) / (1 − Swirr) (Eq. 4); Brooks-Corey Pc = pe·(Sw*)^(−1/λ) (Eq. 3); van Genuchten Pc = (1/α)·(Sw*^(−1/m) − 1)^(1/n) (Eq. 5); proposed form Pc = pe + α₁·((1 − Sw*) / Sw*)^α₂ (Eq. 6); MSE = Σ(Y_pred − Y_obs)² / N (Eq. 7); fits all three to a synthetic MICP dataset via SciPy nonlinear least squares with R² and MSE comparison | Alipour K., Kasha, Sakhaee-Pour, Sadooni & Al-Kuwari, pp. 591–603 |
| `article3_stress_dependent_permeability` | Closed-form three-measurement steady-state inversion for tight-rock (k0, α, β): Darcy mass-flow integral (Eqs. 1–3); exponential closure k = k0·exp(−α·(σ_c − β·p_p)) (Eqs. 4–6); spatially varying k(x) along the plug (Eqs. 7–8); steady-state mass-flow / pressure relation (Eq. 9); Pair 1 (same pu, pd; two confining pressures) yields α from Q₁/Q₂ (Eqs. 10–15); Pair 2 (same σ_c, two different pp_mean values) yields αβ after correcting for the integrated (pu² − pd²) ratio (Eqs. 16–18); k0 follows from any single run. Reproduces the paper's carbonate-source-rock plug exactly: α ≈ 4.7e-4 /psi, β ≈ 0.83, k0 ≈ 100 nD | Zhang, Liu & Duncan, pp. 604–613 |
| `article4_mud_filtrate_invasion_ct` | Pure-analytical analogue of the time-lapse micro-CT analysis pipeline the paper applies to four cores (Leopard sandstone, Nugget sandstone, Texas Cream Limestone, Vuggy Dolomite): capillary number N_ca = v·μ/σ and Bond number N_B = Δρ·g·R_pore²/σ; Brooks-Corey two-phase relative permeabilities; Leverett J(Sw) = Pc·√(k/φ) / (σ·cos θ); fractional flow f_w = (k_rw/μ_w) / (k_rw/μ_w + k_ro/μ_o); Welge-tangent Buckley-Leverett front saturation; Dewan-Chenevert mudcake-controlled invasion-front position x_front(t) = √(2·k_eff·ΔP / (μ_w·φ) · t); default parameters reproduce the paper's Leopard-sandstone N_ca ≈ 2e-5 / 7e-7 spurt-vs-late transition | Schroeder & Torres-Verdín, pp. 614–641 |
| `article5_cement_acid_gas_corrosion` | Class-G oilwell-cement autoclave exposure to 12 % CO₂ + 5 ppm H₂S at 150 °C / 75 MPa for 7 / 14 / 30 days: labelled steady-state gas-Darcy permeability formula k = (2·Q·P₀·μ·L) / (A·(P₁² − P₂²)) (Eq. 1); diffusion-limited reaction-front depth x_f(t) = K·√t with a cylindrical-rim corrosion-fraction geometry; empirical exponential-in-time permeability growth k(t) = k_init · exp(B·t) fitted to the paper's three measurements (~200× rise from day 7 to day 30, matching the reported 3e-4 → 6.46e-2 mD trend); tensile-strength loss as a linear function of corrosion fraction reaching ~ 9.5 MPa at day 30 (paper reports ~ 9.8 MPa) | Zhou, Zeng, Sun, Zhou, Lei, Wan, Luo, Wu, Zhang & Xiao, pp. 642–651 |

DOI pattern: `10.30632/PJV63N5-2022aN` (N = 1 … 5; a presumed a6 — Gao et al. coring method — is listed in the TOC but its body is not implemented)

---

## src2022_12 — Vol. 63, No. 6 (December 2022)

Special Issue: *Best Papers of the 2022 SPWLA Annual Symposium* (Stavanger, Norway, June 11-15, 2022). Seven peer-reviewed extensions of the highest-rated symposium papers spanning fiber-optic DAS VSP full-waveform inversion, sourceless LWD acoustics from drill-bit noise, ultradeep azimuthal resistivity (UDAR) geosteering on the Norwegian Continental Shelf, fractured-carbonate static/dynamic modeling with a spherical self-organizing map facies classifier, dipole-shear reflection imaging combined with Mohr-Coulomb critically-stressed-fracture geomechanics, molecular-dynamics quantification of mineral/fracturing-fluid interfaces, and digital-rock-physics QC of a novel percussion sidewall coring system.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_das_vsp_fwi` | Full-waveform inversion of fiber-optic DAS VSP data: DAS strain observable d_DAS = S·ε·τ along the fiber tangent (Eqs. 4–5); least-squares time-domain misfit J = ½‖d_pred − d_obs‖² (Eq. 6) with adjoint gradient as the time cross-correlation of forward and back-propagated residual wavefields (Eq. 7); critically, DAS residuals are first averaged spatially by Sᵀ and injected as moment-tensor sources (Eq. 9) instead of point forces (Eq. 8); explicit closed-form moment tensors M_vert, M_hor,x and M_45,xz for vertical / horizontal / 45° deviated wells from τ = (τ_x, τ_y, τ_z) (Eqs. 10–11); 1-D reflectivity-domain Gauss-Newton inversion on per-interface log-impedance contrasts as a tractable analogue of the paper's 2-D elastic FWI | Podgornova, Bettinelli, Liang, Le Calvez, Leaney, Perez & Soliman, pp. 576–590 |
| `article2_sourceless_lwd_acoustics` | First open-literature LWD field test of "sourceless" borehole acoustics extracting P, S and Stoneley velocities from drill-bit-generated noise on a six-ring four-azimuth (90° apart) receiver array 36.5 m above a PDC bit; receiver-azimuth recombination into monopole m(t) = (a₀ + a₁ + a₂ + a₃)/4 (Eq. 1), quadrupole q(t) = (a₀ − a₁ + a₂ − a₃)/4 (Eq. 2) and dipole d_x(t) = ((a₀ + a₂) − (a₁ + a₃))/4 (Eq. 3); listening-mode 4,096-sample records at 24 µs sampling (~98 ms, 20.8 kHz Nyquist) acquired ~every 10 ft; multi-receiver semblance over a 50–250 µs/ft slowness grid recovers Vp and Vs from the synthetic record within ±15 % | Bolshakov, Walker, Marksamer, Samano & Reynolds, pp. 591–603 |
| `article3_udar_geosteering` | Snorre-field UDAR-LWD geosteering case history: forward operator for an azimuthal-deep resistivity tool over a layered earth (Gaussian depth-of-investigation kernel of std = 0.6·spacing for spacings 7/15/30/60 m at 2 and 8 kHz); 1-D Occam-style stochastic Metropolis inversion over (resistivity, boundary) at the transmitter measure-point; "geostop" decision rule fires when the 5th-percentile distance-to-base falls below a configurable safety margin (the paper's BCU+Mime-marl scenario; ~50 % acceptance; ~100× misfit reduction relative to a 4-m-offset prior) | Sinha, Walmsley, Clegg, Vicuna, Wu, McGill, Paiva dos Reis, Nygard, Ulfsnes, Constable, Antonsen & Danielsen, pp. 604–633 |
| `article4_fractured_carbonate_som` | Integrated static/dynamic modeling of a Hungarian Triassic carbonate basement: Harrison (1995) Russian-log analogue (φ = 0.40 − 0.0030·NGK clipped to [0, 0.35]; linear Vsh from GK); rectangular Kohonen SOM with Gaussian-neighbourhood competitive learning as a tractable analogue of the paper's spherical SOM (sSOM); majority-vote unit-label assignment with nearest-occupied-unit fill for empty cells; ~99 % held-out accuracy on three fracture facies (macrofracture / microfracture / host) on a synthetic five-feature log dataset (POR, RD, VSh, DTc, CALI-BS); Torabi et al. (2019) damage-zone-width law w_dz = 0.12·d^0.8 with the four-class fault-core / high-DZ / low-DZ / host partition | Ali Akbar, Nemes, Bihari, Soltesz, Barany, Toth, Borka & Ferincz, pp. 634–649 |
| `article5_dipole_shear_mohr` | Dipole-shear reflection imaging coupled with 3-D Mohr-Coulomb critically-stressed-fracture analysis: effective stress tensor (Eq. 1); fracture normal n = (sin θ sin α, sin θ cos α, cos θ)ᵀ (Eq. 2); effective normal stress σ_n on the fracture face (Eq. 3); shear stress τ_n = ‖T − σ_n n‖ (Eq. 4); SH-wave image SH(α) = xx·cos²α − sin α·cos α·(xy + yx) + yy·sin²α (Eq. 5); Mohr-Coulomb criticality τ = S₀ + μ·σ_n (Eq. 6); numerical verification of the paper's analytical claim that the 180° cross-dipole strike ambiguity does NOT change the (σ_n, τ_n) pair, so the imaged set can be filtered by the geomechanical criterion without resolving the azimuth ambiguity | Tang, Wang, Li, Xiong & Zhang, pp. 650–657 |
| `article6_md_mineral_fluid` | Synthetic-trajectory analogue of the paper's all-atom MD analysis pipeline: 2-D Langevin Brownian dynamics for ions/water in a 3-nm reflecting-wall slit pore with an optional "sticky" near-wall subpopulation (D_sticky = 0.05·D_bulk) modelling adsorbed methanol / citric acid on illite or calcite; per-particle density profile across the slit; mineral-fluid wall-contact count as a proxy for the paper's hydrogen-bond statistic; self-diffusion D from the long-time slope of the slit-parallel MSD via D = lim_{t→∞} ⟨|r(t) − r(0)|²⟩ / (2·d·t) (Eq. 1), with d_dim = 1 since the slit-normal MSD saturates at slit_width²/12 | Silveira de Araujo & Heidari, pp. 658–670 |
| `article7_pswc_drp_qc` | Digital-rock-physics QC workflow for a novel percussion sidewall coring (PSWC) bullet benchmarked against rotary plugs: synthetic 3-D voxel sand-pack as the analogue of a binary-segmented micro-CT cube; depth-localised percussion-damage zone induced by injecting small "fines" grains into a slice band; bulk and per-slice porosity from pore-voxel count; specific surface area S_v from grain-pore voxel-interface count; Kozeny-Carman absolute permeability k = φ³ / (c·S_v²·(1−φ)²) in millidarcy from voxel size; damage map showing per-slice porosity and permeability drop within the percussion-damaged band | Lakshtanov, Zapata, Saucier, Cook, Eve, Lancaster, Lane, Gettemy, Sincock, Liu, Geetan, Draper & Gill, pp. 671–684 |

DOI pattern: `10.30632/PJV63N6-2022aN` (N = 1 … 7)

---

## src2023_02 — Vol. 64, No. 1 (February 2023)

Regular (non-themed) issue spanning nine papers: reservoir-fluid geodynamics, carbonate phi-k rock typing, deep-learning borehole-image fracture extraction, hexa-combo LWD operational case study, digital-core elastic moduli, probabilistic geosteering inversion, data-mining permeability, hot-water-injection temperature optimisation, and well-log depth-matching benchmark.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_rfg_petroleum_system` | Reservoir-fluid geodynamics + petroleum-system case study: gravitational term of the Flory-Huggins-Zuo asphaltene EOS (ln(φ_a/φ_ref) = −V_a·g·(ρ_a−ρ_o)·h_above/RT); exponential biodegradation kinetic for n-alkane depletion; two-stage volumetric mixing of resident biodegraded oil with a late condensate charge plus solubility-threshold asphaltene flocculation; WAT correlation (WAT = 5 + 800·f_wax + 0.04·(ρ−700)); reproduces the paradox of lower asphaltene in the more biodegraded Central block and upstructure asphaltene destabilisation at the moving fluid contact | Pierpont, Birkeland, Cely, Yang, Chen, Achourov, Betancourt, Canas, Forsythe, Pomerantz, Yang, Datir & Mullins, pp. 6–17 |
| `article2_carbonate_phi_k` | Three classical φ-k models fitted per carbonate rock type (grainy / mixed / muddy): Amaefule FZI workflow with RQI = 0.0314·√(k/φ), NPI = φ/(1−φ), FZI = RQI/NPI (Eqs. 1–3) — recovered FZI 3.5 / 0.9 / 0.35 µm; Lucia rock-fabric number with log(RFN) = (3.1107+1.8834·log φ_g + log Swi) / (3.0634+1.4045·log φ_g) and log k = (9.7982−12.0838·log RFN) + (8.6711−8.2965·log RFN)·log φ_g (Eqs. 4–5); Winland-Kolodzie log r35 = 0.732 + 0.588·log k − 0.864·log φ (Eq. 6); factor-of-two accuracy metric showing per-RRT fits clearly beat a single global FZI fit | Dernaika, Masalmeh, Mansour, Al Jallad & Koronfol, pp. 18–37 |
| `article3_swin_fracture` | NumPy-only proof-of-concept of the W-shape dual encoder-decoder for sinusoidal-fracture segmentation: W-MSA vs full-MSA FLOP formulas Ω(MSA) = 4hwC² + 2(hw)²C and Ω(W-MSA) = 4hwC² + 2M²·hw·C (Eqs. 1–2); patch-window mean-pool encoder with variance-based attention proxy; second branch implementing a top-K sinusoidal Hough decoder with non-maximum suppression on the depth axis; per-pixel Precision / Recall / IoU / Dice scoring (Eqs. 3–6); dual-branch combiner beats a fixed-threshold baseline on noisy synthetic images | Wang & Zhou, pp. 38–49 |
| `article4_hexa_combo_lwd` | Marrat-style operational case study: synthetic LWD suite over a tight fractured carbonate (GR, multi-DOI Rt, NPHI, RHOB, DTC, DTS, NMR T2 distribution); Vsh and density porosity from triple-combo; effective porosity (φ_e = ½(φ_d+NPHI) − Vsh·φ_sh); Archie water saturation; NMR BVI / FFI partition at a 33 ms T2 cutoff; dynamic K, G, ν, E from RHOB and DTC/DTS (geomechanics); Wang-Gale brittleness index BI = ½·E_norm + ½·(1−Vsh); rule-based perforation-interval picker (φ_e > 0.06 ∧ Sw < 0.40 ∧ BI > 0.55, min 4 ft) | Saleh, Al-Khudari, Al-Azmi, Al-Otaibi, Patnaik, Joshi, Abdulkarim, Aki, Fahri, Sanyal & Sainuddin, pp. 50–66 |
| `article5_digital_core_poisson` | Multi-component 3-D digital core for a Wufeng-Longmaxi-style shale (quartz, clay, calcite, dolomite, pyrite, kerogen, gas-filled pore): two-point spatial autocorrelation Z(r₁)·Z(r₂) (Eq. 1); Voigt-Reuss-Hill bounds on (K, G) for the solid skeleton as a tractable analogue of the paper's FEM elastic-potential minimisation U = ½·Σ D_rp,sq·u_rp·u_sq (Eqs. 2–5); Krief-style porosity softening and Gassmann fluid term; Poisson's ratio ν = (3K−2G) / (2·(3K+G)) (Eq. 6) returning the paper's ~0.24 baseline; bedding-dip sweep reproducing the 45°-minimum-ν behaviour; sensitivity to calcite fraction (most influential mineral) | Liu, Wang, Lai, Wang, Zhang, Zhang & Ou, pp. 67–79 |
| `article6_geosteering_enrml` | Approximate Levenberg-Marquardt Ensemble Randomized Maximum Likelihood (LM-EnRML) joint inversion (Appendix A1) on a three-layer scenario: state vector (φ, Sw per layer, two bed boundaries) updated by m_{n+1} = m_n − C_x·G_nᵀ·(G_n·C_x·G_nᵀ + λ·C_d)⁻¹·(d_pred − d_obs) with ensemble-empirical sensitivity G_n; toy depth-of-investigation forward operators (Gaussian kernels of std 0.9 m at the bit for shallow propagation, 14.9 m and 10 m ahead for extra-deep symmetric EM, 0.4 m for nuclear bulk density) with Archie + density mixing per layer; demonstrates the >2× reduction in chi² misfit and the boundary-uncertainty shrinkage when shallow density is added to deep EM | Jahani, Alyaev, Ambia, Fossum, Suter & Torres-Verdín, pp. 80–91 |
| `article7_dm_permeability` | Seven-step data-mining workflow for permeability prediction in heterogeneous Tarim-Basin-style carbonates: synthetic three-class dataset (dolostone / limestone / anhydritic) with seven features (GR, RHOB, NPHI, DT, log Rt, fracture index, φ); mutual-information / Gini feature ranking; class-aware predictor combining standardised-space k-means + per-class log-linear regression (with Random Forest baseline if sklearn is available); MAE(log10 k) metric — per-class fit improves on global by ~55 % on the synthetic dataset (paper reports ~18 % on real Tarim data) | X. Li, pp. 92–106 |
| `article8_hot_water_injection` | Ramey-style closed-form wellbore fluid temperature profile T_f(z) = T_geo(z) − g·A + (T_inj − T₀ + g·A)·exp(−z/A) (Eqs. 1–3) with relaxation depth A = w·ρ·c·f(t_D) / (2π·r·U) and the Hasan-Kabir dimensionless-time function f(t_D) = ln(exp(−0.2·t_D) + (1.5 − 0.3719·exp(−t_D))·√t_D) for transient formation thermal resistance; two-section variant for an upper insulated tubing length (lower U → larger A); bisection optimiser for the surface T_inj that just delivers T_WAT at bottomhole — recommended ~70 °C bare and ~63 °C with 750 m of insulation on a Liaohe-style parameter set, matching the paper's 60–65 °C engineering band | Yu & S. Zhang, pp. 107–114 |
| `article9_depth_matching` | Empirical benchmark of three signal-alignment algorithms on a synthetic GR pair with a non-linear monotonic depth warp, amplitude scaling, and additive noise: classical Dynamic Time Warping; Constrained DTW with a Sakoe-Chiba warping band (window = 10 % of length); Correlation Optimised Warping with piecewise-linear time re-mapping and greedy sequential boundary search maximising per-segment Pearson correlation; per-method alignment-vs-reference correlation as the scoring metric, recovering the paper's observation that DTW achieves high pointwise correlation but COW produces alignments more consistent with an expert pick | Ezenkwu, Guntoro, Starkey, Vaziri & Addario, pp. 115–129 |

DOI pattern: `10.30632/PJV64N1-2023aN` (N = 1 … 9)

---

## src2023_04 — Vol. 64, No. 2 (April 2023)

Artificial Intelligence and Machine-Learning Special Issue. Four sub-themes: (i) data-driven petrophysical interpretation (DP-based electrofacies clustering, image-based rock classification, symbolic regression for interpretation models), (ii) ML-assisted petrophysical data preprocessing (comparative log prediction methods, unsupervised outlier detection and log editing, removal of borehole-image artefacts), (iii) ML and data analytics for uncertainty modeling (sonic-log imputation with goodness metric, exemplar-guided sedimentary facies modeling, spatial data analytics-assisted subsurface modeling), and (iv) ML-based surrogate modeling (fast deconvolution and convolution methods for induction-log inversion and forward modeling).

| Module | Topic | Reference |
| --- | --- | --- |
| `article01_electrofacies_dp` | Unsupervised electrofacies clustering with dynamic programming: generic objective L_f(X,Y,W) = Σ_t f(x_t, w_{y_t}) (Eq. 1) minimised under constraints on number of clusters C, max transitions N, and minimal block size MinPhi via the recurrence ω_t(n,c) (Eq. 3); Waxman-Smits resistivity 1/Rt = (φ^m*·S_w^n*/a)·(C_w + B·Q_v/S_w) as the per-cluster physical model (Eq. 13) with Dacy-Martin temperature-dependent B (Eq. 14); random-init + dp_path_finder iteration to convergence; ARI-based selection of the most-common assignment across initialisations (Eq. 12) | Sinnathamby, Hou, Gkortsas, Venkataramanan, Datir, Kollien & Fleuret, pp. 137–153 |
| `article02_image_rock_classification` | Image-based rock classification from CT scans and slabbed core photos: per-depth grayscale descriptive statistics — mean, variance, skewness, kurtosis (Eqs. 1–4); HSV channel means from RGB photos (Eq. 5); GLCM-based contrast / energy / correlation textural features (Eqs. 6–8) on Haar-wavelet horizontal-detail coefficients in a sliding window; supervised Random Forest and SVM classifiers with 5-fold CV; unsupervised k-means baseline; class-based permeability-porosity model log10(k) = a + b·φ per facies showing the ~35 % MRE reduction over a single formation-wide model | Gonzalez, Heidari & Lopez, pp. 154–173 |
| `article03_symbolic_regression` | Genetic-programming symbolic regression for petrophysical interpretation models: Pearson and Spearman correlation heatmaps for input-variable selection (top-k by absolute correlation); minimalist GP engine with crossover, point mutation, tournament selection, and elitism over a primitive-function pool (+, −, ×, ÷, log, sqrt, square); Archie-style F = φ^(−m) and SDR-style permeability targets; complexity-penalised fitness (MSE + λ·tree_size) implementing the model-discrimination criterion; ensemble averaging across multiple GP seeds | Chen, Shao, Sheng & Kwak, pp. 174–190 |
| `article04_log_prediction_ml` | Comparative ML methods for missing-log prediction: PAE pointwise fully-connected autoencoder, WAE window-based 1-D convolutional autoencoder (sliding-window stacked-feature MLP), and XGBoost regressor; random input-masking augmentation that lets all three handle missing inputs at inference; standardised target/feature scaling; full metric suite RMSE / MAE / Pearson r / PSNR mirroring Tables 4–8; demonstration of robust prediction when one input curve is fully zeroed out | Simoes, Maniar, Abubakar & Zhao, pp. 192–212 |
| `article05_outlier_detection` | Five-step automated workflow for outlier detection and log editing with uncertainty: (1) standardisation; (2) one-class SVM footprint with RBF kernel; (3) inflection-point algorithm — kneedle criterion on the (outlier-fraction, SVM-score) curve to auto-pick ν; (4) per-well 2-D footprint binning + Jaccard / Overlap inter-well similarity matrices feeding multidimensional scaling (MDS) for unsupervised well clustering; (5) k-NN ensemble regression that returns mean + predictive standard deviation for log reconstruction QC | Akkurt, Conroy, Psaila, Paxton, Low & Spaans, pp. 213–238 |
| `article06_borehole_image_artifacts` | Supervised ML removal of artefacts in oil-based-mud resistivity-imager (OBMRI) borehole images: depth-window × azimuth-window pixel-neighbourhood feature extraction; XGBoost regressor trained on (raw, traditional-processed) image pairs to mimic the moving-window column-baseline subtraction that handles the geometric-factor effect; per-pad operation; demonstration on a synthetic image with a U-shaped per-button standoff offset that produces the depth-invariant artefact of Fig. 4 | Guner, Fouda & Barrett, pp. 239–251 |
| `article07_sonic_log_imputation` | Sonic well-log (DTC, DTS) imputation with ensemble-based uncertainty: mutual-information feature ranking against the response curve; ensemble of gradient-boosted regressors with row-subsampled bagging producing a non-parametric predictive CDF F_y(u_i); accuracy plot indicator ξ(u_i;p) (Eq. 2) and a(p) = 1 − 2·|ξ(p) − p| accuracy term over symmetric probability intervals; combined goodness-aware loss (Eq. 4) blending normalised MAE with the goodness metric; hyperparameter grid search that picks (learning_rate, n_estimators) maximising goodness rather than just MSE | Maldonado-Cruz, Foster & Pyrcz, pp. 253–270 |
| `article08_egfm_facies` | Exemplar-Guided Facies Modeling, simplified non-GAN demonstration of the content/pattern decoupling concept: distance-transform "content field" interpolating positive / negative well-point evidence; Gabor-filter-bank "pattern field" capturing exemplar orientation and energy; Adaptive Feature Fusion Block (AFB) — sigmoid attention weights times learnable γ_c, γ_p factors (Eq. 17) that aggregate the two streams; threshold + hard well-honouring decoder; pattern controllability check showing how swapping a horizontal exemplar for a vertical one shifts ~30 % of the generated facies map | Wu, Hu, Sun, Zhang, Wang & Zhang, pp. 271–286 |
| `article09_spatial_analytics` | End-to-end 2-D geostatistical workflow assisting Duvernay-style mature-data subsurface modelling: Mahalanobis-distance + isolation-forest spatial-outlier identification; Gaussian-kernel moving-window trend modelling; experimental semivariogram γ(h) (Eq. 1) with spherical-model fitting (Nelder-Mead) for nugget / sill / range; simple kriging with kd-tree neighbour search; sequential Gaussian simulation (SGS) drawing from the kriging-mean / kriging-variance distribution along a random path to produce stochastic realisations; collocated cokriging under the Markov-Bayes assumption with a variance-reduction factor for cosimulating a primary feature against a secondary | Salazar, Ochoa, Garland, Lake & Pyrcz, pp. 287–302 |
| `article10_induction_deconvolution` | ML-based deconvolution for fast, high-resolution induction-log inversion: linear deconvolution baseline log(R) = Σ a_k·log(R_app(z_{i+k})) (Eq. 5) with weights from a fixed-window least-squares fit; XGBoost (LightGBM-style) regressor mapping a 21-point, 10-ft sliding window of log(R_app) to log(R_model) at the centre depth; layered earth-model generator with log-uniform thickness (0.1–50 ft) and resistivity (0.1–100 Ω·m); RMSLE evaluation on a training set plus three independent test earth models showing the ML model beats both raw R_app and the linear deconvolution baseline | Hagiwara, pp. 304–311 |
| `article11_induction_convolution` | Companion to Article 10 — ML-based forward "convolution" model that calculates the induction-log apparent resistivity from a layered earth model: 101-point, 50-ft sliding window of log(R_model) input to an XGBoost regressor predicting log(R_app) at the centre depth (the larger window required by the 60° deviated-borehole geometry of the paper); linear-convolution baseline for comparison; window-size scan demonstrating that 50 ft is the elbow beyond which RMSLE no longer improves; orders-of-magnitude speed-up over analytic 1-D forward modelling | Hagiwara, pp. 312–322 |

DOI pattern: `10.30632/PJV64N2-2023aNN` (NN = 1 … 11)

---

## src2023_06 — Vol. 64, No. 3 (June 2023)

Special Issue containing the *Best Papers of the 2022 SCA International Symposium*. Three sub-themes: (i) continuous improvement of core analysis techniques for data quality and turnaround time (initial-water-saturation setting on core, wireless centrifuge resistivity index, overburden FRF/RI models), (ii) new methodologies to address petrophysical challenges (digital-SCAL gas trapping, T1-T2\* shale magnetic resonance, angle-dependent ultrasonic reflectivity), and (iii) emerging technologies for detailed rock imaging and behaviour studies (NMR-mapped dielectric dispersion, terahertz microporosity imaging, time-lapse X-ray radiography of mud-filtrate invasion).

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_hdt` | Hybrid Drainage Technique (HDT) for setting initial water saturation on core samples: Hassler-Brunner centrifuge capillary pressure Pc(r) = ½·Δρ·ω²·(R²−r²) (Eq. 1), Phase-1 viscous-flooding profile generator with capillary end-effect "foot", Phase-2 porous-plate iterative homogenisation that imposes a uniform Pc through a semipermeable plate at the outlet, and the std-dev / max-min profile-quality metrics used in Figs. 10, 12 and 16 to demonstrate CEE removal | Fernandes, Nicot, Pairoys, Bertin, Lachaud & Caubit, pp. 325–339 |
| `article2_wiri` | Wireless Resistivity Index in centrifuge (WiRI) and comparison with porous plate (PP) and ultra-fast Pc-RI (UFPCRI) for Archie's saturation exponent: forward Archie law RI = Sw^(−n), three n-estimators (log-log linear regression for PP/UFPCRI, global least-squares through-origin inversion for WiRI), and a Monte Carlo sensitivity study reproducing Figs. 3 and 5 (random absolute error on produced volumes, random relative error on resistivity) showing the downward bias of PP and the near-unbiased behaviour of WiRI | Danielczick, Nepesov, Rochereau, Lescoulie, De Oliveira Fernandes & Nicot, pp. 340–352 |
| `article3_overburden_frf_ri` | Analytical models for the formation resistivity factor and resistivity index at overburden conditions: Rock Resistivity Modulus RRM = (1/Ro)·dRo/dP and True Resistivity Modulus TRM = (1/Rt)·dRt/dP definitions (Eqs. 5–7), Multi-FRF model FRF(P₂) = FRF₁·exp(−RRM·ΔP) (Eq. 15) fitted from a slope of ln(FRF₂/FRF₁) vs ΔP (Eq. 18), Single-FRF compressibility-based RRM ≈ −m·(Cp − Cb) (Eq. 16), and the analogous RI(P₂) = RI₁·exp(−TRM·ΔP) prediction with a first-order saturation-exponent correction | Nourani, Pruno, Ghasemi, Fazlija, Gonzalez & Rodvelt, pp. 353–366 |
| `article4_gas_trapping` | Advanced digital-SCAL measurements of gas trapped in sandstone: Land trapping model Sgr = Sgi/(1 + C·Sgi) with C = 1/Sgr_max − 1/Sgi_max, exponential ripening / dissolution kinetics Sgr(t) = Sgr_∞ + (Sgr₀ − Sgr_∞)·exp(−t/τ) capturing the continued shrinkage of disconnected gas clusters in pre-equilibrated brine, and a 3-class quantile-threshold segmentation of synthetic micro-CT volumes returning gas / brine / grain volume fractions and the resulting pore-scale gas saturation | Gao, Sorop, Brussee, van der Linde, Coorn, Appel & Berg, pp. 368–383 |
| `article5_shale_t1t2star` | Shale characterization with T1-T2\* magnetic resonance relaxation correlation at low and high field: effective transverse relaxation 1/T2\* = 1/T2 + γ·ΔB₀ + γ·Δχ·B₀ (Eq. 1), Look-Locker effective T1\* with 1/T1\* = 1/T1 − ln(cos α)/τ (Eq. 2a), forward 2-D saturation-recovery + FID signal generator S(τr,t) = Σ Aₖ·(1 − exp(−τr/T1ₖ))·exp(−t/T2\*ₖ) for kerogen / oil / water populations, and a non-negative-projected linear inversion that recovers their amplitudes when the relaxation times are known | Zamiri, Guo, Marica, Romero-Zerón & Balcom, pp. 384–401 |
| `article6_ultrasonic_reflection` | Angle-dependent ultrasonic-wave reflection for high-resolution elastic-property estimation on complex rock samples: closed-form fluid-solid Brekhovskikh / reduced Zoeppritz reflection coefficient \|R(θ)\| with Snell's law and complex sqrt for post-critical angles, P- and S-wave critical-angle calculator θc = arcsin(Vf/Vp,s), and a SciPy least-squares inversion that recovers (Vp, Vs, ρs) from a noisy measured reflection-coefficient curve, reproducing the Berea and Texas Cream Limestone behaviour of Figs. 6–7 | Olszowska, Gallardo-Giozza, Crisafulli & Torres-Verdín, pp. 402–419 |
| `article7_dielectric_nmr` | NMR-mapped distributions of dielectric dispersion in carbonates: Bloembergen-Purcell-Pound (BPP) NMR T1 and T2 from autocorrelation time τc (Eqs. 1–2), complex Debye permittivity ε\* = ε∞ + (εs − ε∞)/(1 + iωτ) (Eq. 5), Havriliak-Negami extension ε\* = ε∞ + (εs − ε∞)/(1 + (iωτ)^α)^β (Eq. 6), linear additive Pore Combination Model εr = ε∞ + φm·εr,matrix + φv·εr,vug (Eq. 7), and the τPCM rule that splits a measured NMR T2 distribution into a fastest-relaxing matrix part and a slow-relaxing vug part to honour an externally measured matrix porosity | Funk, Myers & Hathon, pp. 421–437 |
| `article8_thz_porosity` | Terahertz time-domain spectroscopy (THz-TDS) for lateral microporosity mapping in carbonate rocks: mass-balance bulk porosities φ_total = (m_sat − m_dry)/(ρw·Vb) and φ_micro = (m_cent − m_dry,f)/(ρw·Vb) with φ_macro = φ_total − φ_micro, Beer-Lambert THz attenuation A = −ln(I/I_dry), and a calibration step that scales the per-pixel attenuation map to the measured bulk porosity to deliver lateral φ_total / φ_micro / φ_macro maps from three intensity scans (saturated / centrifuged / dry) | Eichmann, Bouchard, Ow, Petkie & Poitzsch, pp. 438–447 |
| `article9_xray_invasion` | Time-lapse X-ray radiography of mud-filtrate invasion and mudcake deposition: Beer-Lambert per-pixel attenuation, baseline-subtraction map A = −ln(I_now/I_dry), pure-NumPy 3×3 median filter (the noise filter applied throughout the paper), Darcy front-advance solution x_front(t) = √(2·k·ΔP/(μ·φ)·t), a synthetic 2-D radiograph time-series generator with explicit mudcake and invaded zones, and a column-profile threshold detector that returns mudcake-end and invasion-front pixel positions for each frame | Aérens, Torres-Verdín & Espinoza, pp. 448–461 |

DOI pattern: `10.30632/PJV64N3-2023aN` (N = 1 … 9)

---

## src2023_08 — Vol. 64, No. 4 (August 2023)

Mixed-topic issue covering a historical review of casedhole nuclear
surveillance logging, two companion papers on water-based-mud filtrate
invasion in the tight-gas Barik sandstone (a forward compositional
fluid-flow simulation and an iterative MCMC resistivity-inversion
workflow), mineralogical modelling of the Brazilian presalt Barra Velha
Formation, fracture imaging with the new high-definition oil-based-mud
borehole imagers, and a Python Dash application for well-log data quality
control.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_nuclear_logging` | Casedhole nuclear surveillance logging review and quantitative core: Pulsed-Neutron-Capture (PNC) volumetric mixing law Σt = (1−φ)·Σma + φ·(1−Sw)·Σhc + φ·Sw·Σw (Eq. 4) and its inversion for Sw, time-lapse PNC monitoring removing the matrix term (Eq. 5), salinity-to-Σw conversion, and Larionov tertiary / older-rocks shale-volume estimators | Fitz, pp. 473–501 |
| `article2_invasion_simulation` | Mud-filtrate invasion + Archie resistivity workflow for tight-gas sandstones: Sw_in = a·φ^b regression (Eq. 1), Land/Jerauld trapped-gas model (Eq. 2), Brooks-Corey gas and water relative permeabilities (Eqs. 3–4), Brooks-Corey capillary pressure Pc = Pd·Se^(−1/λ) (Eq. 5), Dewan & Chenevert mudcake permeability and porosity time evolution (Eqs. 6–7), Chin mudcake-thickness ODE (Eq. 8), Archie's law (Eq. 9), and a radial Sw / salinity / Rt(r) profile generator | Merletti, Al Hajri, Rabinovich, Farmer, Bennis & Torres-Verdín, pp. 502–517 |
| `article3_mineralogical_inversion` | Multicomponent mineralogical inversion of the Barra Velha Formation (presalt Santos Basin): volumetric photoelectric factor U = PEF·ρb (Eq. 1), Larionov GR clay volumes for younger and older rocks (Eqs. 2–3), NMR clay volume V_NR = (NMRtt − NMReff)/NMRtt (Eq. 4), hybrid GR + NMR clay (Eqs. 6–7), the linear log-response system ML_j = Σi α_ij·V_i (Eq. 8) solved with non-negative least squares under a unit-sum constraint, and the weighted RMS error metric (Eq. 9), with a built-in calcite/dolomite/quartz/clay/stevensite end-member catalogue | Jácomo, Hartmann, Rebelo, Mattos, Batezelli & Leite, pp. 518–543 |
| `article4_obm_imager_inversion` | High-definition oil-based-mud borehole-imager forward + inverse model: series-circuit two-frequency button impedance Z(ω) = Z_mud + Z_fmt with each layer as thickness/(jω·ε₀·ε_r − σ), damped Gauss-Newton inversion for (R_fmt, ε_fmt at F2, sensor standoff), the mud-angle helper arctan(σ/(ωε)) − 90°, and the fracture-equivalent-standoff trend that explains why open mud-filled fractures appear conductive in resistive formations and resistive in conductive formations | Chen, Zhang, Bloemenkamp & Liang, pp. 544–554 |
| `article5_iterative_resistivity` | Iterative resistivity-modelling workflow for deeply-invaded reservoirs: sliding-window first-derivative + variance bed-boundary detector, P5/P50/P95 OBM-equivalent Sw–φ envelope (Sw = a·φ^b for three quantiles, Eq. 1) converted to an Rt envelope through Archie, simplified array-laterolog forward model with depth-of-investigation weights, single-layer Bayesian / Markov-Chain Monte Carlo inversion of (Rt, Rxo) with the Rt envelope as a soft prior, and an outer iterative loop that refines the invasion radius L_xo by grid search | Merletti, Rabinovich, Al Hajri, Dawson, Farmer, Ambia & Torres-Verdín, pp. 555–567 |
| `article6_well_log_qc` | Well-log data validation, visualisation-helper, and repeatability checks for the Plotly-Dash QC application: `ValidationConfig` dataclass + the four-rule integrity check (missing / redundant / units / value-validity), summary-table builder for a Dash DataTable, log-difference (Eq. 1), Pearson correlation r between repeat and main passes (Eq. 2), and depth-shift cross-correlation that finds the optimal shift powering the Fig. 5 repeatability panel | Jin, Xu, Lin, Li & Zeghlache, pp. 568–573 |

DOI pattern: `10.30632/PJV64N4-2023aNN` (NN = 1 … 6)

---

## src2023_10 — Vol. 64, No. 5 (October 2023)

Energy Transition Special Issue covering integrated formation evaluation for
carbon capture and sequestration (site capacity / containment / injectivity,
time-lapse pulsed-neutron CO₂ monitoring), wireline-conveyed deep-borehole
stress measurement, high-resolution probe-based core analysis, flow-rate-
dependent relative permeability scaling, the Potash Identification crossplot,
X-ray radiography of mud invasion, joint SP/resistivity inversion in shaly
sands, numerical core-to-log forward modelling for QC, reservoir-fluid
geodynamics in the deepwater Gulf of Mexico, and chelating-agent acidising of
tight sandstones.

| Module | Topic | Reference |
| --- | --- | --- |
| `article_01_laronga_ccs_evaluation` | Integrated CCS site evaluation across the three "pillars" of capacity, containment, and injectivity: simple supercritical-CO₂ density correlation, Batzle-Wang brine density, DOE/USGS volumetric storage-capacity equation M = A·h·NTG·φ·E·ρ_CO2, Young-Laplace caprock entry pressure Pc = 2σ·cosθ/r and the corresponding maximum buoyant CO₂ column h = Pc/((ρ_b−ρ_CO2)·g), and steady-state radial Darcy injectivity index II = 2π·k·h/(μ·(ln(re/rw)+S)) | Laronga, Borchardt, Hill, Velez, Klemin, S. Haddad, E. Haddad, Chadwick, Mahmoodaghdam & Hamichi, pp. 580–620 |
| `article_02_desroches_stress_measurement` | Wireline micro-fracturing stress-measurement interpretation: synthetic pump-up / shut-in / decline pressure-time generator, fracture closure pressure (FCP) picked by both the √t tangent-intersection method and the Nolte G-function derivative method, instantaneous shut-in pressure (ISIP) extraction, and the Hubbert-Willis breakdown relation Pb = 3·Sh,min − SH,max − Pp + T applied to a multi-test stress profile vs depth | Desroches, Peyret, Gisolf, Wilcox, Di Giovanni, Schram de Jong, Sepehri, Garrard & Giger, pp. 621–639 |
| `article_03_okwoli_probe_screening` | Probe-based high-resolution core screening for energy-transition reservoirs: synthetic mm-scale generator for probe luminance, magnetic susceptibility, P-wave velocity, and mini-permeameter permeability with embedded thin cemented features; boxcar upscaling to plug- and log-scale to demonstrate feature attenuation; multivariate log-linear permeability predictor log10(k) = a·lum + b·log10(MS) + c·Vp + d; and a cross-correlation depth-shift function for probe-to-log alignment | Okwoli & Potter, pp. 640–655 |
| `article_04_karadimitriou_relperm_scaling` | Flow-rate-dependent relative permeability for steady-state two-phase flow on a microfluidic network: Brooks-Corey baseline krw = krw,max·Sw_e^nw, krnw = krnw,max·(1−Sw_e)^nnw with Sw_e = (Sw−Swir)/(1−Swir−Snwr); capillary number Ca = μw·vw/σ and Valavanides-style log-Ca scaling kr(Ca) = kr,BC·(1 + α·log10(Ca/Ca_ref)); plus a tiny pore-network steady-state simulator with throat-radius-weighted Hagen-Poiseuille conductances | Karadimitriou, Valavanides, Mouravas & Steeb, pp. 656–679 |
| `article_05_laronga_pulsed_neutron_ccs` | Time-lapse pulsed-neutron monitoring of CO₂ storage with three independent measurements: forward and inverse models for thermal porosity (TPHI), thermal-neutron capture cross-section (SIGMA, c.u.), and fast-neutron cross-section (FNXS); per-channel ΔSco2 = −Δm/(φ·(m_brine − m_CO2)) inversion; and a three-channel consistency cross-check that flags depths where the independent estimates disagree, indicating endpoint or environmental issues | Laronga, Swager & Bustos, pp. 680–699 |
| `article_06_hill_potash_pid_plot` | The Potash Identification (PID) crossplot for rapid screening of commercial potash from cased-hole gamma-ray and neutron logs alone: mineral library with %K2O, GR (API), and neutron porosity (pu) for sylvite / langbeinite / carnallite / kainite / leonite / polyhalite / halite / anhydrite / gypsum / kieserite / shale; rule-based GR-NPHI quadrant classifier separating commercial (anhydrous) from non-commercial (hydrated) potash; RMA GR→%K2O transform; and grade-thickness aggregation against the BLM ≥ 4 ft / ≥ 4 % K2O standards | Hill, Crain & Teufel, pp. 700–713 |
| `article_07_aerens_xray_mud_invasion` | High-resolution time-lapse X-ray radiography of mud-filtrate invasion: Beer-Lambert attenuation I = I0·exp(−μ_eff·x), pixel-grayscale-to-water-saturation linear conversion between dry and fully-saturated reference frames, Outmans/Dewan-Chenevert √t external mudcake growth h(t) = √(2·k_mc·ΔP·t/(μ·(fc/fs−1))), and 1-D Buckley-Leverett radial-invasion saturation profiles via Welge tangent construction on a Brooks-Corey fractional-flow curve | Aérens, Espinoza & Torres-Verdín, pp. 715–740 |
| `article_08_zhao_sp_resistivity_inversion` | Joint inversion of water saturation and Qv from spontaneous-potential and resistivity logs in low-permeability shaly sandstones: Waxman-Smits oil-bearing resistivity 1/Rt = (φ^m*/(a·Rw))·Sw^n*·(1+B·Qv·Rw/Sw); Smits-style analytical SP membrane potential ΔSP = K_SP·log10(Cw/Cmf)·f_clay(Qv,Cw)·f_sat(Sw); and a derivative-free Particle Swarm Optimisation solver minimising a normalised joint (Rt, ΔSP) residual | Zhao, Wang, Li, Hu, Xie, Duan & Mao, pp. 741–752 |
| `article_09_bennis_corelogs_simulation` | Numerical well-log simulation from core measurements for QC: depth-resolved volumetric mineral + porosity + saturation model; forward operators for GR (linear mixing), bulk density (linear mixing of solids and pore fluids), neutron porosity (mineral and HI-weighted fluid), and Vp (time-average / Wyllie); Gaussian vertical-response convolution to wireline aperture; and chi-square misfit + linear regression bias detection that recovers (slope, intercept) corrections for badly environmentally-corrected logs | Bennis & Torres-Verdín, pp. 753–772 |
| `article_10_mohamed_rfg_connectivity` | Reservoir-fluid geodynamics workflow for hydraulic-connectivity assessment in heavily-faulted reservoirs: iteratively-solved Flory-Huggins-Zuo asphaltene gradient combining a gravitational term V_a·g·(ρ_a−ρ_o)·Δh/(R·T) with a solubility-parameter term ((δ_a−δ_o)²·V_a/(R·T))·((1−φ_a)²−(1−φ_a,ref)²); exponential viscosity-from-asphaltene correlation μ = μ0·exp(k·φ_a); and a greedy piecewise-linear pressure-gradient segmentation that detects fluid contacts and fault-bounded compartments from RFT/MDT pressure surveys | Mohamed, Torres-Verdín & Mullins, pp. 773–795 |
| `article_11_shafiq_chelating_acidizing` | Chelating-agent acidising of tight sandstones (HEDTA / EDTA / GLDA): per-mineral first-order Arrhenius-modulated dissolution X = 1 − exp(−k_eff·t) with k_eff = k0·(C/0.6)·exp(−Ea/R·(1/T−1/Tref)) for calcite, kaolinite, illite, feldspar (quartz inert); porosity update φ' = φ + ΣXᵢ·fᵢ·(1−φ); Kozeny-Carman permeability uplift k'/k = (φ'/φ)³·((1−φ)/(1−φ'))²; and pore-size-distribution shift toward smaller-radius widening | Shafiq, Ben Mahmud, Khan, Gishkori, Wang & Jamil, pp. 796–817 |

DOI pattern: `10.30632/PJV64N5-2023aNN` (NN = 1 … 11)

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
python -m src2023_04.article01_electrofacies_dp
python -m src2023_06.article1_hdt
python -m src2023_08.article1_nuclear_logging
python -m src2023_10.article_01_laronga_ccs_evaluation
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
python -m src2023_04.test_all
python -m src2023_06.test_all
python -m src2023_08.run_all_tests
python -m src2023_10.run_all_tests
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
