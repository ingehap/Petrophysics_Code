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
├── src2020_06/   Vol. 61 No. 3 (Jun 2020)  —  5 modules + test suite
├── src2020_08/   Vol. 61 No. 4 (Aug 2020)  —  4 modules + test suite
├── src2020_10/   Vol. 61 No. 5 (Oct 2020)  —  7 modules + test suite
├── src2020_12/   Vol. 61 No. 6 (Dec 2020)  —  7 modules + test suite
├── src2021_02/   Vol. 62 No. 1 (Feb 2021)  —  9 modules + test suite
├── src2021_04/   Vol. 62 No. 2 (Apr 2021)  —  5 modules + test suite
├── src2021_06/   Vol. 62 No. 3 (Jun 2021)  —  6 modules + test suite
├── src2021_08/   Vol. 62 No. 4 (Aug 2021)  —  8 modules + test suite
├── src2021_10/   Vol. 62 No. 5 (Oct 2021)  —  9 modules + test suite
├── src2021_12/   Vol. 62 No. 6 (Dec 2021)  — 10 modules + test suite
├── src2022_02/   Vol. 63 No. 1 (Feb 2022)  —  6 modules + test suite
├── src2022_04/   Vol. 63 No. 2 (Apr 2022)  —  7 modules + test suite
├── src2022_06/   Vol. 63 No. 3 (Jun 2022)  — 11 modules + test suite
├── src2022_08/   Vol. 63 No. 4 (Aug 2022)  —  5 modules + test suite
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

## src2020_06 — Vol. 61, No. 3 (June 2020)

A regular issue of five papers spanning casedhole formation evaluation along unconventional horizontal wells, the impact of cement quality on carbon/oxygen and elemental pulsed-neutron analysis, reliable relative-permeability measurement in tight gas sands, an analytical relative-permeability-from-resistivity model for fractal porous media, and neural-network estimation of reservoir porosity from drilling parameters. This issue's source PDF (`Petrophysics_2020_06.pdf`) has a text layer, so titles, authors, page ranges, DOIs, equation numbers, variable definitions, and many numeric constants were read directly from the paper bodies; the conversion dropped most typeset formula glyphs (only Article 1's Eq. 1 survived verbatim), so the numbered formulas are faithful standard-form reconstructions from the preserved variables and the standard textbook expressions each paper cites. See the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_casedhole_horizontal_fe` | *Case study:* casedhole FE in laterals — spectral gamma ray γAPI = 4·Th + 8·U + 16·K (Eq. 1, verbatim), M-ANNIE VTI stiffness→engineering moduli E_v/E_h/ν_v/ν_h (Eqs. 2–5), sigma water saturation, acoustic-impedance Z = ρ·Vp gas indicator, and the +2 Ca / −3 Fe / −2 Al wt% elemental corrections | Sullivan, Wang, Bolshakov, Song, Lazorek, Tohidi & Seth, pp. 253–272 |
| `article2_cement_quality_co_pulsed_neutron` | *MCNP modeling + case study (no equations):* the carbon/oxygen ratio and salinity-independent oil saturation, the cement calcium-yield contribution (> 40%) and formation-calcium correction, the OBM-vs-WBM channel C/O bias, and sigma water saturation | Wang, Sullivan, Seth, Barnes, Wilson & Lazorek, pp. 273–285 |
| `article3_relperm_tight_gas_sand` | Relative permeability in tight gas sand: centrifuge capillary pressure Pc = ½·Δρ·ω²·(LR²−(LR−L)²) (Eq. 1), modified Corey-Brooks gas rel-perm (Eq. 2, ng in 0.5–3.75), SDR (NMR T2) brine rel-perm (Eqs. 3–4), and the Klinkenberg gas-slippage correction | Gonzalez, Tandon, Heidari, Gramin & Merle, pp. 286–302 |
| `article4_relperm_resistivity_fractal` | Relative permeability from resistivity for fractal media: pore-size fractal PDF (Eq. 1), pore fractal dimension Df = De − ln φ/ln(rmin/rmax) (Eq. 2 → 2.767 base case), Archie resistivity index (Eq. 11), fractal/Brooks-Corey wetting & nonwetting rel-perm (Eqs. 22, 24, λ = De − Df), and the kr-from-resistivity-index relationship (Eq. 23) | Shi, Meng, Liu, Zhang & Wang, pp. 303–317 |
| `article5_porosity_drilling_ann` | *No equations:* a feed-forward tanh ANN predicting porosity from six drilling parameters (ROP, WOB, RPM, torque, GPM, SPP), scored by the correlation coefficient R and RMSE — reaches R ≈ 0.98 / RMSE ≈ 0.01 (paper: R ≈ 0.94–0.96, RMSE ≈ 0.018–0.035) | Al-AbdulJabbar, Al-Azani & Elkatatny, pp. 318–334 |

DOI pattern: `10.30632/PJV61N3-2020aN` (N = 1 … 5). The source PDF has a text layer, so titles/authors/DOIs/constants are from the paper bodies; most equations are standard-form reconstructions (the typeset glyphs were dropped in extraction). See `src2020_06/README.md`.

---

## src2020_08 — Vol. 61, No. 4 (August 2020)

A compact regular issue of four papers spanning the flexural attenuation technique for cased-hole annulus evaluation, the effect of clay minerals and pore-water conductivity on the saturation exponent of clay-bearing sandstones (digital rock), petrophysical-property improvement of tight reservoirs using thermochemical fluids, and knowledge-driven hierarchical clustering for specific-facies detection. This issue's source PDF (`Petrophysics_2020_08.pdf`) has a text layer, so titles, authors, page ranges, DOIs, equation numbers, variable definitions, and many numeric constants were read directly from the paper bodies; the conversion dropped most typeset formula glyphs (only Article 3's Eq. 5 and reaction survived verbatim), so the numbered formulas are faithful standard-form reconstructions from the preserved variables and the standard textbook expressions each paper cites. See the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_flexural_attenuation_casing` | Flexural attenuation for cased-hole annulus evaluation: plane-wave phase shift / phase velocity (Eqs. 1–2), Snell optimal incidence sin θ = vf/vφ (Eq. 3, reproduces 30° from 1325 / 2650 m/s), amplitude-ratio attenuation 20·log₁₀(A₁/A₂) and coefficient α (Eq. 4), TIE annulus thickness x_a = s_a·cos θ (Eqs. 5–7), and a cosine eccentricity fit (Eq. 8) | Sirevaag, Johansen, Larsen & Holt, pp. 334–351 |
| `article2_saturation_exponent_clay_digitalrock` | Saturation exponent of clay-bearing sandstone: Archie F = a·φ⁻ᵐ and I = Sw⁻ⁿ (Eqs. 1–2), Waxman-Smits C₀ = (Cw + B·Qv)/F* (Eqs. 3–4), cation mobility B(Cw) (Eq. 5), Qv from CEC (Eq. 6), and a partial-saturation conductivity whose log I–log Sw slope gives the apparent n — shows clay lowers n (clean 2 → ~1.0 for high-CEC clay) and high Cw dilutes the effect | Fan, Pan, Guo & Lei, pp. 352–362 |
| `article3_thermochemical_stimulation` | Thermochemical stimulation of tight rocks: the exothermic NaNO₂ + NH₄Cl reaction and its heat (ΔH = 369 kJ/mol), improvement ratios, dynamic moduli E/ν/K/µ from Vp,Vs,ρ (Eqs. 1–2, 6–7), Young-Laplace (Eq. 3) and centrifuge (Eq. 4) capillary pressures, and the scratch energy Ft = E·A (Eq. 5) — reproduces porosity +80% / perm +1359.9% (limestone) and UCS 38.2 → 17.1 MPa | Mustafa, Mahmoud, Abdulraheem, Tariq & Al-Nakhli, pp. 363–382 |
| `article4_kdhc_facies_clustering` | Knowledge-driven hierarchical clustering for facies: neutron-density separation ND (Eq. 1), the expert baffle rule (ZDN > 2.55 and MLR > 15), cluster-area / indicator / purity (P = A_E4/A_C) / decision (D = K/N, stop at 1) metrics (Eqs. 2–6) and the F1 score (Eq. 7), with a k-means + silhouette splitter proxy — reproduces the ~0.98 reservoir F1 | Emelyanova, Peyaud, Dance & Pervukhina, pp. 383–400 |

DOI pattern: `10.30632/PJV61N4-2020aN` (N = 1 … 4). The source PDF has a text layer, so titles/authors/DOIs/constants are from the paper bodies; most equations are standard-form reconstructions (the typeset glyphs were dropped in extraction). See `src2020_08/README.md`.

---

## src2020_10 — Vol. 61, No. 5 (October 2020)

A regular issue of seven papers spanning nanoindentation of shale cuttings and its upscaling to core, classification of shale N₂-adsorption-isotherm curves by pore structure, automatic wellbore cave-in detection by unsupervised clustering, a petrophysically consistent Archie's equation for heterogeneous (vuggy) carbonates, wettability and water-blockage in organic-rich tight rocks, neural-network prediction of sonic transit times from drilling parameters, and an integrated multiphysics rock-classification workflow. This issue's source PDF (`Petrophysics_2020_10.pdf`) has a text layer, so titles, authors, page ranges, DOIs, equation numbers, variable definitions, and many numeric constants were read directly from the paper bodies; the conversion dropped most typeset formula glyphs (keeping the equation numbers and prose), so the numbered formulas are faithful standard-form reconstructions from the preserved variables and the standard textbook expressions each paper cites. See the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_nanoindentation_shale` | Nanoindentation of shale cuttings: Oliver-Pharr hardness H = Pmax/Ac and indentation modulus M = (√π/2)·S/(α·√Ac) (Eqs. 1a–1b, Berkovich α = 1.03), Young's modulus Es = M·(1−ν²) (Eq. 2), ideal Berkovich area Ac = 24.5·hc², and the Johnson plastic-zone radius (Eq. 4) bounding indent spacing — reproduces the ~20 GPa basis and < 6% Poisson sensitivity | Esatyana, Sakhaee-Pour, Sadooni & Al-Kuwari, pp. 404–416 |
| `article2_adsorption_isotherm_classification` | *Classification (no equations):* BET linearization for monolayer volume / surface area, the IUPAC five-type classifier, micro/meso/macro pore-size classes, a sorting class from the PSD spread, and the paper's new three-parameter (shape × size × sorting) scheme giving 27 curve types | Tian, Chen, Yan, Deng & He, pp. 417–433 |
| `article3_cavein_clustering_detection` | Wellbore cave-in detection: rolling coefficient of variation of bulk density (Eq. 5) and caliper rugosity (Eq. 1) as features, with the TICC good/bad-hole clustering (Eqs. 2–4) represented by a Gaussian k-means proxy + temporal smoothing — recovers a planted cave-in zone with > 80% recall, < 10% false flags | Sen, Ong, Kainkaryam & Sharma, pp. 434–449 |
| `article4_archie_carbonate_consistent` | Consistent Archie for vuggy carbonates: Archie / R0 / resistivity index (Eqs. 1a–1c), effective cementation exponent from F (Eq. 6), and a symmetric-Bruggeman homogenization (Eqs. 4–5) — shows m varies with vug fraction (separate vugs elevate m above 2) and the vuggy resistivity index rises with an effective n well below 2 (near unity) | Ramamoorthy, Ramakrishnan, Dasgupta & Raina, pp. 450–472 |
| `article5_wettability_water_blockage` | Wettability & water blockage: Young-Laplace capillary pressure and Washburn pore-throat radius, the water-wet / oil-wet / mixed pore-type fractions from spontaneous-imbibition volumes (Eqs. 1–3), and the trapped-water saturation with the ~1,500 psi threshold to restore oil continuity within a 7,000 psi step-pressurization | Mukherjee, Dang, Rai & Sondergeld, pp. 473–481 |
| `article6_sonic_transit_drilling_nn` | Sonic transit time from drilling parameters: a compact single-hidden-layer tanh ANN predicting Δt from six surface parameters (WOB, RPM, ROP, torque, SPP, GPM) scored by R and AAPE, then dynamic Poisson's ratio and Young's modulus from Vp, Vs, ρ (Eqs. 1–2) — reaches R ≈ 0.99 / AAPE ≈ 1.3% (paper: R ≈ 0.94 / AAPE ≈ 1–1.9%) | Gowida & Elkatatny, pp. 482–494 |
| `article7_multiphysics_rock_classification` | Multiphysics rock classification: mean gray level (Eq. 1), GLCM contrast & energy (Eqs. 2–3), the experimental variogram for window selection (Eq. 5), the silhouette coefficient (Eq. 6), and k-means classification with the permeability cost function (Eq. 7) whose convergence picks the optimum class count (matching the three formations) | Gonzalez, Kanyan, Heidari & Lopez, pp. 495–518 |

DOI pattern: `10.30632/PJV61N5-2020aN` (N = 1 … 7). The source PDF has a text layer, so titles/authors/DOIs/constants are from the paper bodies; most equations are standard-form reconstructions (the typeset glyphs were dropped in extraction). See `src2020_10/README.md`.

---

## src2020_12 — Vol. 61, No. 6 (December 2020)

The **"Pulsed-Neutron Logging in the 2020s: Smarter, Faster, and Much More Powerful"** special issue on nuclear spectroscopy — a historical review followed by six papers spanning formation chlorine / water-salinity measurement, self-compensated pulsed-neutron spectroscopy, two multidetector saturation case studies (Malaysia, Indonesia), through-casing TOC and oil saturation from excess carbon (South Kuwait), and gas-pressure assessment through casing. Unlike the scanned issues, this issue's source PDF (`Petrophysics_2020_12.pdf`) has a text layer, so titles, authors, page ranges, DOIs, equation numbers, variable definitions, and many numeric constants were read directly from the paper bodies; the conversion dropped most typeset formula glyphs (keeping the equation numbers and prose), so the numbered formulas are faithful standard-form reconstructions from the preserved variables and constants — except Article 5's Eq. 3, whose text survived verbatim. See the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_nuclear_spectroscopy_history` | *Historical review (no equations):* the canonical nuclear-logging relations it surveys — macroscopic capture cross section Σ = ΣᵢNᵢσᵢ (capture units), thermal-neutron die-away Σ = 4550/τ, number density from bulk density, the carbon/oxygen ratio, and a K/U/Th spectral-gamma sum — reproduces the ~22 c.u. fresh-water sigma from H/O number densities | Pemper, pp. 523–548 |
| `article2_formation_chlorine_salinity` | Formation chlorine → water salinity: yields-to-weights W = FY2W·S·Y (Eq. 6), chlorine yield split Y_Cl = Y_form + Y_bh (Eq. 1) with the CYDCL / Φ(env)=1/f borehole subtraction (Eqs. 4–9), DWCL → NaCl-salinity / BVW / Sw (Eqs. 11–14) via the 1.649 molar-mass ratio, and macroscopic sigma mixing / Σmax (Eqs. 19–20) — uses the paper's 567 c.u. per (g/cc) Cl, 22 c.u. fluid, 29.4 c.u. shale | Miles, Mossé & Grau, pp. 549–569 |
| `article3_self_compensated_spectroscopy` | Self-compensated spectroscopy: yields-to-weights (Eq. 1), a FY2W predictor from raw measurements (rising with hole size, smaller far-detector inelastic FY2W), and the differential near-over-far dry-weight element that cancels a common borehole contribution (recovers formation Ca independent of cement Ca) | Zhou, Rose, Miles, Gendur, Wang & Sullivan, pp. 570–584 |
| `article4_co_sigma_saturation_casestudy` | *Case study (no equations):* standard pulsed-neutron saturation — salinity-independent C/O-ratio oil saturation by water/oil-endpoint interpolation, sigma water saturation from the volumetric porosity balance, and a near/far multidetector gas indicator | Johare, Mohd Amin, Prasodjo, Afandi & Din, pp. 585–599 |
| `article5_through_casing_toc_saturation` | Through-casing TOC & saturation: linear multimineral response (Eq. 1) solved by closure-constrained weighted least squares (Eq. 2), excess carbon XCarbon = CTot − (CMin + CMat) (Eq. 3, verbatim), and the calibration-free oil saturation So = ρb·Xc/(ρo·Fc·φe) (Eq. 4) | Bouchou, Abughneej, Ghioca, Alarcon & Mendez, pp. 600–609 |
| `article6_pulsed_neutron_gas_pressure` | Gas pressure through casing: bulk gas sigma Σ = ρ_bulk·Σₑ(wₑσₑ) (Eq. 1, proportional to gas density) inverted through a real-gas density law ρ = PM/(zRT) — reproduces the case study's ~2,785 psi from the measured sigma | Cavalleri, Brouwer, Kodri, Rose & Brinks, pp. 610–622 |
| `article7_sigma_gas_saturation_lowporosity` | Sigma gas saturation in low-porosity shaly rock: the clean (Eq. 1) and shaly (Eq. 2) sigma porosity-balance saturations using the paper's endpoints (Σ_ma 7.5, Σ_sh 27, Σ_w 24, Σ_g 3 c.u.; φ ≈ 12 p.u.), plus the low-porosity sensitivity caveat |dΣ/dSg| = φ(Σ_w − Σ_g) | Wijaya, Aulianagara, Guo, Naibaho, Asriwan & Amirudin, pp. 623–632 |

DOI pattern: `10.30632/PJV61N6-2020aN` (N = 1 … 7). The source PDF has a text layer, so titles/authors/DOIs/constants are from the paper bodies; most equations are standard-form reconstructions (the typeset glyphs were dropped in extraction), with Article 5's Eq. 3 verbatim. See `src2020_12/README.md`.

---

## src2021_02 — Vol. 62, No. 1 (February 2021)

A regular issue opening with an invited tutorial on extracting net pay from mudlogs, followed by eight papers spanning downhole-fluid-analysis lateral gradients and reservoir mixing over geologic time, weak bedding planes in the Marcellus Shale, fracture-fill identification with dielectric imaging in oil-based mud, formation-tester sampling of CO₂ and other reactive components, an integrated NMR/resistivity/pressure carbonate case study, high-resolution dual-ultrasonic LWD slowness and imaging, multiwell electromagnetic 3D inversion of sand injectites, and a dual neural network for permeability with uncertainty. This issue's source PDF (`Petrophysics_2021_02.pdf`) has no usable text layer (reading it returns empty text), so the article titles, authors, page ranges, and DOIs are taken verbatim from the official SPWLA issue table of contents and the numbered formulas are faithful standard-form reconstructions of the well-established methods each paper applies. See the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_mudlog_net_pay_tutorial` | *Tutorial:* net pay from mudlogs — gas normalization to rock volume GN = G·Q/(ROP·A), the Haworth wetness/balance/character ratios with their productivity bands, the Pixler light-component ratios, and an integrated gas/porosity/Vsh/Sw cutoff scheme summing net pay and net-to-gross | Malik, Hanson & Clinch, pp. 4–15 |
| `article2_dfa_lateral_gradients_mixing` | DFA fluid gradients & geologic-time mixing: the Flory-Huggins-Zuo gravity term OD(z₂)/OD(z₁) = exp[v_a·g·(ρ_a−ρ_f)·Δz/RT] giving the equilibrium asphaltene gradient, a 1D diffusion model (erfc step front and the H²/(π²D) homogenization time), and an equilibrium-vs-disequilibrium connectivity diagnosis | Chen, Kristensen, Johansen, Achourov, Betancourt & Mullins, pp. 16–30 |
| `article3_marcellus_weak_bedding_planes` | Weak bedding planes: Jaeger's single-plane-of-weakness slip strength, intact Mohr-Coulomb strength, their combination into the U-shaped strength-vs-bedding-angle curve with the minimum at β = 45° + φ_w/2, and a mud-weight floor that suppresses bedding-parallel shear (the mitigation strategy) | Kowan, Schanken & Jacobi, pp. 31–44 |
| `article4_obm_dielectric_fracture_fill` | Fracture fill from dielectric imaging in OBM: the CRIM permittivity mixing law √ε = Σφ_i√ε_i, complex permittivity with the σ/(ωε₀) conduction term, the loss tangent flagging conductive fill, a thin-gap button admittance, and a classifier separating open (oil/mud), calcite-cemented, and conductive (clay/brine) fills | Schlicht, Zhang, Lüling, Graham, Cournot & Sadownyk, pp. 45–64 |
| `article5_formation_tester_co2_sampling` | *Short operational paper:* CO₂/reactive-component sampling proxies — the power-law (V^−5/12) cleanup of OBM contamination, CO₂ phase identification against the critical point (31 °C / 73.8 bar), Henry's-law CO₂ solubility in brine with a Sechenov salting-out factor, and a mass-balance correction recovering the in-situ CO₂ fraction | Piazza, Vieira, Sacorague, Jones, Dai, Pearl & Aguiar, pp. 65–72 |
| `article6_nmr_resistivity_pressure_carbonate` | *Case study:* Archie Sw and formation factor, Timur-Coates and SDR NMR permeability, the Buckles bulk volume water, fluid density from a pressure gradient ρ = (dP/dz)/g, and a fluid contact recovered from two intersecting pressure-gradient lines (round-trips a planted OWC to < 1 m) | Li, Drinkwater, Whittlesey & Condon, pp. 73–88 |
| `article7_lwd_dual_ultrasonic_slowness` | Dual-ultrasonic LWD: slowness-time-coherence (semblance) processing over a receiver array with slowness picking (recovers a planted 80 µs/ft headwave), and acoustic impedance Z = ρv with the normal-incidence reflection coefficient R = (Z₂−Z₁)/(Z₂+Z₁) for pulse-echo imaging | Blyth, Sakiyama, Hori, Yamamoto, Nakajima, Fahim Ud Din, Haecker & Kittridge, pp. 89–108 |
| `article8_injectite_em_3d_inversion` | Multiwell EM 3D inversion of injectites: the EM skin depth δ = 503·√(ρ/f), a straight-path cross-well sensitivity operator, and a Tikhonov (smoothness-regularized) least-squares inversion m = (GᵀG + λLᵀL)⁻¹Gᵀd recovering a resistive injectite from a crossing horizontal+vertical fan survey | Clegg, Eriksen, Best, Tollefsen, Kowicki & Marchant, pp. 109–121 |
| `article9_dual_nn_permeability_uncertainty` | Dual neural network for permeability + uncertainty: a compact two-head MLP (shared tanh hidden layer; mean + log-variance heads) trained with the heteroscedastic Gaussian NLL L = ½·mean[(y−μ)²e^−s + s], predicting log-permeability and a calibrated uncertainty that grows in less-informative low-porosity rock | Kausik, Prado, Gkortsas, Venkataramanan, Datir & Johansen, pp. 122–134 |

DOI pattern: `10.30632/PJV62N1-2021aN` (N = 1 … 8), plus the tutorial `10.30632/PJV62N1-2021t1`. The source PDF has no text layer, so titles/authors/DOIs are from the official SPWLA table of contents and equations are standard-form reconstructions. See `src2021_02/README.md`.

---

## src2021_04 — Vol. 62, No. 2 (April 2021)

A regular issue of five papers spanning NMR pore-structure characterization of a complex carbonate, a deepwater-turbidite rock-typing case study, Thomeer/NMR free-vs-bound porosity partitioning, nonlinear-acoustics noncollinear wave mixing for near-wellbore evaluation, and an integrated NMR continuous/stationary fluid-and-contacts workflow. This issue's source PDF has no usable text layer, so the modules were built by rendering the PDF pages to images and reading them visually — the equations are transcribed from the genuinely rendered math (Article 1's NMR relaxation equations, Article 3's Thomeer/Swanson equations, and Article 4's nonlinear-acoustics wave-mixing equations are verbatim). Articles 2 and 5 are a case study and a workflow paper. See the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_nmr_carbonate_porestructure` | NMR pore-structure of a complex carbonate: multi-exponential relaxation 1/T2 = 1/T2bulk + 1/T2surf + 1/T2diff (Eq. 1), diffusion relaxation rate 1/T2diff = D·γ²·g²·TE²/12 (Eq. 2), porosity correction φ_corr = φ + 0.3·Vol_largepore (Eq. 3); single-pore forward model (sphere S/V = 3/r), large-pore T2-cutoff (847 ms) partition, Timur-Coates / SDR permeability — reproduces the (900/200)² = 20.25 inter-tool diffusion ratio | Saidian, Jain & Milad, pp. 138–155 |
| `article2_turbidite_rock_typing` | *Case study:* deepwater-turbidite rock typing — Winland-R35 regressions from core CT (Eq. 1) and from logs (Eq. 2), R35 pore-throat rock-type classifier (RT-1..RT-4), per-rock-type irreducible-saturation lookup, and the Waxman-Smits Co = (1/F*)(Cw + B·Qv) conductivity line | Angel Restrepo, Gómez-Moncada, Mora Sánchez & Bueno Silva, pp. 156–174 |
| `article3_thomeer_nmr_partitioning` | Thomeer & NMR free-vs-bound partitioning: Thomeer hyperbola Shg = Bv·exp(−G/(logPc−logPd)) (Eq. 1), normalized porosity (Eq. 2), RQI/FZI (Eq. 3), Swanson permeability Ka = 3.8068·G^(−1.3334)·(Bv/Pd)² (Eq. 4) and its inversion for G (Eq. 5), Washburn pore-throat radius, and the NMR↔MICP calibration C = T2·Pc tying the 0.3-µm / 14-ms cutoffs | Gianotten, Rameil, Foyn, Kollien, Marre, Looyestijn, Zhang & Hebing, pp. 175–194 |
| `article4_nonlinear_acoustics_mixing` | Noncollinear acoustic wave mixing: cubic nonlinear stress-strain (Eq. 1), nonlinearity parameter β (Eq. 2), convergence angle (Eq. 3) and scattering angle (Eq. 4), exact (Eqs. 5–8) and approximate (Eq. 9) scattering coefficients, and the frequency-ratio validity rule (Eq. 10) — reproduces Table 1's φ = γ = 47.5° at ω₂/ω₁ = 0.74 | Skelt, TenCate, Guyer, Johnson, Larmat, Le Bas, Nihei & Vu, pp. 195–209 |
| `article5_nmr_fluid_contacts` | *Workflow paper:* canonical NMR relations the workflow relies on — full T2 relaxation (bulk + surface + diffusion), T1 relaxation, hydrogen-index porosity correction (~11% uplift), clay-bound/capillary/free T2-cutoff partition (3 ms, 60 ms), D-T2 fluid typing (gas/water/oil), and the √(stacks) station-stacking SNR gain | Kozlowski, Chakraborty, Jambunathan, Lowrey, Balliet, Engelman, Ånensen, Kotwicki & Johansen, pp. 210–226 |

DOI pattern: `10.30632/PJV62N2-2021aN` (N = 1 … 5). The source PDF had no text layer, so equations were read from rendered pages; descriptive articles (2, 5) implement the quantitative relations the papers rely on. See `src2021_04/README.md`.

---

## src2021_06 — Vol. 62, No. 3 (June 2021)

A regular issue opening with an invited tutorial on sidewall coring, followed by five papers spanning NMR restricted-diffusion pore characterization, AI prediction of acoustic velocities while drilling, machine-learning sonic-shear processing, the first LWD co-located-antenna anisotropy/dip tool, and proactive geosteering with 2D structural analysis. This issue's source PDF uses broken embedded-font encodings (machine text extraction yields multi-font cipher garbage), so the modules were built by rendering the PDF pages to images and reading them visually — meaning the equations are transcribed from the genuinely rendered math (Article 2's NMR Padé equations and Article 4's VTI/ANNIE stiffness relations are verbatim). Articles 1, 5, 6 are descriptive (tutorial / instrument-introduction / case study). See the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_sidewall_coring_tutorial` | *Tutorial (descriptive)* on a century of sidewall coring: the Fig. 14 rotary-coring tool table as structured data; cylindrical core-plug volume V = π(d/2)²L; per-run recovered volume; tool selection by pressure/temperature rating — reproduces worked plug volumes (MaxCOR 4.42 in³, XL-Rock 6.19 in³) | Jackson, pp. 230–243 |
| `article2_nmr_restricted_diffusion` | Pore size / tortuosity / permeability from NMR restricted diffusion: diffusion length L_D = √(D0·t) (Eq. 9), cylindrical S/V = 4/d (Eq. 10), Padé interpolation bridging short-time Mitra and long-time tortuosity limits (Eq. 11), modified Carman-Kozeny k = (φ/32)d²/(BTR²τ) (Eq. 3), Timur-Coates & SDR permeability (Eqs. 1–2), electrical/diffusive tortuosity (Eqs. 4–7) — grid-search round-trips pore size and tortuosity | Wang, Singer, Liu, Chen, Hirasaki & Vinegar, pp. 244–264 |
| `article3_ai_acoustic_velocity` | Real-time Vp/Vs prediction from surface drilling parameters: Spearman rank correlation (Eq. 1), AAPE (Eq. 2), correlation coefficient R (Eq. 3), min-max normalization, the nine Appendix-1 empirical Vs-from-Vp correlations (Pickett/Carroll/Castagna/Brocher), and a numpy linear-regression surrogate for the ANN/RF predictor | Alsaihati & Elkatatny, pp. 265–281 |
| `article4_ml_sonic_shear` | Machine-learning-enabled dipole-flexural shear processing: ANNIE VTI stiffness relations (Eqs. 1–7) incl. Poisson ratio (Eq. 6) and Thomsen γ (Eq. 7), RMAD validation metric (Eq. 9) and inversion misfit (Eq. 10), plus a surrogate flexural-dispersion forward model and DTS inversion standing in for the NN proxy / mode-search solver | Liang & Lei, pp. 282–295 |
| `article5_lwd_colocated_antenna` | *Instrument introduction:* tilted-antenna magnetic-moment projection onto tool axes; 3×3 magnetic-tensor coupling V = m_R·H·m_T; standard propagation-resistivity attenuation / phase-shift relations (skin depth) with apparent-resistivity inversion (the EM forward response itself is from the cited LWD-resistivity literature) | Bittar, Wu, Ma, Pan, Fan, Griffing & Lozinsky, pp. 296–310 |
| `article6_geosteering_2d_structural` | *Case study:* borehole-geometry relations the workflow relies on — MD→TVD, boundary TVD from a distance-to-boundary pick, apparent↔true dip, structural dip from two picks, least-squares fault-plane fit to dip/azimuth (recovers the Table 1 OBc 44°/23° fault), and net-pay (reservoir-contact) accounting along a lateral | Antonov, Kushnir, Martakov, Pazos, Small, Tropin, Maraj, Itter, Nelson & Rabinovich, pp. 311–330 |

DOI pattern: `10.30632/PJV62N3-2021aN` (N = 1 … 5), plus the tutorial `10.30632/PJV62N3-2021t1`. Equations were read from rendered PDF pages (the source PDF's font encoding is broken); descriptive articles (1, 5, 6) implement the quantitative relations the papers rely on. See `src2021_06/README.md`.

---

## src2021_08 — Vol. 62, No. 4 (August 2021)

A regular issue opening with an invited tutorial on thinly bedded formations, followed by seven papers spanning deep-Q-learning depth matching, NMR fluid substitution, borehole-sonic dispersion analysis, a machine-learning synthetic-sonic contest, an oil-based-mud resistivity imager, an acoustic volcanic-rock saturation model, and the capillary-pressure / resistivity-index relationship in tight sandstones. Throughout the issue the typeset equations were image-rendered and did not survive text extraction, so the numbered formulas are faithful standard-form reconstructions of the methods the prose describes; the deep-learning / proprietary methods (articles 2, 4, 5) are represented by compact numpy implementations of the same underlying method. See the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_thinly_bedded_petrophysics` | Tutorial on laminated reservoirs: parallel (horizontal) conductivity 1/Rh = Σ(v_i/R_i) (Eq. 2) and series (vertical) resistivity Rv = Σ(v_i·R_i) (Eq. 3); anisotropy λ = √(Rv/Rh); Moran-Gianzero apparent resistivity vs relative dip (Eq. 1); sand-resistivity inversion from Rh vs Rv (series route robust to shale-volume error); Thomeer capillary-pressure curve (Eq. 4) — reproduces the worked Rh→1.82 / Rv=5.5 / Rss=10 ohm-m | Aldred, pp. 335–352 |
| `article2_depth_matching_deep_q` | Well-log depth matching as an MDP solved with Q-learning: Bellman update Q(s,a) ← Q + α(r + γ·maxQ′ − Q); ε-greedy policy; shift-action space with a stop action; reward driving the agent to the match point — the paper's CNN Rainbow-DQN is replaced by a compact tabular Q-learner (γ=0.99) | Bittar, Wang, Wu & Chen, pp. 353–361 |
| `article3_nmr_fluid_substitution` | Reconstructing Sw=1 NMR T2 distributions from partially-saturated ones: surface relaxation 1/T2 = 1/T2bulk + ρ·(S/V); T2→pore-radius r = Fs·ρ·T2; BVI/BVM split at the 33-ms cutoff; porosity-conserving fluid substitution that re-amplifies the movable-water peak by 1/Sw_eff | Li, Kesserwan, Jin & Ma, pp. 362–378 |
| `article4_sonic_dispersion_dpsm` | Multimode borehole-sonic dispersion via a modified differential-phase semblance: frequency-slowness phase-coherence semblance over a receiver array (phase back-propagation exp(i·2πf·s·z)); group delay T(f) = −(1/2π)dφ/df; slowness extraction — recovers a known slowness from a synthetic array | Wang, Coates & Zhao, pp. 379–392 |
| `article5_synthetic_sonic_ml_contest` | SPWLA 2020 synthetic-sonic ML contest: pooled DTC+DTS RMSE scoring metric (Eq. 1, benchmark 17.93), per-log RMSE and R²; z-score / min-max normalization; log-resistivity transform; numpy linear-regression baseline (five contest models summarized in the README) | Yu, Xu, Misra, Li, Ashby et al., pp. 393–406 |
| `article6_obm_resistivity_imager` | High-fidelity oil-based-mud resistivity imager: parallel-RC element values R = kb·ρ, C = ε·ε₀/kb (Eq. 1); complex button impedance Z = R/(1+jωRC) (Eq. 2); apparent impedivity ξ = Z/kb with low-ρ limit Re(ξ)≈ρ (Eqs. 3–4); capacitive oil-mud term; DC-conductivity / dielectric-loss decoupling σ = σ_DC + ω·ε″·ε₀ (Eq. 6) — reproduces dielectric rollover and ~−90° mud phase | Guner, Fouda, Ewe, Torres & Barrett, pp. 407–421 |
| `article7_volcanic_saturation_gassmann` | Acoustic (not electrical) volcanic-rock saturation model: Gassmann equation (Eq. 1); Wood-Lindsay (Reuss) / Domenico (Voigt) / Brie fluid moduli (Eqs. 2–4); White patchy modulus (Eq. 5); Gassmann-Brie-Patchy blend (Eq. 6); Vp from (K, μ, ρ) — confirms patchy is the upper velocity bound, uniform the lower, converging at Sw = 0 and 1 | Pan, Zhou, Guo, Si & Lin, pp. 422–433 |
| `article8_capillary_resistivity_index` | Capillary pressure vs resistivity index in tight sandstone: Archie index I = Sw^(−n) and formation factor F = φ^(−m) (Eqs. 1–2); Waxman-Smits clay-corrected index (Eq. 5); Li & Williams power-law Pc = Pe·I^β (Eq. 9) and Szabo linear (Eq. 6) models; Toledo fractal Pc = Pew·Sw*^(−1/λ), λ=3−D (Eq. 18); Washburn throat radius; β(k) and b(k) regressions | Xiao, Yang, Li, Yang, Bernabé, Zhao, Li & Ren, pp. 434–446 |

DOI pattern: `10.30632/PJV62N4-2021aN` (N = 1 … 7), plus the tutorial `10.30632/PJV62N4-2021t1`. Equations are standard-form reconstructions (typeset glyphs were image-rendered in the source PDF). See `src2021_08/README.md`.

---

## src2021_10 — Vol. 62, No. 5 (October 2021)

The special issue on **"Applications of 3D Printing and Synthetic Rocks in Petrophysics, Rock Physics, and Rock Mechanics"** — nine papers spanning binder-saturation control of 3D-printed sandstone porosity, image-processing petrophysics education, original-size carbonate pore replication, 3D-printed mudrock micromodels, fractal characterization of digital rocks, pore-volume compressibility of unconsolidated sands, fluid effects on the elastic properties of printed anisotropic rock, joint-roughness shear behavior, and near-wellbore perforation fracturing. Article 9 was only partly present in the source PDF (truncated mid-results; an experimental study with no equations), so its module is a methodology proxy; and throughout the issue the typeset equations were image-rendered and did not survive text extraction, so the numbered formulas are faithful standard-form reconstructions of the methods the prose describes. See the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_binder_saturation_porosity` | Binder-saturation control of 3D-printed sandstone porosity: printed cylinder volume V = π(d/2)²h (Eq. 2); binder volume from burnout; binder volume fraction f_b = V_binder/V_total (Eq. 4); binder saturation level S = f_b/void (Eq. 5); theoretical porosity trend φ = φ₀(1−S) — reproduces 36/34/32% porosity at 10/15/20% saturation | Hodder, Craplewe, Ishutov & Chalaturnyk, pp. 450–462 |
| `article2_image_processing_petrophysics` | Image-processing petrophysics education: porosity as pore-pixel fraction; phase saturation; irreducible/residual saturation; displacement efficiency E_D = (S_oi−S_or)/S_oi (Eq. 1); equivalent grain radius r = √(A/π); contact-angle wettability rule — reproduces φ = 27.01%, S_wir = 0.332, S_or = 0.226, E_D ≈ 66% | Alyafei, Al Musleh, Bautista, Idris & Seers, pp. 463–476 |
| `article3_carbonate_pore_replication` | Original-size carbonate pore replication: pore-size scaling d_model = S·d_original (1:1 vs prior 5×); equivalent spherical pore diameter d_eq = (6V/π)^(1/3); cylinder bulk volume; scaffolding print-time speedup (technical note — no published equations) | Ishutov, Hodder, Chalaturnyk & Zambrano-Narvaez, pp. 477–485 |
| `article4_3dprint_mudrock_micromodel` | 3D-printed mudrock micromodels: Washburn pore-throat diameter D = −4γcos(θ)/P (Eq. 1) matching the paper's anchors (a few psi → tens of µm; 33,000 psi → single-digit nm); Boyle's-law grain volume; porosity φ = (V_bulk−V_grain)/V_bulk; firing dimensional/mass loss | Hasiuk & Harding, pp. 486–499 |
| `article5_fractal_digital_rock` | Fractal characterization of digital rocks: box-counting fractal dimension log N(r) = D·log(1/r)+c (Eq. 3, validated on a Sierpinski carpet → log8/log3 = 1.893); permeability power laws K(φ)/K(D)/K(Su) (Eqs. 10–12); Archie formation factor F = φ^(−m) and m inversion (Eq. 13); gliding-box lacunarity (Eq. 5) | Zhao, Luo, Li, Wu, Mao & Ostadhassan, pp. 500–515 |
| `article6_pore_volume_compressibility` | Pore-volume compressibility of unconsolidated sands: uniaxial compaction coefficient Cm = (1/L)(dL/dσ_a) (Eq. 1); pore-volume compressibility Cp = Cm/φ (Eq. 2); Trask sorting coefficient So = √(GS25/GS75) (Eq. 3); a peaked Cm-vs-effective-stress demonstrator (Regions A/B/C) | Hathon, Myers & Arya, pp. 516–536 |
| `article7_3dprint_anisotropic_elastic` | Fluid effects on 3D-printed anisotropic rock elasticity: saturated density ρ = ρ_m(1−φ)+ρ_f·φ (Eq. 1); velocity from traveltime; isotropic moduli K/G/E/ν; Thomsen ε/γ anisotropy; Gassmann fluid substitution; Vp/Vs and impedance — reproduces G = 0.39 / K = 2.65 GPa / ν = 0.43 and air ε ≈ 0.26 | Dande, Stewart & Dyaur, pp. 537–552 |
| `article8_joint_roughness_shear` | Joint-roughness shear behavior of 3D-printed samples: Z2 RMS profile slope; Tse & Cruden (1979) JRC = 32.2+32.47·log10(Z2); Barton-Bandis peak shear strength τ = σ_n·tan(φ_b + JRC·log10(JCS/σ_n)); Mohr-Coulomb; secant shear stiffness — reproduces the τ-vs-JRC and τ-vs-normal-stress trends | Fereshtenejad, Kim & Song, pp. 553–563 |
| `article9_perforation_fracture_morphology` | *Methodology proxy* for the near-wellbore perforation-fracturing paper (body truncated in available PDF extract; experimental study with no equations): perforation-mode classification (spiral/directional/fixed-plane/interlaced); three-microfracture-type taxonomy; standard near-wellbore Kirsch hoop stress and tensile breakdown pressure P_b = 3σ_h−σ_H−P0+T | Wang, Li, Xu, Jia & Zhang, pp. 564–580 |

DOI pattern: `10.30632/PJV62N5-2021aN` (N = 1 … 9). Equations are standard-form reconstructions (typeset glyphs were image-rendered in the source PDF); Article 9 is a methodology proxy. See `src2021_10/README.md`.

---

## src2021_12 — Vol. 62, No. 6 (December 2021)

The **"Best Papers of the 2021 Symposium"** issue — ten papers spanning data quality for petrophysical machine learning, variational-autoencoder mineral quantification from spectroscopy, the SEAT eigenvector dip-analysis technique, deep-learning sedimentary-geometry interpretation from borehole images, density-tool breakout detection behind slotted liner, NanoTag cuttings depth correlation, acoustic multistring isolation evaluation for P&A, overbalanced-drilling core damage and correction, integrated tight-gas characterization, and resistivity-based rock physics for mudrock saturation. Articles 9–10 were available only as table-of-contents entries plus the editor's narrative in the source PDF (the extract truncated partway through Article 8), so their modules are methodology proxies; and throughout the issue the typeset equations were image-rendered and did not survive text extraction, so the numbered formulas are faithful standard-form reconstructions of the methods the prose describes. See the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article01_data_quality_ml` | Data-quality considerations for petrophysical ML: z-score (Eq. 1) and IQR/box-plot (Eq. 2) outlier detection; simple (Eq. 3) and reference-percentile (Eq. 4, Shier 2004) normalization; precision/recall (Eqs. 5–6); MAE/RMSE (Eqs. 7–8); Gaussian-noise injection (Eqs. 9–10); Pearson r (Eq. 11); sentinel→NaN cleaning — reproduces the Table 3 confusion matrix (precision 0.704, recall 0.909) | McDonald, pp. 585–613 |
| `article02_vae_mineral_spectroscopy` | Variational-autoencoder mineral quantification from spectroscopy elements: forward model e = A·m with a stoichiometric element→mineral sensitivity matrix; heteroscedastic Gaussian negative-log-likelihood cost (y−ŷ)²/(2σ²) + ln σ summed over outputs and samples (Eqs. 1–2); non-negative, closure-constrained simplex inversion as the encoder analogue; element reconstruction as decoder QC; matrix grain density | Craddock, Srivastava, Datir, Rose, Zhou, Mosse & Venkataramanan, pp. 614–629 |
| `article03_seat_dip_eigenvectors` | Statistical Eigenvector Analysis Technique (SEAT) for borehole-image dips: dip→pole-to-bedding unit vectors (R1); orientation/scatter matrix T = (1/N)Σ n·nᵀ (R3); eigen-decomposition with the minimum-eigenvalue eigenvector as the slump-fold symmetry axis (R4); Woodcock K/C (R5) and Vollmer P/G/R (R6) fabric indices; eigenvector→trend/plunge (R8); the paper's tilt-invariance claim (axis trend stable under <40° structural tilt) | Ruehlicke, Uhrin, Veselovsky & Schlaich, pp. 630–635 |
| `article04_borehole_image_cnn_sedimentary` | Deep-learning sedimentary-geometry classification from borehole images: sinusoid model of a planar bed on an unrolled image z(φ) = z0 − r·tan(δ)·cos(φ−φ0) (R9) with a least-squares fit recovering apparent dip and dip azimuth; softmax (R5), categorical cross-entropy (R6), accuracy (R7); the four-level Rubin (1987) bedform hierarchy — CNN represented by its analytic geometric core | Lefranc, Bayraktar, Kristensen, Driss, Le Nir, Marza & Kherroubi, pp. 636–650 |
| `article05_density_breakout_behind_casing` | Openhole-equivalent caliper behind slotted liner from the density tool: radial response J = (ρ_app−ρ_fm)/(ρ_ann−ρ_fm) (Eq. 1); tanh radial-response model J(h) = tanh(λh) (Eq. 2); casing-corrected ρ_cc (Eq. 3) and casing+nominal-cement-corrected ρ_CH (Eq. 4); annulus-thickness inversion exact + Taylor (Eq. 5) using the quoted C_SS3 = 0.52, C_LS3 = 1.78; completion/fluid classification by annulus density | Mosse, Pell & Neville, pp. 651–669 |
| `article06_nanotags_cuttings_depth` | NanoTag cuttings depth correlation: volumetric lag-time algebra — upward t_l = v_a/f (Eq. 1) and downward t_d = v_d/f (Eq. 3) lag; conventional t_g = t_c−t_l (Eq. 2) and NanoTag t_g = t+t_d (Eq. 4) generation times; annular capacity from hole/pipe diameters; depth error = ROP·Δt — reproduces t_d ≈ 17 min and a ~2-ft error per 2-min slip at 60 ft/hr | Poitzsch, Zhu, Antoniv, Aljabri & Marsala, pp. 670–680 |
| `article07_multistring_isolation_acoustic` | Acoustic multistring isolation evaluation for P&A (proprietary inversion → physics demonstrator): impedance Z = ρ·v (R1); reflection coefficient R = (Z₂−Z₁)/(Z₂+Z₁) (R2); transmitted energy 1−R² (R3) reproducing the ~95% energy loss through one tubing layer; casing thickness resonance f_n = n·v/(2d) (R4); cement/liquid/gas impedance classification; operational isolation-qualification logic (continuous + cumulative footage) | Zhang, Mueller, Bryce, Brockway & Iskander, pp. 681–693 |
| `article08_overbalanced_drilling_correction` | Overbalanced-drilling core/log damage and correction (case study → standard relations): overbalance ΔP = P_mud−P_pore (INF-1); mud hydrostatic P = 0.0981·ρ_sg·TVD (INF-2); additive porosity correction (INF-4) reproducing the 33% NMR undercall; k–φ semilog transform/fit (INF-7); Klinkenberg correction (INF-8); fraction-of-original overburden correction (INF-6); damage flag φ > 12 p.u. AND k > 100 md | Mohammadlou, Reppert, Del Negro & Jones, pp. 694–710 |
| `article09_tight_gas_neuquen_integrated` | *Methodology proxy* for the integrated tight-gas characterization paper (body not in available PDF extract): clay volume (linear + Larionov older-rocks); density porosity; Archie and Simandoux saturation; Winland r35 dominant pore-throat radius; RQI / normalized porosity / flow-zone-indicator hydraulic units; overpressure pore-pressure gradient (up to ~50% above hydrostatic) | Carrizo, Santiago & Saldungaray, pp. 711–736 |
| `article10_resistivity_rockphysics_wolfcamp` | *Methodology proxy* for the resistivity rock-physics paper (body not in available PDF extract): Archie baseline; Waxman-Smits dual-conductivity saturation for organic-rich mudrock; core-free inversion for the cementation exponent m from a 100%-water zone; hydrocarbon-pore-volume per acre showing the Archie-vs-new reserve improvement (~33% / +70,000 bbl/acre reported) | Dash & Heidari, pp. 737–751 |

DOI pattern: `10.30632/PJV62N6-2021aN` (N = 1 … 10). Equations are standard-form reconstructions (typeset glyphs were image-rendered in the source PDF); Articles 9–10 are methodology proxies guided by the editor's narrative. See `src2021_12/README.md`.

---

## src2022_02 — Vol. 63, No. 1 (February 2022)

Regular issue of six papers spanning a new in-situ Raman composition-logging tool for EOR / CO₂ / hydrogen-storage monitoring, automated well-log depth matching (1D CNN vs. classic cross correlation), an automated log-data-analytics workflow with cross-correlation and dynamic-time-warping depth matching, ultrasonic (pulse-echo + pitch-catch) logging of creeping shale, sand-injectite reservoir evaluation via a modified Thomas-Stieber method, and core-based closed-retort quantification in the Delaware Basin Bone Spring / Wolfcamp. In the source-PDF extract the typeset equations were image-rendered and did not survive text extraction, so the numbered formulas in these modules are faithful standard-form reconstructions of the methods described in the paper bodies (Article 6 publishes no equations and is implemented with standard petrophysical proxies); see the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_raman_logging_eor_gas_storage` | Downhole in-situ Raman composition-logging tool: linear Raman forward model X = G·M·ρ (Eq. 3); ideal-gas number density ρ_m = f_m·P/(k_B·T) (Eq. 4); Lorentz-Lorenz molar-refractivity excitation-volume correction 1 + 3·r_m·ρ; Σf = 1-constrained composition-plus-gain inversion (Eqs. 5–6); Beer-Lambert cross-absorption response X = X₀·exp(−Σα·ρ·L) (Eqs. 7–8); ideal-gas per-zone / per-component flow allocation with P/T/z corrections (Eqs. 1–2) — synthetic five-gas case recovers planted mole fractions and optical gain to machine precision | Andrews & Speck, pp. 4–11 |
| `article2_cnn_xcorr_depth_matching` | LWD-vs-EWL depth matching, 1D CNN vs. cross correlation: normalized cross-correlation alignment lag = argmax_L c(L) (Eq. 1); compact pure-numpy 1D CNN (2-channel conv → ReLU → average-pooled positional bins → flatten → linear, Eqs. 2–5) regressing the shift, trained with MSE on synthetic bulk-shifted 256-sample windows (±20-sample label range); Pearson r (Eq. 6); Euclidean distance (Eq. 7); Ind1% / Ind4% improvement indicators (Eqs. 8–9) — TensorFlow replaced by a small numpy net | Torres Caceres, Duffaut, Yazidi, Westad & Johansen, pp. 13–34 |
| `article3_log_analytics_dtw_xcorr` | Automated log-data-analytics depth-matching engines + QC: cross correlation with optional stretch/squeeze factor α (Eq. 1); constrained dynamic time warping with a Sakoe-Chiba band, local cost d(i,j) = (x_i−y_j)² and normalized accumulated-distance minimisation (Eqs. 2–4); QC metrics — Pearson (A1.1), trace energy TE = Σx² (A1.2), residual energy RE = Σ(x−y)² (A1.3), predictability P = 1 − RE/TE (A1.4) and Euclidean distance (A1.5); scipy / dtaidistance replaced by direct numpy implementations | Torres Caceres, Duffaut, Westad, Stovas, Johansen & Jenssen, pp. 36–61 |
| `article4_ultrasonic_creeping_shale` | Ultrasonic monitoring of shale creeping onto steel casing: pulse-echo group delay τ(ω) = −dφ/dω with φ = arg(S_P/S_N) (Eq. 1); thickness-resonance frequency f_min = 0.95·v_p/(2d) with S1-mode negative-group-velocity correction (Eq. 2); PE and PC empirical impedance calibrations Z(|τ_min|) and Z(α) (Eqs. 3–4) fit to the paper's quoted anchor pairs; Lamb-wave attenuation rate α = (E_T−E)/L; normal-incidence reflection coefficient R = (Z₂−Z₁)/(Z₂+Z₁) reproducing R ≈ −0.95 (kerosene gap) → −0.82 (bonded shale) | Diez, Johansen & Larsen, pp. 61–82 |
| `article5_sand_injectite_thomas_stieber` | Sand-injectite reservoir evaluation (Froskelår Main, Norwegian North Sea): shale-corrected effective porosity PHIE = PHIT − Vsh·PHITsh (Eq. 1); Herron permeability K = Af·exp(ΣBi·Mi) (Eq. 2); CT-scan porosity and grain-density volumetric mixing (Eqs. 3, 5); constant-BVW saturation Sw = BVW/PHIT (Eq. 4); sand counting with Fsd ≥ 0.30 cutoff (Eq. 6); net thickness for bulk and Thomas-Stieber fractional-FNTG methods (Eqs. 7–8); HVOLH for both (Eqs. 9–10); Thomas-Stieber FNTG helper and Poupon-inversion → Archie sand-phase saturation — synthetic three-facies log confirms Thomas-Stieber recovers more HVOLH than bulk analysis in the breccia | Kotwicki, Baig, Johansen, Leirdal, Aftret, Sandstad, Anthonsen, Gianotten, Hansen & Firinu, pp. 83–104 |
| `article6_closed_retort_core_quant` | Core-based closed-retort quantification (Delaware Basin Bone Spring / Wolfcamp): standard petrophysical proxies for the case study's uncalibrated crossplots — Boyle's-law density porosity φ = 1 − ρ_b/ρ_g; fluid-summation porosity; crushing fluid loss (intact − crushed NMR); mass-balance water/oil saturations; NMR-T2 free/bound-water partition at the 10 ms cutoff; Schmoker TOC-from-density (tunable coefficients); open → closed retort collection-efficiency correction (~80% → ~95%) | Perry, Zumberge & Cheng, pp. 105–121 |

DOI pattern: `10.30632/PJV63N1-2022aN` (N = 1 … 6). Equations are standard-form reconstructions (typeset glyphs were image-rendered in the source PDF); Article 6 uses standard petrophysical proxies. See `src2022_02/README.md`.

---

## src2022_04 — Vol. 63, No. 2 (April 2022)

Regular issue containing one *Best of the 2021 Symposium* paper plus six regular submissions, spanning through-tubing casing-deformation imaging with Bayesian GPR inversion, chalk permeability modelling, pyrite-aware water-saturation with Hashin-Shtrikman mineral mixing, time-lapse micro-CT of filter cakes, methane solubility in oil-based mud, gas-hydrate rock physics, and digital-core wellbore stability. Articles 4–7 were only available as table-of-contents entries in the source PDF, so their modules are methodology proxies guided by the editor's letter; see the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_dec_tool_bayesian_gpr` | Through-tubing deformation-and-eccentricity (DEC) electromagnetic tool: linearised magnetostatic transfer function ΔBr = f(Δμ, Δr, Δt) (Eq. 1); casing/tubing flux-density ratio r_flux = ΦC/ΦT (Eqs. 2–3); eccentricity ratio Ecc = Δe / (IR_casing − OR_tubing) (Eq. 4); deformation factor Def = R_A / R_B (Eq. 5); forward model d_obs = T(P) (Eq. 6); Bayesian inversion via Gaussian Process Regression with a Matérn-5/2 covariance kernel (Eqs. 7–9); cyclic-angle handling via (Ecc·cos θ, Ecc·sin θ, (Def−1)·cos 2γ, (Def−1)·sin 2γ) Cartesian decomposition with polar recovery — recovers (Ecc, θ, Def, γ) within 0.005 / 0.5° / 0.005 / 5° on a 24-Hall-probe synthetic | Yang, Qin, Olson & Rourke, pp. 125–146 |
| `article2_kozeny_permeability_chalk` | Four Kozeny-equation routes for permeability in Lower-Cretaceous Sola/Tuxen marly chalks (Well Boje-2C): base Kozeny k = c·φ³ / S_φ² (Eq. 1) with Mortensen et al. (1998) shielding factor c(φ) = (4·cos(φπ/3))² / 8 (Eq. 2); ternary calcite/silicate/pyrite porosity from density log (Eqs. 3–4); pore-space SSA from mineralogy (Eq. 5); four alternative SSA estimators — spectral GR Sb = x·ρ_b·(Th+K) + y (Eq. 6); Sw and pseudo-water-film thickness pwft (Eqs. 7–8); NMR T2 of the water peak S_φ = 1/(ρ·T2) (Eqs. 9–10); flow-zone-indicator FZI with void-ratio coupling and FZI–Vp regression (Eqs. 11–13) | Storebø, Meireles & Fabricius, pp. 148–171 |
| `article3_pyrite_saturation_hs_bounds` | Pyrite-aware extension of Archie's law combining Clavier dual-water with weighted Hashin-Shtrikman bounds: Archie σ_t = σ_w·φ^m·Sw^n (Eq. 1); Wiener arithmetic and harmonic two-phase bounds (Eqs. 2–3); Hashin-Shtrikman lower / upper bounds for an isotropic two-component medium (Eqs. 4–5); Archie with extra conductivity σ_o = (σ_w + σ_extra)·φ^m·Sw^n (Eq. 6); Waxman-Smits excess conductivity σ_x = β·Qv (Eqs. 7–9); Clavier dual-water mixing (Eqs. 10–13); CEC-based silicate excess conductance (Eq. 32); weighted HS pyrite mixing (Eqs. 48–52) — default constants reproduce the paper's Boje-2C numbers (σ_wb = 82.9 S/m at 91 °C, σ_pyrite = 1500 S/m, w = 0.03) | Storebø, Hjuler, Meireles & Fabricius, pp. 172–198 |
| `article4_microct_filtercake` | *Methodology proxy* for the time-lapse micro-CT filter-cake paper (full body not in available PDF extract): Dewan-Chenevert / Outmans √t mudcake-growth law h_mc(t) = √(2·k_mc·ΔP·t / (μ·(1 − φ_mc))); mudcake-porosity evolution under compaction stress φ_mc(t) = φ_0·(1 + t/τ)^(−c); Kozeny-Carman permeability evolution k(t) = k_0·(φ/φ_0)³·((1−φ_0)/(1−φ))²; synthetic 2-D CT slice with annular mudcake band and threshold-based thickness detector | Schroeder & Torres-Verdín, pp. 199–217 |
| `article5_methane_solubility_obm` | *Methodology proxy* for the OBM methane-solubility paper (full body not in available PDF extract): Henry's-law / Krichevsky-Kasarnovsky form ln(x_CH4) = a + b·ln(P) − ΔH/(R·T); multivariate linear regression for ln(x_CH4) against (P, T, base-oil mass fraction, mud viscosity) — the four design variables identified in the paper; synthetic dataset recovers planted regression coefficients within 2 % | Song, Sukari, Wang, Jiang, Cai, Xu & Huang, pp. 218–236 |
| `article6_gas_hydrate_rock_physics` | *Methodology proxy* for the NGHP-02 gas-hydrate rock-physics paper (full body not in available PDF extract): Voigt-Reuss-Hill mineral mixing; two end-member hydrate models (cementing and Jason grain-supported); Gassmann fluid substitution K_sat / (K_min − K_sat) = K_dry / (K_min − K_dry) + K_fl / (φ·(K_min − K_fl)); Vp and Vs from (K_sat, G, ρ_b); Vp/Vs cross-plot classifier discriminating hydrate-bearing shaly sand from sand, calcite and shale | Kumar, Mishra, Chatterjee, Tiwari & Avadhani, pp. 237–255 |
| `article7_digital_core_wellbore_stability` | *Methodology proxy* for the digital-core wellbore-stability paper (full body not in available PDF extract): 3-D voxel sand-pack as the digital-core analogue; VRH solid moduli; Krief porosity softening; Young's modulus E = 9·K·G / (3·K + G); Plumb-Allen UCS predictor UCS = a·E − b·φ; exponential water-immersion weakening UCS(t) = UCS_dry·(floor + (1−floor)·exp(−t/τ)); Kirsch-stress + Mohr-Coulomb critical-mud-weight check for vertical-well stability | Zhou, Ye, Zhu, Cheng, Song, Wang & Cai, pp. 256–284 |

DOI pattern: `10.30632/PJV63N2-2022aN` (N = 1 … 7). Articles 4–7 implemented as methodology proxies; see `src2022_04/README.md`.

---

## src2022_06 — Vol. 63, No. 3 (June 2022)

Special Issue curated by the NMR Special Interest Group (Guest Editor: Philip Singer), organised into three sub-themes: *Machine Learning and Data Processing* (articles 1–3), *Log Analysis and Tools* (articles 4–7), and *Core Analysis* (articles 8–11). Two trending threads run across the issue: unsupervised / machine-learning methods for fluid typing on T1-T2 maps, and NMR characterisation of unconventional fine-grained reservoirs. Articles 7–11 were only available as Table-of-Contents entries in the source PDF, so their modules are methodology proxies guided by the Guest Editor's narrative; see the per-folder README for details.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_nmf_clustering_t1t2` | Unsupervised NMR T1-T2 fluid-typing on a tight-carbonate stack: Lee-Seung multiplicative-update non-negative matrix factorisation V ≈ W·H to extract pore-fluid end-member spectra W and per-depth mixing weights H from a 30-depth × 1024-pixel synthetic T1-T2 map stack; average-link agglomerative clustering on the end-members; two fluid-typing rules (T1/T2 ≥ 4 → hydrocarbon; T2 ≥ 33 ms → mobile) partition the map into immobile-water / immobile-HC / mobile-HC / mobile-water quadrants; wettability index from the oil-cluster T1/T2 (Eq. 8); body-to-throat ratio combines NMF pore-body distribution with MICP pore-throat distribution | Jiang, Bonnie, Correa, Krueger, Kelly & Wasson, pp. 277–289 |
| `article2_fuzzy_genetic_nmr` | Dual-Echo-Time (DTE) NMR-while-drilling ML stack: effective-T2 diffusion-decay model 1/T2_eff = 1/T2 + (γ·G·TE)²·D/12 used to push WBM filtrate to short T2; triangular-membership fuzzy classifier producing gas / oil / water memberships from (T2, RHOB, log Rt); real-valued genetic algorithm (tournament selection, blended crossover, Gaussian mutation) evolving polynomial-coefficient maps from log features to predicted saturation; distance-weighted k-nearest-neighbours regressor as the benchmark; synthetic Seven-Heads-style dataset of 400 samples with thin-bedded sands carrying high-viscosity residual oil | Cuddy, pp. 290–299 |
| `article3_nmr_processing_toolbox` | Comprehensive reference toolbox for NMR-log data processing: CPMG forward kernel E_i = Σ A_n·exp(−t_i/T2_n) (Eq. 1); exact Tikhonov NNLS inversion on the augmented system [K; λ·I]·A = [E; 0] for 1-D T2 inversion (Eqs. 9–17); 2-D T1-T2 kernel as the Kronecker product of saturation-recovery and CPMG bases (Eqs. 24–32); Timur-Coates K = C·φ^m·(FFV/BFV)^n (Eq. 52); SDR K = a·φ^m·T2_lm^n (Eqs. 56–60); log-mean T2; data-driven ML log-linear permeability predictor (Eq. 62 analogue); inversion benchmark uses the physically-meaningful BVI partition rather than full L2 recovery, consistent with the known ill-conditioning of multi-exponential fitting | Shao & Balliet, pp. 300–338 |
| `article4_bssica_dt2_invasion` | Sidewall-NMR analysis of barite-WBM near-wellbore damage in Greater Burgan: parallel relaxation rates 1/T1 = 1/T1B + 1/T1S (Eq. 1) and 1/T2 = 1/T2B + 1/T2D + 1/T2S (Eq. 2); porosity undercall δφ = φ_open − φ_NMR (Eq. 3); Timur-Coates K = C·φ^m·(FFV/BFV)^n with C = 10, m = 4, n = 2 (Eq. 4); permeability ratio index KRI = K_NMR / K_open (Eq. 5); FastICA implementation with symmetric orthogonalisation for Blind Source Separation under the linear mixing model x = A·s (Eqs. 6–11) — synthetic three-source mixture recovered with > 0.95 component correlation | Romero Rojas, Tagarieva, Panchal, AlTurki & Qubian, pp. 340–351 |
| `article5_nppm_pore_size_perm` | NMR Petrophysical Pore Multimodal (NPPM) analysis for tight-oil mudstones: generalised relaxation rate sum 1/T2 = 1/T2_B + 1/T2_D + S/V · Σ s_f·ρ_f over fluid-fraction-weighted surface relaxivities (Eqs. 1–3); greedy peeling fit of 2-D log-normal Gaussian components on the T1-T2 map yielding (μ_T1, μ_T2, σ_T1, σ_T2, amp) per fluid cluster; apparent surface relaxivity ρ_n,f = r / T2_peak; Kozeny-Carman permeability k = φ³·〈r²〉 / (180·(1−φ)²) (Eqs. 4–6) with a Herron-style exp(−3·V_clay + 0.5·V_carb) mineralogy correction | Ijasan, Macquaker, Luycx, Alzobaidi, Oyewole & Rudnicki, pp. 352–367 |
| `article6_ddtw_mudgas_integration` | Heimdal Sandstone LWD case study: NMR polarisation function S(TW) = S∞·(1 − exp(−TW/T1)) (Eq. 1); variable-matrix-density mix; closed-form Density + DTW (DDTW) gas-zone solution that solves the linear-in-(φ, φ·Sg) system from density apparent porosity and NMR apparent porosity — recovers planted (φ = 0.22, Sg = 0.65) exactly; mud-gas hydrogen-index estimator from C1–C5 molar fractions calibrated so pure methane gives HI ≈ 0.42 (the paper's reported value) | Thern, Kotwicki, Ritzmann, Petersen & Mohnke, pp. 368–388 |
| `article7_slimhole_lwd_factor` | *Methodology proxy* for the slimhole LWD NMR + factor-analysis paper (full body not in available PDF extract): SVD-based factor analysis with Kaiser varimax-style rotation of NMR-log feature matrix; independent log-uniform synthetic weights ensure full-rank recovery of three planted fluid factors with per-factor max correlation > 0.95; LWD-vs-wireline time-lapse difference map quantifies oil-based-mud filtrate invasion contrast inside vs. outside an invaded depth band | Hursan, Silva, Van Steene & Muna, pp. 389–404 |
| `article8_highfield_al_nmr` | *Methodology proxy* for the high-field 27Al MAS NMR mineral characterisation paper (full body not in available PDF extract): 27Al chemical-shift library at ~14 T for common rock-forming minerals (kaolinite, illite, smectite, muscovite, chlorite, K-feldspar, albite, anorthite, analcime, corundum); synthetic Lorentzian spectrum generator; sech-kernel windowed integration for mineral identification reliably resolves AlVI-region clays from AlIV-region feldspars and corundum | Wang, Sun, Yang, Seltzer & Wigand, pp. 405–417 |
| `article9_t2_imbibition_wettability` | *Methodology proxy* for the NMR-based wettability-index paper (full body not in available PDF extract): time-lapse T2 spectra for sequential water then oil imbibition; long-T2 area integral above a 100 ms cutoff; wettability index WI = (A_water_long − A_oil_long) / (A_water_long + A_oil_long) in [−1, +1]; D2O-imbibition subtraction isolates the protonated-phase contribution; water-wet rock returns WI > 0.5, oil-wet rock returns WI < −0.5 | Dick, Veselinovic, Bonnie & Kelly, pp. 418–441 |
| `article10_pcr_nmr_micp_perm` | *Methodology proxy* for the PCA / Principal Component Regression permeability paper (full body not in available PDF extract): combined NMR T2 distribution + MICP cumulative-saturation feature vector + porosity; SVD-based PCA; log-domain PCR onto k principal components; head-to-head comparison against Timur-Coates and SDR baselines on a synthetic dataset where permeability depends on both NMR FFV/BFV AND MICP entry pressure — PCR with 6 PCs cleanly beats both NMR-only baselines | Rios, Azeredo, Moss, Pritchard & Domingues, pp. 442–453 |
| `article11_core_nmr_review` | *Methodology proxy* for the NMR core-analysis review paper (full body not in available PDF extract): three representative concept demonstrations — geometric variable-spaced-tau (VST) schedule sampling slow relaxation tail with an exp(−tau/T2) forward over a log-T2 distribution; finite-thickness slice-selective profile via spatial mask; SPRITE-style 1-D phase-encoded imaging using forward FT at gradient-defined k-axis and inverse FT to recover the source profile | Dick, Veselinovic & Green, pp. 454–end |

DOI pattern: `10.30632/PJV63N3-2022aN` (N = 1 … 11). Articles 7–11 implemented as methodology proxies; see `src2022_06/README.md`.

---

## src2022_08 — Vol. 63, No. 4 (August 2022)

Regular (non-themed) issue with four editorial themes: integration of rock-typing characteristics, resistivity-tool modelling and applications, fluid properties and behaviour, and well-log prediction / interpretation methodology. Five technical articles spanning gas-condensate PVT prediction from fluid pressure gradients, deep-learning inversion of LWD resistivity in faulted formations, sensitivity analysis of electric-dipole geosteering tools, a database-driven Bayesian log-interpretation framework, and change-point + fuzzy-c-means log-facies analysis of basement granitic reservoirs.

| Module | Topic | Reference |
| --- | --- | --- |
| `article1_gas_condensate_fpg` | Hybrid EOS/PVT models that derive in-situ gas-condensate properties from the measured fluid pressure gradient (FPG): adiabatic fluid modulus K_ad = ρ·V_p² (Eq. 1); quadratic-in-density CGR predictor with linear P and T centred-residual corrections log10(CGR) = a0 + a1·ρ + a2·ρ² + a3·(P − 15000) + a4·(T − 360) (Eq. 3); viscosity correlations against in-situ density (Eq. 4) and methane mole fraction X_CH4 (Eqs. 5–6); acoustic velocity-MW regression (Eq. 7); multivariate density and velocity predictors (Eqs. 8–9); Gassmann fluid-modulus expression (Eq. 10); coefficients tuned so the Shearwater Field test case (15,400 psi, 360 °F, ρ = 0.464 g/cm³) recovers the paper's 144.7 STB/MMscf CGR within 1 % | Bryndzia & Kittridge, pp. 488–505 |
| `article2_lwd_dl_inversion` | Four-network deep-learning workflow for 2.5-D inversion of triaxial 2-MHz LWD resistivity: one classifier picks among three earth-model classes (3-layer in host, with bed-boundary crossing, or with vertical-fault crossing); three per-class encoder-decoder regressors invert the layer parameters; 40-dim feature vector (5 Tx-Rx pairs × 4 channels × phase+attenuation); class-specific signatures on the cross-component geosignal and azimuthal channels reproduce the paper's finding that coaxial-only inversion is insufficient and that cross-component data is required to resolve bed boundaries; standard inverse loss augmented with a physics-guided forward-consistency term yields a joint objective L = ‖m_pred − m_true‖² + λ·‖F(m_pred) − d_obs‖² (Eq. 2); ~100 % held-out classification accuracy on clean synthetic data (paper reports 97–99 %) | Noh, Torres-Verdín & Pardo, pp. 506–518 |
| `article3_electric_dipole_sensitivity` | Closed-form electric and magnetic-field responses for arbitrarily-oriented electric current dipoles in 1-D transversely-isotropic media: bed-detection sensitivity definitions δ_E (Eqs. 31, 33) and δ_H (Eqs. 32, 34) as normalised perturbation when a 10,000 Ω·m bed is inserted into a 1 Ω·m host at distance D from the tool (10 m T-R spacing, 100 Hz); key analytical result that *electric-field* sensitivity decays as (L/D)³ while *transverse-magnetic-field* sensitivity decays as (L/D)², extending the H-channel detection range by ~ 2× at a 1 % signal threshold; per-interface reflection coefficient R = (σ_i − σ_{i+1})/(σ_i + σ_{i+1}) for the Appendix-7 multilayer recursion | Bautista-Anguiano & Hagiwara, pp. 519–533 |
| `article4_bayesian_log_db` | Database-driven Bayesian log interpretation: pre-builds a 20,000-realisation database of synthetic formations (mineral volumes via Dirichlet, φ and Sw via uniform draws) with forward-modelled tool responses; Appendix-3 forward operators include volume-weighted GR (Eq. A3.1), bulk density (Eq. A3.2), photoelectric factor (Eq. A3.3), neutron with excavation correction (Eq. A3.4), Wyllie compressional travel-time (Eqs. A3.5–A3.13), and merged Waxman-Smits / Dual-Water resistivity using the Juhasz B-factor (Eqs. A3.14–A3.16); Bayes' theorem (Eq. 1) with Gaussian likelihood weighting (Eq. A1.1) yields posterior mean and uncertainty for (φ, Sw, Vsh, mineral fractions) from a noisy seven-channel observation - test case recovers φ within 0.02 and Sw within 0.05 of truth | Spalburg, pp. 534–548 |
| `article5_cpa_fcm_logfacies` | Two-stage log-facies analysis on synthetic five-curve (GR, RD, DEN, AT, NP) suites: (1) change-point analysis on the GR series using the mean-change-point model x_i = a_i + e_i (Eq. 1), SSE-minimisation breakpoint search (Eq. 2), Q = H/R initial guesses (Eq. 3), greedy add-and-refine via the W functional (Eq. 4), and a jump-magnitude θ statistic with minimum-spacing filter (Eq. 5); (2) Bezdek fuzzy c-means on segment-averaged 5-log features minimising the FCM objective Σ U_ij^m·D_ij² (Eq. 6) with membership updates (Eq. 7) and centroid updates (Eq. 8), fuzzifier m = 2; synthetic test recovers all embedded breakpoints within ±5 samples and the FCM objective drops by > 90 % in ~ 10 iterations | Hua, Yang, Xu, Lei & Zhong, pp. 549–565 |

DOI pattern: `10.30632/PJV63N4-2022aN` (N = 1 … 5)

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
