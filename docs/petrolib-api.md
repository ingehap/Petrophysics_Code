# petrolib API reference

One-line summaries of the public API, generated from the docstrings by
[`tools/gen_petrolib_api.py`](../tools/gen_petrolib_api.py) — regenerate after
any petrolib change.  Full parameter documentation lives in the docstrings
(``help(petrolib.<module>.<function>)``); complete journal citations for every
``srcYYYY_MM`` source tag are in each module docstring's *References* section.


## Modules

- [`petrolib.acoustic_geomech`](#petrolibacoustic_geomech) — Acoustic / elastic conversions, rock-physics mixing, and geomechanics.
- [`petrolib.borehole_image`](#petrolibborehole_image) — Borehole-image analysis: bed sinusoids, dip picking, thresholding, texture.
- [`petrolib.capillary_pressure`](#petrolibcapillary_pressure) — Capillary pressure: Young-Laplace/Washburn, curve models, and scaling.
- [`petrolib.constants`](#petrolibconstants) — Physical and unit-conversion constants — the single source of truth.
- [`petrolib.data_qc_io.clean`](#petrolibdata_qc_ioclean) — Log cleaning: sentinels, gap imputation, outliers, compositional closure.
- [`petrolib.data_qc_io.filt`](#petrolibdata_qc_iofilt) — Depth-series filtering: smoothing, moving statistics, tool response, 2-D median filter, window feature stacks, bed-boundary detection.
- [`petrolib.data_qc_io.io`](#petrolibdata_qc_ioio) — Universal wellbore-data container (Bradley et al. 2023).
- [`petrolib.data_qc_io.scale`](#petrolibdata_qc_ioscale) — Curve normalization against a reference well or target moments.
- [`petrolib.data_qc_io.signal`](#petrolibdata_qc_iosignal) — Signal-to-noise, stacking, and controlled noise injection.
- [`petrolib.data_qc_io.synth`](#petrolibdata_qc_iosynth) — Synthetic test data: blocky logs, log suites, shifted pairs, spectra, sphere packs, and disk images.
- [`petrolib.depth_correction`](#petrolibdepth_correction) — Cable / drill-string depth corrections: elastic stretch, thermal, tension.
- [`petrolib.depth_matching`](#petrolibdepth_matching) — Depth matching / curve alignment: DTW, cross-correlation shift, warping.
- [`petrolib.em_dielectric`](#petrolibem_dielectric) — Electromagnetic / dielectric petrophysics: permittivity, dispersion, mixing.
- [`petrolib.flow_transport`](#petrolibflow_transport) — Single-phase flow, poro-permeability transforms, rock typing, and diffusion.
- [`petrolib.geochem_fluids.adsorption`](#petrolibgeochem_fluidsadsorption) — Gas adsorption isotherms and shale gas-in-place volumetrics.
- [`petrolib.geochem_fluids.asphaltene`](#petrolibgeochem_fluidsasphaltene) — Asphaltene gravity gradients (Flory-Huggins-Zuo) and Yen-Mullins sizes.
- [`petrolib.geochem_fluids.brine`](#petrolibgeochem_fluidsbrine) — Brine resistivity, salinity, capture cross-section, and density.
- [`petrolib.geochem_fluids.contamination`](#petrolibgeochem_fluidscontamination) — OBM/filtrate contamination mixing and cleanup.
- [`petrolib.geochem_fluids.core_geochem`](#petrolibgeochem_fluidscore_geochem) — Core geochemistry: Dean-Stark saturations, oxide closure, and OSI.
- [`petrolib.geochem_fluids.gradients`](#petrolibgeochem_fluidsgradients) — Formation-pressure gradients and fluid contacts.
- [`petrolib.geochem_fluids.mudgas`](#petrolibgeochem_fluidsmudgas) — Mud-gas composition ratios and fluid-type classification.
- [`petrolib.geochem_fluids.pvt`](#petrolibgeochem_fluidspvt) — Gas PVT: pseudo-reduced properties, z-factor, density, and phase equilibrium.
- [`petrolib.geochem_fluids.solubility`](#petrolibgeochem_fluidssolubility) — Gas solubility in brine and oil (Henry / Setschenow / Duan-Sun style).
- [`petrolib.integrity_drilling`](#petrolibintegrity_drilling) — Well integrity and drilling: cement bond, casing condition, leaks, mud gas.
- [`petrolib.inversion_numerics.costs`](#petrolibinversion_numericscosts) — Cost / misfit functions and regularization-parameter schedules.
- [`petrolib.inversion_numerics.fitting`](#petrolibinversion_numericsfitting) — Curve fitting: line / power-law / exponential-approach / cosine.
- [`petrolib.inversion_numerics.linear`](#petrolibinversion_numericslinear) — Linear inversion: least squares, Tikhonov, SVD, unmixing, operators.
- [`petrolib.inversion_numerics.nonlinear`](#petrolibinversion_numericsnonlinear) — Nonlinear inversion: finite-difference derivatives, LM, Occam, search.
- [`petrolib.inversion_numerics.optimize`](#petrolibinversion_numericsoptimize) — Global / gradient optimization: particle swarm and gradient descent.
- [`petrolib.inversion_numerics.pde`](#petrolibinversion_numericspde) — Grid PDE solvers: 2D effective conductivity and 1D diffusion.
- [`petrolib.inversion_numerics.stochastic`](#petrolibinversion_numericsstochastic) — Stochastic inversion: likelihoods, priors, MCMC, MALA, ensemble methods.
- [`petrolib.ml_stats`](#petrolibml_stats) — Statistics and numpy-only machine-learning helpers.
- [`petrolib.nmr`](#petrolibnmr) — Nuclear magnetic resonance: relaxation physics, T2 statistics, forward models, permeability transforms, and fluid typing.
- [`petrolib.nuclear`](#petrolibnuclear) — Nuclear logging: capture cross-section, attenuation, density, GR, neutron.
- [`petrolib.porosity_lithology`](#petrolibporosity_lithology) — Porosity, shale volume, lithology mixing, Thomas-Stieber, TOC and net pay.
- [`petrolib.relperm_wettability`](#petrolibrelperm_wettability) — Relative permeability and displacement: Corey/Brooks-Corey/LET, mobility, fractional flow, and the Buckley-Leverett/Welge construction.
- [`petrolib.saturation_resistivity`](#petrolibsaturation_resistivity) — Water saturation from resistivity: Archie and shaly-sand models.
- [`petrolib.testing`](#petrolibtesting) — Regression-safety helpers for the library migration.
- [`petrolib.units`](#petrolibunits) — Unit conversions for the quantities the article code mixes most.
- [`petrolib.wellbore_geometry`](#petrolibwellbore_geometry) — Wellbore-survey trajectory geometry: dogleg, minimum curvature, MD to TVD.

## `petrolib.acoustic_geomech`

Acoustic / elastic conversions, rock-physics mixing, and geomechanics.

| Name | Summary |
| --- | --- |
| `G_STANDARD` | constant |
| `moduli_from_velocity(vp, vs, rho)` | Bulk and shear modulus from P/S velocities and density -> ``(K, G)``. |
| `velocity_from_moduli(k, g, rho)` | P/S velocities from bulk/shear modulus and density -> ``(Vp, Vs)``. |
| `stiffness_from_velocity(rho, v)` | Stiffness coefficient ``C = rho*V**2`` (the P-wave modulus ``M`` for Vp). |
| `lame_from_velocity(vp, vs, rho)` | Lame parameters from velocities and density -> ``(lambda, mu)``. |
| `velocity_from_slowness(dt, *, dt_unit)` | Velocity (m/s) from acoustic slowness. |
| `slowness_from_velocity(v, *, dt_unit)` | Acoustic slowness from velocity (m/s) -- inverse of ``velocity_from_slowness``. |
| `youngs_poisson_dynamic(vp, vs, rho)` | Dynamic Young's modulus and Poisson's ratio from velocities -> ``(E, nu)``. |
| `youngs_from_kg(k, g)` | Young's modulus from bulk and shear modulus ``E = 9*K*G/(3*K + G)``. |
| `poisson_from_kg(k, g)` | Poisson's ratio from bulk and shear modulus ``nu = (3K - 2G)/(2*(3K + G))``. |
| `poisson_from_velocity(vp, vs)` | Poisson's ratio from P/S velocities ``(Vp**2 - 2*Vs**2)/(2*(Vp**2 - Vs**2))``. |
| `pwave_modulus(k, g)` | P-wave (constrained) modulus ``M = K + 4/3*G``. |
| `voigt(f, m, *, axis)` | Voigt (arithmetic) bound ``sum(f_i*m_i)`` over the mixing axis. |
| `reuss(f, m, *, axis)` | Reuss (harmonic) bound ``1/sum(f_i/m_i)`` over the mixing axis. |
| `voigt_reuss_hill(f, m, *, axis)` | Voigt-Reuss-Hill average ``0.5*(Voigt + Reuss)``. |
| `wood_fluid_modulus(saturations, moduli, *, axis)` | Wood's law effective fluid modulus (Reuss average over fluid phases). |
| `brie_fluid_modulus(sw, k_liquid, k_gas, *, e)` | Brie et al. |
| `gassmann_ksat(*, k_dry, k_mineral, k_fluid, phi)` | Gassmann saturated bulk modulus from the dry-frame modulus. |
| `gassmann_kdry(*, k_sat, k_mineral, k_fluid, phi)` | Gassmann dry-frame bulk modulus from the saturated modulus (inverse). |
| `acoustic_impedance(rho, v, *, out)` | Acoustic impedance ``Z = rho*v``. |
| `reflection_coefficient(z1, z2)` | Normal-incidence reflection coefficient ``(Z2 - Z1)/(Z2 + Z1)``. |
| `transmission_energy(z1, z2)` | Transmitted energy fraction ``1 - R**2`` at a normal-incidence interface. |
| `attenuation_db(a1, a2)` | Attenuation in decibels ``20*log10(A1/A2)`` between two amplitudes. |
| `attenuation_coefficient(a0, ax, x)` | Attenuation coefficient ``alpha`` in ``A = A0*exp(-alpha*x)``. |
| `snell_angle(v1, v2, *, degrees)` | Snell critical/refraction angle ``arcsin(V1/V2)`` from two velocities. |
| `thomsen_epsilon(c11, c33)` | Thomsen ``epsilon = (C11 - C33)/(2*C33)`` (P-wave anisotropy). |
| `thomsen_gamma(c66, c44)` | Thomsen ``gamma = (C66 - C44)/(2*C44)`` (S-wave anisotropy). |
| `thomsen_delta(c13, c33, c44)` | Thomsen ``delta = ((C13 + C44)**2 - (C33 - C44)**2)/(2*C33*(C33 - C44))``. |
| `annie_c13(c11, c66)` | ANNIE approximation ``C13 = C11 - 2*C66`` (assumes ``C13 = C12``). |
| `annie_c11(c33, c44, c66)` | ANNIE approximation ``C11 = 2*(C66 - C44) + C33``. |
| `mannie_c13(c11, c66, *, k)` | Modified-ANNIE ``C13 = k*(C11 - 2*C66)`` (``k=1`` recovers ANNIE). |
| `mannie_c11(c33, c44, c66, *, kp)` | Modified-ANNIE ``C11 = kp*2*(C66 - C44) + C33`` (``kp=1`` recovers ANNIE). |
| `mannie2_c66(c11, c33, c44, *, k)` | Modified-ANNIE-2 shear closure ``C66 = C44*(1 + k*(C11 - C33)/C33)``. |
| `shear_wave_splitting(v_fast, v_slow)` | Shear-wave splitting (anisotropy) ``(V_fast - V_slow)/V_fast``. |
| `vti_engineering_moduli(c11, c33, c44, c66, c13, *, c12)` | VTI engineering moduli from the stiffness constants. |
| `biot_coefficient(k_dry, k_mineral)` | Biot-Willis effective-stress coefficient ``alpha = 1 - K_dry/K_mineral``. |
| `effective_stress(sigma_total, pp, *, biot)` | Terzaghi/Biot effective stress ``sigma_eff = sigma_total - biot*Pp``. |
| `eaton_pore_pressure(overburden, hydrostatic, observed, normal, *, exponent, ...)` | Eaton pore pressure ``Pp = S_ov - (S_ov - P_hydro)*ratio**exponent``. |
| `min_horizontal_stress(sv, pp, nu, *, biot, e, eps_h, eps_H, tectonic)` | Minimum horizontal stress -- poroelastic uniaxial-strain superset. |
| `breakdown_pressure(sh, sH, pp, *, tensile_strength)` | Hubbert-Willis breakdown pressure ``3*sh - sH - Pp + T0``. |
| `reopening_pressure(sh, sH, pp)` | Fracture reopening pressure ``3*sh - sH - Pp`` (breakdown with T0=0). |
| `shmax_from_reopening(pr, sh, pp)` | Maximum horizontal stress from the reopening pressure ``sH = 3*sh - Pr - Pp``. |
| `kirsch_hoop_stress(sH, sh, pw, theta_deg)` | Kirsch tangential (hoop) stress at the borehole wall. |
| `brittleness_rickman(e, nu, *, e_range, nu_range, percent)` | Rickman (2008) brittleness index from Young's modulus and Poisson's ratio. |
| `vs_from_vp(vp, *, method)` | Empirical shear velocity from compressional velocity (Vp in km/s). |
| `overburden_stress(depth, rho_bulk, *, g)` | Overburden (vertical) stress ``sigma_v = rho_bulk*g*depth`` (SI, Pa). |
| `hydrostatic_pressure(depth, *, rho, g)` | Hydrostatic pressure ``P = rho*g*depth`` (SI, Pa). |
| `bowers_pore_pressure(velocity, overburden, *, a, b, v0, unloading, u, sigma_max)` | Bowers (1995) pore pressure from velocity and overburden (field units). |

## `petrolib.borehole_image`

Borehole-image analysis: bed sinusoids, dip picking, thresholding, texture.

| Name | Summary |
| --- | --- |
| `bed_sinusoid(azimuth_deg, z0, radius, dip_deg, dip_azimuth_deg)` | Depth trace of a planar bed on the unrolled borehole wall. |
| `fit_sinusoid(azimuth_deg, z, *, mask)` | Least-squares sinusoid fit on ``[1, cos, sin]`` -> ``(z0, amplitude, phase_deg)``. |
| `dip_from_amplitude(amplitude, radius, *, sample_spacing)` | True dip from a sinusoid amplitude: ``arctan(amplitude*sample_spacing/radius)`` (deg). |
| `apparent_dip(true_dip_deg, section_azimuth_deg)` | Apparent dip at section angle ``beta`` to true dip: ``tan(app)=tan(true)cos(beta)`` (deg). |
| `fit_plane(points_enz)` | SVD plane fit through ``(E, N, TVD)`` points -> ``(dip_deg, dip_azimuth_deg)``. |
| `otsu_threshold(image, *, bins)` | Otsu threshold on the image's native data range, returning a bin-centre value. |
| `class_fractions(image, thresholds)` | Volume fractions between ascending ``thresholds``. |
| `phase_saturation(phase, pore)` | Saturation of a phase within the pore space: ``\|phase & pore\| / \|pore\|``. |
| `porosity_from_mask(pore)` | Porosity as the fraction of voxels flagged pore: ``pore.sum()/pore.size``. |
| `glcm(image, *, levels, offset, symmetric)` | Normalised grey-level co-occurrence matrix. |
| `glcm_features(p)` | Haralick contrast / energy / correlation of a normalised GLCM ``p``. |
| `sobel_gradient(image)` | 3x3 Sobel gradient -> ``(gx, gy, magnitude)`` with zero borders. |

## `petrolib.capillary_pressure`

Capillary pressure: Young-Laplace/Washburn, curve models, and scaling.

| Name | Summary |
| --- | --- |
| `young_laplace_pc(radius, *, sigma, theta_deg, absolute)` | Capillary entry pressure of a tube: ``Pc = 2*sigma*cos(theta)/r``. |
| `washburn_radius(pc, *, sigma, theta_deg, absolute)` | Pore-throat radius from capillary pressure: ``r = 2*sigma*cos(theta)/Pc``. |
| `washburn_diameter(pc, *, sigma, theta_deg, absolute)` | Pore-throat diameter ``d = 4*sigma*cos(theta)/Pc``. |
| `leverett_j(pc, *, sigma, theta_deg, k, phi, absolute)` | Leverett J-function ``J = Pc/(sigma*cos(theta)) * sqrt(k/phi)``. |
| `pc_from_leverett_j(j, *, sigma, theta_deg, k, phi, absolute)` | Invert :func:`leverett_j` for capillary pressure. |
| `pc_convert_system(pc, *, sigma_from, theta_from_deg, sigma_to, ...)` | Convert Pc between fluid systems by the sigma*cos(theta) ratio. |
| `brooks_corey_sw(pc, *, pc_entry, lam, swirr, clip)` | Brooks-Corey drainage saturation ``Sw = Swirr + (1-Swirr)*(Pe/Pc)**lam``. |
| `brooks_corey_pc(sw, *, pc_entry, lam, swirr, snr)` | Brooks-Corey capillary pressure ``Pc = Pe * Swn**(-1/lam)``. |
| `thomeer_shg(pc, *, bv_inf, g, pd, log_base)` | Thomeer hyperbola ``Shg = Bv_inf * exp(-G / log_base(Pc/Pd))``; 0 below Pd. |
| `buoyancy_pc(height, *, delta_rho, g)` | Buoyancy capillary pressure ``Pc = delta_rho * g * h`` (SI: Pa). |
| `height_above_fwl(pc, *, delta_rho, g)` | Height above the free-water level from Pc: inverse of :func:`buoyancy_pc`. |
| `buoyancy_pc_gradient(height_ft, *, sg_water, sg_hc, gradient_psi_per_ft)` | Oilfield buoyancy Pc: ``Pc = gradient*(SGw - SGhc)*h`` (psi, ft, SG). |
| `centrifuge_pc(omega, *, delta_rho, r1, r2)` | Hassler-Brunner inlet-face Pc: ``0.5*delta_rho*omega**2*(r2**2 - r1**2)``. |
| `capillary_rise_height(radius, *, sigma, theta_deg, delta_rho, g)` | Equilibrium capillary rise ``h = 2*sigma*cos(theta)/(delta_rho*g*r)``. |
| `lucas_washburn_length(t, *, sigma, radius, theta_deg, mu)` | Lucas-Washburn imbibition front ``L(t) = sqrt(sigma*r*cos(theta)*t/(2*mu))``. |

## `petrolib.constants`

Physical and unit-conversion constants — the single source of truth.

| Name | Summary |
| --- | --- |
| `KB` | Boltzmann constant [J/K] (exact, 2019 SI). |
| `NA` | Avogadro constant [1/mol] (exact, 2019 SI). |
| `R_GAS` | Molar gas constant [J/(mol*K)] (exact: KB * NA). |
| `EPS0` | Vacuum electric permittivity [F/m] (CODATA 2018). |
| `MU0` | Vacuum magnetic permeability [H/m] (CODATA 2018). |
| `G_STD` | Standard acceleration of gravity [m/s^2] (exact, conventional). |
| `GAMMA_H` | Proton gyromagnetic ratio [rad/(s*T)] (CODATA 2018).  Article modules disagree in the 4th digit: some use 2.675e8, others 2*pi*42.58e6 = 2.6753e8.  This is the CODATA value; migrating call sites whose assertions depend on their local value pass it explicitly. |
| `GAMMA_H_HZ` | Proton gyromagnetic ratio over 2*pi [Hz/T] (CODATA 2018). |
| `M_PER_FT` | Metres per international foot (exact). |
| `M_PER_IN` | Metres per international inch (exact). |
| `PA_PER_PSI` | Pascals per psi (pound-force per square inch; exact by definition of lbf and in). |
| `PA_PER_BAR` | Pascals per bar (exact). |
| `PA_PER_ATM` | Pascals per standard atmosphere (exact). |
| `M2_PER_DARCY` | Square metres per darcy.  The darcy is defined via cP, atm and cm; this is the conventional value (ISO 31-8 / SPE). |
| `KGM3_PER_GCC` | Kilograms per cubic metre per g/cc. |

## `petrolib.data_qc_io.clean`

Log cleaning: sentinels, gap imputation, outliers, compositional closure.

| Name | Summary |
| --- | --- |
| `SENTINELS` | Common LAS/DLIS no-data sentinel values. |
| `MAD_TO_SIGMA` | Gaussian-consistency factor: sigma ~= 1.4826 * MAD for normal data. |
| `sentinels_to_nan(x, sentinels, *, rtol, atol)` | Replace LAS/DLIS no-data sentinels (e.g. |
| `impute_gaps(x, index)` | Fill NaN runs by linear interpolation over the valid samples. |
| `outlier_mask(x, method, *, threshold, k, side)` | Boolean outlier mask by z-score, IQR whiskers, or robust MAD. |
| `despike(x, *, threshold, side, replace)` | Remove spikes flagged by the robust MAD criterion. |
| `closure_residual(*fracs, target)` | Closure residual ``sum(fracs) - target`` (e.g. |
| `renormalize(fracs, axis)` | Rescale fractions to sum to one along ``axis`` (compositional closure). |
| `relative_discrepancy(a, b)` | Relative discrepancy ``2\|a-b\| / (a+b)`` (difference over the mean). |

## `petrolib.data_qc_io.filt`

Depth-series filtering: smoothing, moving statistics, tool response, 2-D median filter, window feature stacks, bed-boundary detection.

| Name | Summary |
| --- | --- |
| `_MOVING_STATS` | constant |
| `smooth(x, window, kind, *, sigma)` | Smooth a 1-D curve with a boxcar, Gaussian, or median filter. |
| `moving_stat(x, window, stat, *, center)` | Moving-window statistic with shrinking edge windows. |
| `tool_response(x, dz, fwhm)` | Convolve a fine-scale curve with a Gaussian tool vertical response. |
| `median_filter2d(img, size)` | Pure-numpy 2-D median filter with edge-replicated padding. |
| `window_features(x, window)` | Edge-padded sliding-window feature stack for depth-indexed ML. |
| `detect_bed_boundaries(curve, window, threshold)` | Bed-boundary indices from a derivative-times-local-variability metric. |

## `petrolib.data_qc_io.io`

Universal wellbore-data container (Bradley et al. 2023).

| Name | Summary |
| --- | --- |
| `class WellboreData` | Hierarchical metadata + n-D channels with units and axes. Methods: `add_channel`, `to_dict`, `to_json`, `from_dict`, `from_json`. |

## `petrolib.data_qc_io.scale`

Curve normalization against a reference well or target moments.

| Name | Summary |
| --- | --- |
| `normalize_to_reference(x, ref_lo, ref_hi, *, in_lo, in_hi, pct)` | Two-point (Shier 2004) normalization onto reference-well endpoints. |
| `match_moments(x, target_mean, target_std)` | Affine transform of ``x`` to the target mean and (population) std. |

## `petrolib.data_qc_io.signal`

Signal-to-noise, stacking, and controlled noise injection.

| Name | Summary |
| --- | --- |
| `snr_db(signal, *, noise, noise_std, guard)` | Signal-to-noise ratio in dB: ``10*log10(mean(s^2)/P_noise)``. |
| `stack_gain(n)` | Amplitude SNR gain ``sqrt(n)`` from stacking ``n`` repeat measurements. |
| `block_stack(x, n, axis)` | Block-mean stacking: average consecutive groups of ``n`` along ``axis``. |
| `add_gaussian_noise(x, sigma_fraction, rng)` | Add zero-mean Gaussian noise with ``sigma = fraction * mean(\|x\|)``. |

## `petrolib.data_qc_io.synth`

Synthetic test data: blocky logs, log suites, shifted pairs, spectra, sphere packs, and disk images.

| Name | Summary |
| --- | --- |
| `LOG_SUITE_CURVES` | Curve order of :func:`log_suite` columns. |
| `LOG_SUITE_FACIES_PROPS` | Per-facies (rows) mean levels for the :data:`LOG_SUITE_CURVES` (columns): granitic, shale, fractured granite, weathered.  Source: src2022_08/article5. |
| `blocky_log(n, n_beds, *, level_range, noise, rng)` | Blocky (square) log of ``n_beds`` constant levels plus Gaussian noise. |
| `log_suite(n, n_facies, *, rng)` | Five-log synthetic well (GR, RD, RHOB, DT, NPHI) with facies blocks. |
| `shifted_pair(n, shift, *, noise, rng)` | Reference curve and a copy lagged by ``shift`` samples. |
| `gaussian_mixture_spectrum(axis, centres, amps, widths, *, noise, log_axis, rng)` | Sum-of-Gaussians spectrum (synthetic NMR T2 / pore-size distributions). |
| `sphere_pack_volume(shape, r_range, target_phi, *, rng, max_grains)` | Random sphere pack on a voxel grid (0 = pore, 1 = grain). |
| `disk_image(size, n_features, *, rng)` | Grey image with darker random disks; returns ``(image, mask)``. |

## `petrolib.depth_correction`

Cable / drill-string depth corrections: elastic stretch, thermal, tension.

| Name | Summary |
| --- | --- |
| `E_STEEL` | Effective Young's modulus of steel drill pipe / wireline cable, Pa. |
| `ALPHA_STEEL` | Linear thermal-expansion coefficient of steel, 1/K. |
| `elastic_stretch(force, length, area, *, E)` | Point-load Hookean stretch ``dL = F L / (E A)``. |
| `distributed_stretch(length, weight_per_length, ea, *, end_load, buoyancy)` | Hanging-string stretch ``dL = (F L + 0.5 b w L^2) / (E A)``. |
| `thermal_elongation(length, dT, *, alpha)` | Thermal elongation ``dL = alpha L dT``. |
| `cable_tension(depth, total_depth, tool_weight, cable_weight_per_length)` | Tension profile ``T(z) = W_tool + w (L - z)`` along the cable. |
| `corrected_depth(measured, *, stretch, thermal, convention)` | Apply stretch and thermal corrections to a measured depth. |

## `petrolib.depth_matching`

Depth matching / curve alignment: DTW, cross-correlation shift, warping.

| Name | Summary |
| --- | --- |
| `class DtwResult` | Result of :func:`dtw` / :func:`rddtw`. |
| `class ShiftResult` | Result of :func:`xcorr_shift`. |
| `dtw(ref, target, *, band, cost, root)` | Banded dynamic time warping of ``target`` onto ``ref``. |
| `rddtw(ref, target, *, tau, lam, band)` | Regularized derivative DTW (RDDTW) with an excessive-warping penalty. |
| `warp_to_reference(target, path, n_ref, *, reduce)` | Collapse a DTW ``path`` into a ``target`` curve aligned to the reference grid. |
| `path_depth_shifts(path, depth_ref, depth_target)` | Per-depth correction from a DTW path: ``(depth_ref[i], depth_target[j]-depth_ref[i])``. |
| `cow(ref, target, *, n_segments, slack)` | Correlation-optimised warping -> ``(aligned, warp)``. |
| `xcorr_shift(ref, target, *, max_lag, edge, return_curve)` | Integer-lag bulk shift of ``target`` onto ``ref`` by maximum correlation. |
| `xcorr_shift_depth(depth_a, a, depth_b, b, *, max_shift, step, use_abs_corr)` | Physical-unit core-to-log homing by interpolated correlation. |
| `local_shifts(ref, target, *, window, step, max_lag)` | Windowed non-wrapping lag profile of ``target`` relative to ``ref``. |
| `apply_integer_shift(log, lag, *, fill)` | Shift ``log`` by an integer ``lag`` samples without wrapping; vacated ends get ``fill``. |
| `apply_depth_shift(values, depth, shift)` | Apply a continuous (possibly per-depth) ``shift`` to ``values`` by interpolation. |

## `petrolib.em_dielectric`

Electromagnetic / dielectric petrophysics: permittivity, dispersion, mixing.

| Name | Summary |
| --- | --- |
| `EPS0` | constant |
| `MU0` | constant |
| `_NEPER_TO_DB` | constant |
| `complex_permittivity(eps_real, *, sigma, freq_hz, eps_imag)` | Complex permittivity ``eps* = eps' - j*(eps'' + sigma/(omega*eps0))``. |
| `imag_permittivity_from_sigma(sigma, freq_hz)` | Loss permittivity from conductivity: ``eps'' = sigma/(2*pi*f*eps0)``. |
| `sigma_from_imag_permittivity(eps_imag, freq_hz)` | Conductivity from loss permittivity: ``sigma = eps''*2*pi*f*eps0``. |
| `loss_tangent(eps_star)` | Loss tangent ``tan(delta) = -Im(eps*)/Re(eps*)`` (-j convention). |
| `impedivity(eps_r, sigma, freq_hz)` | Complex impedivity ``1/(j*omega*eps0*eps_r + sigma)`` (inverse admittivity). |
| `water_permittivity(rw, freq_hz, *, eps_real, eps_dl)` | Brine complex permittivity from ``rw`` (ohm.m) at GHz-band frequencies. |
| `debye(freq_hz, *, eps_inf, eps_s, tau)` | Single-pole Debye relaxation ``eps_inf + (eps_s-eps_inf)/(1 + j*omega*tau)``. |
| `cole_cole(freq_hz, *, eps_inf, eps_s, tau, alpha)` | Cole-Cole relaxation with distribution exponent ``alpha`` (0 -> Debye). |
| `havriliak_negami(freq_hz, *, eps_inf, eps_s, tau, alpha, beta)` | Havriliak-Negami relaxation (``alpha=beta=1`` -> Debye). |
| `cole_cole_resistivity(freq_hz, *, rho0, chargeability, tau, c)` | Pelton (1978) Cole-Cole complex resistivity (induced-polarization form). |
| `mix_power_law(fractions, eps_components, *, alpha)` | Lichtenecker-Rother power-law mixing over the leading (component) axis. |
| `crim(phi, sw, *, eps_w, eps_hc, eps_matrix, alpha)` | CRIM 3-component rock forward model (water / hydrocarbon / matrix). |
| `sw_from_permittivity(eps_meas, phi, *, eps_w, eps_hc, eps_matrix, alpha, clip)` | Water saturation by inverting :func:`crim` for ``Sw``. |
| `bvw_from_permittivity(eps_meas, phi, *, eps_w, eps_hc, eps_matrix, alpha)` | Bulk volume water ``phi*Sw`` from :func:`crim`, unclipped (salinity-robust). |
| `water_filled_porosity(eps_meas, *, eps_matrix, eps_w, alpha, clip)` | Water-filled porosity from a two-component (water/matrix) CRIM inversion. |
| `maxwell_garnett(eps_host, eps_incl, f_incl, *, depol)` | Maxwell-Garnett effective permittivity with depolarization ``depol``. |
| `bruggeman_symmetric(fractions, eps_components, *, iterations)` | Symmetric Bruggeman effective-medium permittivity (N phases, spheres). |
| `hanai_bruggeman(eps_host, eps_incl, f_incl, *, iterations)` | Asymmetric (Hanai) Bruggeman effective permittivity by robust Newton solve. |
| `depolarization_spheroid(aspect)` | Osborn depolarization factors ``(Lx, Ly, Lz)`` for a spheroid. |
| `skin_depth(rho, freq_hz, *, mu_r)` | EM skin depth ``delta = sqrt(2*rho/(omega*mu))`` (``mu = mu0*mu_r``). |
| `induction_number(spacing_m, rho, freq_hz)` | Induction number ``L/delta`` (near-field vs far-field discriminator). |
| `complex_wavenumber(freq_hz, sigma, eps_r, *, mu_r)` | Exact lossy-medium wavenumber ``(k_r, k_i) = (beta, alpha)``. |
| `sigma_eps_from_wavenumber(k_r, k_i, freq_hz, *, mu_r)` | Closed-form inverse of :func:`complex_wavenumber` -> ``(sigma, eps_r)``. |
| `phase_shift_deg(rho, freq_hz, spacing_m)` | Induction-limit phase shift over ``spacing_m``: ``degrees(L/delta)``. |
| `attenuation_db(rho, freq_hz, spacing_m)` | Induction-limit amplitude attenuation over ``spacing_m``: ``8.686*L/delta`` dB. |
| `resistivity_from_phase(phase_deg, freq_hz, spacing_m)` | Induction-limit inverse of :func:`phase_shift_deg` -> apparent ``rho``. |
| `attenuation_phase_from_voltages(v_near, v_far)` | Attenuation (dB) and phase shift (deg) from near/far complex voltages. |
| `anisotropy_coefficient(rh, rv)` | Resistivity anisotropy coefficient ``lambda = sqrt(Rv/Rh)`` (canonical). |
| `apparent_resistivity_dip(rh, rv, dip_deg)` | Apparent resistivity at relative dip ``theta``: ``Rh*sqrt(cos^2 + (Rv/Rh)*sin^2)``. |
| `doll_radial_geometric_factor(r, spacing_m)` | Doll radial geometric factor ``G = r^2/(r^2 + (L/2)^2)`` (integrated to radius r). |
| `apparent_conductivity_two_zone(sigma_xo, sigma_t, r_invasion, spacing_m)` | Doll two-zone (invaded + virgin) apparent conductivity. |

## `petrolib.flow_transport`

Single-phase flow, poro-permeability transforms, rock typing, and diffusion.

| Name | Summary |
| --- | --- |
| `_TINY` | constant |
| `darcy_permeability(q, *, mu, length, area, dp)` | Darcy permeability of a core plug ``k = q*mu*L/(A*dP)``. |
| `darcy_rate(k, *, area, dp, mu, length)` | Darcy volumetric flow rate ``q = k*A*dP/(mu*L)`` (the inverse of :func:`darcy_permeability`). |
| `darcy_pressure_drop(q, *, mu, k, area, length, kr)` | Pressure drop for Darcy flow ``dP = q*mu*L/(k*kr*A)``. |
| `darcy_gas_permeability(q, *, mu, length, area, p_up, p_down, p_ref)` | Compressible (gas) Darcy permeability, the pressure-squared form ``k = 2*q*mu*L*P_ref / (A*(P_up**2 - P_down**2))``. |
| `klinkenberg_apparent(k_inf, *, b, p_mean, c2)` | Apparent gas permeability ``k_app = k_inf*(1 + b/P + c2/P**2)``. |
| `klinkenberg_corrected(k_app, *, b, p_mean)` | Liquid-equivalent permeability ``k_inf = k_app/(1 + b/P)`` — the explicit inverse of :func:`klinkenberg_apparent` (no c2 term). |
| `fit_klinkenberg(p_mean, k_app)` | Fit ``(k_inf, b)`` from the Klinkenberg plot of ``k_app`` vs ``1/P``. |
| `mean_free_path(*, pressure, temperature, mu, molar_mass, d_collision)` | Gas mean free path, in one of two physically distinct forms. |
| `knudsen_number(mfp, pore_size)` | Knudsen number ``Kn = lambda/L_pore``. |
| `flow_regime(kn, *, cuts)` | Gas flow regime from the Knudsen number: ``"continuum"`` (Kn < cuts[0]), ``"slip"`` (< cuts[1]), ``"transition"`` (< cuts[2]), else ``"free-molecular"``. |
| `stress_permeability(k0, *, gamma, ncs, ncs0)` | Exponential permeability decline ``k = k0*exp(-gamma*(NCS - NCS0))``. |
| `net_confining_stress(total_stress, *, pore_pressure, biot)` | Terzaghi/Biot net confining stress ``sigma_eff = sigma_total - biot*Pp``. |
| `fit_stress_permeability(ncs, k)` | Fit ``(k0, gamma)`` from measured ``k`` vs net confining stress, a line in ``(NCS, ln k)``: ``k0 = exp(intercept)``, ``gamma = -slope``. |
| `kozeny_carman(phi, *, specific_surface, tau, c, grain_term)` | Kozeny-Carman permeability from porosity and specific surface area. |
| `kozeny_carman_ratio(k0, phi, phi0, *, grain_term)` | Kozeny-Carman porosity-permeability sensitivity (ratio/update form). |
| `winland_r35(k_md, phi)` | Winland-Kolodzie r35 pore-throat radius (um) at 35% mercury saturation. |
| `winland_permeability(r35, phi)` | Permeability (mD) from Winland r35 and porosity — the inverse of :func:`winland_r35`: ``log10(k) = (log10(r35) - 0.732 + 0.864*log10(phi_pct))/0.588`` (phi a fraction). |
| `swanson_permeability(sb_pc_apex, *, c, d)` | Swanson (1981) permeability from the MICP apex ``k = c*(Sb/Pc)_apex**d``. |
| `micp_apex(shg, pc)` | Pittman (1992) MICP apex: the maximum ``Shg/Pc`` ratio and its index. |
| `lucia_permeability(phi_g, rfn)` | Lucia (2007) rock-fabric-number permeability transform. |
| `lucia_rfn_from_swi(phi_g, swi)` | Lucia rock-fabric number from interparticle porosity and irreducible water saturation: ``log(RFN) = (3.1107 + 1.8834*log(phi_g) + log(Swi)) / (3.0634 + 1.4045*log(phi_g))``. |
| `poroperm_powerlaw(phi, *, a, b)` | Log-log poro-perm power law ``k = 10**(a + b*log10(phi))``. |
| `fit_poroperm(phi, k_md)` | Fit ``(a, b)`` for :func:`poroperm_powerlaw` by least squares of ``log10(k)`` on ``log10(phi)``. |
| `rqi(k_md, phi, *, c)` | Amaefule Reservoir Quality Index ``RQI = c*sqrt(k_md/phi)`` (um). |
| `phi_z(phi)` | Normalized porosity index ``phi_z = phi/(1 - phi)``. |
| `fzi(k_md, phi, *, c)` | Amaefule Flow Zone Indicator ``FZI = RQI/phi_z`` (um). |
| `k_from_fzi(phi, fzi_val, *, c)` | Permeability (mD) from FZI and porosity — the inverse of :func:`fzi`: ``k = fzi**2*phi**3 / (c**2*(1-phi)**2)``. |
| `classify_hfu(fzi_values, *, n_units)` | Classify samples into hydraulic flow units by log-spaced FZI thresholds. |
| `permeability_average(k, *, method, weights)` | Average permeability by ``method`` = ``"arithmetic"`` (parallel flow), ``"harmonic"`` (series flow), or ``"geometric"`` (default). |
| `wiener_bounds(k)` | Wiener (harmonic-arithmetic) bounds on the effective permeability of a heterogeneous medium: ``(k_harmonic, k_arithmetic)`` — the series lower and parallel upper bounds. |
| `diffusion_length(D, t, *, geometry_factor)` | Characteristic diffusion length ``L = sqrt(f*D*t)``. |
| `diffusion_time(L, D, *, geometry_factor)` | Diffusion / equilibration time ``t = L**2/(f*D)`` (inverse of :func:`diffusion_length`). |
| `stokes_einstein(T_K, mu, radius)` | Stokes-Einstein diffusivity ``D = kB*T/(6*pi*mu*r)`` of a sphere of radius ``r`` in a fluid of viscosity ``mu``. |
| `fick_flux(D, dc, dx)` | Fick's first law steady flux ``J = -D*dc/dx``. |
| `erfc_profile(c0, x, *, D, t)` | Semi-infinite diffusion concentration profile ``C(x,t) = C0*erfc(x/(2*sqrt(D*t)))`` (fixed ``C0`` at ``x=0``, far-field 0). |
| `early_time_uptake(D, t, half_length)` | Early-time fractional uptake ``Mt/Minf = (2/L)*sqrt(D*t/pi)`` for a slab of half-length ``L`` (valid to ~0.6). |
| `millington_quirk(D0, phi, sw)` | Millington-Quirk effective diffusivity ``D_eff = (phi*Sw)**(10/3)/phi**2 * D0``. |
| `pore_velocity(u_darcy, phi, sw)` | Interstitial (pore) velocity ``v = u_darcy/(phi*Sw)``. |
| `advect_disperse_1d(c_in, *, length, n_cells, t_total, v, D, k_rxn, n_steps)` | Explicit finite-difference 1-D advection-dispersion-reaction transport. |

## `petrolib.geochem_fluids.adsorption`

Gas adsorption isotherms and shale gas-in-place volumetrics.

| Name | Summary |
| --- | --- |
| `N_AVOGADRO` | constant |
| `V_MOLAR_STP_CM3` | constant |
| `N2_CROSS_M2` | constant |
| `langmuir(p, v_l, p_l, *, rho_b)` | Langmuir isotherm ``V = V_L * P/(P + P_L)`` (half capacity at ``P = P_L``). |
| `gibbs_excess(gc, rho_free, rho_ads)` | Gibbs excess correction ``G_excess = Gc*(1 - rho_free/rho_adsorbed)``. |
| `bet_isotherm(x_rel, vm, c)` | BET isotherm ``V = Vm*C*x/((1-x)*(1 + (C-1)*x))`` with ``x = P/P0``. |
| `bet_fit(x_rel, v_ads, *, cross_nm2)` | Linear BET fit -> ``(Vm, C, SSA_m2_g)``. |
| `free_gas(phi, sw, bg)` | Free-gas volume per bulk volume ``= phi*(1 - Sw)/Bg``. |
| `gas_in_place(area_m2, h_m, phi, sg, bg)` | Volumetric free gas-in-place ``= A*h*phi*Sg/Bg``. |

## `petrolib.geochem_fluids.asphaltene`

Asphaltene gravity gradients (Flory-Huggins-Zuo) and Yen-Mullins sizes.

| Name | Summary |
| --- | --- |
| `R_GAS` | constant |
| `G_ACCEL` | constant |
| `N_AVOGADRO` | constant |
| `YEN_MULLINS_DIAMETERS_M` | constant |
| `molar_volume_from_diameter(d_m)` | Molar volume (m3/mol) of a sphere of diameter ``d_m`` ``= (pi/6)*d**3 * NA``. |
| `diameter_from_molar_volume(va_m3mol)` | Sphere diameter (m) from molar volume ``= (6*(Va/NA)/pi)**(1/3)``. |
| `fhz_ratio(dz_m, va_m3mol, delta_rho, temp_k, *, entropy, dsol2_pa)` | Flory-Huggins-Zuo concentration ratio between two depths (gravity + optional terms). |
| `fhz_profile(depth_m, od_ref, depth_ref, va_m3mol, delta_rho, temp_k)` | Optical-density profile ``OD(z) = OD_ref * fhz_ratio(z - z_ref, ...)`` (gravity-only). |
| `fhz_invert_molar_volume(od1, z1, od2, z2, delta_rho, temp_k)` | Recover the asphaltene molar volume (m3/mol) from two OD/depth points. |
| `nearest_yen_mullins(d_m, *, rtol)` | Nearest Yen-Mullins class for a particle diameter (m) -> ``(name, ref_d, agrees)``. |

## `petrolib.geochem_fluids.brine`

Brine resistivity, salinity, capture cross-section, and density.

| Name | Summary |
| --- | --- |
| `ARPS_C` | constant |
| `_M_NACL` | constant |
| `rw75_from_salinity(nacl_ppm)` | Bateman-Konen water resistivity at 75 degF ``R75 = 0.0123 + 3647.5/C**0.955``. |
| `salinity_from_rw75(rw75_ohmm)` | NaCl ppm from the 75 degF resistivity -- exact inverse of ``rw75_from_salinity``. |
| `arps_correct(r1, t1, t2, *, unit)` | Arps temperature correction ``R2 = R1*(T1+c)/(T2+c)``. |
| `rw_from_salinity(nacl_ppm, temp, *, unit)` | Water resistivity at ``temp`` from NaCl ppm (75 degF reference). |
| `salinity_from_rw(rw, temp, *, unit)` | NaCl ppm from water resistivity at ``temp`` -- inverse of ``rw_from_salinity``. |
| `sigma_w_from_salinity(nacl_ppm, temp_c)` | Thermal-neutron capture cross-section of brine (c.u.). |
| `nacl_meq_per_liter(nacl_ppm)` | NaCl concentration in meq/L ``= ppm/58.44`` (monovalent, brine density ~1). |
| `brine_density_bw92(nacl_ppm, temp_c, press_mpa)` | Simplified Batzle-Wang (1992) brine density (kg/m3). |

## `petrolib.geochem_fluids.contamination`

OBM/filtrate contamination mixing and cleanup.

| Name | Summary |
| --- | --- |
| `mix_linear(p_v, p_f, eta)` | Linear mixing ``P = (1 - eta)*P_virgin + eta*P_filtrate``. |
| `contamination_fraction(p, p_v, p_f)` | Contamination fraction ``eta = (P - P_v)/(P_f - P_v)`` (inverse of the mixing rule). |
| `cleanup_powerlaw(v, eta0, v_star, *, exponent)` | Power-law cleanup ``eta(V) = eta0*(1 + V/V_star)**(-exponent)``. |
| `volume_to_target(eta0, v_star, eta_t, *, exponent)` | Pumped volume to reach contamination ``eta_t`` -- inverse of :func:`cleanup_powerlaw`. |

## `petrolib.geochem_fluids.core_geochem`

Core geochemistry: Dean-Stark saturations, oxide closure, and OSI.

| Name | Summary |
| --- | --- |
| `dean_stark(v_water, v_hc, *, v_bulk, v_pore)` | Dean-Stark porosity and saturations -> ``(phi, Sw, S_hc)``. |
| `oxide_closure(oxides, *, axis)` | Normalize an oxide (or elemental) suite to sum to one -> ``(closed, factor)``. |
| `osi(s1_mg_g, toc_wt_pct)` | Oil saturation index ``OSI = 100*S1/TOC`` (mg HC / g TOC; Jarvie producibility). |

## `petrolib.geochem_fluids.gradients`

Formation-pressure gradients and fluid contacts.

| Name | Summary |
| --- | --- |
| `G_STANDARD` | constant |
| `_PSI_TO_PA` | constant |
| `_FT_TO_M` | constant |
| `_BAR_TO_PA` | constant |
| `fit_pressure_gradient(depth_m, p)` | Least-squares pressure-depth gradient -> ``(dP/dz, P0)`` (slope, intercept). |
| `density_from_gradient(dpdz, *, p_unit)` | Fluid density (kg/m3) from a pressure gradient (no clipping). |
| `fluid_contact(depth_a, p_a, depth_b, p_b)` | Contact depth (m) where two fitted pressure gradients intersect. |

## `petrolib.geochem_fluids.mudgas`

Mud-gas composition ratios and fluid-type classification.

| Name | Summary |
| --- | --- |
| `normalize_composition(comps, *, axis)` | Sum-to-one (zero-safe) closure of a composition along ``axis``. |
| `wetness_ratio(c1, c2, c3, c4, c5, *, percent)` | Haworth wetness ``Wh = (C2+C3+C4+C5)/(C1+..+C5)`` (percent by default). |
| `balance_ratio(c1, c2, c3, c4, c5)` | Haworth balance ``Bh = (C1+..+C5)/(C3+C4+C5)`` (inf where the denominator is 0). |
| `character_ratio(c3, c4, c5)` | Haworth character ``Ch = (C4+C5)/C3`` (0 where C3 is 0). |
| `pixler_ratios(c1, c2, c3, c4, c5)` | Pixler ratios ``{C1/C2, C1/C3, C1/C4, C1/C5}`` (inf where a denominator is 0). |
| `bernard_ratio(c1, c2, c3)` | Bernard ratio ``C1/(C2+C3)`` (biogenic vs thermogenic indicator). |
| `apply_eec(comps, alphas)` | Extraction-efficiency correction: divide each component by its ``alpha``. |
| `classify_fluid_gor(gor_sm3, *, thresholds)` | Fluid type from GOR (Sm3/Sm3) against ascending thresholds. |
| `classify_fluid_wetness(wh_pct, bh)` | Haworth wetness/balance fluid typing (``wh`` in PERCENT). |

## `petrolib.geochem_fluids.pvt`

Gas PVT: pseudo-reduced properties, z-factor, density, and phase equilibrium.

| Name | Summary |
| --- | --- |
| `R_GAS` | constant |
| `MW_AIR` | constant |
| `MW_METHANE` | constant |
| `pseudo_reduced(p, t, ppc, tpc)` | Pseudo-reduced pressure and temperature ``(P/Ppc, T/Tpc)``. |
| `z_beggs_brill(ppr, tpr)` | Beggs-Brill (1973) gas compressibility factor from pseudo-reduced P/T. |
| `z_peng_robinson(p_pa, t_k, tc_k, pc_pa, omega, *, phase)` | Peng-Robinson z-factor (scalar) for the vapor or liquid root. |
| `gas_density(p_pa, t_k, *, m_kg_mol, z)` | Real-gas density ``rho = P*M/(z*R*T)`` (kg/m3). |
| `pressure_from_gas_density(rho, t_k, *, m_kg_mol, z)` | Pressure (Pa) from gas density -- inverse of ``gas_density``. |
| `mixture_mw(y, mw, *, axis)` | Mixture molecular weight ``sum(y_i*MW_i)`` over ``axis``. |
| `gas_gravity(mw_kg_mol)` | Gas specific gravity ``MW_gas/MW_air`` (air = 0.028964 kg/mol). |
| `wilson_k(p_pa, t_k, pc_pa, tc_k, omega)` | Wilson (1969) K-value estimate ``(Pc/P)*exp(5.373*(1+omega)*(1 - Tc/T))``. |
| `rachford_rice(z, k, *, tol, max_iter)` | Rachford-Rice vapor fraction ``beta`` solving ``sum(z*(k-1)/(1+beta*(k-1))) = 0``. |

## `petrolib.geochem_fluids.solubility`

Gas solubility in brine and oil (Henry / Setschenow / Duan-Sun style).

| Name | Summary |
| --- | --- |
| `R_GAS` | constant |
| `henry_constant_co2(t_k)` | CO2 Henry constant (MPa/molality), Duan-Sun style vs temperature (K). |
| `setschenow_factor(m_nacl, t_k, *, ks25)` | Setschenow salting-out activity factor ``exp(2*ks*m)``. |
| `co2_solubility_brine(p_mpa, t_k, m_nacl, *, m_ch4)` | CO2 solubility in brine (mol/kg): ``P/(H*gamma) * 1/(1 + 0.6*m_CH4)``. |
| `henry_solubility_ln(p_mpa, t_k, a, b, dh_j_mol)` | Krichevsky-Kasarnovsky ``ln(x) = a + b*ln(P) - dH/(R*T)`` (gas-in-oil mole fraction). |

## `petrolib.integrity_drilling`

Well integrity and drilling: cement bond, casing condition, leaks, mud gas.

| Name | Summary |
| --- | --- |
| `V_STEEL` | Compressional velocity of casing steel, m/s. |
| `PSI_PER_FT_PER_SG` | Fresh-water pressure gradient per specific gravity, psi/ft. |
| `BAR_PER_SG_M` | Hydrostatic gradient per specific gravity, bar/m (rho_w * g / 1e5). |
| `G_STD` | Standard gravity, m/s^2. |
| `acoustic_impedance(rho, v, *, rho_unit)` | Acoustic impedance ``Z = rho*v`` in MRayl. |
| `reflection_coefficient(z1, z2)` | Normal-incidence reflection coefficient ``R = (Z2 - Z1)/(Z2 + Z1)`` (z1 incident side). |
| `transmission_energy(z1, z2)` | Transmitted energy fraction across an interface, ``1 - R^2``. |
| `attenuation_db(a_near, a_far, spacing_m)` | Amplitude attenuation ``20*log10(A_near/A_far)`` in dB (dB/m if ``spacing_m`` given). |
| `attenuation_coefficient(a0, ax, x_m)` | Exponential attenuation coefficient ``ln(A0/Ax)/x`` (1/m) from ``A = A0*exp(-alpha*x)``. |
| `bond_index(measured, free_pipe, well_bonded, *, method, ...)` | Cement bond index in [0, 1] from a CBL measurement and its two anchors. |
| `bond_index_combined(bi_a, bi_b, *, weights, corrector)` | Weighted combination of two bond indicators, clipped to [0, 1]. |
| `classify_annulus(z_mrayl, *, gas_max, liquid_max, cement_min)` | Classify the annulus fill from its acoustic impedance (MRayl). |
| `CEMENT_QUALITY_THRESHOLDS` | Cement-quality (good, fair) impedance thresholds in MRayl by cement type. |
| `cement_quality_score(z_mrayl, *, cement_type, thresholds)` | Continuous cement-quality score in [0, 1] from annulus impedance (MRayl). |
| `classify_cement_from_cbl(relative_amp, *, good_max, medium_max)` | CBL relative-amplitude cement classes ``'Good'``/``'Medium'``/``'Poor'``. |
| `casing_resonance_frequency(thickness_m, *, v, n, correction)` | Casing plate thickness-resonance frequency ``f = corr * n * v / (2 d)`` (Hz). |
| `casing_thickness_from_resonance(freq_hz, *, v, n, correction)` | Casing wall thickness (m) from the thickness-resonance frequency (Hz). |
| `metal_loss_pct(measured, nominal)` | Casing metal loss ``(1 - measured/nominal)*100``, clipped to [0, 100] %. |
| `casing_condition(loss_pct, *, bands)` | Casing condition class from metal loss %: good / fair / poor / critical. |
| `remaining_life_years(thickness_mm, min_acceptable_mm, rate_mm_per_yr)` | Remaining casing life ``(t - t_min)/rate`` (years), floored at 0. |
| `corrosion_front_depth(t, K)` | Diffusion-limited corrosion front ``x = K*sqrt(t)`` (units follow ``K``). |
| `microannulus_omega(r_casing_m, aperture_m)` | Annular-gap fourth moment ``Omega`` (m^4) of a microannulus. |
| `leak_rate_liquid(aperture_m, r_casing_m, dp_pa, length_m, mu_pa_s, *, ...)` | Liquid microannulus leak rate (m^3/s), gravity-corrected Hagen-Poiseuille. |
| `leak_rate_gas(aperture_m, r_casing_m, p_in_pa, p_out_pa, length_m, ...)` | Gas microannulus leak rate (m^3/s at outlet), isothermal compressible flow. |
| `cubic_law_conductivity(aperture_m, *, rho, mu, g)` | Parallel-plate crack hydraulic conductivity ``K = rho g w^3 / (12 mu)`` (m/s). |
| `haworth_ratios(c1, c2, c3, c4, c5, *, percent)` | Haworth wetness / balance / character ratios ``(Wh, Bh, Ch)``. |
| `pixler_ratios(c1, c2, c3, c4, c5)` | Pixler component ratios ``C1/C2 .. |
| `classify_fluid_haworth(wh, bh, ch, *, n_classes)` | Classify reservoir fluid from Haworth ratios (``wh`` in percent). |
| `normalize_gas(total_gas, rop, flow, bit_diameter, *, mud_weight, ...)` | Normalize a total-gas reading for drilling parameters. |
| `hydrostatic_pressure(tvd_m, *, rho)` | Hydrostatic pressure ``rho * g * TVD`` (Pa), g = 9.80665. |
| `hydrostatic_pressure_psi(tvd_ft, *, sg)` | Hydrostatic pressure in psi from TVD in ft: ``0.433 * SG * TVD``. |
| `hydrostatic_pressure_bar(tvd_m, *, sg)` | Hydrostatic pressure in bar from TVD in m: ``0.0980665 * SG * TVD``. |
| `overburden_pressure(tvd_m, rho_bulk, *, water_depth_m, rho_sw)` | Overburden pressure (Pa): seawater column plus constant-density sediment. |
| `overburden_pressure_psi(water_depth_ft, sediment_depth_ft, *, sw_sg, sediment_sg)` | Deepwater overburden in psi: ``0.433 * (SG_sw*wd + SG_sed*sd)``. |
| `eaton_pore_pressure(overburden, hydrostatic, observed, normal, *, exponent, ...)` | Eaton pore pressure ``Pp = OB - (OB - Pn) * ratio^exponent``. |
| `bowers_pore_pressure(velocity, overburden, *, A, B, unloading, U, v0, sigma_max)` | Bowers pore pressure from sonic velocity (ft/s) and overburden (psi). |
| `drilling_window_margin(pore, frac)` | Drilling-window width ``frac - pore`` (same units as the inputs). |
| `within_drilling_window(ecd, pore, frac)` | True when the ECD sits strictly inside the pore/frac window. |
| `mudcake_thickness(t_s, *, k_mc_m2, dp_pa, mu_pa_s, solids_ratio, model, ...)` | Mudcake thickness growth (m) in one of the corpus conventions. |

## `petrolib.inversion_numerics.costs`

Cost / misfit functions and regularization-parameter schedules.

| Name | Summary |
| --- | --- |
| `misfit(sim, obs, *, weights, kind, log_space)` | Data misfit between a simulation and observations. |
| `reg_lambda_multiplicative(misfit_value, alpha, beta, lam_max)` | Habashy-Abubakar multiplicative cooling ``lam = min(alpha*misfit^beta, lam_max)``. |
| `reg_lambda_brd(a, b, chi2_target, bracket, *, max_iter)` | Discrepancy-principle regularization weight by geometric bisection. |

## `petrolib.inversion_numerics.fitting`

Curve fitting: line / power-law / exponential-approach / cosine.

| Name | Summary |
| --- | --- |
| `class LineFit` | Result of :func:`fit_line` (in the transformed space). |
| `fit_line(x, y, *, xform, yform)` | Degree-1 least-squares fit ``y = slope*x + intercept`` with R^2. |
| `fit_powerlaw_decay(x, y, exponent)` | Fit ``y = a * x**(-b)`` -> ``(a, b)``. |
| `fit_exponential_approach(t, y, three_point)` | Fit an exponential approach ``y = asymptote + (y0-asymptote)*exp(-t/tau)``. |
| `fit_cosine(az, y)` | Azimuthal cosine-harmonic fit ``y = mean + amp*cos(az - phase)``. |

## `petrolib.inversion_numerics.linear`

Linear inversion: least squares, Tikhonov, SVD, unmixing, operators.

| Name | Summary |
| --- | --- |
| `difference_operator(n, order)` | Finite-difference smoothness operator ``L`` of shape ``(n-order, n)``. |
| `condition_number(a)` | 2-norm condition number ``s_max/s_min`` from the singular values. |
| `svd_solve(a, b, rank)` | Least-squares / truncated-SVD solve of ``A x = b`` (rank-limited pseudo-inverse). |
| `tikhonov_solve(a, b, lam, *, reg_op, x_ref, sigma, nonneg)` | Tikhonov-regularized least squares. |
| `map_estimate(g, d, noise_var, prior_strength, ell, m_prior)` | Bayesian MAP estimate and posterior covariance for a linear-Gaussian model. |
| `nnls_solve(a, b, reg)` | Non-negative least squares (scipy), optionally ridge-augmented by ``reg``. |
| `ista_l1(a, b, eta, nonneg, max_iter)` | Sparse L1 inversion by ISTA (iterative soft-thresholding). |
| `project_simplex(v, total)` | Euclidean projection of ``v`` onto the simplex ``{x>=0, sum(x)=total}``. |
| `unmix(measured, response_matrix, *, sigma, closure, nonneg, ...)` | Mineral / spectral unmixing ``measured ~ response_matrix @ x``. |
| `convolution_matrix(kernel, n)` | Row-normalized ``(n, n)`` convolution (blurring) matrix for a centered kernel. |
| `deconvolve(d, g, rank)` | Deconvolve ``d = G x`` for ``x`` via (rank-limited) pseudo-inverse. |

## `petrolib.inversion_numerics.nonlinear`

Nonlinear inversion: finite-difference derivatives, LM, Occam, search.

| Name | Summary |
| --- | --- |
| `class InvResult` | Result of a deterministic nonlinear inversion. |
| `fd_jacobian(forward, m, eps, scheme, relative)` | Finite-difference Jacobian ``d forward / d m`` (``n_data x n_param``). |
| `fd_gradient(f, x, **kw)` | Finite-difference gradient of a scalar objective ``f``. |
| `levenberg_marquardt(forward, data, m0, *, bounds, log_params, lam0, lam_up, ...)` | Levenberg-Marquardt least-squares inversion of ``forward(m) ~ data``. |
| `occam(forward, data, m0, noise_level, *, reg_order, lam0, ...)` | Occam smoothest-model iteration toward a target data misfit. |
| `grid_search(forward, data, grids, misfit)` | Brute-force grid search minimizing the misfit over the Cartesian ``grids``. |
| `multistart(solver, bounds, n_starts, seed, aggregate)` | Run ``solver`` from many random starts and aggregate the models. |
| `feasible_set_sampling(forward, data, m_center, bounds, noise_level, ...)` | Equivalence / feasible-set sampling around a solution for uncertainty bounds. |

## `petrolib.inversion_numerics.optimize`

Global / gradient optimization: particle swarm and gradient descent.

| Name | Summary |
| --- | --- |
| `pso(objective, bounds, n_particles, n_iter, omega, ...)` | Particle-swarm optimization over box ``bounds``. |
| `gradient_descent(f, x0, lr, backtracking, max_iter, tol)` | Finite-difference gradient descent of a scalar objective ``f``. |

## `petrolib.inversion_numerics.pde`

Grid PDE solvers: 2D effective conductivity and 1D diffusion.

| Name | Summary |
| --- | --- |
| `cfl_number(alpha, dt, dx)` | Diffusion CFL number ``alpha*dt/dx^2`` (explicit stability needs ``<= 0.5``). |
| `effective_conductivity_2d(sigma_map, n_iter, tol)` | Effective vertical conductivity of a 2D conductivity map. |
| `diffusion_step_1d(u, alpha, dt, dx, source, bc)` | One explicit finite-difference step of the 1D diffusion equation. |

## `petrolib.inversion_numerics.stochastic`

Stochastic inversion: likelihoods, priors, MCMC, MALA, ensemble methods.

| Name | Summary |
| --- | --- |
| `class Chain` | MCMC output: posterior ``samples`` and the ``acceptance`` fraction. |
| `gaussian_loglik(obs, pred, sigma, weights, log_space)` | Gaussian log-likelihood ``-0.5*sum(w*((obs-pred)/sigma)^2)``. |
| `uniform_logprior(x, bounds)` | Uniform (box) log-prior: ``0`` inside ``bounds``, ``-inf`` outside. |
| `soft_envelope_logprior(x, lo, hi)` | Soft box log-prior: ``0`` inside ``[lo, hi]`` and a quadratic penalty outside. |
| `metropolis(log_post, x0, step, n_samples, *, log_space, burn_in, seed)` | Random-walk Metropolis sampler of ``log_post``. |
| `mala(log_post, x0, step, n_samples, seed)` | Metropolis-adjusted Langevin (MALA) sampler with a finite-difference drift. |
| `enkf_update(ens, obs, obs_cov, obs_op, seed)` | Stochastic (perturbed-observation) ensemble-Kalman update of ``ens``. |
| `lm_enrml(prior_mean, prior_cov, obs, obs_cov, forward, n_ens, ...)` | Levenberg-Marquardt ensemble randomized maximum likelihood (LM-EnRML). |

## `petrolib.ml_stats`

Statistics and numpy-only machine-learning helpers.

| Name | Summary |
| --- | --- |
| `rmse(y_true, y_pred)` | Root-mean-square error. |
| `mae(y_true, y_pred)` | Mean absolute error. |
| `rss(y_true, y_pred)` | Residual sum of squares. |
| `mape(y_true, y_pred, *, as_percent)` | Mean absolute percentage error (the articles' AAPE). |
| `r2_score(y_true, y_pred, *, eps)` | Coefficient of determination R² (NOT the correlation coefficient R). |
| `pearson_r(x, y, *, eps)` | Pearson correlation coefficient R (NOT R²). |
| `accuracy(y_true, y_pred)` | Fraction of matching labels. |
| `confusion_matrix(y_true, y_pred, *, labels)` | Confusion matrix ``C[i, j]`` = count of true class i predicted as j. |
| `precision_recall_f1(y_true, y_pred, *, positive, zero_division)` | Binary precision, recall and F1 for the ``positive`` class. |
| `zscore(x, *, axis, eps)` | Standardize to zero mean and unit standard deviation. |
| `minmax(x, *, axis, lo, hi, eps)` | Rescale to the range [lo, hi] (min-max normalization). |
| `affine_rescale(x, *, src_lo, src_hi, dst_lo, dst_hi)` | Map the source range [src_lo, src_hi] linearly onto [dst_lo, dst_hi]. |
| `fit_line(x, y)` | Degree-1 least-squares fit; returns ``(slope, intercept)`` — in that order, always. |
| `fit_powerlaw(x, y)` | Fit ``y = coefficient * x**exponent``; returns ``(coefficient, exponent)``. |
| `ols(X, y, *, intercept)` | Multivariate ordinary least squares; returns ``(coef, intercept)``. |
| `predict_linear(X, coef, intercept)` | Evaluate the linear model from :func:`ols`. |
| `kmeans(x, k, *, weights, max_iter)` | Deterministic k-means; returns ``(labels, centroids)``. |
| `silhouette_score(x, labels)` | Mean silhouette coefficient over all samples (O(N²) pairwise). |
| `pca(x, *, n_components)` | Principal component analysis via SVD of the centered data. |

## `petrolib.nmr`

Nuclear magnetic resonance: relaxation physics, T2 statistics, forward models, permeability transforms, and fluid typing.

| Name | Summary |
| --- | --- |
| `_TINY` | constant |
| `diffusion_relaxation_rate(D, *, G, TE, gamma)` | Diffusion relaxation rate ``1/T2_D = (gamma*G*TE)**2 * D / 12``. |
| `relaxation_rate(*, t2_bulk, rho, s_over_v, D, G, TE, gamma)` | Brownstein-Tarr relaxation rate (bulk + surface + diffusion). |
| `t2_apparent(*, t2_bulk, rho, s_over_v, D, G, TE, gamma)` | Apparent relaxation time ``T2 = 1/relaxation_rate(...)`` — the inverse of :func:`relaxation_rate` with the same keywords. |
| `combine_relaxation_times(*times_ms)` | Combine relaxation times as parallel rates ``1/T = sum(1/Ti)``. |
| `surface_to_volume(t2_ms, *, rho, t2_bulk)` | Surface-to-volume ratio from a relaxation time, inverting Brownstein-Tarr. |
| `pore_radius_from_t2(t2_ms, *, rho, shape_factor, t2_bulk)` | Pore radius from a relaxation time ``r = shape_factor/(S/V)``. |
| `surface_relaxivity_from_pore(t2_ms, radius_um, *, shape_factor)` | Surface relaxivity from a relaxation time and known pore size. |
| `larmor_frequency(b0_T, *, gamma)` | Larmor frequency ``f = gamma*B0/(2*pi)`` (Hz). |
| `t2_logmean(t2_ms, amplitude)` | Amplitude-weighted geometric mean ``T2LM = exp(sum(A*ln T2)/sum(A))``. |
| `total_porosity(amplitude)` | Total NMR porosity as the summed T2 amplitude ``phi = sum(A)``. |
| `t2_partition(t2_ms, amplitude, *, cutoffs_ms, fractions)` | Partition T2 amplitude into ``len(cutoffs)+1`` bands by T2 cutoffs. |
| `bvi_ffi(t2_ms, amplitude, *, cutoff_ms)` | Bound (BVI) and free (FFI) fluid split at a single T2 cutoff. |
| `cpmg_kernel(t_ms, t2_grid_ms)` | CPMG forward kernel ``K[i,j] = exp(-t_i/T2_j)``. |
| `multiexp_decay(t_ms, amplitudes, t2_ms, *, noise, rng)` | Multi-exponential CPMG decay ``M(t) = sum_i A_i*exp(-t/T2_i)``. |
| `t1_saturation_recovery(t_ms, m0, t1_ms)` | Saturation-recovery magnetization ``M(t) = M0*(1 - exp(-t/T1))``. |
| `t1_inversion_recovery(t_ms, m0, t1_ms)` | Inversion-recovery magnetization ``M(t) = M0*(1 - 2*exp(-t/T1))``. |
| `t1t2_kernel(t_echo_ms, t_wait_ms, t1_grid_ms, t2_grid_ms, *, mode)` | 2D T1-T2 Kronecker kernel ``kron(K_T1, K_T2)``. |
| `fit_t1(t_ms, signal, *, model)` | Fit ``(M0, T1)`` from a T1 recovery curve (scipy ``curve_fit``, lazy). |
| `timur_coates(phi, ffi, bvi, *, C, m, n, form)` | Timur-Coates NMR permeability (mD). |
| `sdr(phi, t2lm_ms, *, a, m, n, rho_um_s)` | Schlumberger-Doll-Research NMR permeability (mD). |
| `timur(phi, swirr, *, a, b, c)` | Timur (1968) NMR permeability ``k = a*phi**b/Swirr**c`` (mD). |
| `t1_t2_ratio(t1, t2)` | T1/T2 ratio for fluid typing. |
| `classify_t1t2(t1t2, *, cutoff)` | Boolean hydrocarbon flag ``T1/T2 >= cutoff`` (True = hydrocarbon/bound). |
| `partition_t1t2_map(t1t2, amplitudes, *, cutoff)` | Split a T1-T2 amplitude map into ``(v_hc, v_water)`` at a T1/T2 cutoff. |
| `nmr_saturation(v_fluid, v_total)` | NMR fluid saturation ``S = V_fluid/V_total``. |
| `hydrogen_index(rho, n_protons, mol_weight, *, rho_w, n_w, m_w)` | Hydrogen index vs water ``HI = (rho*n/M)/(rho_w*n_w/M_w)``. |
| `porosity_hi_correction(phi_apparent, hi)` | HI-corrected porosity ``phi_true = phi_apparent/HI``. |
| `bpp_spectral_density(omega, tau_c)` | Bloembergen-Purcell-Pound spectral density ``J = tau_c/(1+(omega*tau_c)**2)``. |
| `bpp_t1_t2(omega0, tau_c, *, dipolar_constant)` | BPP relaxation times ``(T1, T2)`` from the correlation time. |
| `mitra_short_time(d0, t, s_over_v, *, normalized)` | Mitra short-time restricted diffusion ``D(t)/D0 = 1 - (4/(9*sqrt(pi)))*(S/V)*sqrt(D0*t)``. |
| `tortuosity(d0, d_inf)` | Diffusive tortuosity ``tau = D0/D_inf`` (free / restricted-plateau diffusivity). |

## `petrolib.nuclear`

Nuclear logging: capture cross-section, attenuation, density, GR, neutron.

| Name | Summary |
| --- | --- |
| `NA` | constant |
| `SIGMA_TAU_CONST` | constant |
| `THERMAL_NEUTRON_VELOCITY_CM_S` | constant |
| `RHOE_TO_RHOB_SLOPE` | constant |
| `RHOE_TO_RHOB_OFFSET` | constant |
| `sigma_forward(phi, sw, *, sigma_ma, sigma_w, sigma_hc, vsh, sigma_sh)` | Volumetric capture cross-section ``Sigma`` (c.u.) of a rock. |
| `sigma_forward_3phase(phi, so, sg, sw, *, sigma_oil, sigma_gas, sigma_w, ...)` | Three-phase (oil/gas/water) capture cross-section ``Sigma`` (c.u.). |
| `sw_from_sigma(sigma_t, phi, *, sigma_ma, sigma_w, sigma_hc, vsh, ...)` | Water saturation by inverting :func:`sigma_forward` for ``Sw``. |
| `delta_sw_timelapse(sigma_base, sigma_mon, phi, *, sigma_w_base, ...)` | Monitor-survey water saturation from a time-lapse Sigma pair. |
| `sigma_sensitivity(phi, sigma_w, sigma_hc)` | Saturation sensitivity ``dSigma/dSw = phi*(Sigma_w - Sigma_hc)`` (c.u.). |
| `sigma_w_from_salinity(ppm_nacl, *, temperature_c, model)` | Brine capture cross-section ``Sigma_w`` (c.u.) from NaCl salinity (ppm). |
| `number_density(rho_g_cc, wfrac, atomic_mass)` | Atomic number density ``N = rho*NA*w/A`` (1/cm3) for a mass fraction ``w``. |
| `macroscopic_sigma(number_densities, micro_barns, *, units)` | Macroscopic capture ``Sigma`` from number densities and micro cross-sections. |
| `sigma_from_tau(tau_us)` | Capture cross-section from thermal decay time ``Sigma = 4550/tau`` (c.u.). |
| `tau_from_sigma(sigma_cu)` | Thermal decay time from capture cross-section ``tau = 4550/Sigma`` (us). |
| `pnc_decay(t_us, n0, sigma_cu, *, background)` | Pulsed-neutron capture decay ``N(t) = N0*exp(-Sigma*v*t) + background``. |
| `sigma_from_decay_fit(t_us, counts, *, fit_window)` | Capture cross-section from a log-linear fit of a capture-decay curve. |
| `beer_lambert(i0, mu, x)` | Beer-Lambert transmitted intensity ``I = I0*exp(-mu*x)``. |
| `mu_from_intensity(i0, i, x)` | Linear attenuation coefficient ``mu = ln(I0/I)/x`` from an intensity pair. |
| `attenuation_map(i, i_ref)` | Optical-density attenuation ``-ln(I/I_ref)`` (zero-guarded ratio). |
| `gamma_count(rho, *, n0, mu_mass, spacing_cm)` | Gamma count rate vs density ``N = N0*exp(-mu_mass*rho*spacing)``. |
| `density_from_count(count, *, n0, mu_mass, spacing_cm)` | Density from a gamma count ``rho = ln(N0/N)/(mu_mass*spacing)`` (inverse). |
| `dual_detector_density(near, far, *, a, b)` | Dual-detector density ``rho = a + b*ln(near/far)`` (count-ratio calibration). |
| `spine_ribs(rho_ls, rho_ss, *, rib_coeffs)` | Spine-and-ribs compensated density -> ``(rho_b, drho)``. |
| `electron_density_index(z, a, rho_m)` | Electron density ``rho_e = (2*Z/A)*rho_m`` of a single element/compound. |
| `electron_density_mixture(z, a, mass_frac, rho_m)` | Mixture electron density ``rho_e = 2*sum(w_i*Z_i/A_i)*rho_m``. |
| `rhob_from_rhoe(rho_e)` | Bulk density from electron density ``rho_b = 1.0704*rho_e - 0.1883``. |
| `rhoe_from_rhob(rho_b)` | Electron density from bulk density ``rho_e = (rho_b + 0.1883)/1.0704`` (inverse). |
| `gr_api(k_pct, u_ppm, th_ppm, *, coeff)` | Total (spectral) gamma ray API ``SGR = ck*K + cu*U + cth*Th``. |
| `cgr_api(k_pct, th_ppm, *, coeff)` | Computed (uranium-free) gamma ray API ``CGR = ck*K + cth*Th``. |
| `hydrogen_index_fluid(fluid, *, rho_gas, hi_oil)` | Hydrogen index of a pore fluid (``HI``, not ``HI*phi``). |
| `hydrogen_index_chemical(rho, n_protons, mol_weight, *, rho_ref, n_ref, mw_ref)` | Hydrogen index from chemistry, normalised to fresh water. |
| `hi_mix(phi, *, hi_fluid, hi_matrix)` | Volumetric hydrogen-index log ``HI = phi*HI_fluid + (1-phi)*HI_matrix``. |
| `phi_from_hi(hi_log, *, hi_matrix, hi_fluid)` | Porosity from a hydrogen-index log ``phi = (HI - HI_matrix)/(HI_fluid - HI_matrix)``. |
| `phi_hi_correction(phi_apparent, hi)` | Hydrogen-index porosity correction ``phi = phi_apparent/HI`` (neutron & NMR). |
| `collision_parameter(a_mass)` | Neutron collision parameter ``alpha = ((A-1)/(A+1))**2``. |
| `average_lethargy_gain(a_mass)` | Mean logarithmic energy decrement ``xi = 1 + alpha/(1-alpha)*ln(alpha)``. |
| `moderating_power(a_mass, sigma_s)` | Moderating power ``xi*Sigma_s`` (slowing-down power). |
| `slowing_down_length_empirical(phi, e_mev, *, ls0, e_ref)` | Empirical slowing-down length ``Ls = Ls0*(1 - 0.6*phi)*sqrt(E/E_ref)`` (cm). |
| `phi_from_ls(ls, e_mev, *, ls0, e_ref)` | Porosity from slowing-down length (inverse of :func:`slowing_down_length_empirical`). |
| `transport_length_mix(vol_fracs, lengths)` | Harmonic transport/migration-length mix ``L = 1/sum(v_i/L_i)``. |
| `_LM_ENDPOINTS` | constant |
| `phi_n_from_lm(lm_star, *, lithology)` | Neutron porosity from migration length ``phi = (Lm_ma - Lm)/(Lm_ma - Lm_fl)``. |
| `compensated_neutron_porosity(near, far, *, a, b, c)` | Compensated neutron porosity ``phi = a + b*lnR + c*lnR**2`` (p.u.), ``R=near/far``. |
| `co_ratio(c_yield, o_yield)` | Carbon-oxygen ratio ``COR = Y_C/Y_O``. |
| `so_from_co(cor, cor_water, cor_oil, *, clip)` | Oil saturation from C/O endpoint interpolation ``So = (COR-COR_w)/(COR_o-COR_w)``. |
| `co_forward_3phase(phi, so, sg, sw, *, c_oil, c_gas, c_mat, o_w, o_mat)` | Inelastic C/O forward model ``COR = C/(O+1e-9)`` from a 3-phase saturation. |
| `yields_to_weights(fy2w, s, y)` | Elemental weight fraction from a spectroscopy yield ``W = FY2W*S*Y``. |
| `weights_to_yields(fy2w, s, w)` | Spectroscopy yield from a weight fraction ``Y = W/(FY2W*S)`` (inverse). |
| `toc_from_yield(y_toc, calib)` | Total organic carbon from a carbon yield ``TOC = calib*Y_TOC``. |
| `counting_precision(counts)` | Relative counting precision ``1/sqrt(N)`` (Poisson). |
| `counting_sigma(value, counts)` | Absolute counting uncertainty ``value/sqrt(N)`` (Poisson). |
| `decay_constant(half_life)` | Radioactive decay constant ``lambda = ln(2)/T_half``. |
| `radioactive_decay(n0, t, half_life)` | Radioactive decay ``N = N0*exp(-lambda*t)`` with ``lambda = ln(2)/T_half``. |
| `activity(n, half_life)` | Radioactive activity ``A = lambda*N`` with ``lambda = ln(2)/T_half``. |

## `petrolib.porosity_lithology`

Porosity, shale volume, lithology mixing, Thomas-Stieber, TOC and net pay.

| Name | Summary |
| --- | --- |
| `gamma_ray_index(gr, gr_clean, gr_shale, *, clip)` | Gamma-ray shale index ``IGR = (GR-GR_clean)/(GR_shale-GR_clean)``. |
| `vshale_from_gr(gr, gr_clean, gr_shale, *, method, gcur, clip)` | Shale volume from the gamma-ray index by one of several transforms. |
| `vshale_neutron_density(phi_n, phi_d, phi_n_sh, phi_d_sh, *, clip)` | Shale volume from neutron-density separation. |
| `combine_clay_indicators(*vcl, how)` | Combine several clay-volume indicators into one estimate. |
| `density_porosity(rho_b, rho_ma, rho_fl, *, clip)` | Density porosity ``phi_D = (rho_ma-rho_b)/(rho_ma-rho_fl)``. |
| `neutron_density_porosity(phi_n, phi_d, *, method)` | Combine neutron and density porosity. |
| `effective_porosity(phi_t, vsh, phi_sh, *, clip)` | Shale-corrected effective porosity ``phi_e = phi_t - Vsh*phi_sh``. |
| `porosity_from_volumes(v_bulk, v_grain)` | Core porosity ``phi = (V_bulk - V_grain)/V_bulk``. |
| `boyle_porosity(rho_bulk, rho_grain)` | Boyle's-law density porosity ``phi = 1 - rho_bulk/rho_grain``. |
| `boyle_grain_volume(v_cell, v_expansion, p1, p2)` | Boyle's-law grain volume ``V_grain = V_cell - V_exp/(p1/p2 - 1)``. |
| `fluid_summation_porosity(bv_oil, bv_water, bv_gas)` | Fluid-summation porosity ``phi = BV_oil + BV_water + BV_gas``. |
| `porosity_from_voxel_count(pore_voxels, total_voxels)` | Digital-rock porosity ``phi = pore_voxels/total_voxels``. |
| `gravimetric_porosity(m_dry, m_sat, v_bulk, *, rho_fluid)` | Mass-balance porosity ``phi = (m_sat-m_dry)/(rho_fluid*V_bulk)``. |
| `ct_porosity(mu, mu_grain, mu_fluid, *, clip)` | Porosity from CT attenuation ``phi = (mu_grain-mu)/(mu_grain-mu_fluid)``. |
| `ct_saturation(mu, mu_dry, mu_sat, *, clip)` | Saturation from CT attenuation ``S = (mu-mu_dry)/(mu_sat-mu_dry)``. |
| `matrix_density_from_volumes(v, rho)` | Grain density by volume-weighted (arithmetic) mixing ``sum(v_i*rho_i)``. |
| `matrix_density_from_masses(w, rho, *, w_kerogen, rho_kerogen)` | Grain density by mass-weighted (harmonic) mixing. |
| `fluid_density(saturations, rhos)` | Pore-fluid density ``rho_fl = sum(S_i*rho_i)`` over the last axis. |
| `bulk_density(phi, rho_ma, rho_fl, *, v_k, rho_k)` | Forward bulk density ``rho_b = (1-phi-v_k)*rho_ma + v_k*rho_k + phi*rho_fl``. |
| `log_response(volumes, endpoints)` | Linear tool-response mixing ``M = sum(V_j * R_j)`` = ``endpoints @ volumes``. |
| `electron_density_to_bulk(rho_e, *, a, b)` | Electron- to bulk-density calibration ``rho_b = a*rho_e + b``. |
| `volume_to_weight_fractions(v, rho)` | Convert volume fractions to weight fractions ``w_i = v_i*rho_i/sum(v*rho)``. |
| `weight_to_volume_fractions(w, rho)` | Convert weight fractions to volume fractions ``v_i = (w_i/rho_i)/sum(w/rho)``. |
| `thomas_stieber_phit(v_lam, phi_sand, phi_sh, *, v_disp)` | Thomas-Stieber total porosity vs laminar shale volume. |
| `thomas_stieber_vlam(phi_t, phi_sand, phi_sh, *, clip)` | Inverse Thomas-Stieber laminar shale ``V_lam=(phi_sand-phi_t)/(phi_sand-phi_sh)``. |
| `thomas_stieber_sand_porosity(phi_t, v_lam, phi_sh)` | Sand (net) porosity ``phi_sand = (phi_t - V_lam*phi_sh)/(1 - V_lam)``. |
| `kerogen_mass_fraction(toc, *, k)` | Organic-matter mass fraction from TOC ``OM = k*TOC`` (carbon fraction 1/k). |
| `kerogen_volume_from_toc(toc, rho_ref, *, rho_k, carbon_frac)` | Kerogen volume fraction ``V_k = (TOC/carbon_frac)*rho_ref/rho_k``. |
| `toc_schmoker(rho_b, *, a, b)` | Schmoker density TOC ``TOC[wt%] = a/rho_b - b``. |
| `toc_passey_dlogr(rt, overlay, rt_base, overlay_base, *, lom, k_overlay)` | Passey Delta-log-R TOC ``TOC = dlogR * 10**(2.297 - 0.1688*LOM)``. |
| `multimineral_solve(measured, endpoints, *, sigma, closure, closure_weight, ...)` | Solve tool responses ``endpoints @ v = measured`` for mineral volumes ``v``. |
| `bulk_volume_water(phi, sw)` | Bulk volume water ``BVW = phi*Sw`` (constant in a Buckles zone). |
| `hydrocarbon_pore_volume(phi, sw)` | Hydrocarbon pore volume fraction ``HCPV = phi*(1-Sw)``. |
| `pay_flag(phi, vsh, sw, *, phi_cut, vsh_cut, sw_cut)` | Boolean net-pay flag from porosity, shale, and saturation cutoffs. |
| `interval_thickness(depth, flag)` | Summed thickness ``sum(\|dz\|)`` over samples where ``flag`` is true. |
| `net_to_gross(depth, net_flag, gross_flag)` | Net-to-gross ratio ``net_thickness / gross_thickness``. |

## `petrolib.relperm_wettability`

Relative permeability and displacement: Corey/Brooks-Corey/LET, mobility, fractional flow, and the Buckley-Leverett/Welge construction.

| Name | Summary |
| --- | --- |
| `_TINY` | constant |
| `normalized_saturation(s, sr, snr, *, clip)` | Normalized (effective) saturation ``Se = (S - Sr)/(1 - Sr - Snr)``. |
| `corey_krw(sw, *, swr, sor, krw_max, nw, clip)` | Corey water relative permeability ``krw = krw_max * Se**nw``. |
| `corey_kro(sw, *, swr, sor, kro_max, no, clip)` | Corey oil relative permeability ``kro = kro_max * (1 - Se)**no``. |
| `corey_kr(sw, *, swr, sor, krw_max, kro_max, nw, no, clip)` | Two-phase Corey pair ``(krw, kro)`` on a shared Se. |
| `corey_krg(sg, *, sgc, swc, sorg, krg_max, ng, clip)` | Corey gas relative permeability ``krg = krg_max * Sg*_**ng``. |
| `brooks_corey_burdine_kr(sw, *, swr, lam, snwr, clip)` | Brooks-Corey (Burdine) kr pair from the pore-size-distribution index. |
| `let_kr(sw, *, swr, L, E, T, kr_max, phase, clip)` | Lomeland-Ebeltoft-Thomas (LET) relative permeability. |
| `phase_mobility(kr, mu)` | Phase mobility ``lambda = kr / mu``. |
| `total_mobility(kr_w, kr_nw, *, mu_w, mu_nw)` | Total two-phase mobility ``lambda_t = kr_w/mu_w + kr_nw/mu_nw``. |
| `fractional_flow(kr_w, kr_nw, *, mu_w, mu_nw)` | Buckley-Leverett fractional flow of the wetting phase (zero-safe). |
| `fractional_flow_curve(*, swr, sor, krw_max, kro_max, nw, no, mu_w, mu_nw, n)` | Corey fractional-flow curve ``(sw, fw)`` over the mobile window. |
| `welge_shock(sw, fw, swc)` | Welge tangent construction: ``(swf, fwf, sw_avg)``. |
| `endpoint_mobility_ratio(krw_max, kro_max, *, mu_w, mu_o)` | Endpoint mobility ratio ``M = (krw_max/mu_w) / (kro_max/mu_o)``. |
| `fit_corey(sw, krw_obs, krnw_obs, *, swr, sor)` | Least-squares Corey endpoints and exponents from observed kr. |
| `capillary_number(*, mu, v, sigma)` | Capillary number ``Nca = mu*v/sigma`` (viscous / capillary). |
| `bond_number(*, drho, k, sigma, g)` | Bond number ``Nb = drho*g*k/sigma`` (gravity / capillary). |
| `trapping_number(nca, nb)` | Total trapping number ``Nt = Nca + Nb`` (the additive form used repo-wide). |
| `capillary_desaturation(n, *, sor_max, sor_min, n_crit, exponent)` | Capillary desaturation curve ``Sor(N)``. |
| `land_c(s_i_max, s_r_max)` | Land trapping coefficient ``C = 1/Sr_max - 1/Si_max``. |
| `land_trapped(s_i, *, C, s_r_max)` | Land trapped saturation ``Sr = Si/(1 + C*Si)``. |
| `amott_indices(vw_spont, vw_forced, vo_spont, vo_forced)` | Amott-Harvey indices ``(Iw, Io, Iah)`` from displaced volumes. |
| `usbm_index(area_drainage, area_imbibition)` | USBM wettability index ``W = log10(A_drainage / A_imbibition)``. |
| `nmr_wettability_index(w_signal, o_signal)` | NMR wettability index ``Iw = (w - o)/(w + o)``. |
| `young_contact_angle(sigma_so, sigma_sw, sigma_wo)` | Young's-law contact angle (degrees) from the three interfacial tensions. |
| `wenzel_angle(theta_young_deg, roughness)` | Wenzel apparent contact angle (degrees): ``cos(theta_app) = r*cos(theta)``. |
| `work_of_adhesion(sigma_wo, contact_angle_deg)` | Young-Dupre work of adhesion ``W = sigma_wo*(1 + cos(theta))``. |
| `classify_wettability_angle(theta_deg, *, cuts)` | Classify wettability from the contact angle: water-wet / intermediate / oil-wet. |
| `classify_wettability_index(i, *, scheme)` | Classify wettability from an Amott/USBM-style index in [-1, 1]. |
| `displacement_efficiency(soi, sor)` | Microscopic displacement efficiency ``Ed = (Soi - Sor)/Soi``. |

## `petrolib.saturation_resistivity`

Water saturation from resistivity: Archie and shaly-sand models.

| Name | Summary |
| --- | --- |
| `formation_factor(phi, *, a, m)` | Archie formation resistivity factor ``F = a / phi**m``. |
| `archie_sw(rt, rw, *, phi, a, m, n, clip)` | Archie water saturation ``Sw = (a*Rw / (phi**m * Rt))**(1/n)``. |
| `archie_rt(sw, rw, *, phi, a, m, n)` | Forward Archie resistivity ``Rt = a*Rw / (phi**m * Sw**n)``. |
| `archie_conductivity(sw, cw, *, phi, m, n)` | Conductivity-domain Archie ``Ct = Cw * phi**m * Sw**n``. |
| `archie_sw_from_conductivity(ct, cw, *, phi, m, n, clip)` | Invert conductivity-domain Archie: ``Sw = (Ct/(Cw*phi**m))**(1/n)``. |
| `resistivity_index(rt, ro)` | Resistivity index by definition, ``RI = Rt/Ro``. |
| `resistivity_index_from_sw(sw, *, n, b)` | Resistivity-index power law ``RI = b * Sw**-n`` (b=1 is Archie). |
| `sw_from_resistivity_index(ri, *, n, b)` | Water saturation from the resistivity index, ``Sw = (RI/b)**(-1/n)``. |
| `fit_cementation_exponent(phi, ff)` | Fit ``(m, a)`` from a log-log formation-factor vs porosity regression. |
| `fit_saturation_exponent(sw, ri)` | Fit ``n`` from a log-log resistivity-index vs saturation regression. |
| `cementation_exponent_at_point(phi, ff, *, a)` | Single-point inversion ``m = log(F/a) / log(1/phi)``. |
| `bulk_volume_water(phi, sw)` | Bulk volume water ``BVW = phi * Sw``. |
| `apparent_water_resistivity(rt, phi, *, a, m)` | Apparent water resistivity ``Rwa = Rt * phi**m / a``. |
| `waxman_smits_conductivity(sw, *, cw, qv, b, phi, m_star, n_star)` | Waxman-Smits total conductivity. |
| `dual_water_conductivity(sw, *, cw, cb, swb, phi, m, n)` | Dual-water total conductivity, bound-water form (Clavier et al.). |
| `dual_water_conductivity_qv(sw, *, cw, qv, alpha_vqh, beta, phi, m0, n)` | Dual-water total conductivity, Qv-based form. |
| `waxman_smits_sw(ct, *, cw, qv, b, phi, m_star, n_star, lo, hi, n_iter, ...)` | Invert :func:`waxman_smits_conductivity` for Sw by bisection. |
| `dual_water_sw(ct, *, cw, cb, swb, phi, m, n, lo, hi, n_iter, tol, clip)` | Invert the bound-water :func:`dual_water_conductivity` for Sw. |
| `dual_water_sw_qv(ct, *, cw, qv, alpha_vqh, beta, phi, m0, n, lo, hi, ...)` | Invert the Qv-form :func:`dual_water_conductivity_qv` for Sw. |
| `simandoux_sw(rt, rw, *, phi, vsh, rsh, a, m, clip)` | Simandoux (1963) water saturation, closed-form for n = 2. |
| `indonesia_sw(rt, rw, *, phi, vcl, rcl, a, m, n, clip)` | Indonesia (Poupon-Leveaux) water saturation, closed form. |
| `qv_from_cec(cec, *, rho_grain, phi)` | Cation exchange capacity per pore volume, grain-based. |
| `cec_from_qv(qv, *, rho_grain, phi)` | Inverse of :func:`qv_from_cec`. |
| `qv_juhasz(v_clay_dry, *, rho_clay, cec_clay, phit)` | Juhasz clay-based Qv normalization. |

## `petrolib.testing`

Regression-safety helpers for the library migration.

| Name | Summary |
| --- | --- |
| `assert_matches_original(original, replacement, cases, *, rtol, atol)` | Assert ``replacement(*case)`` matches ``original(*case)`` for every case. |

## `petrolib.units`

Unit conversions for the quantities the article code mixes most.

| Name | Summary |
| --- | --- |
| `_TEMPERATURE_ALIASES` | constant |
| `convert(values, from_unit, to_unit)` | Convert ``values`` between two units of the same quantity family. |
| `slowness_to_velocity(slowness, unit)` | Sonic slowness (``us/ft`` or ``us/m``) to velocity in m/s. |
| `velocity_to_slowness(velocity, unit)` | Velocity in m/s to sonic slowness (``us/ft`` or ``us/m``). |

## `petrolib.wellbore_geometry`

Wellbore-survey trajectory geometry: dogleg, minimum curvature, MD to TVD.

| Name | Summary |
| --- | --- |
| `dogleg_angle(inc1, azi1, inc2, azi2)` | Dogleg (total curvature) angle between two survey stations, in radians. |
| `ratio_factor(dogleg_rad)` | Minimum-curvature ratio (radius) factor ``RF = (2/DL) tan(DL/2)``. |
| `minimum_curvature_step(md1, inc1, azi1, md2, inc2, azi2)` | One minimum-curvature step, returning ``(dTVD, dNorth, dEast)``. |
| `survey_to_path(md, inc, azi)` | Cumulative minimum-curvature trajectory, shape ``(n, 3)``. |
| `md_to_tvd(md, inc_deg, *, method)` | Cumulative true vertical depth from measured depth and inclination. |
