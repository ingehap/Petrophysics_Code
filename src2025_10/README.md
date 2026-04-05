# Petrophysics Vol. 66 No. 5 (October 2025) — Python Modules

Python implementations of the ideas from each article in
*Petrophysics*, Vol. 66, No. 5 (October 2025), published by SPWLA.

## Modules

| Module | Article | Key Ideas Implemented |
|--------|---------|----------------------|
| `a1_log_interpretation.py` | Proestakis & Fabricius, pp. 705–727 (DOI: 10.30632/PJV66N5-2025a1) | Kozeny permeability, Archie m-exponent from surface area, parallel conduction model, iso-frame elastic model, Gassmann substitution, Biot coefficient |
| `a2_damage_model.py` | Liu et al., pp. 728–740 (DOI: 10.30632/PJV66N5-2025a2) | M-integral computation, local mechanical failure driving factor, initial/microscopic/total damage, Weibull-based damage constitutive model |
| `a3_youngs_modulus.py` | Al-Dousari et al., pp. 741–762 (DOI: 10.30632/PJV66N5-2025a3) | Dynamic/static Young's modulus, Mullen lithology models, Steiber Vsh, FZI/DRT rock typing, nonlinear regression model, simple BPNN |
| `a4_multimodal_permeability.py` | Fang et al., pp. 764–784 (DOI: 10.30632/PJV66N5-2025a4) | LSTM for time-series logs, 1-D CNN for NMR T2 images, DNN for text features, explicit tensor interaction (binary planes + ternary core) |
| `a5_missing_log_prediction.py` | Oppong et al., pp. 785–806 (DOI: 10.30632/PJV66N5-2025a5) | 1-D U-Net encoder-decoder with skip connections, LSTM depth-trend module, hybrid fusion for missing-log prediction |
| `a6_carbonate_petrophysics.py` | Fadhil, pp. 807–838 (DOI: 10.30632/PJV66N5-2025a6) | Shale volume (linear/Larionov), density-neutron porosity, water saturation (Archie/Indonesian/Simandoux), Timur permeability, net-pay flagging |
| `a7_nmr_porosity_correction.py` | Zhu et al., pp. 840–857 (DOI: 10.30632/PJV66N5-2025a7) | Rock magnetic susceptibility from minerals, internal gradient field, NMR T2 relaxation (bulk + surface + diffusion), T2 spectrum correction, porosity correction model |
| `a8_digital_core_conductivity.py` | Feng & Zou, pp. 858–871 (DOI: 10.30632/PJV66N5-2025a8) | Archie's first/second laws with directional anisotropy, bimodal saturation exponent, wettability/salinity effects, 3-D digital core generation, resistivity simulation |
| `a9_cementing_quality.py` | Pan et al., pp. 872–885 (DOI: 10.30632/PJV66N5-2025a9) | Slip interface boundary conditions, coupling stiffness matrix, relative amplitude vs. shear stiffness/USA, stiffness inversion from amplitude, cement quality classification |
| `a10_neutron_log_shale.py` | Rasmus, pp. 887–893 (DOI: 10.30632/PJV66N5-2025a10) | Migration/slowing-down/diffusion lengths (SNUPAR-like), neutron porosity transforms (SS/LS/DOL), effective Lm*, nonlinear shale response modelling |
| `a11_fracture_identification.py` | Lee et al., pp. 894–914 (DOI: 10.30632/PJV66N5-2025a11) | Synthetic borehole image generation with sinusoidal fractures, feature extraction (gradient, variance), threshold & CNN-based detectors, F1 score with depth tolerance |

## Running

Every module is standalone:

```bash
python a1_log_interpretation.py    # runs built-in demo
python a6_carbonate_petrophysics.py
# etc.
```

Run the full test suite (81 assertions across all 11 modules):

```bash
python test_all.py
```

## Dependencies

- **NumPy** (required for all modules)
- **SciPy** (only `a9_cementing_quality.py` uses `scipy.special.jv`)

No deep-learning frameworks are needed; the ML modules (articles 4, 5, 11) use pure-NumPy implementations of LSTM, CNN, U-Net, and DNN architectures.
