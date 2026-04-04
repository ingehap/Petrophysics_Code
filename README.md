# Petrophysics_Code

Unofficial Python implementations of articles published in
[*Petrophysics*](https://www.spwla.org/petrophysics-journal) — the journal of the
Society of Petrophysicists and Well Log Analysts (SPWLA).

Each module translates the key algorithms and equations from a single journal
article into self-contained Python code with synthetic-data demonstrations.
The implementations are meant for learning and experimentation, not as a
replacement for the original papers.

## Requirements

- Python 3.9+
- NumPy ≥ 1.24
- SciPy ≥ 1.10

## Repository layout

```
Petrophysics_Code/
├── src2025_12/   Vol. 66 No. 6 (Dec 2025)  — 13 modules
├── src2026_02/   Vol. 67 No. 1 (Feb 2026)  — 12 modules + test suite
└── src2026_04/   Vol. 67 No. 2 (Apr 2026)  — 12 modules
```

---

## src2025_12 — Vol. 66, No. 6 (December 2025)

Best Papers of the 2024 SCA International Symposium.

| Module | Topic | Reference |
|--------|-------|-----------|
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
|--------|-------|-----------|
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
| `test_all` | Test suite covering all modules in this package | — |

DOI pattern: `10.30632/PJV67N1-2026a{1..15}`

---

## src2026_04 — Vol. 67, No. 2 (April 2026)

| Module | Topic | Reference |
|--------|-------|-----------|
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

```bash
python -m src2025_12.pgs_rock_typing
python -m src2026_02.depth_alignment
python -m src2026_04.a01_sponge_core_saturation_uncertainty
```

The `src2026_04` modules each export an `example_workflow()` function that
demonstrates the key algorithms with synthetic data:

```python
from src2026_04 import a12_depth_shifting_ml

a12_depth_shifting_ml.example_workflow()
```

The `src2026_02` package includes a test suite:

```bash
python -m src2026_02.test_all
```

## Disclaimer

These implementations are *unofficial* and are not affiliated with or endorsed
by the SPWLA or any of the original authors. They are intended as educational
companions to the published articles. Please refer to the original papers for
authoritative descriptions of the methods.
