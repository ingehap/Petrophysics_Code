# Petrophysics — October 2023 (Vol. 64 No. 5) — Energy Transition Special Issue

A Python module per article in the SPWLA Petrophysics journal,
October 2023 issue. Each module:

* runs as a standalone script (`python article_NN_*.py`)
* exposes a `test_all()` function that exercises the implementation
  on synthetic data
* implements the central quantitative ideas of the article — not
  every figure or sub-experiment, but the equations and workflow
  that define the paper's contribution.

## Running everything

```
python run_all_tests.py
```

## Modules and source articles

| Module | Article |
|---|---|
| `article_01_laronga_ccs_evaluation.py` | Laronga, R., et al. (2023). *Integrated Formation Evaluation for Site-Specific Evaluation, Optimization, and Permitting of Carbon Storage Projects.* Petrophysics 64(5), 580–620. DOI: 10.30632/PJV64N5-2023a1 |
| `article_02_desroches_stress_measurement.py` | Desroches, J., et al. (2023). *Stress Measurement Campaign in Scientific Deep Boreholes: Focus on Tools and Methods.* Petrophysics 64(5), 621–639. DOI: 10.30632/PJV64N5-2023a2 |
| `article_03_okwoli_probe_screening.py` | Okwoli, E. and Potter, D. K. (2023). *Probe Screening Techniques for Rapid, High-Resolution Core Analysis and Their Potential Usefulness for Energy Transition Applications.* Petrophysics 64(5), 640–655. DOI: 10.30632/PJV64N5-2023a3 |
| `article_04_karadimitriou_relperm_scaling.py` | Karadimitriou, N., Valavanides, M. S., Mouravas, K., and Steeb, H. (2023). *Flow-Dependent Relative Permeability Scaling for Steady-State Two-Phase Flow in Porous Media: Laboratory Validation on a Microfluidic Network.* Petrophysics 64(5), 656–679. DOI: 10.30632/PJV64N5-2023a4 |
| `article_05_laronga_pulsed_neutron_ccs.py` | Laronga, R., Swager, L., and Bustos, U. (2023). *Time-Lapse Pulsed-Neutron Logs for Carbon Capture and Sequestration: Practical Learnings and Key Insights.* Petrophysics 64(5), 680–699. DOI: 10.30632/PJV64N5-2023a5 |
| `article_06_hill_potash_pid_plot.py` | Hill, D. G., Crain, E. R., and Teufel, L. W. (2023). *The Potash Identification (PID) Plot: A Rapid Screening Crossplot for Discrimination of Commercial Potash.* Petrophysics 64(5), 700–713. DOI: 10.30632/PJV64N5-2023a6 |
| `article_07_aerens_xray_mud_invasion.py` | Aérens, P., Espinoza, D. N., and Torres-Verdín, C. (2023). *High-Resolution Time-Lapse Monitoring of Mud Invasion in Spatially Complex Rocks Using In-Situ X-Ray Radiography.* Petrophysics 64(5), 715–740. DOI: 10.30632/PJV64N5-2023a7 |
| `article_08_zhao_sp_resistivity_inversion.py` | Zhao, P., Wang, Y., Li, G., Hu, C., Xie, J., Duan, W., and Mao, Z. (2023). *Joint Inversion of Saturation and Qv in Low-Permeability Sandstones Using Spontaneous Potential and Resistivity Logs.* Petrophysics 64(5), 741–752. DOI: 10.30632/PJV64N5-2023a8 |
| `article_09_bennis_corelogs_simulation.py` | Bennis, M. and Torres-Verdín, C. (2023). *Numerical Simulation of Well Logs Based on Core Measurements: An Effective Method for Data Quality Control and Improved Petrophysical Interpretation.* Petrophysics 64(5), 753–772. DOI: 10.30632/PJV64N5-2023a9 |
| `article_10_mohamed_rfg_connectivity.py` | Mohamed, T. S., Torres-Verdín, C., and Mullins, O. C. (2023). *Enhanced Reservoir Description via Areal Data Integration and Reservoir Fluid Geodynamics: A Case Study From Deepwater Gulf of Mexico.* Petrophysics 64(5), 773–795. DOI: 10.30632/PJV64N5-2023a10 |
| `article_11_shafiq_chelating_acidizing.py` | Shafiq, M. U., Ben Mahmud, H., Khan, M., Gishkori, S. N., Wang, L., and Jamil, M. (2023). *Effect of Chelating Agents on Tight Sandstone Formation Mineralogy During Sandstone Acidizing.* Petrophysics 64(5), 796–817. DOI: 10.30632/PJV64N5-2023a11 |

## Dependencies

Only NumPy is required. Tested with Python 3.11 and NumPy 2.x.

## Design notes

* The implementations are deliberately compact and pedagogical: where
  the paper specifies a particular fitting constant, kinetic rate, or
  endpoint, that value is used; where the paper relies on
  proprietary/measured datasets, representative literature values are
  substituted so the synthetic tests are reproducible.
* Each `test_all()` function builds its own synthetic input and uses
  assertions against physical expectations (monotonicity, bounded
  ranges, recovery of known parameters from inversions, etc.) rather
  than against fixed numerical fingerprints.
* No paper text or figures are reproduced; the code is an original
  implementation of the documented methods, with citations.
