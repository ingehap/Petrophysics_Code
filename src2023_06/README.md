# Petrophysics — Vol. 64, No. 3 (June 2023)
### Python modules implementing the technical ideas of every article

This bundle contains one Python module per article in the *Petrophysics*
journal issue **Vol. 64, No. 3 (June 2023)** — *Best Papers of the 2022
SCA International Symposium* — published by the Society of
Petrophysicists and Well Log Analysts (SPWLA).

## Contents

| File | Article | DOI |
|---|---|---|
| `article1_hdt.py` | Fernandes, V. *et al.* — *Hybrid Technique for Setting Initial Water Saturation on Core Samples* | [10.30632/PJV64N3-2023a1](https://doi.org/10.30632/PJV64N3-2023a1) |
| `article2_wiri.py` | Danielczick, Q. *et al.* — *Wireless Acquisition for Resistivity Index in Centrifuge — WiRI* | [10.30632/PJV64N3-2023a2](https://doi.org/10.30632/PJV64N3-2023a2) |
| `article3_overburden_frf_ri.py` | Nourani, M. *et al.* — *Analytical Models for Predicting the Formation Resistivity Factor and Resistivity Index at Overburden Conditions* | [10.30632/PJV64N3-2023a3](https://doi.org/10.30632/PJV64N3-2023a3) |
| `article4_gas_trapping.py` | Gao, Y. *et al.* — *Advanced Digital-SCAL Measurements of Gas Trapped in Sandstone* | [10.30632/PJV64N3-2023a4](https://doi.org/10.30632/PJV64N3-2023a4) |
| `article5_shale_t1t2star.py` | Zamiri, M. S. *et al.* — *Shale Characterization Using T1-T2\* Magnetic Resonance Relaxation Correlation Measurement at Low and High Magnetic Fields* | [10.30632/PJV64N3-2023a5](https://doi.org/10.30632/PJV64N3-2023a5) |
| `article6_ultrasonic_reflection.py` | Olszowska, D. *et al.* — *Angle-Dependent Ultrasonic Wave Reflection for Estimating High-Resolution Elastic Properties of Complex Rock Samples* | [10.30632/PJV64N3-2023a6](https://doi.org/10.30632/PJV64N3-2023a6) |
| `article7_dielectric_nmr.py` | Funk, J. *et al.* — *NMR-Mapped Distributions of Dielectric Dispersion* | [10.30632/PJV64N3-2023a7](https://doi.org/10.30632/PJV64N3-2023a7) |
| `article8_thz_porosity.py` | Eichmann, S. L. *et al.* — *THz Imaging to Map the Lateral Microporosity Distribution in Carbonate Rocks* | [10.30632/PJV64N3-2023a8](https://doi.org/10.30632/PJV64N3-2023a8) |
| `article9_xray_invasion.py` | Aérens, P. *et al.* — *Experimental Time-Lapse Visualization of Mud-Filtrate Invasion and Mudcake Deposition Using X-Ray Radiography* | [10.30632/PJV64N3-2023a9](https://doi.org/10.30632/PJV64N3-2023a9) |
| `test_all.py` | Orchestrator — imports every module and runs its `test_all()` function | — |

The full journal issue is available from SPWLA at:
<https://www.spwla.org/SPWLA/Publications/Journals/Recent_Petrophysics_Journals.aspx>

## Rules respected

1. **Every module can be run as a standalone script.**
   ```bash
   python article1_hdt.py
   python article2_wiri.py
   ...
   ```

2. **Every module exposes a `test_all()` function** that exercises the
   implemented ideas with synthetic data and asserts that the recovered
   quantities match the expected ones (e.g. Archie *n* near 2,
   recovered RRM within tolerance, uniform Sw after the porous-plate
   phase, Darcy invasion-front position matching the analytic
   prediction within a few percent, etc.).

To run all tests at once:

```bash
python test_all.py
```

Expected output ends with:
```
9/9 modules passed.
```

## Dependencies

* Python ≥ 3.10
* `numpy`
* `scipy` (only used by `article6_ultrasonic_reflection.py` for the
  non-linear least-squares inversion)

```bash
pip install numpy scipy
```

## Scope and disclaimer

These modules are **didactic re-implementations** of the published
ideas, not exact reproductions of the original experimental codes.
They focus on the core formulas and workflows and use synthetic data
designed to demonstrate the key behaviours described in each paper.
For the full experimental procedures, calibration details, and
quantitative comparisons with field data, please refer to the
original articles.
