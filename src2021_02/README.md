# Petrophysics February 2021 - Vol. 62, No. 1

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 62, No. 1 (February 2021) - a regular issue opening with an invited
tutorial on extracting net pay from mudlogs, followed by eight papers spanning
DFA lateral fluid gradients and reservoir mixing over geologic time, weak
bedding planes in the Marcellus Shale, fracture-fill identification with
dielectric imaging in oil-based mud, formation-tester sampling of CO2 and other
reactive components, an integrated NMR/resistivity/pressure carbonate case
study, high-resolution dual-ultrasonic LWD slowness and imaging, multiwell
electromagnetic 3D inversion of sand injectites, and a dual neural network for
permeability with associated uncertainty.

## Quick start

```bash
pip install numpy scipy

# Run all 9 module tests
python test_all.py

# Or run a single article
python article9_dual_nn_permeability_uncertainty.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_mudlog_net_pay_tutorial.py` | *Tutorial:* Maximizing Value From Mudlogs - Integrated Approach to Determine Net Pay | Malik, Hanson, Clinch | 10.30632/PJV62N1-2021t1 |
| `article2_dfa_lateral_gradients_mixing.py` | Analysis of Lateral Fluid Gradients From DFA Measurements and Simulation of Reservoir Fluid Mixing Processes Over Geologic Time | Chen, Kristensen, Johansen, Achourov, Betancourt, Mullins | 10.30632/PJV62N1-2021a1 |
| `article3_marcellus_weak_bedding_planes.py` | Conclusive Proof of Weak Bedding Planes in the Marcellus Shale and Proposed Mitigation Strategies | Kowan, Schanken, Jacobi | 10.30632/PJV62N1-2021a2 |
| `article4_obm_dielectric_fracture_fill.py` | Identifying Fracture-Filling Material in Oil-Based Mud With Dielectric Borehole Imaging | Schlicht, Zhang, Lüling, Graham, Cournot, Sadownyk | 10.30632/PJV62N1-2021a3 |
| `article5_formation_tester_co2_sampling.py` | Innovative Formation Tester Sampling Procedures for Carbon Dioxide and Other Reactive Components | Piazza, Vieira, Sacorague, Jones, Dai, Pearl, Aguiar | 10.30632/PJV62N1-2021a4 |
| `article6_nmr_resistivity_pressure_carbonate.py` | Formation Evaluation With NMR, Resistivity, and Pressure Data: A Case Study of a Carbonate Oil Field Offshore West Africa | Li, Drinkwater, Whittlesey, Condon | 10.30632/PJV62N1-2021a5 |
| `article7_lwd_dual_ultrasonic_slowness.py` | Revealing Hidden Information: High-Resolution Logging-While-Drilling Slowness Measurements and Imaging Using Advanced Dual Ultrasonic Technology | Blyth, Sakiyama, Hori, Yamamoto, Nakajima, Fahim Ud Din, Haecker, Kittridge | 10.30632/PJV62N1-2021a6 |
| `article8_injectite_em_3d_inversion.py` | Mapping Complex Injectite Bodies With Multiwell Electromagnetic 3D Inversion Data | Clegg, Eriksen, Best, Tollefsen, Kowicki, Marchant | 10.30632/PJV62N1-2021a7 |
| `article9_dual_nn_permeability_uncertainty.py` | Dual Neural Network Architecture for Determining Permeability and Associated Uncertainty | Kausik, Prado, Gkortsas, Venkataramanan, Datir, Johansen | 10.30632/PJV62N1-2021a8 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2021_02.pdf`)
> has no usable text layer - reading it returns empty text, consistent with the
> image-rendered typesetting seen across these issues. The article titles,
> authors, page ranges, and DOIs above are taken verbatim from the official
> SPWLA issue table of contents; the numbered formulas in the modules are
> therefore **faithful standard-form reconstructions** of the well-established
> methods each paper applies, not transcriptions of the typeset equations.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions. A few practical substitutions:

- **Article 1 (Malik et al.)** *(tutorial)*: gas normalization to rock volume
  (GN = G·Q/(ROP·A) removing the drilling-rate dilution), the Haworth
  wetness / balance / character ratios with their productivity bands, the
  Pixler light-component ratios, and an integrated gas/porosity/Vsh/Sw cutoff
  scheme that sums net pay and net-to-gross.

- **Article 2 (Chen et al.)**: the Flory-Huggins-Zuo gravity term giving the
  equilibrium asphaltene (optical-density) gradient with depth, a 1D diffusion
  model (erfc step front and the H²/(π²D) column homogenization time) for
  mixing over geologic time, and an equilibrium-vs-disequilibrium connectivity
  diagnosis. The full FHZ EOS adds solubility and entropy terms; the gravity
  term dominates the asphaltene-nanoaggregate distribution.

- **Article 3 (Kowan et al.)**: Jaeger's single-plane-of-weakness theory - the
  weak-plane slip strength, intact Mohr-Coulomb strength, their combination
  into the U-shaped strength-vs-bedding-angle curve with the minimum at
  β = 45° + φ_w/2, and a mud-weight floor that suppresses bedding-parallel
  shear (the mitigation strategy).

- **Article 4 (Schlicht et al.)**: the CRIM permittivity mixing law, complex
  permittivity with the σ/(ωε₀) conduction term, the loss tangent that flags
  conductive fill, a thin-gap button admittance, and a classifier separating
  open (oil/mud), calcite-cemented, and conductive (clay/brine) fracture fills.

- **Article 5 (Piazza et al.)** *(short operational paper, no equations)*:
  standard PVT/sampling proxies - the power-law (V^−5/12) cleanup of OBM
  contamination, CO2 phase identification against the critical point
  (31 °C / 73.8 bar), Henry's-law CO2 solubility in brine with a Sechenov
  salting-out factor, and a mass-balance correction recovering the in-situ CO2
  fraction from a depleted sample.

- **Article 6 (Li et al.)** *(case study)*: Archie water saturation and
  formation factor, Timur-Coates and SDR NMR permeability, the Buckles bulk
  volume water, fluid density from a pressure gradient, and a fluid contact
  recovered from two intersecting pressure-gradient lines (round-trips a
  planted OWC to < 1 m).

- **Article 7 (Blyth et al.)**: slowness-time-coherence (semblance) processing
  over a receiver array with slowness picking (recovers a planted 80 µs/ft
  headwave), plus acoustic impedance and the normal-incidence reflection
  coefficient for pulse-echo borehole imaging.

- **Article 8 (Clegg et al.)**: the EM skin depth (503·√(ρ/f)), a straight-path
  cross-well sensitivity operator, and a Tikhonov (smoothness-regularized)
  least-squares inversion that recovers a resistive injectite body from a
  multiwell (crossing horizontal + vertical fan) survey. Full 3D Maxwell
  modelling is replaced by the tomographic operator demonstrating the same
  ill-posed inverse and its regularization.

- **Article 9 (Kausik et al.)**: a compact two-head MLP (shared tanh hidden
  layer; mean + log-variance heads) trained with the heteroscedastic Gaussian
  negative-log-likelihood, so it predicts both log-permeability and a
  calibrated uncertainty that grows in the less-informative low-porosity rock.
  A numpy network with manual backpropagation stands in for the published deep
  network.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2021)
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
