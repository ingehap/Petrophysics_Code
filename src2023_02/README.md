# Petrophysics February 2023 - Vol. 64, No. 1

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 64, No. 1 (February 2023).  Regular (non-themed) issue: nine papers
spanning field case studies and methodology development across reservoir
fluids, carbonate petrophysics, deep-learning image interpretation, LWD
operations, digital rock physics, geosteering inversion, data-driven
permeability, thermal recovery, and well-log depth matching.

## Quick start

```bash
pip install numpy scipy scikit-learn   # sklearn used only by article 7

# Run all 9 module tests
python test_all.py

# Or run a single article
python article2_carbonate_phi_k.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_rfg_petroleum_system.py` | Enigmatic Reservoir Properties Deciphered Using Petroleum System Modeling and Reservoir Fluid Geodynamics | Pierpont, Birkeland, Cely, Yang, Chen, Achourov, Betancourt, Canas, Forsythe, Pomerantz, Yang, Datir, Mullins | 10.30632/PJV64N1-2023a1 |
| `article2_carbonate_phi_k.py` | Modeling Permeability in Different Carbonate Rock Types | Dernaika, Masalmeh, Mansour, Al Jallad, Koronfol | 10.30632/PJV64N1-2023a2 |
| `article3_swin_fracture.py` | Fracture Extraction From Logging Image Using a Dual Encoder-Decoder Architecture With Swin Transformer | Wang, Zhou | 10.30632/PJV64N1-2023a3 |
| `article4_hexa_combo_lwd.py` | First Hexa-Combo Logging-While-Drilling Run in Kuwait: A Case Study | Saleh, Al-Khudari, Al-Azmi, Al-Otaibi, Patnaik, Joshi, Abdulkarim, Aki, Fahri, Sanyal, Sainuddin | 10.30632/PJV64N1-2023a4 |
| `article5_digital_core_poisson.py` | Analysis of Influencing Factors of Poisson's Ratio in Deep Shale Gas Reservoir Based on Digital Core Simulation | Liu, Wang, Lai, Wang, Zhang, Zhang, Ou | 10.30632/PJV64N1-2023a5 |
| `article6_geosteering_enrml.py` | Enhancing the Detectability of Deep-Sensing Borehole EM Instruments by Joint Inversion of Multiple Logs Within a Probabilistic Geosteering Workflow | Jahani, Alyaev, Ambia, Fossum, Suter, Torres-Verdín | 10.30632/PJV64N1-2023a6 |
| `article7_dm_permeability.py` | Permeability Calculation of Complex Carbonate Reservoirs Based on Data Mining Techniques | Li | 10.30632/PJV64N1-2023a7 |
| `article8_hot_water_injection.py` | An Algorithm to Optimize Water Injection Temperature for Thermal Recovery of High Pour Point Oil | Yu, Zhang | 10.30632/PJV64N1-2023a8 |
| `article9_depth_matching.py` | Automated Well-Log Pattern Alignment and Depth-Matching Techniques: An Empirical Review and Recommendations | Ezenkwu, Guntoro, Starkey, Vaziri, Addario | 10.30632/PJV64N1-2023a9 |
| `test_all.py` | Master test runner | - | - |

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (Pierpont et al.)**: the paper combines basin-scale Tissot-
  Welte petroleum-system modelling with Downhole Fluid Analysis and the
  Mullins/Zuo asphaltene equation of state.  Here we implement the
  gravitational term of the Flory-Huggins-Zuo asphaltene EOS, an
  exponential biodegradation kinetic, a two-stage volumetric mixing of
  the resident oil with a late condensate charge, and a simple WAT
  correlation - enough to reproduce the paradox of *lower* asphaltene
  in the *more* biodegraded oil and the resulting upstructure asphaltene
  destabilisation.

- **Article 3 (Wang & Zhou)**: the paper trains a Swin-Transformer-based
  W-shape dual encoder-decoder in PyTorch.  Here we replace the trained
  network with a NumPy-only proof-of-concept consisting of a
  patch-window attention proxy and a top-K sinusoidal Hough decoder,
  evaluated with the SAME Dice / mIoU / Precision / Recall metrics
  (Eqs. 3-6 of the paper).  Eqs. 1-2 (the W-MSA complexity) are also
  reproduced for sanity-checking against full MSA.

- **Article 4 (Saleh et al.)**: an operational case study, not a method
  paper.  The module synthesises a Marrat-style Hexa-Combo LWD log
  suite (GR, multi-DOI Rt, NPHI, RHOB, DTC, DTS, NMR T2) over a
  fractured tight carbonate, then runs the canonical interpretation
  workflow (Vsh, density porosity, Archie Sw, NMR BVI/FFI partition,
  dynamic K/G/E/nu, brittleness index, perforation picker).

- **Article 5 (Liu et al.)**: the paper computes effective elastic
  moduli of a multi-component digital core by finite-element
  minimisation of total elastic potential energy (Eqs. 2-5).  As a
  tractable analogue we use a Voigt-Reuss-Hill average over the
  mineral skeleton plus a Krief-style porosity softening and a
  Gassmann-style fluid term; the resulting Poisson's ratio matches the
  paper's reservoir target of ~0.24 and reproduces the
  45-deg-bedding-angle minimum-nu behaviour.

- **Article 6 (Jahani et al.)**: implements an approximate LM-EnRML
  ensemble inversion (Appendix A1 of the paper) for a three-layer
  geosteering scenario with toy depth-of-investigation kernels.  The
  full paper uses a proprietary nuclear-density forward model plus an
  ML surrogate for extra-deep EM; here we use Gaussian-kernel proxies
  for shallow propagation, extra-deep symmetric EM, and nuclear bulk
  density, which is sufficient to demonstrate the >2x misfit
  reduction and uncertainty shrinkage that motivate the workflow.

- **Article 7 (Li)**: the survey describes many data-mining
  alternatives (decision trees, random forests, rough sets, SVM,
  Bayesian nets, ANNs).  The reference implementation here uses
  scikit-learn's KMeans and RandomForest where available, with a
  pure-NumPy log-linear fallback.  Mutual-information feature ranking,
  per-class log-linear regression, and an MAE(log10 k) metric let
  you reproduce the "~18 %" improvement-over-global claim.

- **Article 8 (Yu & Zhang)**: a closed-form Ramey-style wellbore-fluid
  temperature profile with the Hasan-Kabir dimensionless-time function
  for transient formation resistance.  A two-section variant supports
  an upper insulated tubing length.  The bisection optimiser returns
  the surface T that delivers the wax-appearance-temperature floor at
  bottomhole; on the default Liaohe-style parameter set it lands in
  the paper's reported 60-65 degC band.

- **Article 9 (Ezenkwu et al.)**: classical DTW, Constrained-DTW
  (Sakoe-Chiba band), and Correlation Optimised Warping (greedy
  Nielsen variant) are evaluated on a synthetic GR pair with a
  non-linear depth warp, amplitude scaling, and additive noise.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2023)
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
