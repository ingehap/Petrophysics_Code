# Petrophysics December 2018 - Vol. 59, No. 6

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 59, No. 6 (December 2018) - the **"Special Issue: Data-Driven Analytics in
Logging and Petrophysics"** (Petrophysical Data-Driven Analytics, PDDA).  It
opens with the third capillary-pressure tutorial, then a suite of
machine-learning / data-driven papers.

## Quick start

```bash
pip install numpy

# Run all 12 module tests
python test_all.py

# Or run a single article
python article5_ultradeep_resistivity_transdim_inversion.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_capillary_pressure_tutorial_part3.py` | *Tutorial:* Capillary Pressure Tutorial Part 3 - The Jungle Gives Us Many Things | Murphy | 10.30632/PJV59N6Y2018t1 |
| `article2_geological_feature_image_ml.py` | Geological Feature Prediction Using Image-Based Machine Learning | Jobe, Vital-Brazil, Khait | 10.30632/PJV59N6Y2018a1 |
| `article3_poisson_ratio_functional_network.py` | A Rigorous Data-Driven Approach to Predict Poisson's Ratio of Carbonate Rocks Using a Functional Network | Tariq, Abdulraheem, Mahmoud, Ahmed | 10.30632/PJV59N6Y2018a2 |
| `article4_borehole_resistivity_ml.py` | Borehole Resistivity Measurement Modeling Using Machine-Learning Techniques | Xu, Sun, Xie, Zhong, Mirto, Feng, Hong | 10.30632/PJV59N6Y2018a3 |
| `article5_ultradeep_resistivity_transdim_inversion.py` | Data-Driven Interpretation of Ultradeep Azimuthal Propagation Resistivity Measurements: Transdimensional Stochastic Inversion and Uncertainty Quantification | Shen, Chen, Wang | 10.30632/PJV59N6Y2018a4 |
| `article6_lithology_cnn.py` | Intelligent Logging Lithological Interpretation With Convolution Neural Networks | Zhu, Li, Yang, Li, Ao | 10.30632/PJV59N6Y2018a5 |
| `article7_hydraulic_fracture_optimization.py` | Use of Data Analytics to Optimize Hydraulic Fracture Locations Along Borehole | Gupta, Rai, Devegowda, Sondergeld | 10.30632/PJV59N6Y2018a6 |
| `article8_shallow_learning_sonic_logs.py` | Comparative Study of Shallow Learning Models for Generating Compressional and Shear Traveltime Logs | He, Misra, Li | 10.30632/PJV59N6Y2018a7 |
| `article9_fluid_optical_database_reconstruction.py` | Fluid Optical Database Reconstruction With Validated Mapping from External Oil and Gas Information Source | Chen, Jones, Dai, van Zuilekom | 10.30632/PJV59N6Y2018a8 |
| `article10_ml_depth_matching.py` | Machine-Learning-Based Automatic Well-Log Depth Matching | Zimmermann, Liang, Zeroug | 10.30632/PJV59N6Y2018a9 |
| `article11_data_preconditioning.py` | Data Preconditioning for Predictive and Interpretive Algorithms: Importance in Data-Driven Analytics and Methods for Application | Frost, Quinn | 10.30632/PJV59N6Y2018a10 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2018_12.pdf`)
> has a text layer, so titles, authors, page ranges, and DOIs were read from the
> contents page and bodies (the DOI suffix is printed `PJV59N6Y2018…`, with `Y`
> in place of a hyphen). The machine extraction captured the full bodies of the
> **tutorial and articles a1-a8** but **truncated after a8** (page ~848), so
> **articles a9-a11 were available only as table-of-contents entries** and are
> implemented as **methodology proxies**. As with the other issues, the typeset
> formula glyphs were dropped in extraction, so the numbered formulas are
> faithful standard-form reconstructions; the deep/data-driven models are
> represented by compact numpy implementations of the same method.

## Implementation notes & substitutions

- **Article 1 (Murphy)** *(tutorial)*: Young-Laplace capillary pressure, the
  Leverett J-function that collapses curves across rocks, the saturation-height
  relation, and a Brooks-Corey saturation-height curve.

- **Article 2 (Jobe et al.)**: image texture features (mean, gradient energy,
  orientation contrast) and a logistic classifier separating bedded from chaotic
  fabric.

- **Article 3 (Tariq et al.)**: a functional network (basis expansion + least
  squares) predicting the dynamic Poisson's ratio of carbonates (R ≈ 0.99).

- **Article 4 (Xu et al.)**: a physics-based apparent-resistivity forward model
  (shoulder-bed averaging) and an NN surrogate that reproduces it quickly.

- **Article 5 (Shen et al.)**: a transdimensional (reversible-jump) MCMC that
  inverts a layered-resistivity profile of unknown layer count, recovering the
  layering and quantifying uncertainty (higher posterior spread near boundaries).

- **Article 6 (Zhu et al.)**: a CNN-style lithology classifier - 1D
  convolutional features (global-average-pooled) combined with window statistics
  and a softmax head (3-class accuracy ~1.0 on synthetic logs).

- **Article 7 (Gupta et al.)**: the Rickman brittleness index, the poroelastic
  minimum-horizontal-stress profile, a completion-quality score, and a
  minimum-spacing stage-placement optimizer.

- **Article 8 (He et al.)**: shallow models (OLS and k-nearest-neighbor)
  predicting the DTC sonic log from conventional logs, scored by R and RMSE.

- **Article 9 (Chen et al.)** *(methodology proxy)*: Beer-Lambert optical
  density, a composition→OD-spectrum forward mapping, and a validated
  least-squares inversion of OD back to composition.

- **Article 10 (Zimmermann et al.)** *(methodology proxy)*: cross-correlation
  and DTW depth matching with a non-wrapping windowed local-shift estimator.

- **Article 11 (Frost & Quinn)** *(methodology proxy)*: z-score / min-max
  scaling, z-score and IQR outlier detection, gap imputation, and a downstream
  RMSE improvement from preconditioning a sentinel-corrupted log.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2018)
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
