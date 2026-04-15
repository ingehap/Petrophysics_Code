# Petrophysics April 2023 — AI/ML Special Issue

Python modules implementing the methods from each article in
*Petrophysics*, Vol. 64, No. 2 (April 2023), the SPWLA AI & ML special issue.

## Quick start

```bash
# Install the only non-standard dependency
pip install xgboost scikit-image scipy scikit-learn numpy

# Run all 11 module tests
python test_all.py

# Or run a single article's tests
python article01_electrofacies_dp.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article01_electrofacies_dp.py` | Unsupervised Electrofacies Clustering Based on Parameterization of Petrophysical Properties: A Dynamic Programming Approach | Sinnathamby, Hou, Gkortsas, Venkataramanan, Datir, Kollien, Fleuret | 10.30632/PJV64N2-2023a1 |
| `article02_image_rock_classification.py` | Data-Driven Algorithms for Image-Based Rock Classification and Formation Evaluation in Formations With Rapid Spatial Variation in Rock Fabric | Gonzalez, Heidari, Lopez | 10.30632/PJV64N2-2023a2 |
| `article03_symbolic_regression.py` | Use of Symbolic Regression for Developing Petrophysical Interpretation Models | Chen, Shao, Sheng, Kwak | 10.30632/PJV64N2-2023a3 |
| `article04_log_prediction_ml.py` | Comparative Study of Machine-Learning-Based Methods for Log Prediction | Simoes, Maniar, Abubakar, Zhao | 10.30632/PJV64N2-2023a4 |
| `article05_outlier_detection.py` | An Unsupervised Machine-Learning Workflow for Outlier Detection and Log Editing With Prediction Uncertainty | Akkurt, Conroy, Psaila, Paxton, Low, Spaans | 10.30632/PJV64N2-2023a5 |
| `article06_borehole_image_artifacts.py` | Removal of Artifacts in Borehole Images Using Machine Learning | Guner, Fouda, Barrett | 10.30632/PJV64N2-2023a6 |
| `article07_sonic_log_imputation.py` | Sonic Well-Log Imputation Through Machine-Learning-Based Uncertainty Models | Maldonado-Cruz, Foster, Pyrcz | 10.30632/PJV64N2-2023a7 |
| `article08_egfm_facies.py` | Exemplar-Guided Sedimentary Facies Modeling for Bridging Pattern Controllability Gap | Wu, Hu, Sun, Zhang, Wang, Zhang | 10.30632/PJV64N2-2023a8 |
| `article09_spatial_analytics.py` | Spatial Data Analytics-Assisted Subsurface Modeling: A Duvernay Case Study | Salazar, Ochoa, Garland, Lake, Pyrcz | 10.30632/PJV64N2-2023a9 |
| `article10_induction_deconvolution.py` | Machine-Learning-Based Deconvolution Method Provides High-Resolution Fast Inversion of Induction Log Data | Hagiwara | 10.30632/PJV64N2-2023a10 |
| `article11_induction_convolution.py` | Machine-Learning-Based Convolution Method for Fast Forward Modeling of Induction Log | Hagiwara | 10.30632/PJV64N2-2023a11 |
| `test_all.py` | Master test runner | — | — |

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** —
not byte-perfect reproductions. A few practical substitutions:

- **Article 4 (Simoes et al.)**: WAE/PAE are implemented as scikit-learn
  `MLPRegressor`s instead of TensorFlow/PyTorch CNN/U-Net, so they train in
  seconds and avoid a heavyweight dependency. The PAE/WAE distinction is
  preserved (pointwise vs. windowed inputs).
- **Article 7 (Maldonado-Cruz et al.)**: ensemble built from sklearn
  `GradientBoostingRegressor`s with row subsampling, instead of NGBoost.
- **Article 8 (Wu et al.)**: the full paper trains a CGAN with two encoders,
  three discriminators, and an Adaptive Feature Fusion Block. This module
  implements the *content/pattern decoupling concept* using a distance
  transform + Gabor filter bank + attention-weighted fusion. It is a
  proof-of-concept of the architecture, not a trained generator.
- **Articles 10 & 11 (Hagiwara)**: DataRobot/LightGBM is replaced with
  `xgboost`. The 2C40 forward model is approximated by a Gaussian
  smoothing of log-conductivity (the right qualitative shape; not the exact
  Doll geometric factor).
- **Article 6 (Guner et al.)**: per the paper the ML model is trained to
  reproduce a "traditional" processing pipeline. Here the traditional
  output is a column-baseline subtraction.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2023)
DOI: <doi>
"""

# imports
# implementation functions
# def synthetic_data(...): ...
# def test_all(): ...

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a library,
and self-tests with synthetic data.

## Reference

*Petrophysics* — The SPWLA Journal of Formation Evaluation and Reservoir
Description — Vol. 64, No. 2 (April 2023). Society of Petrophysicists and
Well Log Analysts. ISSN 1529-9074.
