# Petrophysics August 2019 - Vol. 60, No. 4

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 60, No. 4 (August 2019) - a regular issue of six papers spanning
machine-learning well-log correlation, neural-network TOC from XRF data, the
shale-gas compressibility factor at the core scale, an NMR wettability index,
the effect of the aging protocol on relative-permeability measurements, and
coupled smart-water-CO2 flooding.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article3_shale_gas_z_factor.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_ml_well_log_correlation.py` | A Machine-Learning-Based Approach to Assistive Well-Log Correlation | Brazell, Bayeh, Ashby, Burton | 10.30632/PJV60N4-2019a1 |
| `article2_toc_xrf_neural_network.py` | Total Organic Carbon Characterization Using Neural-Network Analysis of XRF Data | Lawal, Mahmoud, Alade, Abdulraheem | 10.30632/PJV60N4-2019a2 |
| `article3_shale_gas_z_factor.py` | The Compressibility Factor (Z) of Shale Gas at the Core Scale | Tran, Sakhaee-Pour | 10.30632/PJV60N4-2019a3 |
| `article4_nmr_wettability_index.py` | Practical Approach to Derive Wettability Index by NMR in Core Analysis Experiments | Looyestijn | 10.30632/PJV60N4-2019a4 |
| `article5_relperm_aging_highthroughput.py` | In-Situ Investigation of Aging Protocol Effect on Relative Permeability Measurements Using High-Throughput Experimentation Methods | Mascle, Youssef, Deschamps, Vizika | 10.30632/PJV60N4-2019a5 |
| `article6_smart_water_co2_flooding.py` | Novel Coupling Smart Water-CO2 Flooding for Sandstone Reservoirs | Al-Saedi, Flori | 10.30632/PJV60N4-2019a6 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2019_08.pdf`,
> ~110 MB) is a scanned issue with **no usable text layer** - reading it returns
> empty text. The article titles, authors, page ranges, and DOIs above were
> therefore obtained from the journal's metadata (Crossref / the issue table of
> contents), and the numbered formulas in the modules are **faithful
> standard-form reconstructions** of the well-established methods each paper's
> topic uses, not transcriptions of the original equations. DOI pattern:
> `10.30632/PJV60N4-2019aN` (N = 1 … 6).

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions. A few practical notes:

- **Article 1 (Brazell et al.)**: cross-correlation and dynamic-time-warping
  alignment of offset-well logs, a DTW warping path that maps a marker depth
  from a reference well to an offset well, and a logistic tie-confidence score -
  recovers a planted inter-well depth offset.

- **Article 2 (Lawal et al.)**: a neural network regressing TOC from XRF
  elements (redox-sensitive Mo/S/V vs detrital Si/Al/Ca), with the Schmoker
  density and Passey delta-log-R methods as baselines (NN R ≈ 0.99 on synthetic
  data).

- **Article 3 (Tran & Sakhaee-Pour)**: pseudo-reduced pressure/temperature, the
  Beggs-Brill explicit Z-factor correlation, real-gas density ρ = PM/(ZRT), and
  a confinement shift of the critical properties that moves the core-scale Z -
  Z → 1 at low pressure and dips at moderate reduced pressure.

- **Article 4 (Looyestijn)**: the NMR surface-relaxation wettability index
  WI = (Rw − Ro)/(Rw + Ro), where each phase's surface-relaxation rate
  1/T2_surf = 1/T2_obs − 1/T2_bulk reflects whether it wets the pore surface
  (+1 water-wet, −1 oil-wet, ~0 mixed).

- **Article 5 (Mascle et al.)**: the Corey water/oil relative-permeability model
  with its crossover saturation, and an aging transform of the wettability
  parameters (water-wet → mixed-wet) that shifts the crossover to lower water
  saturation and raises the water endpoint - quantifying the aging-protocol
  effect.

- **Article 6 (Al-Saedi & Flori)**: Buckley-Leverett fractional flow, the
  end-point mobility ratio and displacement efficiency, a smart-water
  (low-salinity) residual-oil reduction, a CO2 oil-viscosity reduction, and the
  coupled recovery factor - the coupled process recovers more than either method
  alone.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2019)
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
