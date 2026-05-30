# Petrophysics June 2019 - Vol. 60, No. 3

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 60, No. 3 (June 2019) - a regular issue opening with an organic-mudstone
storage-capacity tutorial, followed by eight articles spanning shale composition
and gas adsorption, wellsite-tomography Bayesian inversion, finite-volume shale
tortuosity/permeability, a fast NMR T1 measurement, a reconsideration of
Klinkenberg's permeability data, the appropriate cementation exponent for
vuggy/fractured carbonates, perched water contacts, and LWD wellbore
positioning.

## Quick start

```bash
pip install numpy scipy

# Run all 9 module tests
python test_all.py

# Or run a single article
python article4_shale_tortuosity_permeability_fvm.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_organic_mudstone_storage_tutorial.py` | *Tutorial:* Organic-Mudstone Petrophysics: Part 3: Workflow to Estimate Storage Capacity | Newsham, Comisky, Chemali | 10.30632/PJV60N3-2019t1 |
| `article2_niutitang_shale_pore_adsorption.py` | Composition of the Shales in Niutitang Formation at Huijunba Syncline and its Influence on Microscopic Pore Structure and Gas Adsorption | Fu, Xu, Tian, Qin, Yang | 10.30632/PJV60N3-2019a1 |
| `article3_wellsite_tomography_bayesian.py` | Accelerated Whole-Core Analysis Optimization With Wellsite Tomography Instrumentation and Bayesian Inversion | Mendoza, Roininen, Girolami, Heikkinen, Haario | 10.30632/PJV60N3-2019a2 |
| `article4_shale_tortuosity_permeability_fvm.py` | Finite-Volume Computations of Shale Tortuosity and Permeability From 3D Pore Networks Extracted From Scanning Electron Tomographic Images | Almasoodi, Reza | 10.30632/PJV60N3-2019a3 |
| `article5_fast_nmr_t1.py` | Application of a Fast NMR T1 Relaxation Time Measurement to Sedimentary Rock Cores | Mitchell, Valori | 10.30632/PJV60N3-2019a4 |
| `article6_reconsidering_klinkenberg.py` | Reconsidering Klinkenberg's Permeability Data | Ruth, Arabjamaloei | 10.30632/PJV60N3-2019a5 |
| `article7_carbonate_m_vugs_fractures.py` | Determination of the Appropriate Value of m for Evaluation of Carbonate Reservoirs With Vugs and Fractures at the Well-Log Scale | Wang, Peng | 10.30632/PJV60N3-2019a6 |
| `article8_perched_water_contacts.py` | Perched Water Contacts: Understanding Fundamental Controls | Hulea | 10.30632/PJV60N3-2019a7 |
| `article9_wellbore_positioning_lwd.py` | Wellbore Positioning While Drilling With LWD Measurements | Poedjono, Nwosu, Martin | 10.30632/PJV60N3-2019a8 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2019_06.pdf`,
> ~228 MB) is a scanned issue with **no usable text layer** - reading it returns
> empty text. The article titles, authors, page ranges, and DOIs above were
> therefore obtained from the journal's metadata (Crossref / the issue table of
> contents), and the numbered formulas in the modules are **faithful
> standard-form reconstructions** of the well-established methods each paper's
> topic uses, not transcriptions of the original equations. DOI pattern:
> `10.30632/PJV60N3-2019aN` (a1 … a8) plus the tutorial `…-2019t1`.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions. A few practical notes:

- **Article 1 (Newsham et al.)** *(tutorial)*: kerogen volume from TOC, effective
  porosity, free gas G_free = φ_e(1−Sw)/Bg, Langmuir adsorbed gas, and the
  free/adsorbed storage partition - the storage-capacity workflow.

- **Article 2 (Fu et al.)**: the BET surface area, the FHH fractal dimension from
  the N2 isotherm, the Langmuir methane-adsorption isotherm, and a composition
  (TOC + clay) control on the Langmuir volume.

- **Article 3 (Mendoza et al.)**: a Bayesian (MAP) linear inversion with a
  Gaussian smoothness prior for accelerated whole-core CT - it beats unregularized
  least squares on a few noisy projections and yields a posterior covariance.

- **Article 4 (Almasoodi & Reza)**: a finite-volume Laplace solver on a pore grid,
  effective conductivity from the steady flux, tortuosity τ = φ·σ_fluid/σ_eff
  (1 for an open slab, > 1 for a tortuous path), and Kozeny-Carman permeability.

- **Article 5 (Mitchell & Valori)**: inversion- and saturation-recovery T1
  models, a full nonlinear T1 fit, and a fast two-point T1 estimate that matches
  the full fit at a fraction of the acquisition time.

- **Article 6 (Ruth & Arabjamaloei)**: the first-order Klinkenberg
  k_app = k_l(1+b/Pm) and a second-order k_app = k_l(1+b/Pm+c/Pm²) model, both
  fit by regression against 1/Pm - the second-order term captures the
  low-pressure curvature the first-order misses.

- **Article 7 (Wang & Peng)**: Archie formation factor and effective cementation
  exponent, with separate vugs raising the effective m above 2 and conductive
  fractures lowering it - and the resulting water-saturation bias from assuming
  m = 2.

- **Article 8 (Hulea)**: the buoyancy capillary pressure vs height, a Brooks-Corey
  saturation-height function, the entry height, and the condition under which a
  high-entry-pressure barrier perches water above the regional free-water level.

- **Article 9 (Poedjono et al.)**: the minimum-curvature survey method - dogleg
  and ratio factor, the TVD/north/east station increments, the cumulative 3D well
  path, and along-hole position-uncertainty growth.

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
