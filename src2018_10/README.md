# Petrophysics October 2018 - Vol. 59, No. 5

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 59, No. 5 (October 2018) - the **"Best of 2018 SPWLA Symposium"** issue: a
capillary-pressure tutorial, nine symposium papers, and one regular submission.

## Quick start

```bash
pip install numpy

# Run all 11 module tests
python test_all.py

# Or run a single article
python article2_xray_sourceless_density.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_capillary_pressure_tutorial_part2.py` | *Tutorial:* Capillary Pressure Tutorial Part 2 - The Path Out of the Jungle | Thomas | 10.30632/PJV59N5-2018t1 |
| `article2_xray_sourceless_density.py` | A Novel X-Ray Tool for True Sourceless Density Logging | Simon, Tkabladze, Beekman, Atobatele, De Looz, Grover, Hamichi, Jundt, McFarland, Mlcak, Reijonen, Revol, Stewart, Yeboah, Zhang | 10.30632/PJV59N5-2018a1 |
| `article3_kerogen_log_geomechanics.py` | Integrating Measured Kerogen Properties With Log Analysis for Petrophysics and Geomechanics in Unconventional Resources | Craddock, Mossé, Prioul, Miles, Loan, Pirie, Rylander, Lewis, Pomerantz | 10.30632/PJV59N5-2018a2 |
| `article4_fast_pressure_decay_permeability.py` | Fast Pressure-Decay Core Permeability Measurement for Tight Rocks | Gan, Griffin, Dacy, Xie, Lee | 10.30632/PJV59N5-2018a3 |
| `article5_unsupervised_nmr_t1t2_fluid_volumes.py` | An Unsupervised Learning Algorithm to Compute Fluid Volumes From NMR T1-T2 Logs in Unconventional Reservoirs | Venkataramanan, Evirgen, Allen, Mutina, Cai, Johnson, Green, Jiang | 10.30632/PJV59N5-2018a4 |
| `article6_proxy_stochastic_fluid_sampling.py` | Proxy-Enabled Stochastic Interpretation of Downhole Fluid Sampling Under Immiscible Flow Conditions | Kristensen, Chugunov, Cig, Jackson | 10.30632/PJV59N5-2018a5 |
| `article7_dfa_gas_chromatography.py` | Downhole Fluid Analysis and Gas Chromatography; a Powerful Combination for Reservoir Evaluation | Mullins, Forsythe, Pomerantz, et al. | 10.30632/PJV59N5-2018a6 |
| `article8_permeability_nmr_electric_rockfabric.py` | Integrated Workflow to Estimate Permeability Through Quantification of Rock Fabric Using Joint Interpretation of NMR and Electric Measurements | Garcia, Han, Heidari | 10.30632/PJV59N5-2018a7 |
| `article9_digital_rock_porosity_upscaling.py` | Upscaling of Digital Rock Porosities by Correlation With Whole-Core CT-Scan Histograms | Hertel, Rydzy, Anger, Berg, Appel, de Jong | 10.30632/PJV59N5-2018a8 |
| `article10_resistivity_mixedwet_rocks.py` | Improved Interpretation of Electrical Resistivity Measurements in Mixed-Wet Rocks | Newgord, Garcia, Rostami, Heidari | 10.30632/PJV59N5-2018a9 |
| `article11_hierarchical_rock_classification.py` | A New Hierarchical Method for Rock Classification Using Well-Log-Based Rock Fabric Quantification | Purba, Garcia, Heidari | 10.30632/PJV59N5-2018a10 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2018_10.pdf`,
> ~49 MB) has a text layer, so titles, authors, page ranges, and DOIs were read
> from the contents page and bodies. The machine extraction captured the full
> bodies of the **tutorial and articles a1-a6** but **truncated after a6** (page
> ~671), so **articles a7-a10 were available only as table-of-contents entries**
> and are implemented as **methodology proxies**. As with the other issues, the
> typeset formula glyphs were dropped in extraction, so the numbered formulas are
> faithful standard-form reconstructions. DOI pattern: `10.30632/PJV59N5-2018aN`
> (a1 … a10) plus the tutorial `…t1`.

## Implementation notes & substitutions

- **Article 1 (Thomas)** *(tutorial)*: lab-to-reservoir capillary-pressure
  conversion via the |σ·cosθ| ratio, the saturation-height function, the
  Leverett J-function, and drainage/imbibition hysteresis.

- **Article 2 (Simon et al.)**: the Compton density response, density from the
  count ratio, a spine-and-ribs mudcake/standoff correction (DRHO), and a
  photoelectric-factor proxy from the soft/hard window ratio.

- **Article 3 (Craddock et al.)**: TOC→kerogen volume, three-component bulk
  density, Voigt-Reuss-Hill modulus mixing with a soft kerogen component, and the
  dynamic Young's modulus / Poisson's ratio.

- **Article 4 (Gan et al.)**: the pulse-decay pressure relaxation, the decay
  time constant, permeability from the fitted decay rate (recovers a planted
  microdarcy value), and the Klinkenberg correction.

- **Article 5 (Venkataramanan et al.)**: an unsupervised (weighted k-means)
  clustering of the NMR T1-T2 map into fluid populations and their volumes -
  recovers planted clay-bound / capillary / movable volumes.

- **Article 6 (Kristensen et al.)**: the power-law contamination cleanup proxy
  and a Monte-Carlo (Bayesian) posterior on the cleanup parameters and the
  pumpout volume, with uncertainty.

- **Article 7 (Mullins et al.)**: gas/oil ratio from a C1-C7+ composition, the
  FHZ asphaltene optical-density gradient with depth, and an equilibrium /
  connectivity check.

- **Article 8 (Garcia et al.)** *(methodology proxy)*: Timur-Coates NMR
  permeability, the cementation exponent from the formation factor, and a joint
  NMR + electrical permeability scaled by connectivity.

- **Article 9 (Hertel et al.)** *(methodology proxy)*: CT porosity from
  attenuation, the running-average REV convergence, histogram porosity, and a
  linear digital-rock→whole-core upscaling correlation.

- **Article 10 (Newgord et al.)** *(methodology proxy)*: the Archie resistivity
  index, a wettability-dependent saturation exponent n(oil-wet fraction), and the
  Sw bias from assuming a fixed water-wet n.

- **Article 11 (Purba et al.)** *(methodology proxy)*: average-linkage
  agglomerative hierarchical clustering of rock-fabric features, with a
  silhouette validation and a dendrogram cut into rock classes.

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
