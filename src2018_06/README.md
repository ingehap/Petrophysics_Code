# Petrophysics June 2018 - Vol. 59, No. 3

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 59, No. 3 (June 2018): the third **Shaly Sand** tutorial, five
formation-evaluation articles (organic-shale porosity, shale wettability,
clay-network resistivity, wideband EM, and dielectric CEC logging), and three
regular submissions (carbonate permeability heterogeneity, saturation-height
stress corrections, and NMR of magnetic nanoparticles).

## Quick start

```bash
pip install numpy

# Run all 9 module tests
python test_all.py

# Or run a single article
python article4_clay_network_resistivity_saturation.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_shaly_sand_tutorial_part3.py` | *Tutorial:* What is it about Shaly Sands? Shaly Sand Tutorial No. 3 of 3 | Thomas | 10.30632/PJV59N3-2018t1 |
| `article2_matrix_adjusted_shale_porosity.py` | Matrix-Adjusted Shale Porosity Measured in Horizontal Wells | Craddock, Mossé, Bernhardt, Ortiz, Gonzalez Tomassini, Pirie, Saldungaray, Pomerantz | 10.30632/PJV59N3-2018a1 |
| `article3_nmr_wettability_index_shales.py` | Water-Wet or Oil-Wet: is it Really That Simple in Shales? | Gupta, Jernigen, Curtis, Rai, Sondergeld | 10.30632/PJV59N3-2018a2 |
| `article4_clay_network_resistivity_saturation.py` | A New Resistivity-Based Model for Improved Hydrocarbon Saturation Assessment in Clay-Rich Formations Using Quantitative Geometry of the Clay Network | Garcia, Jagadisan, Rostami, Heidari | 10.30632/PJV59N3-2018a3 |
| `article5_wideband_em_dem_permittivity.py` | Coherent Interpretation of Wideband Electromagnetic Measurements in the Millihertz to Gigahertz Frequency Range | Seleznev, Hou, Freed, Habashy, Feng, Fellah, Xu, Nadeev | 10.30632/PJV59N3-2018a4 |
| `article6_dielectric_cec_shaly_sand.py` | A Physics-Based Model for the Dielectric Response of Shaly Sands and Continuous CEC Logging | Freed, Seleznev, Hou, Fellah, Little, Dumy, Sen | 10.30632/PJV59N3-2018a5 |
| `article7_carbonate_permeability_heterogeneity.py` | Digital and Conventional Techniques to Study Permeability Heterogeneity in Complex Carbonate Rocks | Dernaika, Al Mansoori, Singh, Al Dayyani, Kalam, Bhakta, Koronfol, Uddin | 10.30632/PJV59N3-2018a6 |
| `article8_saturation_height_stress_correction.py` | Saturation-Height Modeling: Assessing Capillary Pressure Stress Corrections | Hulea | 10.30632/PJV59N3-2018a7 |
| `article9_nmr_fe3o4_nanoparticle_relaxation.py` | NMR Relaxation of Surface-Functionalized Fe3O4 Nanoparticles | Zhu, Ko, Daigle, Zhang | 10.30632/PJV59N3-2018a8 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2018_06.pdf`,
> ~40 MB) has a text layer, so titles, authors, page ranges, and DOIs were read
> from the contents page and bodies. The machine extraction captured the full
> bodies of the **tutorial and articles a1-a5** (through journal page ~372) but
> **truncated after a5**, so **articles a6-a8 were available only as
> table-of-contents entries** and are implemented as **methodology proxies**
> (their DOI suffixes a6-a8 are inferred from the confirmed pattern, not read
> from an article header). As with the other issues, the typeset formula glyphs
> were dropped in extraction, so the numbered formulas are faithful standard-form
> reconstructions. DOI pattern: `10.30632/PJV59N3-2018aN` (a1 ... a8) plus the
> tutorial `...t1` (prefix `PJV59N3`, capital N, hyphen separator).

## Implementation notes & substitutions

- **Article 1 (Thomas)** *(tutorial)*: the Waxman-Smits shaly-sand model in
  conductivity space - the clean-sand line C0 = Cw/F, the shaly offset
  C0 = (Cw + B·Qv)/F*, the formation factor F* = phi^-m*, the saturation
  conductivity, and a Thomas-Stieber laminated-shale porosity example.

- **Article 2 (Craddock et al.)**: organic-shale total porosity - the density
  porosity, a kerogen-inclusive matrix density by reciprocal mass mixing, the
  electron-to-bulk density conversion (Eq. 5, verbatim), and the kerogen mass
  fraction from TOC.

- **Article 3 (Gupta et al.)**: shale wettability - the NMR spontaneous-imbibition
  wettability index, its two-sequence average, and the paper's TOC (5 wt%) and
  clay (10 / 65 wt%) percolation thresholds, plus a spectral-GR synthesis.

- **Article 4 (Garcia et al.)**: image-based clay-network resistivity - directional
  electrical tortuosity, the percolating clay-network conductivity, a
  Maxwell-Garnett inclusion mixing for the other components, the summed total
  conductivity, and the Archie water saturation.

- **Article 5 (Seleznev et al.)**: wideband (mHz-GHz) EM - the complex permittivity
  with its low-frequency conductivity term, spheroid depolarization factors, a
  Bruggeman effective-medium mixing, and the Archie F = phi^-m limit.

- **Article 6 (Freed et al.)**: a physics-based dielectric model for continuous CEC
  logging - the CEC↔surface-conductivity relation (with the Stern-layer fraction
  and Nernst-Einstein mobility), the whole-rock CEC, and the complex permittivity;
  reduces to the uncharged case at CEC = 0.

- **Article 7 (Dernaika et al.)** *(methodology proxy)*: carbonate permeability
  heterogeneity - arithmetic (horizontal) vs. harmonic (vertical) layer averaging,
  the Kv/Kh anisotropy ratio, and the Dykstra-Parsons and Lorenz heterogeneity
  coefficients.

- **Article 8 (Hulea)** *(methodology proxy)*: saturation-height modeling - the
  Leverett J-function, the saturation-height function, a net-stress
  permeability/porosity correction that rescales capillary pressure, and a
  Brooks-Corey saturation curve.

- **Article 9 (Zhu et al.)** *(methodology proxy)*: NMR of magnetic nanoparticles -
  the surface (fast-diffusion) relaxation from S/V, the concentration-linear
  relaxivity law 1/T = 1/T0 + r·C, the relaxivity recovered by a linear fit, and
  the r2/r1 ratio.

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
