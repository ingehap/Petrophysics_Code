# Petrophysics April 2018 - Vol. 59, No. 2

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 59, No. 2 (April 2018): the second **Shaly Sand** tutorial, six
formation-evaluation articles (silt in LRLC pay, NMR pore coupling, neutron/
X-ray imaging, shale total porosity, dielectric matrix calibration, and Bakken
dielectric dispersion), and three regular submissions (2D directional-resistivity
imaging, downhole relative permeability, and PNN lithofacies identification).

## Quick start

```bash
pip install numpy

# Run all 10 module tests
python test_all.py

# Or run a single article
python article5_shale_total_porosity_elemental.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_shaly_sand_tutorial_part2.py` | *Tutorial:* What is it about Shaly Sands? Shaly Sand Tutorial No. 2 of 3 | Thomas | 10.30632/PJV59N2-2018t1 |
| `article2_silt_lrlc_thomas_stieber.py` | The Problem With Silt in Low-Resistivity Low-Contrast (LRLC) Pay Reservoirs | Belevich, Bal | 10.30632/PJV59N2-2018a1 |
| `article3_nmr_pore_coupling.py` | Nuclear Magnetic Resonance and Pore Coupling in Clay-Coated Sandstones... Água Grande Formation, Recôncavo Basin, Brazil | Jácomo, Trindade, de Oliveira, Leite, Montrazi, Andreeta, Bonagamba | 10.30632/PJV59N2-2018a2 |
| `article4_neutron_xray_imaging.py` | Simultaneous Neutron and X-Ray Imaging of 3D Structure of Organic Matter and Fracture in Shales | Chiang, LaManna, Hussey, Jacobson, Liu, Zhang, Georgi, Kone, Chen | 10.30632/PJV59N2-2018a3 |
| `article5_shale_total_porosity_elemental.py` | Calculating the Total Porosity of Shale Reservoirs by Combining Conventional Logging and Elemental Logging to Eliminate the Effects of Gas Saturation | Zhu, Zhang, Guo, Jiao, Chen, Zhou, Zhang, Zhang | 10.30632/PJV59N2-2018a4 |
| `article6_dielectric_matrix_crim_cda.py` | Improving Dielectric Interpretation by Calibrating Matrix Permittivity and Solving Dielectric Mixing Laws With a New Graphical Method | Wang, Wang, Toumelin, Brown, Crousse | 10.30632/PJV59N2-2018a5 |
| `article7_bakken_dielectric_dispersion.py` | Bakken Petroleum System Characterization Using Dielectric-Dispersion Logs | Han, Misra | 10.30632/PJV59N2-2018a6 |
| `article8_2d_directional_resistivity_imaging.py` | 2D Reservoir Imaging Using Deep Directional Resistivity Measurements | Thiel, Bower, Omeragic | 10.30632/PJV59N2-2018a7 |
| `article9_downhole_relative_permeability.py` | Downhole Estimation of Relative Permeability With Integration of Formation-Tester Measurements and Advanced Well Logs | Hadibeik, Azari, Kalawina, Ramakrishna, Eyuboglu, Khan, Al-Rushaid, Al-Rashidi, Ahmad | 10.30632/PJV59N2-2018a8 |
| `article10_pnn_lithofacies.py` | Complex Lithofacies Identification Using Improved Probabilistic Neural Networks | Gu, Bao, Rui | 10.30632/PJV59N2-2018a9 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2018_04.pdf`,
> ~36 MB) has a text layer, so titles, authors, page ranges, and DOIs were read
> from the contents page and bodies. The machine extraction captured the full
> bodies of the **tutorial and articles a1-a6** but **truncated after a6** (the
> a7 DOI header was captured but not its body, and a8-a9 were absent), so
> **articles a7-a9 are implemented as methodology proxies** (their a8-a9 DOI
> suffixes are inferred from the confirmed pattern; a7's was read from its
> header). As with the other issues, the typeset formula glyphs were dropped in
> extraction, so the numbered formulas are faithful standard-form
> reconstructions. DOI pattern: `10.30632/PJV59N2-2018aN` (a1 ... a9) plus the
> tutorial `...t1` (prefix `PJV59N2`, capital N, hyphen separator).

## Implementation notes & substitutions

- **Article 1 (Thomas)** *(tutorial)*: why clay perturbs the porosity logs - the
  neutron-porosity overstatement in shale (phi_N = phi_w + Vclay·HI_clay), the
  1:1-vs-2:1 clay hydrogen indices, the spectral gamma ray, and the Vsh != Vclay
  caution.

- **Article 2 (Belevich & Bal)**: silt in LRLC pay - the dispersed/laminated/
  structural shale-distribution porosities, the total porosity, the Thomas-Stieber
  sand-lamina porosity, and the Rv/Rh resistivity-anisotropy discriminator.

- **Article 3 (Jácomo et al.)**: NMR pore coupling - the fast-diffusion surface
  relaxation (1/T2 = rho2·S/V), the multiexponential magnetization decay, the
  field-gradient diffusion relaxation, and the pore S/V from a measured T2.

- **Article 4 (Chiang et al.)**: simultaneous neutron + X-ray imaging - Lambert-Beer
  attenuation, the cross-section-weighted attenuation coefficient, optical density,
  and the orthogonal-contrast voxel segmentation (organic / mineral / void).

- **Article 5 (Zhu et al.)**: shale total porosity - the organic-matter volume from
  TOC and a five-component density+neutron response solved as a 2x2 system that
  eliminates the hydrocarbon saturation, recovering both phi and Sh.

- **Article 6 (Wang et al.)**: dielectric matrix calibration - the CRIM matrix
  permittivity from mineralogy (kerogen included), the CRIM mixing law, the
  simplified-CRIM water-filled-porosity inversion, and the matrix-permittivity
  sensitivity that motivates Complex-Domain Analysis.

- **Article 7 (Han & Misra)**: Bakken dielectric dispersion - the Lichtenecker-Rother
  power-law mixing with the homogeneity index alpha (CRI at alpha=0.5), the complex
  water permittivity, and the inversion for water saturation.

- **Article 8 (Thiel et al.)** *(methodology proxy)*: deep directional resistivity -
  a DOI-weighted two-bed apparent resistivity, the distance-to-boundary inversion,
  the azimuthal geosignal, and 2D boundary-image assembly.

- **Article 9 (Hadibeik et al.)** *(methodology proxy)*: downhole relative
  permeability - the Corey/Brooks-Corey water and oil relative permeabilities,
  the water fractional flow, and the end-point mobility ratio.

- **Article 10 (Gu et al.)** *(methodology proxy)*: lithofacies identification - a
  Specht probabilistic neural network (Gaussian Parzen-window densities, Bayes
  decision) with leave-one-out smoothing-parameter selection.

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
