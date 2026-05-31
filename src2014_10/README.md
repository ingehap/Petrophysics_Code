# Petrophysics October 2014 - Vol. 55, No. 5

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 55, No. 5 (October 2014) - the **Best of the 2014 SPWLA Annual Logging
Symposium**: inversion-based interpretation of LWD resistivity and nuclear
measurements in high-angle and horizontal wells, core-data quality control for
elemental-spectroscopy log interpretation, an assessment of nuclear-based
alternatives to chemical-source bulk density, kerogen/maturity/mineralogy and
clay typing from DRIFTS, the impact of petrophysical properties on near-wellbore
nanoparticle distribution, and an oil-movability quicklook from dielectric
measurements at four depths of investigation.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article1_lwd_inversion_anisotropy.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_lwd_inversion_anisotropy.py` | Inversion-Based Interpretation of Logging-While-Drilling Resistivity and Nuclear Measurements: Field Examples of Application in High-Angle and Horizontal Wells | Ijasan, Torres-Verdín, Preeg, Rasmus, Stockhausen | 374-391 |
| `article2_elemental_spectroscopy_qc.py` | Application and Quality Control of Core Data for the Development and Validation of Elemental Spectroscopy Log Interpretation | S. Herron, M. Herron, Pirie, Saldungaray, Craddock, Charsky, Polyakov, Shray, Li | 392-414 |
| `article3_nuclear_density_alternatives.py` | An Assessment of Fundamentals of Nuclear-Based Alternatives to Conventional Chemical-Source Bulk-Density Measurement | Badruzzaman | 415-434 |
| `article4_drifts_kerogen_mineralogy.py` | Kerogen Content and Maturity, Mineralogy and Clay Typing from DRIFTS Analysis of Cuttings or Core | M. Herron, Loan, Charsky, S. Herron, Pomerantz, Polyakov | 435-446 |
| `article5_nanoparticle_transport.py` | Quantifying the Impact of Petrophysical Properties on Spatial Distribution of Contrasting Nanoparticle Agents in the Near-Wellbore Region | Cheng, Aderibigbe, Alfi, Heidari, Killough | 447-460 |
| `article6_dielectric_oil_movability.py` | Application of an Oil-Movability Quicklook Technique Using Dielectric Measurements at Four Depths of Investigation | Grayson, Hemingway | 461-469 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 55 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2014_10.pdf`,
> ~10 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly. The equation availability varies: Article 2's
> organic-correction equations (A-1 to A-5) and Article 3's Compton transmission
> (Eq. 1) survived as inline text, while many display-equation bodies were
> dropped in extraction (Article 1's anisotropy Eq. A-3/A-4, Article 2's grain
> density Eq. A-6, Article 3's Eq. 2/3, and most of Article 5's Eqs. 1-25) and
> are faithfully reconstructed from the surviving variable definitions and
> nomenclature in standard form. Articles 4 and 6 carry no numbered display
> equations; their methods (DRIFTS Kubelka-Munk linear-combination inversion and
> maturity-scaled TOC; simplified CRIM water-filled porosity and the moved-oil
> quicklook) are transcribed from the prose. The cover features Article 1
> (Ijasan et al.). (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Ijasan et al.)**: LWD inversion & anisotropy - the Hagiwara
  (1996) horizontal conductivity (Eq. A-2) and vertical resistivity (Eq. A-4)
  net-to-gross mixing, the anisotropy coefficient `lambda = sqrt(Rv/Rh)`, the
  net-sand Archie water saturation with an N/G net-pay cutoff, a quadratic
  data-misfit cost, and a linear net/shale conductivity inversion. Field
  Example II/III resistivity values are used in the demo.

- **Article 2 (Herron et al.)**: elemental spectroscopy core QC - the
  element->oxide association factors and oxide closure (proportionality factor
  F), pyrite from sulfur (`100*S/53`), QCMIN element reconstruction from a
  mineralogy with its ad/aad/score metrics, organic matter from TOC (Eq. A-1),
  organic-dilution removal (Eq. A-2), the iron spectral-interference correction
  (`Fe + 0.14*Al`, Eq. A-3), and the TOC-based grain density (Eq. A-6).

- **Article 3 (Badruzzaman)**: nuclear density alternatives - the Compton
  transmission (Eq. 1) and attenuation coefficient (Eq. 2), the neutron-gamma
  capture fraction over a post-burst window (Eq. 3), the neutron-gamma density
  from the log of the two-detector inelastic count ratio (Neuman et al., 1999),
  and the counting-statistics precision. Effective-Z values (quartz 11.78,
  dolomite 13.74, limestone 15.71) are from the paper.

- **Article 4 (Herron et al.)**: DRIFTS - the Kubelka-Munk remission function,
  the weighted non-negative least-squares mineral inversion as a linear
  combination of pure mineral-standard spectra, the mineral-stripped organic
  signal over the 2,800-3,000 cm^-1 aliphatic C-H band, the maturity-scaled TOC
  (`TOC ~ Ro*signal`), and a clay-abundance CEC estimate.

- **Article 5 (Cheng et al.)**: nanoparticle transport - the Stokes-Einstein
  diffusion (Eq. 12), Millington-Quirk effective diffusion (Eqs. 13-14), the
  pore velocity (Eq. 8), the combined dispersion coefficient (Eqs. 9, 16), the
  first-order deposition rate (Eq. 10), and a 1D advection-dispersion-filtration
  finite-difference solver (Eqs. 11, 25). The deposition coefficient
  `k_dep = 2.3e-6 1/s` is the paper's history-matched value.

- **Article 6 (Grayson & Hemingway)**: dielectric oil movability - the CRIM
  mixing permittivity, the simplified-CRIM water-filled porosity from the
  apparent permittivity, the water/oil saturations (`Sw = phi_water/phi_total`),
  the radial saturation profile from the four depths of investigation, and the
  moved-oil quicklook `dSo = So(deep) - So(shallow)`.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2014)
Reference: Petrophysics Vol. 55, No. 5, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
