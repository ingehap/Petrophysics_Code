# Petrophysics August 2015 - Vol. 56, No. 4

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 56, No. 4 (August 2015): four articles plus a technical note, spanning
subsurface fluid characterization with NMR T1-T2 maps and pore-scale imaging,
in-situ vapor evaluation via condensed vapor gamma, gas diffusion into oil with
reservoir baffling and tar mats, an inversion-based interpretation of
neutron-induced gamma-ray spectroscopy, and the Bateman-Konen resistivity-salinity
transform.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article4_spectroscopy_inversion.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_nmr_t1t2_fluid_characterization.py` | Subsurface Fluid Characterization Using Downhole and Core NMR T1-T2 Maps Combined with Pore-Scale Imaging Techniques | Lessenger, Merkel, Medina, Ramakrishna, Chen, Balliet, Xie, Bhattad, Carnerup, Knackstedt | 313-333 |
| `article2_condensed_vapor_gamma.py` | In-Situ Evaluation of Vapor Properties Using Condensed Vapor Gamma | O'Sullivan | 334-345 |
| `article3_gas_diffusion_tar_mats.py` | Gas Diffusion into Oil, Reservoir Baffling and Tar Mats Analyzed by Downhole Fluid Analysis, Pressure Transients, Core Extracts and DSTs | Achourov, Pfeiffer, Kollien, Betancourt, Zuo, di Primio, Mullins | 346-357 |
| `article4_spectroscopy_inversion.py` | Petrophysical Interpretation of LWD, Neutron-Induced Gamma-Ray Spectroscopy Measurements: An Inversion-Based Approach | Ajayi, Torres-Verdín, Preeg | 358-378 |
| `article5_bateman_konen_resistivity_salinity.py` | *Technical Note:* The Bateman-Konen Resistivity-Salinity Transform II | Kennedy | 379-381 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 56 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2015_08.pdf`,
> ~20 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly. **Articles 1-4 have full bodies** and their
> numbered relations survived as inline text (the NMR relaxation/echo relations,
> the radon-condensation physics, the diffusion/FHZ relations, and the
> regularized-inversion/mixing-law/porosity relations); the typeset
> display-equation glyphs were dropped and are faithful standard-form
> reconstructions. **The technical note (Article 5) was beyond the text
> extraction** (the source truncates within Article 4), so it is implemented as
> a methodology proxy from the standard resistivity-salinity-temperature
> transforms it concerns, consistent with how other truncated items are handled
> in this repository.

## Implementation notes & substitutions

- **Article 1 (Lessenger et al.)**: NMR T1-T2 fluid characterization - the
  bulk/surface T1 (Eq. 1), the apparent and intrinsic T2 with the diffusion term
  (Eqs. 2-3), the diffusion relaxation rate (gamma*G*TE)^2*D/12, the
  multiexponential CPMG echo-decay forward model (Eq. 4), and T1/T2 fluid typing.

- **Article 2 (O'Sullivan)**: condensed vapor gamma - the Clausius-Clapeyron
  vapor pressure vs. temperature, the radon-222 decay constant / radioactive
  decay / activity (3.82-day half-life), the radon vapor-liquid partition
  (Ostwald solubility, ~10x for hydrocarbon), and the condensation concentration
  factor producing the anomalous gamma amplitude.

- **Article 3 (Achourov et al.)**: gas diffusion & tar mats - the diffusion
  length sqrt(D*t) and baffle equilibration time L^2/D, the Fickian gas-
  concentration profile C0*erfc(x/(2 sqrt(D t))) into a semi-infinite oil column,
  the FHZ asphaltene gravity gradient (Yen-Mullins cluster sizes), and the
  tar-mat onset where asphaltene content exceeds its solubility limit.

- **Article 4 (Ajayi et al.)**: spectroscopy inversion - the regularized
  (Tikhonov / Occam) least-squares inversion of elemental yields (Eqs. 1, 3),
  the linear mixing law for matrix-sensitive properties (Eq. 2), the matrix
  gamma ray from K/U/Th, the volume-to-weight-fraction conversion (Eq. 6), and
  the density porosity from the inverted matrix density (Eq. 7).

- **Article 5 (Kennedy)** *(technical note, methodology proxy - body beyond
  extraction)*: Bateman-Konen resistivity-salinity - the Arps temperature
  conversion of resistivity, the salinity<->Rw transform (chart fit), the
  apparent water resistivity Rwa, and Archie water saturation.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2015)
Reference: Petrophysics Vol. 56, No. 4, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
