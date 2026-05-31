# Petrophysics October 2015 - Vol. 56, No. 5

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 56, No. 5 (October 2015): six articles spanning the untangling of acoustic
anisotropy, reservoir fluid geodynamics (differing equilibration times of GOR,
asphaltenes and biomarkers), a robust Bakken petrophysical model, EMAT downhole
cement evaluation, integrated rock classification in the McElroy Field, and a
consistent evaluation approach to thin-bedded sands in a Gulf of Mexico
deepwater field.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article2_reservoir_fluid_geodynamics.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_acoustic_anisotropy.py` | Untangling Acoustic Anisotropy | Market, Mejia, Mutlu, Shahri, Tudge | 420-439 |
| `article2_reservoir_fluid_geodynamics.py` | Differing Equilibration Times of GOR, Asphaltenes and Biomarkers as Determined by Charge History and Reservoir Fluid Geodynamics | Wang, Kauerauf, Zuo, Chen, Dong, Elshahawi, Mullins | 440-456 |
| `article3_bakken_petrophysical_model.py` | Using Advanced Logging Measurements to Develop a Robust Petrophysical Model for the Bakken Petroleum System | Simpson, Hohman, Pirie, Horkowitz | 457-478 |
| `article4_emat_cement_evaluation.py` | Utilization of Electromagnetic Acoustic Transducers in Downhole Cement Evaluation | Patterson, Bolshakov, Matuszyk | 479-492 |
| `article5_mcelroy_rock_classification.py` | Integrated Petrophysical Rock Classification in the McElroy Field, West Texas, USA | Saneifar, Skalinski, Theologou, Kenter, Cuffey, Salazar-Tio | 493-510 |
| `article6_thinbedded_sands_gom.py` | Applying a Consistent Evaluation Approach to Thin-Bedded Sands in a Gulf of Mexico Deepwater Field | Salunke, Hamman | 511-520 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 56 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2015_10.pdf`,
> ~7 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly. **Articles 1-4 have full bodies** and their
> numbered relations survived as inline text (the Alford-rotation/anisotropy
> tools, the FHZ gravity term and diffusion relations, the multimineral/SDR/Rh-Rv
> Bakken relations, and the SH/Lamb cement-wave relations); the typeset
> display-equation glyphs were dropped and are faithful standard-form
> reconstructions. **Articles 5 and 6 were beyond the text extraction** (the
> source truncates within Article 4), so they are implemented as methodology
> proxies from the standard methods their titles/abstracts describe (carbonate
> rock classification; Thomas-Stieber + Rh-Rv thin-bed analysis), consistent
> with how other truncated articles are handled in this repository. (This issue
> has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Market et al.)**: untangling acoustic anisotropy - the shear
  anisotropy magnitude from fast/slow velocities (and slowness), Alford rotation
  of crossed-dipole (XX/XY/YX/YY) to fast/slow principal waveforms, the
  fast-shear azimuth by cross-component energy minimization, and the Thomsen
  shear-anisotropy parameter gamma.

- **Article 2 (Wang et al.)**: reservoir fluid geodynamics - the gravitational
  force on an asphaltene particle (Eq. 1), the Flory-Huggins-Zuo asphaltene
  (optical-density) gravity gradient (Eq. 2, dominant term, Yen-Mullins particle
  sizes), the diffusion length sqrt(D*t), and the diffusive equilibration time
  L^2/D (so GOR equilibrates long before asphaltene clusters).

- **Article 3 (Simpson et al.)**: Bakken petrophysical model - the linear
  volumetric log response (Eqs. 1-2) and multimineral inversion, the carbonate
  SDR NMR permeability (Eq. 3), the NMR pore surface-to-volume from T2 (Eq. 4),
  and the bimodal Rh (parallel) / Rv (series) thin-bed model and solver for the
  reservoir resistivity in the laminated Three Forks (Eqs. 5-7).

- **Article 4 (Patterson et al.)**: EMAT cement evaluation - the SH plate-mode
  cutoff frequency and group velocity (Eqs. 1-2; SH0 nondispersive, higher
  orders dispersive), the shear modulus G = rho*Vs^2, an SH attenuation model
  vs. cement shear modulus (zero against a liquid), and the Rayleigh-Lamb
  dispersion residual for the symmetric/antisymmetric (A0) modes (Eq. 3).

- **Article 5 (Saneifar et al.)** *(methodology proxy - body beyond
  extraction)*: McElroy rock classification - the Winland r35 pore-throat
  radius, the reservoir-quality / flow-zone indicators (RQI, FZI), and a
  k-means electrofacies clustering of samples into petrophysical rock classes.

- **Article 6 (Salunke & Hamman)** *(methodology proxy - body beyond
  extraction)*: thin-bedded sands - the Thomas-Stieber laminated/dispersed
  shale porosity trends, the recovery of laminar-shale volume and sand porosity,
  the laminated Rh/Rv resistivity with the sand-resistivity inversion, and the
  Archie sand water saturation (avoiding underestimation of low-resistivity
  thin-bed pay).

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2015)
Reference: Petrophysics Vol. 56, No. 5, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
