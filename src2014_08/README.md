# Petrophysics August 2014 - Vol. 55, No. 4

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 55, No. 4 (August 2014) - the **Best of the 2013 SCA Symposium** plus two
regular contributions: drainage three-phase relative permeability on oil-wet
carbonates, direct hydrodynamic simulation of multiphase flow in porous rock,
multiphase flow imaged under dynamic conditions with fast X-ray
microtomography, the impact of wettability on residual oil saturation and
capillary desaturation curves, dielectric permittivity as a petrophysical
parameter for shales, and a petrophysical analysis of siliceous-ooze sediments
in the Møre Basin.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article1_threephase_relperm_carbonate.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_threephase_relperm_carbonate.py` | Drainage Three-Phase Flow Relative Permeability on Oil-Wet Carbonate Reservoir Rock Types | Egermann, Mejdoub, Lombard, Vizika, Kalam | 287-293 |
| `article2_dhd_multiphase_simulation.py` | Direct Hydrodynamic Simulation of Multiphase Flow in Porous Rock | Koroteev, Dinariev, Evseev, Klemin, Nadeev, Safonov, Gurpinar, Berg, van Kruijsdijk, Armstrong, Myers, Hathon, de Jong | 294-303 |
| `article3_microct_haines_jumps.py` | Multiphase Flow in Porous Rock Imaged Under Dynamic Flow Conditions with Fast X-Ray Computed Microtomography | Berg, Armstrong, Ott, Georgiadis, Klapp, Schwing, Neiteler, Brussee, Makurat, Leu, Enzmann, Schwarz, Wolf, Khan, Kersten, Irvine, Stampanoni | 304-312 |
| `article4_wettability_capillary_desaturation.py` | Impact of Wettability on Residual Oil Saturation and Capillary Desaturation Curves | Humphry, Suijkerbuijk, van der Linde, Pieterse, Masalmeh | 313-318 |
| `article5_dielectric_permittivity_shales.py` | Dielectric Permittivity: A Petrophysical Parameter for Shales | Josh | 319-332 |
| `article6_siliceous_ooze_petrophysics.py` | Petrophysical Analysis of Siliceous-Ooze Sediments, Møre Basin, Norwegian Sea | Awadalkarim, Sørensen, Fabricius | 333-348 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 55 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2014_08.pdf`,
> ~6 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly. Equation availability varies widely: Article 5's
> empirical correlations (Eqs. 1, 4, 5, 6, A1) and Article 6's grain-density /
> hydrogen-index relations (Eqs. 1, 3, 7, 9, 10) survived verbatim, while many
> display-equation bodies were dropped in extraction and reconstructed in
> standard form from the surviving variable definitions and nomenclature.
> Articles 1 and 2 render no display equations at all - Article 1 only *names*
> the three-phase correlations it compares against (Stone I/II, Baker), and
> Article 2's density-functional governing equations are cited to Demianov et
> al. (2011); both are written here in their standard textbook forms. The cover
> features Article 3 (Berg et al.). (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Egermann et al.)**: three-phase rel-perm - Corey two-phase
  endpoints (krw, krow, krog, krg), the Stone I (1970) and Stone II (1973)
  three-phase oil relative permeabilities, and the Baker (1988) saturation-
  weighted interpolation. The paper renders no equations, so these are standard
  reconstructions; an oil-wet Amott index WI = -0.7 is reported.

- **Article 2 (Koroteev et al.)**: direct hydrodynamic simulation - the square-
  gradient (van der Waals) Helmholtz free-energy density and its chemical
  potential, a 1D diffuse-interface (tanh) density profile, the capillary number
  `Ca = mu*v/sigma`, and relative permeability from a steady Darcy flux. The DHD
  governing equations are cited to Demianov et al. (2011) and reconstructed in
  standard density-functional form; a validation sandstone has phi = 0.23,
  k = 1,150 mD.

- **Article 3 (Berg et al.)**: micro-CT Haines jumps - the capillary number
  `Ncap = v*mu/sigma`, the Haines-jump pressure-volume work `W = integral(p dV)`
  (Eq. 1), the pore-scale energy balance splitting reversible interfacial energy
  (~36%) from dissipation (~64%) (Eq. 2), the invasion-percolation event-size
  power law `N ~ (dV/Vpore)^-n` with measured n = 1.0 (Eq. 3), and imaging-
  derived porosity/saturation from voxel counts.

- **Article 4 (Humphry et al.)**: wettability & CDC - the capillary number
  `N_Ca = v_b*mu_b/gamma` (Eq. 1), Bond number `N_Bo = drho*a*k/gamma` (Eq. 2),
  the trapping number, the Ma et al. (1999) dimensionless imbibition time
  (Eq. 3), and a capillary desaturation curve `Sor(N)` with a critical (onset)
  number modelled as a logistic-in-log-N transition.

- **Article 5 (Josh)**: dielectric permittivity of shales - the equivalent
  imaginary permittivity (polarization + conduction loss), the Debye and Cole-
  Cole relaxation models, and the empirical correlations of CEC (Eq. 1), specific
  surface area (Eqs. 4, 5), P-wave velocity (Eq. 6) and the analytic clay SSA
  (Eq. A1) against the paste permittivity.

- **Article 6 (Awadalkarim et al.)**: siliceous-ooze petrophysics - the opal
  fraction from grain density (Eq. 1), non-opal volume from gamma ray (Eq. 2),
  corrected grain density (Eq. 3), structural-water moles in SiO2.nH2O (Eq. 4),
  the corrected bulk density (Eq. 7) and density porosity (Eq. 8), the hydrogen-
  index neutron correction (Eqs. 9-11), and the Biot coefficient
  `beta = 1 - K_dry/K_o` (Eq. 12).

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2014)
Reference: Petrophysics Vol. 55, No. 4, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
