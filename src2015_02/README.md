# Petrophysics February 2015 - Vol. 56, No. 1

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 56, No. 1 (February 2015) - the Best Papers of the 2014 SCA Symposium plus
two regular submissions: onset of oil mobilization and nonwetting-phase
cluster-size distribution, CO2 EOR by diffusive mixing in fractured reservoirs,
coupled multiphase-hydrodynamic / NMR pore-scale modeling, petrophysical
characterization of Permian Wolfcamp pore space, and recharacterization /
validation of through-the-bit-logging tool measurements.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article3_nmr_pore_scale_modeling.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_oil_mobilization_clusters.py` | Onset of Oil Mobilization and Nonwetting-Phase Cluster-Size Distribution | Berg, Armstrong, Georgiadis, Ott, Schwing, Neiteler, Brussee, Makurat, Rücker, Leu, Wolf, Khan, Enzmann, Kersten | 15-22 |
| `article2_co2_diffusive_mixing.py` | CO2 EOR by Diffusive Mixing in Fractured Reservoirs | Eide, Ersland, Brattekås, Haugen, Graue, Fernø | 23-31 |
| `article3_nmr_pore_scale_modeling.py` | Coupling Multiphase Hydrodynamic and NMR Pore-Scale Modeling for Advanced Characterization of Saturated Rocks | Evseev, Dinariev, Hürlimann, Safonov | 32-44 |
| `article4_wolfcamp_pore_space.py` | Petrophysical Characterization of the Pore Space in Permian Wolfcamp Rocks | Rafatian, Capsan | 45-57 |
| `article5_through_the_bit_logging.py` | Recharacterization and Validation of Through-the-Bit-Logging Tool Measurements | Slocombe, Bammi, Hunka, Reischman, Schmid | 58-71 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 56 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2015_02.pdf`,
> ~15 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly; **all five articles have full bodies**. The
> equation-bearing relations survived as inline text (the macroscopic capillary
> number, the Fickian sqrt-time recovery, the Bloch-Torrey / surface-relaxation
> physics, the Washburn / characteristic-radius permeability models, and the
> spine-and-ribs density relations); the typeset display-equation glyphs were
> dropped in extraction, so they are faithful standard-form reconstructions.
> The cover features Article 3 (Evseev et al.). (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Berg et al.)**: oil mobilization & clusters - the macroscopic
  (cluster-based) capillary number (Eq. 1), the microscopic capillary number,
  logarithmic bin edges for the cluster-size histogram (Eq. 2), and a
  power-law cluster-size distribution with a log-binned exponent fit (the
  measured exponent is the true exponent minus one).

- **Article 2 (Eide et al.)**: CO2 EOR by diffusive mixing - the diffusion
  length sqrt(D*t), the diffusion-controlled square-root-of-time recovery and
  its slope fit, the Fickian early-time fractional recovery from a matrix block,
  and the effective diffusion coefficient from the recovery slope.

- **Article 3 (Evseev et al.)**: NMR pore-scale modeling - the magnetization
  decay, the fast-diffusion surface relaxation (1/T2 = 1/T2bulk + rho*(S/V)),
  the Mitra short-time restricted-diffusion coefficient, and a 1D Bloch-Torrey
  finite-difference simulation of a slab pore with surface-relaxation walls.

- **Article 4 (Rafatian & Capsan)**: Wolfcamp pore space - the Washburn
  pore-throat radius from injection pressure (Eq. 7), the Winland r35 radius and
  its inverse permeability, the Swanson (Sb/Pc)max-apex permeability, and the
  shared characteristic-radius permeability form k = c*phi^a*R^2.

- **Article 5 (Slocombe et al.)**: through-the-bit logging - the log density
  from the electron density index (Eq. 1), the electron density index for a
  single element and a mixture (Eq. 2), and the spine-and-ribs mudcake/standoff
  density compensation combining the long- and short-spacing densities
  (Eqs. 3-5).

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2015)
Reference: Petrophysics Vol. 56, No. 1, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
