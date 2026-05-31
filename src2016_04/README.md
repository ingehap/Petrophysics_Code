# Petrophysics April 2016 - Vol. 57, No. 2

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 57, No. 2 (April 2016): five articles spanning the Reservoir Producibility
Index for tight-oil reservoir quality, integrated petrofacies characterization
of the Bakken shale, a new approach to measuring organic (kerogen) density, a
multilevel iterative method for pore-confinement phase equilibrium, and acoustic
anisotropy interpretation in shales when the Stoneley-wave velocity is missing.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article1_reservoir_producibility_index.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_reservoir_producibility_index.py` | The Reservoir Producibility Index: a Metric to Assess Reservoir Quality in Tight-Oil Plays from Logs | Reeder, Craddock, Rylander, Pirie, Lewis, Kausik, Kleinberg, Yang, Pomerantz | 83-95 |
| `article2_bakken_petrofacies.py` | Integrated Petrofacies Characterization and Interpretation of Depositional Environment of the Bakken Shale in the Williston Basin, North America | Bhattacharya, Carr | 96-111 |
| `article3_organic_density.py` | A New Approach to Measuring Organic Density | Dang, Sondergeld, Rai | 112-120 |
| `article4_pore_confinement_phase_equilibrium.py` | A Multilevel Iterative Method to Quantify Effects of Pore-Size Distribution on Phase Equilibrium of Multicomponent Fluids in Unconventional Plays | Li, Mezzatesta, Li, Ma, Jamili | 121-139 |
| `article5_acoustic_anisotropy_no_stoneley.py` | Method for Acoustic Anisotropy Interpretation in Shales When the Stoneley-Wave Velocity is Missing | Gu, Quirein, Murphy, Rivera Barraza, Ou | 140-156 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 57 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2016_04.pdf`,
> ~8 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly; **all five articles have full bodies**. The
> numbered relations survived as inline text (the OSI/RPI definitions, the
> Schmoker TOC and multimineral inversion, the organic-density mass balance and
> regression, the flash / capillary / parachor relations, and the VTI-moduli /
> Thomsen / V-reg / M-ANNIE 2 closures); the typeset display-equation glyphs
> were dropped and are faithful standard-form reconstructions from the surviving
> variable definitions. (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Reeder et al.)**: Reservoir Producibility Index - the Oil
  Saturation Index OSI = 100*S1/TOC (Eq. 1), the RPI = WC_oil^2/WC_org (Eq. 2),
  the dry-weight TOC -> WC_org conversion (Eq. 3), the NMR oil content WC_oil
  (Eqs. 4-5, T2 > 3 ms with bitumen / clay-bound water removed), and the
  clay-bound-water term.

- **Article 2 (Bhattacharya & Carr)**: Bakken petrofacies - the Schmoker-Hester
  density TOC (Eq. 1), the averaged clay volume (Eq. 2), a stochastic
  multimineral linear inversion (with the unity constraint), the five-petrofacies
  classification from the quartz/carbonate ratio (3 and 1/3) and clay (30%), and
  a chi-square statistic for petrofacies vs. depositional environment.

- **Article 3 (Dang et al.)**: organic density - the bulk-density mass balance
  (Eq. 1), the total grain density from mineral + kerogen grains (Eq. 4), the
  TOM = TOC/K relation (Eq. 5), and the paper's method: regressing 1/rho_gt
  against TOC to extract the mineral and organic (kerogen) grain densities.

- **Article 4 (Li et al.)**: pore-confinement phase equilibrium - the bulk
  vapor-liquid flash (Wilson K-values, Rachford-Rice; Eq. 1), the Young-Laplace
  capillary pressure and the Macleod-Sugden (parachor) interfacial tension that
  drive the confined-fluid model (Eqs. 2-4), and a Peng-Robinson
  compressibility-factor kernel. The full multilevel critical-pore-radius
  iteration is built on these kernels.

- **Article 5 (Gu et al.)**: acoustic anisotropy without Stoneley - the VTI
  Young's moduli / Poisson's ratios (Eqs. 3-4, reducing to isotropic), the
  positive-definite stiffness check (Eqs. 2, 6), the Thomsen parameters, the
  M-ANNIE 2 closure C66 from gamma = 0.93*epsilon, the V-reg off-axis velocity
  prediction with C11/C13 from the 90 deg / 45 deg velocities, and the minimum
  horizontal (closure) stress (Eqs. 8-9).

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2016)
Reference: Petrophysics Vol. 57, No. 2, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
