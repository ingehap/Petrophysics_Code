# Petrophysics April 2014 - Vol. 55, No. 2

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 55, No. 2 (April 2014) - the **Special Issue on Deepwater** (Gulf of
Mexico): deepwater exploration and production challenges and opportunities, the
origin and characteristics of turbidite sediments, the dynamics of reservoir
fluids and their systematic variations, fault-block migrations inferred from
asphaltene gradients, formation-evaluation challenges and opportunities in
deepwater, and quantifying the effect of kerogen on resistivity measurements in
organic-rich mudrocks.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article4_asphaltene_fault_block_migration.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_deepwater_gom_overview.py` | Deepwater Exploration and Production in the Gulf of Mexico - Challenges and Opportunities | Elshahawi | 81-87 |
| `article2_turbidite_sediments.py` | Consideration of the Origin and Characteristics of Turbidite Sediments | Dribus | 88-95 |
| `article3_reservoir_fluid_dynamics.py` | The Dynamics of Reservoir Fluids and their Substantial Systematic Variations | Mullins, Zuo, Wang, Hammond, De Santo, Dumont, Mishra, Chen, Pomerantz, Dong, Elshahawi, Seifert | 96-112 |
| `article4_asphaltene_fault_block_migration.py` | Fault Block Migrations Inferred from Asphaltene Gradients | Dong, Hows, Cornelisse, Elshahawi | 113-123 |
| `article5_deepwater_formation_evaluation.py` | Formation-Evaluation Challenges and Opportunities in Deepwater | Chemali, Samec, Balliet, Cooper, Torres, Jones | 124-135 |
| `article6_kerogen_resistivity_mudrocks.py` | Quantifying the Effect of Kerogen on Resistivity Measurements in Organic-Rich Mudrocks | Kethireddy, Chen, Heidari | 136-146 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 55 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2014_04.pdf`,
> ~10 MB) has a text layer, but **every numbered display equation was stripped in
> extraction** (only the equation numbers survived); the bodies are reconstructed
> from the surviving running text and nomenclature in standard form. This is a
> deepwater special issue, so three articles are narrative reviews: Articles 1
> (Elshahawi) and 2 (Dribus) carry no equations and are implemented from their
> genuine quantitative content (deepwater pressure/overburden framing; the Bouma
> sequence and the Ross-Formation net-to-gross / pore-contact contrast), and
> Article 5 (Chemali et al.) is an applied review whose referenced models
> (Eaton, Thomas-Stieber, Coates) are written in standard form. Articles 3, 4
> and 6 are quantitative (FHZ asphaltene EoS, gravity currents, the kerogen
> conductivity model). The cover features Article 1's Gulf of Mexico field map.
> (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Elshahawi)**: deepwater GoM overview - a narrative review;
  implemented as the standard relations that frame its "challenges": water-depth
  classification, the seawater hydrostatic head, the overburden pressure /
  gradient reduced by the water column (the narrow-margin problem), the
  porosity-permeability decoupling (Wilcox permeability spans ~3 orders of
  magnitude), a reservoir-quality lookup by GoM play / geologic epoch
  (Pleistocene, Pliocene, Miocene, Wilcox depth/porosity/permeability ranges),
  and the dual-gradient mud-line pressure relief that widens the narrow margin.

- **Article 2 (Dribus)**: turbidite sediments - a geology review; the Bouma
  (1962) Ta-Te fining-upward facies sequence, the net-to-gross / sand-to-sand
  pore-contact contrast between layered (45%/3%) and amalgamated (90%/67%)
  architectures (Ross Formation) with a connectivity/recovery proxy, the
  submarine-fan facies belts (inner / middle channel-levee / outer fan) with a
  reservoir-quality rank (the distal outer fan is the best target), and the
  mass-transport-deposit progression (slide -> slump -> debris flow -> turbidity
  current) with its rock-property preservation ranking.

- **Article 3 (Mullins et al.)**: reservoir-fluid dynamics - the 1D diffusion
  length/time `x^2 = 2*D*t` (Eq. 1), the buoyancy-driven gravity-current velocity
  (Eq. 2), the asphaltene gravitational (Boltzmann) distribution (Eq. 3), the
  asphaltene-gradient half-height, the Yen-Mullins particle sizes, the
  diffusive-front vs gas-oil-contact displacement comparison (the ~20 m
  "Goldilocks" length), and a reservoir-equilibrium classifier
  (young / moderately aged / aged) from a vertical property profile.

- **Article 4 (Dong et al.)**: asphaltene fault-block migration - the
  Flory-Huggins-Zuo asphaltene EoS with gravity, entropy and solubility terms
  (Eq. 1), its gravity-only black-oil limit, the molar-volume / particle-size
  inference from two depth points, the fault-block vertical-offset diagnostic,
  the FHZ connectivity test (do two stations lie on one equilibrium curve?), and
  the tar-mat onset flag from the 35-40 wt% asphaltene cutoff.

- **Article 5 (Chemali et al.)**: deepwater formation evaluation - an applied
  review; the ECD drilling-window check, the Eaton pore-pressure prediction
  (resistivity / acoustic forms), the resistivity anisotropy ratio
  `lambda = sqrt(Rv/Rh)`, the Thomas-Stieber laminated-sand horizontal/vertical
  resistivities (plus the shale-intrinsic-anisotropy-corrected form, Clavaud)
  and Archie sand saturation, the NMR hydrogen index, the Coates/Timur NMR
  permeability, and the T1 fluid-typing contrast.

- **Article 6 (Kethireddy et al.)**: kerogen effect on resistivity - the water/
  gas saturations and kerogen concentration/porosity (Eqs. 1-6), the linear
  plagioclase-quartz mineral constraint (Eq. 8) and TOC->kerogen conversion
  (Eq. 9), the exponential effective kerogen resistivity (Eq. 10), a
  finite-difference Laplace conductivity solver `div(sigma*grad V)=0` (Eq. 7) for
  the effective resistivity of a kerogen-bearing rock, the ~1000 Ohm*m
  kerogen-conductivity threshold above which the rock resistivity is unaffected,
  and the Archie water-saturation overestimation when conductive kerogen is
  ignored.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2014)
Reference: Petrophysics Vol. 55, No. 2, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.

## Changelog - comprehensiveness pass

The modules were reviewed against the full article text of
`Petrophysics_2014_04.pdf` and extended so each captures more of its article's
quantitative content. The original equations were already faithful; the
additions below cover methods and data anchors that were described in the
articles but not yet implemented. Every addition is exercised by the module's
`test_all()` and all six modules pass.

- **Article 1**: added `reservoir_quality(play)` (Pleistocene / Pliocene /
  Miocene / Wilcox depth, thickness, porosity and permeability ranges) and the
  dual-gradient vs single-gradient mud-line pressure comparison
  (`dual_gradient_mudline_pressure`, `single_gradient_mudline_pressure`).
- **Article 2**: added the submarine-fan facies model (`fan_facies`,
  `best_reservoir_facies`) with a reservoir-quality rank and the proximal-levee
  >100 md anchor, plus the mass-transport-deposit progression
  (`mtd_progression`).
- **Article 3**: added `diffusion_vs_displacement` (diffusive front vs GOC
  displacement over the ~20 m characteristic length) and
  `reservoir_equilibrium_state` (young / moderately aged / aged).
- **Article 4**: added `on_same_fhz_curve` (the FHZ connectivity / fault-seal
  test) and `tar_mat_likely` (35-40 wt% asphaltene cutoff).
- **Article 5**: added `eaton_pore_pressure`,
  `laminated_resistivity_anisotropic` (shale intrinsic-anisotropy correction)
  and `coates_permeability` (NMR free-fluid permeability).
- **Article 6**: added `plagioclase_from_quartz` (Eq. 8, completing the Eqs.
  1-10 set) and `kerogen_affects_resistivity` (the ~1000 Ohm*m kerogen
  conductivity threshold).
