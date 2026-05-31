# Petrophysics June 2017 - Vol. 58, No. 3

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 58, No. 3 (June 2017): six articles spanning Tuscaloosa Marine Shale NMR,
total gas-in-place from magnetic resonance logs, forward mineral modeling by
SVD/ridge regression, recovering elastic properties from rock fragments,
complex-resistivity dispersion logging, and an integrated carbonate pore-system
case study.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article2_tgip_nmr_gas_shale.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_tms_nmr_characterization.py` | Application of Laboratory and Field NMR to Characterize the Tuscaloosa Marine Shale | Besov, Tinni, Sondergeld, Rai, Paul, Ebnother, Smagala | 221-231 |
| `article2_tgip_nmr_gas_shale.py` | A Novel Determination of Total Gas-In-Place (TGIP) for Gas Shale From Magnetic Resonance Logs | Kausik, Kleinberg, Rylander, Lewis, Sibbit, Westacott | 232-241 |
| `article3_forward_mineral_svd.py` | Forward Mineral Modeling Using Regularized Least-Squares Regression With Singular Value Decomposition: Case Study From Qusaiba Shale | Xu, McCormick, Herron, Cheshire, Al-Salim, Almarzouq | 242-269 |
| `article4_elastic_from_fragments.py` | Recovering Elastic Properties From Rock Fragments | Dang, Gupta, Chakravarty, Bhoumick, Taneja, Sondergeld, Rai | 270-280 |
| `article5_complex_resistivity_dispersion.py` | Borehole Measurements of the Complex-Resistivity Dispersion Spectrum: A New Logging Method to Identify Low-Resistivity Reservoirs | Jiang, Ke, Kang, Sun, Yin | 281-288 |
| `article6_carbonate_pore_system.py` | A Case Study on Integrated Petrophysical Characterization of a Carbonate Reservoir Pore System in the Offshore Red River Basin of Vietnam | Giao, Chung | 289-301 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 58 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2017_06.pdf`,
> ~21 MB) has a text layer, so titles, authors, and page ranges were read from
> the contents page and bodies. The machine extraction captured the full bodies
> of **articles 1-5** but **truncated article 6 at its "Methodology of Study"
> heading** (abstract/intro only), so **article 6 is implemented as a
> methodology proxy**. As with the other issues, the typeset formula glyphs were
> dropped in extraction, so the numbered formulas are faithful standard-form
> reconstructions. (This issue has no tutorial; the p302 "Historical Note" is not
> an article.)

## Implementation notes & substitutions

- **Article 1 (Besov et al.)**: Tuscaloosa Marine Shale NMR - the matrix vs
  microfracture porosity from a 10-ms T2 cutoff, the irreducible-water fraction,
  the Washburn pore-throat radius, and a volumetric recoverable-oil estimate
  (7758*A*h*phi*So/Bo).

- **Article 2 (Kausik et al.)**: TGIP from magnetic resonance - the hydrogen
  index, the mean protons per molecule and mixture molecular weight, the gas
  specific gravity, the moles-to-scf conversion (Vscf = 0.8305e6*nu), and the
  TGIP per unit formation volume.

- **Article 3 (Xu et al.)**: forward mineral modeling - the organic-free
  elemental correction, the forward model M = E*x by least squares, the
  truncated-SVD pseudoinverse, the ridge (L2) solution, and the condition number.

- **Article 4 (Dang et al.)**: elastic properties from fragments - the bulk
  modulus from MICP compressibility at 5,000 psi, Young's modulus from the
  nanoindentation reduced modulus, the hardness, and the dynamic bulk modulus
  K = rho*(Vp^2 - 4/3*Vs^2).

- **Article 5 (Jiang et al.)**: complex-resistivity dispersion - the Cole-Cole
  model, the characteristic frequency (Fb = 1/tau), the power-law water-filled
  porosity from Fb, and the water saturation.

- **Article 6 (Giao & Chung)** *(methodology proxy)*: carbonate pore system -
  density porosity, PEF rock typing (limestone vs dolostone), the
  vuggy/interparticle porosity partition, the Lucia rock-fabric permeability, and
  a dual-porosity fracture porosity.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2017)
Reference: Petrophysics Vol. 58, No. 3, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
