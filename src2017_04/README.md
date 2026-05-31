# Petrophysics April 2017 - Vol. 58, No. 2

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 58, No. 2 (April 2017): five articles spanning NMF of NMR T1-T2 logs,
hydrocarbon storage in organic-rich mudstones, NMR relaxation and pore size in
carbonates, rock-fluid affinity from the T1/T2 ratio, and tar-mat formation by
asphaltene phase transition.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article3_nmr_pore_size_shape_factor.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_nmf_t1t2_fluid_signatures.py` | Unlocking the Potential of Unconventional Reservoirs Through New Generation NMR T1/T2 Logging Measurements Integrated with Advanced Wireline Logs | Anand, Ali, Abubakar, Grover, Neto, Pirie, Gonzalez Iglesias | 81-96 |
| `article2_mudstone_hc_storage_kinetics.py` | Characterizing Hydrocarbon Storage in Organic-Rich Mudstones by Integrating Core Measurements, Kinetic Modeling, and Pore-Scale Observations: Application to South Texas Organic-Rich Mudstones | Capsan, Sanchez-Ramirez | 97-115 |
| `article3_nmr_pore_size_shape_factor.py` | A Laboratory Study of the Link Between NMR Relaxation Data and Pore Size In Carbonate Skeletal Grains and Micrite | El-Husseiny, Knight | 116-125 |
| `article4_t1t2_affinity_gassmann.py` | Low-Field NMR Spectrometry of Chalk and Argillaceous Sandstones: Rock-Fluid Affinity Assessed from T1/T2 Ratio | Katika, Saidian, Prasad, Fabricius | 126-140 |
| `article5_tarmat_asphaltene_phase.py` | Scanning Electron Micrographs of Tar-Mat Intervals Formed by Asphaltene Phase Transition | Pfeiffer, Di Primio, Achourov, Mullins | 141-152 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 58 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2017_04.pdf`,
> ~16 MB) has a text layer, so titles, authors, and page ranges were read from
> the contents page and bodies; **all five articles have full bodies**. As with
> the other issues, the typeset formula glyphs were dropped in extraction, so the
> numbered formulas are faithful standard-form reconstructions. Article 5 is a
> conceptual/SEM paper with no typeset equations, so its module implements the
> standard FHZ gravity gradient and DFA optical relation it relies on, plus the
> asphaltene-weight thresholds it reports. (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Anand et al.)**: blind-source-separation of NMR T1-T2 maps - a
  non-negative matrix factorization (V ~ W*H by multiplicative updates), the
  reconstruction error, rank selection from the volume-matrix eigenvalues, and a
  carbon-to-hydrocarbon volume conversion.

- **Article 2 (Capsan & Sanchez-Ramirez)**: mudstone hydrocarbon storage - the
  hydrocarbon pore volume, the Dean-Stark formation volume factor, HPV from oil
  and from pyrolysis S1, an Arrhenius first-order kerogen conversion, and the max
  organic nanoporosity from the converted kerogen mass.

- **Article 3 (El-Husseiny & Knight)**: NMR relaxation and pore size - the
  surface relaxation rate (1/T2 = rho*S/V), the shape-factor S/V = alpha/r, the
  pore radius r = alpha*rho*T2, the shape-factor calibration, and a
  multiexponential decay.

- **Article 4 (Katika et al.)**: rock-fluid affinity from T1/T2 - the T1/T2
  ratio and wettability classification, the elastic moduli from velocities, the
  Voigt/Reuss/Hill fluid-modulus averages selected by wettability, and the
  Gassmann saturated bulk modulus.

- **Article 5 (Pfeiffer et al.)** *(conceptual paper)*: tar-mat asphaltene phase
  transition - the FHZ gravity asphaltene gradient, the linear optical-density /
  asphaltene relation, the asphaltene-weight classification (suspension / onset /
  solid tar mat), and the qualitative solvency-vs-GOR trend.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2017)
Reference: Petrophysics Vol. 58, No. 2, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
