# Petrophysics December 2015 - Vol. 56, No. 6

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 56, No. 6 (December 2015) - a carbonate-characterization special issue of
four case-study articles spanning the multiscale heterogeneity and NMR
petrophysics of the presalt Sag carbonates (Campos Basin), presalt carbonate
evaluation for Santos Basin, the bitumen-saturated karsted Grosmont Formation,
and rock typing of the giant Tengiz carbonate field.

## Quick start

```bash
pip install numpy

# Run all 4 module tests
python test_all.py

# Or run a single article
python article4_tengiz_rock_typing.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_presalt_sag_nmr_petrophysics.py` | Reservoir Characterization Challenges Due to the Multiscale Spatial Heterogeneity in the Presalt Carbonate Sag Formation, North Campos Basin, Brazil | Chitale, Alabi, Gramin, Lepley, Piccoli | 552-576 |
| `article2_santos_presalt_evaluation.py` | Presalt Carbonate Evaluation for Santos Basin, Offshore Brazil | Boyd, Souza, Carneiro, Machado, Trevizan, Santos, Neto, Bagueira, Polinski, Bertolini | 577-591 |
| `article3_grosmont_bitumen_carbonates.py` | Petrophysical Characterization of Bitumen-Saturated Karsted Carbonates: Case Study of the Multibillion Barrel Upper Devonian Grosmont Formation, Northern Alberta, Canada | MacNeil | 592-614 |
| `article4_tengiz_rock_typing.py` | Petrophysical Challenges in Giant Carbonate Tengiz Field, Republic of Kazakhstan | Skalinski, Se, Playton, Theologou, Narr, Sullivan, Mallan | 615-647 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 56 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2015_12.pdf`,
> ~19 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly; **all four articles have full bodies**. These are
> carbonate characterization *case studies* with relatively few display
> equations; each module implements the standard petrophysics the paper relies
> on and applies. The typeset display-equation glyphs were dropped in
> extraction, so the relations are faithful standard-form reconstructions.
> (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Chitale et al.)**: presalt Sag NMR petrophysics - the NMR T2
  distribution partition into bound (BVI) and free (FFI) fluid at a carbonate T2
  cutoff (~90-200 ms), the total-porosity model, the T2 logarithmic mean, and
  the Coates (Timur-Coates) and SDR NMR permeability transforms.

- **Article 2 (Boyd et al.)**: Santos presalt evaluation - the SDR NMR
  permeability, Archie water saturation with variable m and n, the saturation
  exponent recovered from the dielectric textural exponent (water tortuosity
  factor m*n -> n = m*n/m with m from NMR partitioning), the microporosity
  water-saturation estimate, and the macro/vug porosity indicator from the
  sonic-vs-total porosity deficit.

- **Article 3 (MacNeil)**: Grosmont bitumen-saturated carbonates - the density
  porosity against a dolomite grain density (~2.85 g/cm^3) with a bitumen pore
  fluid, Archie water saturation with variable m and n, the (immobile) bitumen
  saturation Sb = 1 - Sw, the Dean-Stark core porosity and saturations from the
  water and bitumen volumes, and the Rmf/Rw ratio for laterolog suitability.

- **Article 4 (Skalinski et al.)**: Tengiz carbonate rock typing - the
  reservoir quality index (RQI), normalized porosity index and flow zone
  indicator (FZI) hydraulic-unit framework, permeability from FZI, the rock-type
  permeability-porosity transform, and the pore-type saturation-height function
  with the rock-type bulk-volume-water (BVW) approach.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2015)
Reference: Petrophysics Vol. 56, No. 6, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
