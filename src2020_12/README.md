# Petrophysics December 2020 - Vol. 61, No. 6

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 61, No. 6 (December 2020) - the **"Pulsed-Neutron Logging in the 2020s:
Smarter, Faster, and Much More Powerful"** special issue on nuclear
spectroscopy.  The issue opens with a historical review of nuclear spectroscopy
in well logging, then covers formation chlorine / water-salinity measurement,
self-compensated pulsed-neutron spectroscopy, two multidetector case studies
(C/O and sigma saturation), through-casing TOC and oil saturation from excess
carbon, and gas-pressure assessment through casing.

## Quick start

```bash
pip install numpy

# Run all 7 module tests
python test_all.py

# Or run a single article
python article2_formation_chlorine_salinity.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_nuclear_spectroscopy_history.py` | A History of Nuclear Spectroscopy in Well Logging | Pemper | 10.30632/PJV61N6-2020a1 |
| `article2_formation_chlorine_salinity.py` | Formation Chlorine Measurement From Spectroscopy Enables Water Salinity Interpretation: Theory, Modeling, and Applications | Miles, Mossé, Grau | 10.30632/PJV61N6-2020a2 |
| `article3_self_compensated_spectroscopy.py` | Self-Compensated Pulsed-Neutron Spectroscopy Measurements | Zhou, Rose, Miles, Gendur, Wang, Sullivan | 10.30632/PJV61N6-2020a3 |
| `article4_co_sigma_saturation_casestudy.py` | New Generation of Pulsed-Neutron Multidetector Comparison in a Challenging Multistack Clastic Reservoir: A Case Study in a Brown Field, Malaysia | Johare, Mohd Amin, Prasodjo, Afandi, Din | 10.30632/PJV61N6-2020a4 |
| `article5_through_casing_toc_saturation.py` | Integration of Nuclear Spectroscopy Technology and Core Data Results for Through-Casing TOC Measurement and Saturation Analysis: A Case Study in Najmah-Sargelu Reservoir, South Kuwait | Bouchou, Abughneej, Ghioca, Alarcon, Mendez | 10.30632/PJV61N6-2020a5 |
| `article6_pulsed_neutron_gas_pressure.py` | Maximizing the Value of Pulsed-Neutron Logs: A Complex Case Study of Gas Pressure Assessment Through Casing | Cavalleri, Brouwer, Kodri, Rose, Brinks | 10.30632/PJV61N6-2020a6 |
| `article7_sigma_gas_saturation_lowporosity.py` | Multidetector Pulsed-Neutron Tool Application in a Low-Porosity Reservoir: A Case Study in Mutiara Field, Indonesia | Wijaya, Aulianagara, Guo, Naibaho, Asriwan, Amirudin | 10.30632/PJV61N6-2020a7 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** Unlike the scanned issues, this issue's source PDF
> (`Petrophysics_2020_12.pdf`) has a text layer, so the article titles, authors,
> page ranges, DOIs, equation numbers, variable definitions, and many numeric
> constants were read directly from the paper bodies. The PDF-to-text
> conversion did, however, drop most typeset formula *glyphs* (it kept the
> equation numbers and the surrounding prose), so the numbered formulas in the
> modules are **faithful standard-form reconstructions** built from the
> preserved variable definitions and constants - except Article 5's Eq. 3
> (`XCarbon = CTot − (CMin + CMat)`), whose text survived verbatim and is
> implemented exactly. The paper-quoted constants below are reproduced in the
> tests.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions. A few practical notes:

- **Article 1 (Pemper)** *(historical review, no equations)*: the canonical
  nuclear-logging relations the review surveys - the macroscopic capture cross
  section Σ = ΣᵢNᵢσᵢ in capture units, the thermal-neutron die-away conversion
  Σ = 4550/τ, number density from bulk density, the carbon/oxygen ratio, and a
  K/U/Th spectral-gamma sum. Reproduces the ~22 c.u. fresh-water sigma from H/O
  number densities.

- **Article 2 (Miles et al.)**: the yields-to-weights relation W = FY2W·S·Y
  (Eq. 6), the chlorine yield split Y_Cl = Y_form + Y_borehole (Eq. 1) with the
  CYDCL standard and Φ(env) = 1/f borehole subtraction (Eqs. 4-9), the DWCL →
  NaCl-salinity / BVW / Sw conversions (Eqs. 11-14) using the molar-mass ratio
  1.649, and the macroscopic sigma mixing / Σmax model (Eqs. 19-20) with the
  paper's 567 c.u. per (g/cc) Cl, 22 c.u. fluid, 29.4 c.u. shale.

- **Article 3 (Zhou et al.)**: the yields-to-weights relation (Eq. 1), a FY2W
  predictor from raw measurements that rises with hole size and gives a smaller
  far-detector inelastic FY2W (deeper depth of investigation), and the
  differential near-over-far dry-weight element that cancels a common borehole
  contribution (demonstrated by recovering formation Ca independent of the
  cement Ca).

- **Article 4 (Johare et al.)** *(case study, no equations)*: the standard
  pulsed-neutron saturation relations - salinity-independent C/O-ratio oil
  saturation by water/oil-endpoint interpolation, sigma water saturation from
  the volumetric porosity balance, and a near/far multidetector ratio gas
  indicator.

- **Article 5 (Bouchou et al.)**: the linear multimineral log-response model
  (Eq. 1) solved by a closure-constrained weighted-least-squares inversion
  (Eq. 2, recovers planted mineral volumes), the excess-carbon relation
  XCarbon = CTot − (CMin + CMat) (Eq. 3, verbatim), and the calibration-free
  oil saturation So = ρb·Xc/(ρo·Fc·φe) (Eq. 4).

- **Article 6 (Cavalleri et al.)**: the bulk gas sigma Σ = ρ_bulk·Σₑ(wₑσₑ)
  (Eq. 1, proportional to gas density), inverted through a real-gas density law
  ρ = PM/(zRT) to assess formation gas pressure - reproduces the case study's
  ~2,785 psi from the measured sigma. The per-mass elemental coefficients are
  set so water returns ~22 c.u. at 1 g/cc.

- **Article 7 (Wijaya et al.)**: the clean and shaly sigma gas-saturation
  equations (Eqs. 1-2) from the porosity balance, using the paper's quoted
  endpoints (Σ_matrix 7.5, Σ_shale 27, Σ_water 24, Σ_gas 3 c.u.; φ ≈ 12 p.u.),
  plus the low-porosity sensitivity caveat |dΣ/dSg| = φ(Σ_w − Σ_g) that makes
  the answer fragile in tight rock.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2020)
DOI: <doi>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
