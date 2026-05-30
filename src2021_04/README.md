# Petrophysics April 2021 - Vol. 62, No. 2

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 62, No. 2 (April 2021) - a regular issue of five papers spanning NMR
pore-structure characterization of a complex carbonate, a deepwater-turbidite
rock-typing case study, Thomeer/NMR free-vs-bound porosity partitioning,
nonlinear-acoustics noncollinear wave mixing for near-wellbore evaluation, and
an integrated NMR continuous/stationary fluid-and-contacts workflow.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article4_nonlinear_acoustics_mixing.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_nmr_carbonate_porestructure.py` | Pore-Structure Characterization of a Complex Carbonate Reservoir in South Iraq Using Advanced Interpretation of NMR Logs | Saidian, Jain, Milad | 10.30632/PJV62N2-2021a1 |
| `article2_turbidite_rock_typing.py` | Challenges in the Petrophysical and Dynamic Characterization of Deepwater Turbidite Deposits of the Colombian Caribbean Offshore - A Case Study | Angel Restrepo, Gómez-Moncada, Mora Sánchez, Bueno Silva | 10.30632/PJV62N2-2021a2 |
| `article3_thomeer_nmr_partitioning.py` | Free or Bound? Thomeer and NMR Porosity Partitioning in Carbonate Reservoirs, Alta Discovery, Southwestern Barents Sea | Gianotten, Rameil, Foyn, Kollien, Marre, Looyestijn, Zhang, Hebing | 10.30632/PJV62N2-2021a3 |
| `article4_nonlinear_acoustics_mixing.py` | Nonlinear Acoustics Applications for Near-Wellbore Formation Evaluation | Skelt, TenCate, Guyer, Johnson, Larmat, Le Bas, Nihei, Vu | 10.30632/PJV62N2-2021a4 |
| `article5_nmr_fluid_contacts.py` | An Integrated Petrophysical Workflow for Fluid Characterization and Contacts Identification Using NMR Continuous and Stationary Measurements in a High-Porosity Sandstone Formation, Offshore Norway | Kozlowski, Chakraborty, Jambunathan, Lowrey, Balliet, Engelman, Ånensen, Kotwicki, Johansen | 10.30632/PJV62N2-2021a5 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF has no usable text layer, so
> the modules were built by **rendering the PDF pages to images and reading
> them visually**.  The equations here are therefore transcribed from the
> genuinely rendered mathematics (Article 1's NMR relaxation equations, Article
> 3's Thomeer/Swanson equations, and Article 4's nonlinear-acoustics wave-mixing
> equations are verbatim).  Articles 2 and 5 are a case study and a workflow
> paper with few or no numbered equations; their modules implement the
> quantitative relations the papers rely on (and standard NMR forms where the
> paper cites them by reference), clearly flagged.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (Saidian et al.)**: the multi-exponential relaxation sum (Eq. 1),
  the diffusion relaxation rate 1/T2diff = D*gamma^2*g^2*TE^2/12 (Eq. 2), the
  NMR porosity correction phi_corr = phi + 0.3*Vol_largepore (Eq. 3), a single-
  pore forward model (sphere S/V = 3/r) showing how large echo spacing saturates
  pore-size sensitivity, the large-pore T2-cutoff (847 ms) partition, and
  Timur-Coates / SDR permeability.  Reproduces the (900/200)^2 = 20.25 diffusion
  ratio between the two tools.

- **Article 2 (Angel Restrepo et al.)** *(case study)*: the two field-specific
  Winland-R35 regressions - from core CT (Eq. 1) and from logs (Eq. 2) - the
  R35 pore-throat rock-type classifier (RT-1..RT-4), the per-rock-type
  irreducible-saturation lookup, and the Waxman-Smits Co = (1/F*)(Cw + B*Qv)
  conductivity line.  Reproduces the worked Co = 0.433 mho/m.

- **Article 3 (Gianotten et al.)**: the Thomeer capillary-pressure hyperbola
  (Eq. 1), normalized porosity (Eq. 2), RQI/FZI (Eq. 3), the Swanson-type
  permeability Ka = 3.8068*G^-1.3334*(Bv/Pd)^2 (Eq. 4) and its inversion for the
  shape factor G (Eq. 5, round-trips exactly), plus the Washburn pore-throat
  radius and the NMR<->MICP calibration C = T2*Pc tying the 0.3-micron / 14-ms
  free-vs-bound cutoffs together.

- **Article 4 (Skelt et al.)**: the cubic nonlinear stress-strain law (Eq. 1),
  the nonlinearity parameter beta (Eq. 2), the noncollinear-mixing convergence
  angle (Eq. 3) and scattering angle (Eq. 4), the exact (Eqs. 5-8) and
  approximate (Eq. 9) scattering coefficients, and the frequency-ratio validity
  rule (Eq. 10).  Reproduces Table 1's phi = gamma = 47.5 deg at omega2/omega1 =
  0.74 and the large negative rock beta.

- **Article 5 (Kozlowski et al.)** *(workflow paper)*: canonical NMR relations
  the workflow relies on - the full T2 relaxation (bulk + surface + diffusion),
  T1 relaxation, the hydrogen-index porosity correction (~11% uplift), the
  clay-bound / capillary / free T2-cutoff partition (3 ms, 60 ms), D-T2 fluid
  typing (gas / water / oil), and the sqrt(stacks) station-stacking SNR gain.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2021)
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
