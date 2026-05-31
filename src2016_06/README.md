# Petrophysics June 2016 - Vol. 57, No. 3

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 57, No. 3 (June 2016): five articles spanning heterogeneous-carbonate
saturation-height modeling with dynamic data, combining hydraulic and electrical
conductivity for pore-space characterization, permeability interpretation from
wireline formation testing with effective thickness, multiscale leaky-P removal
for shear-wave anisotropy inversion, and a wireline-depth elastic-stretch
correction.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article2_hydraulic_electrical_pore_space.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_carbonate_saturation_height.py` | Heterogeneous Carbonate Reservoirs: Ensuring Consistency of Subsurface Models by Maximizing the use of Saturation-Height Models and Dynamic Data | Hulea, Frese, Ramaswami | 223-232 |
| `article2_hydraulic_electrical_pore_space.py` | Combining Hydraulic and Electrical Conductivity for Pore-Space Characterization in Carbonate Rocks | Müller-Huber, Schön, Börner | 233-250 |
| `article3_wft_permeability_effective_thickness.py` | Permeability Interpretation from Wireline Formation Testing Measurements with Consideration of Effective Thickness | Yang, Yang | 251-269 |
| `article4_shearwave_anisotropy_leakyP.py` | An Improved Multiscale and Leaky P-Wave Removal Analysis for Shear-Wave Anisotropy Inversion with Crossed-Dipole Logs | Li, Tao, Wang, Zhang, Vega | 270-293 |
| `article5_wireline_depth_elastic_stretch.py` | Wireline Logging Depth Quality Improvement: Methodology Review and Elastic-Stretch Correction | Bolt | 294-310 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 57 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2016_06.pdf`,
> ~13 MB) has a text layer, so titles, authors, page ranges and the article
> bodies were read directly. Articles 1-4 have full bodies and their numbered
> relations survived as inline text (the Brooks-Corey SHM, the capillary-channel
> hydraulic/electrical model, the Brooks-Corey WFT relations, and the Alford
> rotation / leaky-P / inversion objective); the typeset display-equation glyphs
> were dropped and are faithful standard-form reconstructions. **Article 5
> (Bolt) was beyond the text extraction** (the source truncates within Article
> 4), so it is implemented as a methodology proxy from the standard Hooke's-law
> cable-stretch correction its title describes, consistent with how other
> truncated articles are handled in this repository. (This issue has no
> tutorial.)

## Implementation notes & substitutions

- **Article 1 (Hulea et al.)**: carbonate saturation height - the Brooks-Corey
  saturation-height model (Eq. 1, with entry pressure, irreducible water and
  shape factor), a buoyancy capillary-pressure (height-above-free-water-level)
  helper, permeability averaging per rock type (arithmetic / geometric /
  harmonic), a Lucia-type permeability transform, and the WFT
  mobility-to-permeability relation (Mdd = k/viscosity).

- **Article 2 (Müller-Huber et al.)**: hydraulic + electrical pore space - the
  Hagen-Poiseuille flow (Eq. 1), the capillary-channel permeability k =
  phi*r^2/(8*tau^2) (Eq. 4), Archie's formation factor F = tau^2/phi (Eq. 6),
  the tortuosity-free combination k = r^2/(8*F) (Eq. 7), and the variable pore
  radius / pore-shape factor of the rb/rt model (Eqs. 8-9).

- **Article 3 (Yang & Yang)**: WFT permeability & effective thickness - the
  Brooks-Corey relative-permeability and capillary-pressure relations
  (Eqs. 1-3), the single-probe spherical drawdown permeability (Moran-Finklea
  form), the spherical permeability ks = (kh^2*kv)^(1/3), and a
  pressure-derivative flow-regime slope (spherical ~ -1/2, radial flattens).

- **Article 4 (Li et al.)**: shear-wave anisotropy + leaky-P - the analytic
  (complex) wavelet Hilbert pair (Eq. 1), Alford rotation to fast/slow principal
  waveforms (Eqs. 3-4), the leaky-P contamination model (Eqs. 5-6), a
  fast-shear-azimuth estimator by cross-component energy minimization, and the
  waveform-similarity inversion objective (Eq. 7).

- **Article 5 (Bolt)** *(methodology proxy - body beyond extraction)*: wireline
  depth quality - the cable tension vs. depth, the stretch coefficient ks =
  1/(E*A), the total elastic stretch dL = ks*(W_tool*L + 0.5*w_cable*L^2) from
  integrating Hooke's law along the cable, and the stretch-corrected depth.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2016)
Reference: Petrophysics Vol. 57, No. 3, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
