# Petrophysics June 2021 - Vol. 62, No. 3

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 62, No. 3 (June 2021) - a regular issue opening with an invited tutorial
on sidewall coring, followed by five papers spanning NMR restricted-diffusion
pore characterization, AI prediction of acoustic velocities while drilling,
machine-learning sonic-shear processing, the first LWD co-located-antenna
anisotropy/dip tool, and proactive geosteering with 2D structural analysis.

## Quick start

```bash
pip install numpy

# Run all 6 module tests
python test_all.py

# Or run a single article
python article2_nmr_restricted_diffusion.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_sidewall_coring_tutorial.py` | Tutorial: A Century of Sidewall Coring Evolution and Challenges, From Shallow Land to Deep Water | Jackson | 10.30632/PJV62N3-2021t1 |
| `article2_nmr_restricted_diffusion.py` | Pore Size, Tortuosity, and Permeability From NMR Restricted Diffusion in Organic-Rich Chalks | Wang, Singer, Liu, Chen, Hirasaki, Vinegar | 10.30632/PJV62N3-2021a1 |
| `article3_ai_acoustic_velocity.py` | Real-Time Prediction of Acoustic Velocities While Drilling Vertical Complex Lithology Using AI Technique | Alsaihati, Elkatatny | 10.30632/PJV62N3-2021a2 |
| `article4_ml_sonic_shear.py` | Machine-Learning-Enabled Automatic Sonic Shear Processing | Liang, Lei | 10.30632/PJV62N3-2021a3 |
| `article5_lwd_colocated_antenna.py` | First LWD Co-Located Antenna Sensors for Real-Time Anisotropy and Dip Angle Determination, Yielding Better Look-Ahead Detection | Bittar, Wu, Ma, Pan, Fan, Griffing, Lozinsky | 10.30632/PJV62N3-2021a4 |
| `article6_geosteering_2d_structural.py` | Maximizing Net Pay in Penta-Lateral Well With Advanced Proactive Geosteering and 2D Structural Analysis Using Azimuthal Resistivity Measurements | Antonov, Kushnir, Martakov, Pazos, Small, Tropin, Maraj, Itter, Nelson, Rabinovich | 10.30632/PJV62N3-2021a5 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF uses broken embedded-font
> encodings, so machine text extraction produced multi-font cipher garbage.
> The modules were therefore built by **rendering the PDF pages to images and
> reading them visually** - which means the equations here are transcribed
> from the genuinely rendered mathematics (Article 2's NMR Pade equations and
> Article 4's VTI/ANNIE stiffness relations are verbatim), not reconstructed.
> Articles 1, 5, and 6 are descriptive (tutorial / instrument-introduction /
> case study) with few or no numbered equations; their modules implement the
> quantitative relations the papers rely on, clearly flagged where standard
> forms are supplied.

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (Jackson, tutorial)** *(descriptive)*: the Fig. 14 rotary-coring
  tool table as structured data, cylindrical core-plug volume V = pi(d/2)^2 L,
  per-run recovered volume, and tool selection by pressure/temperature rating.
  Reproduces the worked plug volumes (MaxCOR 4.42 in^3, XL-Rock 6.19 in^3).

- **Article 2 (Wang et al.)**: the NMR restricted-diffusion workflow - diffusion
  length L_D = sqrt(D0 t) (Eq. 9), cylindrical S/V = 4/d (Eq. 10), the Pade
  interpolation bridging the short-time Mitra and long-time tortuosity limits
  (Eq. 11), the modified Carman-Kozeny permeability (Eq. 3), Timur-Coates and
  SDR permeability (Eqs. 1-2), and electrical/diffusive tortuosity (Eqs. 4-7).
  A grid-search round-trips planted pore size and tortuosity.

- **Article 3 (Alsaihati & Elkatatny)**: Spearman rank correlation (Eq. 1),
  AAPE (Eq. 2) and the correlation coefficient R (Eq. 3), min-max normalization,
  the nine Appendix-1 empirical Vs-from-Vp correlations (Pickett, Carroll,
  Castagna, Brocher), and a numpy linear-regression surrogate for the ANN/RF
  predictor (the MATLAB-trained networks cannot be reproduced exactly).

- **Article 4 (Liang & Lei)**: the ANNIE VTI stiffness relations (Eqs. 1-7)
  including the Poisson-ratio (Eq. 6) and Thomsen-gamma (Eq. 7) definitions, the
  RMAD validation metric (Eq. 9) and inversion misfit (Eq. 10), plus a surrogate
  dipole-flexural dispersion and DTS inversion standing in for the paper's
  neural-network forward proxy / mode-search solver.

- **Article 5 (Bittar et al.)** *(instrument introduction)*: the tilted-antenna
  magnetic-moment projection onto tool axes, the 3x3 magnetic-tensor coupling
  V = m_R . H . m_T, and the standard propagation-resistivity attenuation /
  phase-shift relations (skin depth) with apparent-resistivity inversion - the
  EM forward response itself is from the cited LWD-resistivity literature.

- **Article 6 (Antonov et al.)** *(case study)*: the borehole-geometry relations
  the workflow relies on - MD->TVD, boundary TVD from a distance-to-boundary
  pick, apparent<->true dip, structural dip from two picks, least-squares
  fault-plane fitting to dip/azimuth (recovers the Table 1 OBc 44 deg / 23 deg
  fault), and net-pay (reservoir-contact) accounting along a lateral.

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
