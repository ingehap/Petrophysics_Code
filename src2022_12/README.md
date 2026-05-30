# Petrophysics December 2022 - Vol. 63, No. 6

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 63, No. 6 (December 2022), the **Best Papers of the 2022 SPWLA Annual
Symposium** special issue (Stavanger, Norway, June 11-15, 2022).  Seven
peer-reviewed extensions of the highest-rated symposium papers:
DAS-fiber-optic VSP FWI, sourceless LWD acoustics from drill-bit noise,
ultradeep azimuthal resistivity (UDAR) geosteering on the Norwegian
Continental Shelf, fractured-carbonate static/dynamic modeling, dipole-
shear reflection imaging combined with Mohr-Coulomb geomechanics,
molecular-dynamics quantification of mineral/fracturing-fluid interfaces,
and digital-rock-physics QC of a novel percussion sidewall coring system.

## Quick start

```bash
pip install numpy scipy

# Run all 7 module tests
python test_all.py

# Or run a single article
python article5_dipole_shear_mohr.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_das_vsp_fwi.py` | Full-Waveform Inversion of Fiber-Optic VSP Data From Deviated Wells | Podgornova, Bettinelli, Liang, Le Calvez, Leaney, Perez, Soliman | 10.30632/PJV63N6-2022a1 |
| `article2_sourceless_lwd_acoustics.py` | Sourceless LWD Borehole Acoustics: Field Testing the Concept | Bolshakov, Walker, Marksamer, Samano, Reynolds | 10.30632/PJV63N6-2022a2 |
| `article3_udar_geosteering.py` | Past, Present, and Future Applications of Ultradeep Directional Resistivity Measurements: A Case History From the Norwegian Continental Shelf | Sinha, Walmsley, Clegg, Vicuna, Wu, McGill, Paiva dos Reis, Nygard, Ulfsnes, Constable, Antonsen, Danielsen | 10.30632/PJV63N6-2022a3 |
| `article4_fractured_carbonate_som.py` | Naturally Fractured Carbonate Reservoir Characterization: A Case Study of a Mature High-Pour Point Oil Field in Hungary | Ali Akbar, Nemes, Bihari, Soltesz, Barany, Toth, Borka, Ferincz | 10.30632/PJV63N6-2022a4 |
| `article5_dipole_shear_mohr.py` | Fracture Characterization Combining Borehole Acoustic Reflection Imaging and Geomechanical Analyses | Tang, Wang, Li, Xiong, Zhang | 10.30632/PJV63N6-2022a5 |
| `article6_md_mineral_fluid.py` | Quantifying Interfacial Interactions Between Minerals and Reservoir/Fracturing Fluids | Silveira de Araujo, Heidari | 10.30632/PJV63N6-2022a6 |
| `article7_pswc_drp_qc.py` | Using Digital Rock Physics to Evaluate Novel Percussion Core Quality | Lakshtanov, Zapata, Saucier, Cook, Eve, Lancaster, Lane, Gettemy, Sincock, Liu, Geetan, Draper, Gill | 10.30632/PJV63N6-2022a7 |
| `test_all.py` | Master test runner | - | - |

## Implementation notes & substitutions

These are working, runnable, faithful demonstrations of the **methods** -
not byte-perfect reproductions.  A few practical substitutions:

- **Article 1 (Podgornova et al.)**: a faithful 2-D / 3-D elastic FWI is
  far outside the scope of a self-contained demo module.  This module
  reproduces the *measurement model* (DAS strain observable, Eqs. 4-5)
  and the inversion problem (least-squares misfit, Eq. 6, Gauss-Newton
  on per-interface impedance contrasts) in 1-D using a reflectivity-
  based forward operator.  The closed-form moment tensors M_vert,
  M_hor,x and M_45,xz (Eqs. 10-11) are evaluated symbolically and
  checked against ||tau||^2 = 1.

- **Article 2 (Bolshakov et al.)**: the listening-mode drill-bit
  excitation is modelled as a monopole + dipole + quadrupole linear
  combination (Wang et al., 2011) on a six-ring four-azimuth array.
  Multipole recombination (Eqs. 1-3) and a standard multi-receiver
  semblance recover Vp and Vs from the synthetic record.

- **Article 3 (Sinha et al.)**: the proprietary 2.5-D / 3-D EM
  inversion of the paper is replaced by a 1-D Occam-style stochastic
  Metropolis sampler over (resistivity, boundary) for an eight-curve
  UDAR vector built from four spacings (7 / 15 / 30 / 60 m) at two
  frequencies (2 / 8 kHz).  The "geostop" decision rule fires when the
  5th-percentile distance-to-base falls below a configurable safety
  margin.

- **Article 4 (Ali Akbar et al.)**: the Spherical Self-Organizing Map
  of the paper is approximated by a standard rectangular Kohonen SOM
  with Gaussian neighbourhood updates - spherical topology mostly
  removes the border-effect bias that small rectangular grids suffer
  from.  The Harrison (1995) Russian-log porosity transform and the
  Torabi et al. (2019) damage-zone-width law are exposed as standalone
  functions for downstream use.

- **Article 5 (Tang et al.)**: implements the 3-D Mohr-Coulomb
  critically-stressed-fracture analysis exactly as written in Eqs. 1-6
  (effective stresses, fracture normal, sigma_n, tau_n,
  cohesion-friction envelope).  The SH-wave cross-dipole image formula
  (Eq. 5) is used to recover dipole strike from a synthetic 4C wave
  set, and the paper's analytical claim that the 180-deg strike
  ambiguity does NOT change the (sigma_n, tau_n) pair is verified
  numerically.

- **Article 6 (Silveira de Araujo & Heidari)**: a true RASPA /
  CLAYFF / SPC-E / GAFF molecular-dynamics run is not reproducible in
  a self-contained NumPy module.  Instead, the paper's *analysis*
  pipeline is reproduced on a synthetic Langevin trajectory through a
  slit pore: density profiles, mineral-fluid contact counts, and the
  MSD-slope diffusion estimator (Eq. 1) are all faithful to the
  paper.  A "sticky" subset of particles models the effect of an
  adsorbed layer on the wall (the methanol / citric-acid behaviour the
  paper studies).

- **Article 7 (Lakshtanov et al.)**: synthetic 3-D voxel sand-pack as
  the analogue of a binary-segmented micro-CT cube; a depth-localised
  percussion-damage zone is induced by injecting small "fines" grains
  in a slice band.  Bulk and per-slice porosity, specific surface
  area, and Kozeny-Carman absolute permeability are computed directly
  from voxel counts and grain-pore interface counts - the same
  pipeline DRP packages use on real micro-CT data, scaled down to a
  size that runs in seconds.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2022)
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
