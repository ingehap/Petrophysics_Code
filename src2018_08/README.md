# Petrophysics August 2018 - Vol. 59, No. 4

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 59, No. 4 (August 2018) - the **"Special Issue on Flow Diagnostics"**: a
capillary-pressure tutorial, six flow-diagnostics articles, and two regular
submissions.

## Quick start

```bash
pip install numpy

# Run all 9 module tests
python test_all.py

# Or run a single article
python article8_crushedrock_flowregime_permeability.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_capillary_pressure_tutorial_part1.py` | *Tutorial:* Capillary Pressure Tutorial Part 1 - It's a Jungle in Here | Thomas | 10.30632/PJV59V4-2018t1 |
| `article2_acoustic_flowrate_model.py` | Improved Aero/Hydro Flow-Rate Model Using Acoustics | Seshadri, Freund, Jha, Venna, Walters, Jagannathan | 10.30632/PJV59V4-2018a1 |
| `article3_multiphase_pl_holdup_correction.py` | Refining Interpretation Models of Multiphase Flow for Existing and Next-Generation Production Logging Sensors | Manzar, Sun, Chace | 10.30632/PJV59V4-2018a2 |
| `article4_ultracompact_flow_array_doppler.py` | Efficiency Improvements in Production Profiling Using Ultracompact Flow Array Sensing Technology | Abbassi, Tavernier, Donzier, Gysen, Gysen, Chen, Zeid, Cedillo | 10.30632/PJV59V4-2018a3 |
| `article5_downhole_sand_production_rate.py` | Downhole Sand-Production Evaluation for Sand-Management Applications | Swarnanto, Srihirunrusmee, Lilaprathuang, Panmamuang, Wuthicharn, Mukerji, Duangprasert, Puttisounthorn, Millot, Saavedra, Nollet | 10.30632/PJV59V4-2018a4 |
| `article6_distributed_sensing_flow_monitoring.py` | Production Monitoring Using Next-Generation Distributed Sensing Systems | Naldrett, Cerrahoglu, Mahue | 10.30632/PJV59V4-2018a5 |
| `article7_acg_downhole_surveillance.py` | ACG - 20 Years of Downhole Surveillance History | Sheydayev, Atakishiyev, Zett, Schoepf, Thiruvenkatanathan | 10.30632/PJV59V4-2018a6 |
| `article8_crushedrock_flowregime_permeability.py` | Incorporating Flow Regimes Into Crushed-Rock Analysis to Better Understand Matrix Permeability and Pore Structure in Shales | Royer, Hobbs, Bonar | 10.30632/PJV59V4-2018a7 |
| `article9_chargeability_metallic_particles.py` | Chargeability of Porous Rocks With or Without Metallic Particles | Revil, Tartrat, Abdulsamad, Ghorbani, Coperey | 10.30632/PJV59V4-2018a8 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2018_08.pdf`,
> ~45 MB) has a text layer, so titles, authors, page ranges, and DOIs were read
> from the contents page and bodies; the full bodies of **all nine items** were
> captured. The typeset display-equation glyphs were dropped in extraction
> (only the bare equation numbers survived for most articles), so the numbered
> formulas are faithful standard-form reconstructions from the surviving prose,
> variable definitions, and worked numbers - except Article 9 (Revil), whose
> equations largely survived as inline text. DOI pattern: `10.30632/PJV59V4-2018aN`
> (a1 ... a8) plus the tutorial `...t1` - note the prefix is `PJV59V4` (letter
> **V**), not `N4`, as printed in the source.

## Implementation notes & substitutions

- **Article 1 (Thomas)** *(tutorial)*: capillary pressure from first principles -
  the Young-Laplace pressure jump, the capillary-rise force balance
  (h = 2*sigma*cos(theta)/(rho*g*r)), capillary pressure as the difference of the
  two hydrostatic columns, and Pc from the pore-throat radius for rock typing.

- **Article 2 (Seshadri et al.)**: leak rate from hydrophone amplitude - the
  Bernoulli liquid-leak rate, the choked-gas critical pressure ratio and mass
  rate, the leak Mach number, the monopole-dipole-quadrupole amplitude scaling
  (p'^2 ~ C2*M^2 + C3*M^3 + C4*M^4), and a calibrated amplitude->rate inversion.
  The exact Eqs. 13-16 coefficients were not in the text layer, so the inversion
  is fit on synthetic calibration data.

- **Article 3 (Manzar et al.)**: array-PL holdup - the linear water holdup from
  array resistance, the paper's nonlinear velocity-dependent correction (Eq. 7
  survived verbatim; the low-holdup curves are reconstructed between the two
  flow-loop velocities), and cross-section integration into per-phase flow rates.

- **Article 4 (Abbassi et al.)** *(methodology proxy)*: the ultrasonic Doppler
  flow-speed relation (VF = Vs*(Fr-Fe)/(2*Fe*cos(alpha))), the Appendix-6 digital
  Doppler speed from the spectral-peak position, conductivity-probe holdup, and
  the area-weighted array profile mean.

- **Article 5 (Swarnanto et al.)**: piezoelectric sand counting - single-grain
  volume, the 0.4572 m vertical-resolution correction, the volumetric sand rate
  (VSPR), and the mass sand rate (SPR). The conversion chain is transcribed from
  the paper.

- **Article 6 (Naldrett et al.)**: DTS/DAS flow allocation - the thermal-mixing
  energy balance and the two-zone flow split, the Joule-Thomson dT = JTC*dP, the
  gauge-length max detectable frequency (fmax = c/(2*GL)), and fluid typing from
  the acoustic sound speed.

- **Article 7 (Sheydayev et al.)** *(methodology proxy)*: a field-history paper
  with no published equations, so this implements the standard surveillance
  computations its PDHG/DFO workflow uses - a productivity index, a moving-average
  transient-event detector, and the distributed-sensing data-rate budget.

- **Article 8 (Royer et al.)**: flow-regime crushed-rock permeability - Darcy
  flow, the gas mean free path, the Knudsen number and regime, the Klinkenberg
  correction, and the paper's "lambda plot" extrapolated to a 1-nm mean free path
  (k1lambda) with an effective pore diameter from the slope.

- **Article 9 (Revil et al.)**: induced-polarization chargeability - Seigel's
  chargeability from the conductivity dispersion, the Stern-layer surface
  conductivity, the background chargeability and the universal R = lambda/B, and a
  volume-weighted mixture chargeability with a metallic-particle contribution.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2018)
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
