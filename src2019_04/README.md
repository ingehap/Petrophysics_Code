# Petrophysics April 2019 - Vol. 60, No. 2

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 60, No. 2 (April 2019) - the **"Best Papers of the 2018 SCA International
Symposium"** issue: two tutorials, seven Society-of-Core-Analysts papers, and
three regular submissions.  Topics span organic-mudstone storage capacity, a
resistivity-principles primer, trapped-oil capillary desaturation, image-
segmentation uncertainty, NMR wettability, waterflood initialization, in-situ
saturation monitoring, the intercept method for relative permeability,
temperature-array core monitoring, invasion-zone log inversion, loading effects
on gas relative permeability, and borehole acoustic reflection imaging.

## Quick start

```bash
pip install numpy

# Run all 12 module tests
python test_all.py

# Or run a single article
python article2_resistivity_principles_tutorial.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_organic_mudstone_storage_part2_tutorial.py` | *Tutorial:* Organic Mudstone Petrophysics, Part 2: Workflow to Estimate Storage Capacity | Newsham, Comisky, Chemali | 10.30632/PJV60N2-2019t1 |
| `article2_resistivity_principles_tutorial.py` | *Tutorial:* Introduction to Resistivity Principles for Formation Evaluation: A Tutorial Primer | Kennedy, Garcia | 10.30632/PJV60N2-2019t2 |
| `article3_trapped_oil_capillary_desaturation.py` | Pore-Scale Insights on Trapped Oil During Waterflooding of Sandstone Rocks of Varying Wettability States | Berthet, Hebert, Barbouteau, Andriamananjaona, Rivenq | 10.30632/PJV60N2-2019a1 |
| `article4_image_segmentation_uncertainty.py` | Uncertainty Quantification in Image Segmentation for Image-Based Rock Physics in a Shaly Sandstone | Howard, Lin, Zhang | 10.30632/PJV60N2-2019a2 |
| `article5_nmr_wettability_review.py` | A Review of 60 Years of NMR Wettability | Valori, Nicot | 10.30632/PJV60N2-2019a3 |
| `article6_waterflood_init_wettability.py` | A New Waterflood Initialization Protocol With Wettability Alteration for Pore-Scale Multiphase Flow Experiments | Lin, Bijeljic, Krevor, Blunt, Rücker, Berg, Coorn, van der Linde, Georgiadis, Wilson | 10.30632/PJV60N2-2019a4 |
| `article7_issm_saturation_monitoring.py` | In-Situ Saturation Monitoring (ISSM) — Recommendations for Improved Processing | Reed, Cense | 10.30632/PJV60N2-2019a5 |
| `article8_intercept_method_relperm.py` | Review of the Intercept Method for Relative Permeability Correction a Variety of Case Study Data | Reed, Maas | 10.30632/PJV60N2-2019a6 |
| `article9_temperature_array_monitoring.py` | Monitoring Core Measurements With High-Resolution Temperature Arrays | Howard, Hester | 10.30632/PJV60N2-2019a7 |
| `article10_invasion_zone_log_inversion.py` | How the Invasion Zone Can Contribute to the Estimation of Petrophysical Properties From Log Inversion at Well Scale? | Vandamme, Caroli, Gratton | 10.30632/PJV60N2-2019a8 |
| `article11_loading_gas_relperm.py` | Loading Effects on Gas Relative Permeability of a Low-Permeability Sandstone | Agostini, Egermann, Jeannin, Portier, Skoczylas, Wang | 10.30632/PJV60N2-2019a9 |
| `article12_borehole_acoustic_stc_raytracing.py` | Borehole Acoustic Imaging Using 3D STC and Ray Tracing to Determine Far-Field Reflector Dip and Azimuth | Bennett, Donald, Ghadiry, Nassar, Kumar, Biswas | 10.30632/PJV60N2-2019a10 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2019_04.pdf`)
> has a text layer, so the titles, authors, page ranges, and DOIs were read from
> the contents page and paper bodies. The machine extraction captured the full
> bodies of the **two tutorials and articles a1-a5** but **truncated after
> article a5** (page ~282), so **articles a6-a10 were available only as
> table-of-contents entries** and are implemented as **methodology proxies** of
> the standard methods their titles describe. As with the other issues, the
> typeset formula glyphs were dropped in extraction, so the numbered formulas are
> faithful standard-form reconstructions. DOI pattern: `10.30632/PJV60N2-2019aN`
> (a1 … a10) plus tutorials `…-2019t1`, `…-2019t2`.

## Implementation notes & substitutions

- **Article 1 (Newsham et al.)** *(tutorial)*: kerogen volume from TOC, the
  porosity partition (clay-bound / capillary / free hydrocarbon), bulk volume
  hydrocarbon, water saturation, and gas/oil-in-place.

- **Article 2 (Kennedy & Garcia)** *(tutorial)*: the Archie formation factor,
  R0, resistivity index, and water-saturation equation, plus empirical fitting
  of m (from F-φ) and n (from I-Sw).

- **Article 3 (Berthet et al.)**: the capillary number, the capillary
  desaturation curve Sor(Nc), the wettability-state shift of the residual oil
  and critical Nc, and trapping efficiency.

- **Article 4 (Howard et al.)**: porosity from a grayscale threshold, an Otsu
  threshold from the histogram, the propagated porosity uncertainty from the
  threshold uncertainty, and three-phase (pore/clay/grain) fractions.

- **Article 5 (Valori & Nicot)**: surface relaxation 1/T2 = 1/T2_bulk + ρ(S/V),
  the effective relaxivity vs contact angle, and an Amott-style NMR wettability
  index between water-wet and oil-wet end states.

- **Article 6 (Lin et al.)**: the Young-Laplace capillary pressure, the
  primary-drainage initial water saturation from a threshold Pc, and aging that
  alters the contact angle (and Amott wettability) toward oil-wet.

- **Article 7 (Reed & Cense)**: the Beer-Lambert attenuation law, water
  saturation from attenuation between dry/saturated calibrations, and a
  dual-energy two-fluid solve.

- **Article 8 (Reed & Maas)** *(methodology proxy)*: Darcy apparent kr, the
  rate-dependent capillary end effect, and the intercept extrapolation to
  1/Q = 0 that recovers the end-effect-free relative permeability.

- **Article 9 (Howard & Hester)** *(methodology proxy)*: 1D transient
  heat conduction, thermal diffusivity, the CFL stability number, and thermal
  front / hot-spot localization from the temperature array.

- **Article 10 (Vandamme et al.)** *(methodology proxy)*: Archie in the flushed
  and virgin zones, a radial two-zone resistivity model vs depth of
  investigation, and a grid-search inversion recovering (Rt, Rxo, invasion
  radius) from multi-DOI logs.

- **Article 11 (Agostini et al.)** *(methodology proxy)*: Biot effective stress,
  stress-dependent permeability k = k0·exp(−c·σ_eff), Klinkenberg gas slippage,
  and Corey gas relative permeability.

- **Article 12 (Bennett et al.)** *(methodology proxy)*: slowness-time-coherence
  picking, reflector azimuth from the azimuthal amplitude lobe, ray-traced
  reflector distance, and reflector dip from the depth moveout.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2019)
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
