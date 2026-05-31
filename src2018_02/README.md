# Petrophysics February 2018 - Vol. 59, No. 1

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 59, No. 1 (February 2018) - the **"Best Papers of the 2017 SCA
International Symposium"** issue: the first **Shaly Sand** tutorial, seven SCA
best papers (digital-rock benchmarking, stress-sensitive MICP, stress-dependent
permeability, relative-permeability QC, densitometer fluid-volume logging,
salt-bearing digital rock, and core-restoration optimization), and two regular
submissions (geostress effects on resistivity and shale gas adsorption).

## Quick start

```bash
pip install numpy

# Run all 10 module tests
python test_all.py

# Or run a single article
python article6_densitometer_fluid_volume.py
```

## Modules

| File | Article | Authors | DOI |
|------|---------|---------|-----|
| `article1_shaly_sand_tutorial_part1.py` | *Tutorial:* What Is It About Shaly Sands? Shaly Sand Tutorial 1 of 3 | Thomas | 10.30632/petro_059_1_t1 |
| `article2_drp_blind_study_pc.py` | A Blind Study of Four Digital Rock Physics Vendor Laboratories on Porosity, Absolute Permeability, and Primary Drainage Capillary Pressure Data on Tight Outcrops | Chhatre, Sahoo, Leonardi, Vidal, Rainey, Braun, Patel | 10.30632/petro_059_1_a1 |
| `article3_stress_sensitivity_micp.py` | Stress Sensitivity of Mercury-Injection Measurements | Guise, Grattoni, Allshorn, Fisher, Schiffer | 10.30632/petro_059_1_a2 |
| `article4_stress_dependent_permeability.py` | Microstructural Investigation of Stress-Dependent Permeability in Tight-Oil Rocks | King, Sansone, Kortunov, Xu, Callen, Chhatre, Sahoo, Buono | 10.30632/petro_059_1_a3 |
| `article5_drt_relperm_qc.py` | Using Digital Rock Technology to Quality Control and Reduce Uncertainty in Relative Permeability Measurements | Schembre-McCabe, Kamath | 10.30632/petro_059_1_a4 |
| `article6_densitometer_fluid_volume.py` | Using a Densitometer for Quantitative Determinations of Fluid Density and Fluid Volume in Coreflooding Experiments at Reservoir Conditions | Olsen | 10.30632/petro_059_1_a5 |
| `article7_salt_bearing_digital_rock.py` | Investigation of Salt-Bearing Sediments Through Digital Rock Technology Together With Experimental Core Analysis | Rydzy, Anger, Hertel, Dietderich, Patino, Appel | 10.30632/petro_059_1_a6 |
| `article8_core_restoration_rsm.py` | Application of an Optimization Method for the Restoration of Core Samples for SCAL Experiments | Sripal, James | 10.30632/petro_059_1_a7 |
| `article9_geostress_resistivity_correction.py` | Study on the Mechanism of Geostress Difference Effect on Tight Sandstone Resistivity and Its Correction Method | Liu, Zhang, Zheng, Xin | 10.30632/petro_059_1_a8 |
| `article10_shale_gas_adsorption.py` | New Perspectives on the Effects of Gas Adsorption on Storage and Production of Natural Gas From Shale Formations | Tinni, Sondergeld, Rai | 10.30632/petro_059_1_a9 |
| `test_all.py` | Master test runner | - | - |

> **Note on extraction.** This issue's source PDF (`Petrophysics_2018_02.pdf`,
> ~16 MB) has a text layer, so titles, authors, and page ranges were read from
> the contents page and bodies. The machine extraction captured the full bodies
> of the **tutorial and articles a1-a6** and a truncated body for **a7**, but
> **articles a8-a9 (the regular submissions) appear only as table-of-contents
> entries** and are implemented as **methodology proxies**. The PDF text layer
> contains **no article DOIs**, so the DOIs above were taken from CrossRef (the
> authoritative SPWLA registry): this issue uses the older `10.30632/petro_059_1_*`
> scheme (`t1`, `a1` ... `a9`), not the later `PJVxxNx` style. As with the other
> issues, the typeset formula glyphs were dropped in extraction, so the numbered
> formulas are faithful standard-form reconstructions.

## Implementation notes & substitutions

- **Article 1 (Thomas)** *(tutorial)*: the Archie baseline and its validity
  conditions - water saturation, the formation factor, the 5-50% shaly-sand
  classification, and the clay specific-surface ratio.

- **Article 2 (Chhatre et al.)**: digital-rock blind study - the Young-Laplace
  pore-throat radius, normalized water saturation, a power-law drainage
  capillary-pressure curve, and interfacial-tension rescaling.

- **Article 3 (Guise et al.)**: stress-sensitive MICP - the Washburn pore
  diameter, the Swanson permeability from the MICP apex, and threshold-pressure
  detection.

- **Article 4 (King et al.)**: stress-dependent permeability - the net confining
  stress, the exponential permeability decline with stress, the matrix gas
  permeability (k = D*mu/B), and the Klinkenberg correction.

- **Article 5 (Schembre-McCabe & Kamath)**: relative-permeability QC - Corey
  water/oil relative permeabilities with wettability-dependent endpoints, the
  water-wet/oil-wet bounding envelope, an outlier flag, and the fractional flow.

- **Article 6 (Olsen)**: densitometer fluid-volume logging - the water fraction
  from mixture density, cumulative produced water by integration, and produced
  oil by volume closure. (Most equation-rich article.)

- **Article 7 (Rydzy et al.)**: salt-bearing digital rock - resolved porosity,
  paleoporosity, salt saturation and its classes, the exponential permeability
  decline with salt saturation, and size-distribution percentiles.

- **Article 8 (Sripal & James)**: core-restoration optimization - the USBM
  wettability index, a Box-Behnken design, a second-order response-surface
  least-squares fit, and optimization of the fitted surface.

- **Article 9 (Liu et al.)** *(methodology proxy)*: geostress effect on
  resistivity - a stress-dependent resistivity, the inverse correction, the
  Archie saturation, and the saturation bias from skipping the correction.

- **Article 10 (Tinni et al.)** *(methodology proxy)*: shale gas adsorption - the
  Langmuir isotherm, free + adsorbed gas in place, the adsorbed fraction, and the
  gas desorbed by a pressure drawdown.

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
