# Petrophysics February 2014 - Vol. 55, No. 1

Python modules implementing the methods from each article in *Petrophysics*,
Vol. 55, No. 1 (February 2014) - solving the complex dual-water equation with
dielectric-NMR-spectroscopy and conventional logs, capillary pressure and
resistivity-index measurements in a mixed-wet carbonate, spontaneous imbibition
of water into oil-wet carbonate cores using nanofluid, desorbed canister gas
sampling and gas isotopic analysis of two coalbed-methane wells, and
thermal-conductivity estimation from elastic-wave velocity with a
petrographic-coded model.

## Quick start

```bash
pip install numpy

# Run all 5 module tests
python test_all.py

# Or run a single article
python article1_dualwater_dielectric_nmr.py
```

## Modules

| File | Article | Authors | Pages |
|------|---------|---------|-------|
| `article1_dualwater_dielectric_nmr.py` | Solving Complex Dual-Water Equation using Dielectric-NMR-Spectroscopy and Conventional Logs | Tan, Lafferty, Neville | 14-23 |
| `article2_pc_resistivity_index_carbonate.py` | Capillary Pressure and Resistivity Index Measurements in a Mixed-Wet Carbonate Reservoir | Dernaika, Efnik, Koronfol, Skjaeveland, Al Mansoori, Hafez, Kalam | 24-30 |
| `article3_nanofluid_imbibition.py` | An Evaluation of Spontaneous Imbibition of Water into Oil-Wet Carbonate Reservoir Cores Using Nanofluid | Roustaei | 31-37 |
| `article4_canister_gas_isotopes.py` | Desorbed Canister Gas Sampling and Gas Isotopic Analysis Procedures and Practices: A Case Study of Two Coalbed Methane Wells from the Lower Saxony Basin, Germany | Spears, Alles, Makhonin | 38-50 |
| `article5_thermal_conductivity_velocity.py` | Thermal Conductivity Estimation From Elastic-Wave Velocity - Application of a Petrographic-Coded Model | Gegenhuber, Schön | 51-56 |
| `test_all.py` | Master test runner | - | - |

> **Note on DOIs.** This issue **predates SPWLA DOI assignment** - the PDF
> carries no article DOIs and CrossRef has none registered for Vol. 55 (the
> first DOIs appear in February 2018, using the older `10.30632/petro_059_1_*`
> scheme). Articles are therefore cited by volume/issue/page rather than DOI.
>
> **Note on extraction.** This issue's source PDF (`Petrophysics_2014_02.pdf`,
> ~9 MB) has a text layer, but **almost every numbered display equation was
> dropped in extraction** (only the equation numbers survived); the bodies are
> reconstructed from the surviving running text, worked examples and nomenclature
> in standard form. Article 1 (dual water) is the best constrained - both
> conductivity terms survived with a verbatim worked example - and Article 5
> (thermal conductivity) is fully reconstructable from its named source theories
> (Hill, Budiansky-O'Connell, Sen, Berryman). Articles 2 (SCAL) and 4 (canister
> gas) are largely experimental/procedural with no display equations, implemented
> from their genuine quantitative content. The cover features Article 4 (coalbed-
> methane gas isotopes). (This issue has no tutorial.)

## Implementation notes & substitutions

- **Article 1 (Tan et al.)**: dual-water joint inversion - the dual-water
  conductivity (Eq. 1) with its excess-clay (`beta*Qv`) and effective-connate-
  water (`(1-alpha*vQH*Qv)*Cw`) terms (reproducing the paper's worked example),
  the clay-bound-water saturation (Eq. 3), Qv from CEC (Eq. 4), the inversion for
  the clay-dependent cementation exponent m0 (Eq. 5), and a bisection solve for
  Sw. The dielectric tool is a black box (no CRIM law appears). The NMR and
  spectroscopy paths of the workflow are completed by the inverse relations
  `Qv = Swb/(alpha*vQH)` (Eq. 3) and `CEC = Qv*phi/(rho_g*(1-phi))` (Eq. 4) and a
  clay-bound-water porosity from an NMR T2 distribution (T2 < 3 msec cutoff). The
  paper's headline six-step joint inversion is wrapped end-to-end
  (`solve_sw_workflow`: dielectric Sxo/Cmfe -> NMR Qv -> spectroscopy phi ->
  microresistivity Cxo -> invert m0 -> uninvaded-zone Sw, with the dielectric
  salinity-validity limit of ~50 ppt), and the Appendix n-vs-m trade-off
  (`archie_cementation_for_sw`) reproduces the tabulated pairs that hold Sw fixed
  (e.g. Sw = 0.8: n=1.5->m=2.00, n=2.5->m=1.87, n=3.5->m=1.73).

- **Article 2 (Dernaika et al.)**: Pc & resistivity index - the resistivity
  index `RI = Rt/Ro = Sw^-n` and a saturation-exponent fit, plus the Archie
  formation factor and a cementation-exponent fit. Reported n increases through
  the displacement cycles (PD 1.99 -> FI 2.28 for the high-perm RRTs), so a
  `saturation_exponents_by_cycle` helper fits n per cycle and reproduces the
  paper's PD ~ SI < FI ordering. The measured exponent is applied through the
  inverse `Sw = RI^(-1/n)` and a full Archie water saturation. The SCAL
  endpoints are completed by the movable-oil saturation `1 - Swi - Sor` (Sor
  converging to ~20% for high-perm and ~27-30% for tight RRTs) and a
  capillary-pressure unit reconciliation between the bar scale (SI, max 7 bar)
  and the psi scale (FI, max 80 psi).

- **Article 3 (Roustaei)**: nanofluid imbibition - Young's law contact angle and
  its cosine (Eq. 1), a wettability classification, the Young-Dupre work of
  adhesion, the Young-Laplace capillary pressure `Pc = 2*sigma_wo*cos(theta)/r`
  (the paper's central mechanism: wettability alteration turns the capillary
  force "from a barrier to a driving force" - `Pc < 0` oil-wet vs `Pc > 0`
  water-wet - and a higher oil-water IFT raises the driving `Pc`), the
  capillary-vs-gravity discrimination from the recovery-curve shape (curved ->
  capillary, linear -> gravity), and a first-order spontaneous-imbibition
  recovery curve. The nanofluid raises the oil-water IFT (2.65 -> 9.21 mN/m),
  lowers `cos(theta)` so the contact angle increases toward water-wet, and lifts
  the final recovery above ~50% IOIP (brine ~4.3%, surfactant ~46%).

- **Article 4 (Spears et al.)**: canister gas - the atmospheric air-contamination
  correction (N2:O2 = 3.73, after Jin et al., 2010) and an air-contamination
  fraction, a square-root-of-time lost-gas estimate (Direct Method, following
  the desorption guidelines the paper cites) and total gas content, the
  isotope/GC quality-control checks (delta-13C limits, 50 mV-sec CH4 peak area),
  the canister-handling thresholds (the <=10 cm^3/day shipping criterion and the
  ~1 bar/15 psi venting pressure), a Tedlar (PVF) bag hold-time check (<24 h)
  backed by the paper's relative-permeation finding (O2 ~10x N2, He ~15x CO2),
  and a biogenic/thermogenic gas-origin classification.

- **Article 5 (Gegenhuber & Schön)**: thermal conductivity from velocity - the
  Hill average, the crack density in both its forms - by count `eps = (N/V)*r^3`
  (Eq. 3) and from porosity/aspect ratio `eps = phi/((4/3)*pi*alpha)` (Eq. 4,
  with its inverse crack porosity) - the Budiansky-O'Connell self-consistent
  cracked moduli and P-wave velocity, the Sen plate-like depolarization
  exponents, and the Clausius-Mossotti effective thermal conductivity. The two
  halves are joined by the headline two-step estimator: invert the elastic model
  for the crack density that matches a measured Vp, then map that (shared) crack
  porosity to thermal conductivity - estimating `lambda` directly from `Vp`.
  Best-fit aspect ratios: granite/gneiss/sandstone 0.20, basic magmatic 0.25.

## Module conventions

Every module follows the same structure:

```python
"""Article N: <Title>
<Authors> (2014)
Reference: Petrophysics Vol. 55, No. 1, pp. <pages>
"""

# imports
# implementation functions (equation numbers in docstrings)
# def test_all(): ...   # synthetic-data demo with assertions

if __name__ == "__main__":
    test_all()
```

So each module is runnable as a standalone script, importable as a
library, and self-tests with synthetic data.
