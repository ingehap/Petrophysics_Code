"""
petrophysics_spwla_2026_06
==========================
Python implementations of the methods published in:

    PETROPHYSICS – The SPWLA Journal, Vol. 67, No. 3 (June 2026)
    "Best Petrophysics Papers From MEOS GEO 2025"
    Society of Petrophysicists and Well Log Analysts (SPWLA)
    ISSN 2641-4112

One module per technical article:

Article 1  (p. 470) – a01_carbonate_pore_type_dielectric.py
    AlZoukani, Al-Hamad & Abdallah — Effect of pore types (moldic,
    interparticle, intercrystalline) on dielectric permittivity in
    carbonates; complex permittivity, porosity-normalised permittivity
    index, and DIA descriptors (PoA, DOMSize, AR).
    DOI: 10.30632/PJV67N3-2026a1

Article 2  (p. 482) – a02_mf_dielectric_fracture_sensitivity.py
    Al-Qouzi, Hassan, Attia, El-Husseiny & Mahmoud — Sensitivity of
    multifrequency dielectric measurements to hydraulic fractures in
    sandstone and carbonate; Cole-Cole / Maxwell-Wagner model.
    DOI: 10.30632/PJV67N3-2026a2

Article 3  (p. 509) – a03_pore_size_fluid_movement.py
    Manuaba, Najrani, Cavalleri, Moge & Chapura — Effect of pore-size
    distribution on fluid movement; multiphysics inversion, NMR T2
    partitioning, Archie with dielectric textural exponent MN.
    DOI: 10.30632/PJV67N3-2026a3

Article 4  (p. 525) – a04_cretaceous_depositional_model.py
    Sultan et al. — Constructing depositional models of the Cretaceous
    reservoirs in SE Iraq from borehole images, logs and core; Dunham
    classification, synthetic resistivity, electrofacies propagation.
    DOI: 10.30632/PJV67N3-2026a4

Article 5  (p. 544) – a05_udar_anisotropy_sensitivity.py
    Bower, Xie, Wang, Leveque & Dolan — Anisotropy sensitivity in
    ultradeep azimuthal resistivity (UDAR) technologies vs spacing,
    frequency, resistivity, and near/far field.
    DOI: 10.30632/PJV67N3-2026a5

Article 6  (p. 560) – a06_deterministic_inversion_uncertainty.py
    Bower, Xie, Cuevas, Hong, Harms, Gremillion & Viandante — Estimating
    uncertainty of deterministic inversion via multistart (~50 guesses)
    and a-posteriori feasible-set sampling (P5–P95).
    DOI: 10.30632/PJV67N3-2026a6

Article 7  (p. 571) – a07_3d_lookahead_em_inversion.py
    El-Khamry, Ma, Clegg, Lozinsky & Bikchandaev — 3D look-ahead EM
    inversion in near-vertical wells; distance-to-boundary ahead of the
    bit and 3D boundary geometry.
    DOI: 10.30632/PJV67N3-2026a7

Article 8  (p. 582) – a08_mud_gas_ratio_fluid_id.py
    Luo, Li, Lu & Qubaisi — Improved mud gas ratio method (Haworth
    wetness/balance/character ratios) discriminating eight fluid types
    while drilling.
    DOI: 10.30632/PJV67N3-2026a8

Article 9  (p. 594) – a09_mf_dielectric_emulsion.py
    Albenayyan, Hassan, El-Husseiny & Mahmoud — Characterising oil-water
    emulsions with the multifrequency dielectric technique; effective-
    medium mixing and W/O vs O/W discrimination (cover paper).
    DOI: 10.30632/PJV67N3-2026a9

Article 10 (p. 619) – a10_acoustic_emission_multiphase.py
    Zeghlache, Aidagulov & Sindt — Acoustic emission monitoring of
    multiphase flow in intelligent completions; AE features, energy-rate
    calibration, and breakthrough detection.
    DOI: 10.30632/PJV67N3-2026a10

Requirements
------------
numpy >= 1.24

All modules are self-contained and can be run as scripts:
    python a01_carbonate_pore_type_dielectric.py
    python a10_acoustic_emission_multiphase.py
    ...

Each module exports an ``example_workflow()`` function demonstrating the key
algorithms with synthetic data.
"""

__version__ = "0.1.0"
__journal__ = "Petrophysics, Vol. 67, No. 3, June 2026"

from . import a01_carbonate_pore_type_dielectric
from . import a02_mf_dielectric_fracture_sensitivity
from . import a03_pore_size_fluid_movement
from . import a04_cretaceous_depositional_model
from . import a05_udar_anisotropy_sensitivity
from . import a06_deterministic_inversion_uncertainty
from . import a07_3d_lookahead_em_inversion
from . import a08_mud_gas_ratio_fluid_id
from . import a09_mf_dielectric_emulsion
from . import a10_acoustic_emission_multiphase
