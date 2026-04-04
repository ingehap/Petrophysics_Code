"""
petrophysics_spwla_2026
=======================
Python implementations of the methods published in:

    PETROPHYSICS – The SPWLA Journal, Vol. 67, No. 2 (April 2026)
    Society of Petrophysicists and Well Log Analysts (SPWLA)
    ISSN 2641-4112

One module per article:

Article 1  (p. 248) – a01_sponge_core_saturation_uncertainty.py
    Alghazal & Krinis — Monte Carlo uncertainty quantification of
    lab-computed saturation data from sponge cores.
    DOI: 10.30632/PJV67N2-2026a1

Article 2  (p. 263) – a02_nmr_wettability_pore_partitioning.py
    Aljishi, Chitrala, Dang & Rai — NMR T₂-based wettability pore
    partitioning and effects on oil recovery in unconventional reservoirs.
    DOI: 10.30632/PJV67N2-2026a2

Article 3  (p. 280) – a03_water_rock_mechanical_ae.py
    Zhao — Water-rock interactions, mechanical property degradation, and
    acoustic emission characteristics of sandstones.
    DOI: 10.30632/PJV67N2-2026a3

Article 4  (p. 295) – a04_wireline_anomaly_diagnosis.py
    Liu, Zhang, Fan et al. — Dual-signal (tension + vibration) fusion for
    diagnosing downhole wireline logging instrument anomalies.
    DOI: 10.30632/PJV67N2-2026a4

Article 5  (p. 318) – a05_ail_hierarchical_correction.py
    Qiao, Wang, Deng, Xu & Yuan — Hierarchical correction of array
    induction logging data in horizontal wells (thickness → invasion →
    anisotropy).
    DOI: 10.30632/PJV67N2-2026a5

Article 6  (p. 336) – a06_bioclastic_limestone_classification.py
    Guo, Duan, Du et al. — Novel integrated geological + petrophysical
    lithological classification for marine bioclastic limestones.
    DOI: 10.30632/PJV67N2-2026a6

Article 7  (p. 351) – a07_knowledge_guided_dcdnn.py
    Yu, Pan, Guo et al. — Knowledge-guided dilated convolutional DNN
    (DCDNN) with Archie/Timur augmentation for reservoir parameter
    prediction (porosity, Sw, permeability).
    DOI: 10.30632/PJV67N2-2026a7

Article 8  (p. 374) – a08_shale_induced_stress_fracture.py
    Ci — Induced-stress-difference field modelling for double fracturing
    in Qiongzhusi deep shale; pumping-rate and volume optimisation.
    DOI: 10.30632/PJV67N2-2026a8

Article 9  (p. 386) – a09_acid_fracturing_cbm.py
    Zhao, Jin, Zhen & Li — Acid fracturing fracture propagation in deep
    coalbed methane wells; concentration and perforation-location
    optimisation.
    DOI: 10.30632/PJV67N2-2026a9

Article 10 (p. 404) – a10_interlayer_fracture_propagation.py
    Zhao, Jin, Guo et al. — Dynamic interlayer fracture propagation in
    interbedded coal-bearing strata; FDEM-proxy analysis for Taiyuan
    Formation, Jiyang Depression.
    DOI: 10.30632/PJV67N2-2026a10

Article 11 (p. 421) – a11_awi_cement_evaluation.py
    Zhang, Zhang, Zhang et al. — New method for evaluating anti-water-
    invasion (AWI) ability of cement slurry; conductivity-jump detection
    and water-invasion rate modelling.
    DOI: 10.30632/PJV67N2-2026a11

Article 12 (p. 437) – a12_depth_shifting_ml.py
    Pan, Fu, Xu et al. — Automatic well-log depth shifting: DTW, cross-
    correlation, ridge regression, and 1-D CNN approaches (SPWLA PDDA
    2023 ML contest summary).
    DOI: 10.30632/PJV67N2-2026a12

Requirements
------------
numpy >= 1.24
scipy >= 1.10

All modules are self-contained and can be run as scripts:
    python a01_sponge_core_saturation_uncertainty.py
    python a12_depth_shifting_ml.py
    ...

Each module exports an ``example_workflow()`` function demonstrating
the key algorithms with synthetic data.
"""
