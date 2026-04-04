"""
Petrophysics 2026 - Python implementations of ideas from
Petrophysics Journal, Vol. 67, No. 1 (February 2026).

Best papers from the SPWLA 66th Annual Symposium, Dubai, May 17-21, 2025.

Modules
-------
drill_cuttings_ai
    Article 1: Enhanced Reservoir Characterization Using Drill-Cuttings-Based
    Image and Elemental Analysis With AI (Kriscautzky et al.)
dts_co2_monitoring
    Article 2: Real-Time CO2 Injection Monitoring Through Fiber Optics:
    Physics-Based Modeling of DTS Data (Pirrone & Mantegazza)
nmr_discrete_inversion
    Article 3: Discrete Inversion Method for NMR Data Processing and Its
    Applications to Fluid Typing (Gao et al.)
depth_alignment
    Article 4: Dynamic Depth Alignment of Well Logs: A Continuous
    Optimization Framework (Westeng et al.)
fluid_identification
    Article 5: Beyond Gas Bubbles in Norwegian Oil Fields: An Integrated
    Technique to Understand Reservoir Fluid Distribution (Bravo et al.)
multiphysics_inversion
    Article 6: Advanced Logging Techniques for Characterizing a Complex
    Turbidite Reservoir (Datir et al.)
nmr_bitumen
    Article 7: Petrophysical Characterization of Secondary Organic Matter
    and Hydrocarbons Using Laboratory NMR Techniques (Al Mershed et al.)
co2_sequestration
    Article 8: Effect of CO2 Sequestration on Carbonate Formation Integrity
    (Al-Hamad et al.)
tortuosity_permeability
    Article 9: Tortuosity Assessment for Reliable Permeability
    Quantification (Arrieta et al.)
pgs_type_curve
    Article 10: A Novel Type Curve for Sandstone Rock Typing (Musu et al.)
udar_joint_inversion
    Article 11: A Robust Joint Inversion for Improved Structural Mapping
    in UDAR Applications (Wu et al.)
udar_multidim_inversion
    Article 12: Recent Developments and Verifications of Multidimensional
    Inversion of Borehole UDAR Measurements (Saputra et al.)
sand_injectites
    Article 13: New Insights Into the Understanding of Sand Injectite
    Complexes (Ahmad et al.)
resistivity_ranging
    Article 14: Case Studies of Active Resistivity Ranging in Near-Parallel
    Wells (Salim et al.)
udar_look_ahead
    Article 15: UDAR Horizontal Look-Ahead Mapping Technology Identifies
    Fault Ahead of the Bit (Ma et al.)

References
----------
Petrophysics, Vol. 67, No. 1 (February 2026).
DOI prefix: 10.30632/PJV67N1-2026a{1..15}
Published by the Society of Petrophysicists and Well Log Analysts (SPWLA).
"""

__version__ = "0.1.0"
__journal__ = "Petrophysics, Vol. 67, No. 1, February 2026"

from . import drill_cuttings_ai
from . import dts_co2_monitoring
from . import nmr_discrete_inversion
from . import depth_alignment
from . import fluid_identification
from . import multiphysics_inversion
from . import nmr_bitumen
from . import co2_sequestration
from . import tortuosity_permeability
from . import pgs_type_curve
from . import udar_methods
