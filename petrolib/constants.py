"""Physical and unit-conversion constants — the single source of truth.

The duplication analysis behind LIBRARY_MERGE_PLAN.md found these values
redefined per article module, sometimes inconsistently (GAMMA_H differs in
the 4th digit between files; EPS0 appears in at least two directories).
Article code migrates to these names during the domain trains; new code
imports them from here.

Physical constants follow CODATA 2018 / the 2019 SI exact definitions.
Field-approximation constants (e.g. the 0.433 psi/ft freshwater gradient)
deliberately do NOT live here — they are parameters of the domain functions
that use them, never invisible global defaults (CONVENTIONS.md rule 5).

Sources: repeated definitions across src2014_02 … src2026_06; see
LIBRARY_MERGE_PLAN.md section 9 for the GAMMA_H hazard note.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2014_02 -- Petrophysics Vol. 55 No. 1 (Feb 2014) (issue-level reference).
src2026_06 -- Petrophysics Vol. 67 No. 3 (June 2026) — "Best Petrophysics Papers From MEOS GEO
  2025" (issue-level reference).
"""

from __future__ import annotations

# --- SI defining / CODATA 2018 physical constants --------------------------

#: Boltzmann constant [J/K] (exact, 2019 SI).
KB = 1.380649e-23

#: Avogadro constant [1/mol] (exact, 2019 SI).
NA = 6.02214076e23

#: Molar gas constant [J/(mol*K)] (exact: KB * NA).
R_GAS = 8.31446261815324

#: Vacuum electric permittivity [F/m] (CODATA 2018).
EPS0 = 8.8541878128e-12

#: Vacuum magnetic permeability [H/m] (CODATA 2018).
MU0 = 1.25663706212e-6

#: Standard acceleration of gravity [m/s^2] (exact, conventional).
G_STD = 9.80665

#: Proton gyromagnetic ratio [rad/(s*T)] (CODATA 2018).
#:
#: Article modules disagree in the 4th digit: some use 2.675e8, others
#: 2*pi*42.58e6 = 2.6753e8.  This is the CODATA value; migrating call
#: sites whose assertions depend on their local value pass it explicitly.
GAMMA_H = 2.6752218744e8

#: Proton gyromagnetic ratio over 2*pi [Hz/T] (CODATA 2018).
GAMMA_H_HZ = 42.577478518e6

# --- Exact unit-conversion factors ------------------------------------------

#: Metres per international foot (exact).
M_PER_FT = 0.3048

#: Metres per international inch (exact).
M_PER_IN = 0.0254

#: Pascals per psi (pound-force per square inch; exact by definition
#: of lbf and in).
PA_PER_PSI = 6894.757293168361

#: Pascals per bar (exact).
PA_PER_BAR = 1.0e5

#: Pascals per standard atmosphere (exact).
PA_PER_ATM = 101325.0

#: Square metres per darcy.  The darcy is defined via cP, atm and cm;
#: this is the conventional value (ISO 31-8 / SPE).
M2_PER_DARCY = 9.869233e-13

#: Kilograms per cubic metre per g/cc.
KGM3_PER_GCC = 1000.0
