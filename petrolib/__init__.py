"""petrolib — common petrophysics library for the Petrophysics_Code repository.

Shared physics, constants, unit handling, and utilities extracted from the
article implementations in the ``srcYYYY_MM`` issue directories, so that the
scripts import one canonical implementation instead of re-implementing it
(see LIBRARY_MERGE_PLAN.md at the repository root).

Import policy: ``import petrolib`` must succeed with numpy alone.  Submodules
load lazily (PEP 562), and any function needing scipy or another optional
dependency imports it inside the function with a clear error message.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__version__ = "0.1.0"

# Grows one entry per domain train (LIBRARY_MERGE_PLAN.md section 7).
_SUBMODULES = frozenset(
    {
        "acoustic_geomech",
        "borehole_image",
        "capillary_pressure",
        "constants",
        "depth_correction",
        "depth_matching",
        "em_dielectric",
        "flow_transport",
        "geochem_fluids",
        "integrity_drilling",
        "inversion_numerics",
        "ml_stats",
        "nmr",
        "nuclear",
        "porosity_lithology",
        "relperm_wettability",
        "saturation_resistivity",
        "testing",
        "units",
        "wellbore_geometry",
    }
)


def __getattr__(name: str) -> ModuleType:
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULES)
