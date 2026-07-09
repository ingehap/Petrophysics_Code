"""Geochemistry and reservoir-fluids: brine, asphaltene, mud gas, adsorption,
PVT, pressure gradients, solubility, contamination, and core geochemistry.

A package because the domain is large and heterogeneous (LIBRARY_MERGE_PLAN.md).
Each submodule is numpy-vectorized and keyword-explicit; scipy, when needed, is
imported lazily inside the function.  Submodules are exposed as attributes:
``petrolib.geochem_fluids.brine.rw_from_salinity(...)``.
"""

from __future__ import annotations

from . import (
    adsorption,
    asphaltene,
    brine,
    contamination,
    core_geochem,
    gradients,
    mudgas,
    pvt,
    solubility,
)

__all__ = [
    "adsorption",
    "asphaltene",
    "brine",
    "contamination",
    "core_geochem",
    "gradients",
    "mudgas",
    "pvt",
    "solubility",
]
