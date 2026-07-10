"""Log data QC, cleaning, filtering, synthetic data, and wellbore IO.

The cross-cutting data-handling layer the articles re-implement around their
physics (see LIBRARY_MERGE_PLAN.md).  Submodules:

* :mod:`~petrolib.data_qc_io.clean` -- no-data sentinels, gap imputation,
  outlier masks, despiking, and compositional closure checks.
* :mod:`~petrolib.data_qc_io.scale` -- reference-well (Shier) normalization
  and moment matching.
* :mod:`~petrolib.data_qc_io.filt` -- smoothing, moving-window statistics,
  Gaussian tool response, 2-D median filter, depth-window feature stacks,
  and bed-boundary detection.
* :mod:`~petrolib.data_qc_io.signal` -- SNR, stacking gain, block stacking,
  Gaussian noise injection.
* :mod:`~petrolib.data_qc_io.synth` -- synthetic blocky logs, log suites,
  shifted curve pairs, Gaussian-mixture spectra, sphere packs, disk images.
* :mod:`~petrolib.data_qc_io.io` -- the Bradley et al. (2023) universal
  wellbore-data JSON container.

Related canon (deliberately NOT duplicated here): unit conversions live in
:mod:`petrolib.units`; error metrics and z-score / min-max feature scaling in
:mod:`petrolib.ml_stats`; depth alignment (cross-correlation, DTW, shifts) in
:mod:`petrolib.depth_matching`; pay flags, interval thickness, and
net-to-gross in :mod:`petrolib.porosity_lithology`.
"""

from __future__ import annotations

from . import clean, filt, io, scale, signal, synth
from .io import WellboreData

__all__ = [
    "WellboreData",
    "clean",
    "filt",
    "io",
    "scale",
    "signal",
    "synth",
]
