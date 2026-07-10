"""Universal wellbore-data container (Bradley et al. 2023).

A minimal JSON-based format for storing and exchanging wellbore data:
hierarchical metadata plus named n-dimensional channel arrays with units and
axis labels, covering simple 1-D logs through multi-dimensional measurements
such as ultradeep azimuthal resistivity.  Source: src2023_12/
bradley_wellbore_format.py (Petrophysics 64(6):823-836).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class WellboreData:
    """Hierarchical metadata + n-D channels with units and axes.

    ``meta`` holds ``well_name``, ``uwi``, and the format ``version``;
    ``channels`` maps name -> ``{"data", "units", "axes", "shape"}``.
    Round-trip via ``to_json``/``from_json`` (or ``to_dict``/``from_dict``);
    ``shape`` is recomputed on reload.
    """

    def __init__(self, well_name: str, uwi: str | None = None) -> None:
        self.meta: dict[str, Any] = {"well_name": well_name, "uwi": uwi, "version": "1.0"}
        self.channels: dict[str, dict[str, Any]] = {}

    def add_channel(self, name: str, data: ArrayLike, units: str, axes: list[str]) -> None:
        """Store an n-D channel; ``axes`` must name every array dimension."""
        arr = np.asarray(data)
        if len(axes) != arr.ndim:
            raise ValueError(f"axes count {len(axes)} != ndim {arr.ndim}")
        self.channels[name] = {
            "data": arr,
            "units": units,
            "axes": axes,
            "shape": arr.shape,
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict; channel arrays become nested lists."""
        return {
            "meta": self.meta,
            "channels": {
                n: {**{k: v for k, v in c.items() if k != "data"}, "data": c["data"].tolist()}
                for n, c in self.channels.items()
            },
        }

    def to_json(self, path: str | None = None) -> str:
        """Serialize to a JSON string (``indent=2``); optionally write ``path``."""
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as fh:
                fh.write(s)
        return s

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WellboreData:
        """Rebuild from :meth:`to_dict` output (or parsed JSON)."""
        w = cls(d["meta"]["well_name"], d["meta"].get("uwi"))
        w.meta = d["meta"]
        for n, c in d["channels"].items():
            w.add_channel(n, np.array(c["data"]), c["units"], c["axes"])
        return w

    @classmethod
    def from_json(cls, s: str) -> WellboreData:
        """Rebuild from a JSON string produced by :meth:`to_json`."""
        return cls.from_dict(json.loads(s))
