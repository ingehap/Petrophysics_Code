"""
Bradley et al. (2023), Petrophysics 64(6): 823-836.
A proposed universal format for storing/distributing wellbore data, capable of
representing simple 1D logs and complex multi-dimensional measurements
(e.g. ultradeep azimuthal resistivity).

This module implements a minimal JSON-based container that mirrors the paper's
ideas: hierarchical metadata + n-dimensional channel arrays + units + axes.
"""
import json
import numpy as np


class WellboreData:
    def __init__(self, well_name, uwi=None):
        self.meta = {"well_name": well_name, "uwi": uwi, "version": "1.0"}
        self.channels = {}  # name -> dict(data, units, axes, dims)

    def add_channel(self, name, data, units, axes):
        arr = np.asarray(data)
        if len(axes) != arr.ndim:
            raise ValueError(f"axes count {len(axes)} != ndim {arr.ndim}")
        self.channels[name] = {"data": arr, "units": units, "axes": axes,
                               "shape": arr.shape}

    def to_dict(self):
        return {"meta": self.meta,
                "channels": {n: {**{k: v for k, v in c.items() if k != "data"},
                                 "data": c["data"].tolist()}
                             for n, c in self.channels.items()}}

    def to_json(self, path=None):
        s = json.dumps(self.to_dict(), indent=2)
        if path: open(path, "w").write(s)
        return s

    @classmethod
    def from_dict(cls, d):
        w = cls(d["meta"]["well_name"], d["meta"].get("uwi"))
        w.meta = d["meta"]
        for n, c in d["channels"].items():
            w.add_channel(n, np.array(c["data"]), c["units"], c["axes"])
        return w


def test_all():
    w = WellboreData("Synthetic-1", uwi="00/00-00-000-00W0/0")
    md = np.linspace(1000, 1100, 50)
    w.add_channel("GR", 50 + 30 * np.sin(md / 5), "gAPI", ["MD"])
    # 3D: depth x azimuth x DOI (ultradeep azimuthal resistivity)
    udar = np.random.default_rng(1).uniform(1, 100, (50, 16, 5))
    w.add_channel("UDAR", udar, "ohm.m", ["MD", "azimuth", "DOI"])
    s = w.to_json()
    w2 = WellboreData.from_dict(json.loads(s))
    print("Bradley et al. universal format:")
    print(f"  channels: {list(w2.channels.keys())}")
    print(f"  UDAR shape: {w2.channels['UDAR']['data'].shape}")
    assert w2.channels["UDAR"]["data"].shape == (50, 16, 5)
    print("  PASS")


if __name__ == "__main__":
    test_all()
