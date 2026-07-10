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

try:
    import petrolib
except ImportError:  # bare clone, not installed
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import petrolib


class WellboreData(petrolib.data_qc_io.WellboreData):
    """JSON wellbore container -- canonical implementation in petrolib."""


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
