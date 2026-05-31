"""
Article 8: 2D Reservoir Imaging Using Deep Directional Resistivity Measurements
Thiel, Bower, Omeragic (2018)
DOI: 10.30632/PJV59N2-2018a7  (header confirmed; body beyond extraction)

Deep directional (geosteering) resistivity tools see resistivity contrasts tens
of feet from the borehole, letting a 2D image of the reservoir be inverted from
the bedding response.  This *methodology proxy* implements the standard
distance-to-boundary forward/inverse relations the imaging workflow relies on:
a directional tool reads a depth-of-investigation-weighted blend of the local
and adjacent bed resistivities, the blend inverts for the distance to the
boundary, and stacking the per-station distances builds the 2D boundary image.

Implements:

  - Two-bed directional apparent resistivity (DOI-weighted blend)
  - Distance-to-boundary inversion from the apparent resistivity
  - Directional (azimuthal) signal from the bed contrast
  - 2D boundary-image assembly from per-station distances

Note: this article's body was beyond this issue's machine extraction (only the
DOI header line was captured), so - as with the other methodology proxies in
this repository - the relations below are the standard directional-resistivity
formulas the described imaging workflow uses, not formulas transcribed from the
paper.  Resistivities in ohm.m, distances in m.
"""

import numpy as np


# ---------------------------------------------- forward --------------

def directional_apparent_resistivity(r_local, r_bed, distance, doi):
    """Apparent resistivity as a DOI-weighted blend of two beds (log domain)

        log(Ra) = w*log(R_local) + (1-w)*log(R_bed),  w = 0.5*(1 + tanh(d/DOI)).

    Far inside the local bed (d >> DOI) the tool reads R_local; approaching the
    boundary it pulls toward the adjacent bed R_bed.
    """
    w = 0.5 * (1.0 + np.tanh(np.asarray(distance, float) / doi))
    return np.exp(w * np.log(r_local) + (1.0 - w) * np.log(r_bed))


def distance_to_boundary(ra, r_local, r_bed, doi):
    """Invert the directional response for the distance to the boundary

        d = DOI*atanh(2*w - 1),  w = (log(Ra)-log(R_bed))/(log(R_local)-log(R_bed)).
    """
    w = (np.log(ra) - np.log(r_bed)) / (np.log(r_local) - np.log(r_bed))
    w = np.clip(w, 1e-6, 1 - 1e-6)
    return doi * np.arctanh(2.0 * w - 1.0)


def azimuthal_signal(r_local, r_bed):
    """Directional (geosignal) amplitude from the bed contrast  = ln(R_local/R_bed).

    Zero when the two beds match (no detectable boundary); signed by which side
    is more resistive.
    """
    return np.log(r_local / r_bed)


# ---------------------------------------------- imaging --------------

def boundary_image(ra_stations, r_local, r_bed, doi):
    """Assemble a 2D boundary image: distance to the boundary at each station."""
    return np.array([distance_to_boundary(ra, r_local, r_bed, doi) for ra in ra_stations])


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 8: 2D Directional Resistivity Imaging (proxy)")
    print("=" * 60)

    r_local, r_bed, doi = 50.0, 2.0, 5.0
    # Approaching a conductive bed pulls the apparent resistivity down
    ra_far = directional_apparent_resistivity(r_local, r_bed, 12.0, doi)
    ra_near = directional_apparent_resistivity(r_local, r_bed, 1.0, doi)
    print(f"  Ra far / near boundary = {ra_far:.2f} / {ra_near:.2f} ohm.m")
    assert r_bed < ra_near < ra_far < r_local

    # Inversion recovers the planted distance to the boundary
    d_true = 3.5
    ra = directional_apparent_resistivity(r_local, r_bed, d_true, doi)
    d = distance_to_boundary(ra, r_local, r_bed, doi)
    print(f"  recovered distance     = {d:.3f} m  (true 3.500)")
    assert np.isclose(d, d_true, atol=1e-6)

    # No contrast -> no directional signal
    assert np.isclose(azimuthal_signal(10.0, 10.0), 0.0) and azimuthal_signal(50.0, 2.0) > 0

    # A multi-station image returns one distance per station
    img = boundary_image([ra, ra_far, ra_near], r_local, r_bed, doi)
    assert img.shape == (3,) and np.all(np.isfinite(img))
    print("  PASS")
    return {"distance": float(d), "Ra_near": float(ra_near)}


if __name__ == "__main__":
    test_all()
