"""
Article 3: NMR Fluid Substitution - A New Method of Reconstructing T2
           Distributions Under Primary Drainage and Imbibition Conditions
Li, Kesserwan, Jin, Ma (2021)
DOI: 10.30632/PJV62N4-2021a2

NMR-derived petrophysical models are calibrated on T2 distributions measured
at full water saturation (Sw=1), but downhole the rock is partially saturated.
This paper reconstructs the Sw=1 T2 distribution from a partially-saturated one
by keeping the irreducible-water peak fixed and shifting / re-amplifying the
movable-water peak under a total-porosity-conservation constraint.

Implements:

  - Surface relaxation  1/T2 = 1/T2bulk + rho*(S/V),  S/V = Fs/r
  - T2 -> pore radius  r = Fs * rho * T2
  - BVI / BVM split at a T2 cutoff (classic 33 ms)
  - Total porosity from a T2 distribution
  - Porosity-conserving fluid substitution (partial -> full saturation)

Note: the journal's Eqs. 1-12 were image-rendered and not in the text; the
forms here are faithful reconstructions of the surface-relaxation physics and
the porosity-conserving reconstruction the paper describes.  T2 in ms,
surface relaxivity in um/s, radius in um.
"""

import numpy as np

T2_CUTOFF_SANDSTONE = 33.0      # ms, classic BVI cutoff


# ---------------------------------------------- relaxation physics ------

def surface_relaxation_rate(T2_bulk, rho_um_s, s_over_v_per_um):
    """1/T2 = 1/T2bulk + rho*(S/V).  rho in um/s, S/V in 1/um -> 1/ms."""
    rho_um_ms = rho_um_s * 1e-3            # um/s -> um/ms
    return 1.0 / T2_bulk + rho_um_ms * s_over_v_per_um


def t2_to_pore_radius(T2_ms, rho_um_s, shape_factor=3.0):
    """Pore radius from T2 (surface-limited):  r = Fs * rho * T2.

    From 1/T2 ~ rho*(S/V) with S/V = Fs/r.  Returns um.
    """
    rho_um_ms = rho_um_s * 1e-3
    return shape_factor * rho_um_ms * np.asarray(T2_ms, float)


# ---------------------------------------------- porosity / BVI ----------

def total_porosity(amplitudes):
    """Total porosity = sum of the T2-distribution amplitudes."""
    return float(np.sum(amplitudes))


def bvi_bvm(T2_ms, amplitudes, cutoff=T2_CUTOFF_SANDSTONE):
    """Split a T2 distribution into bound (BVI) and movable (BVM) volumes."""
    T2 = np.asarray(T2_ms, float)
    amp = np.asarray(amplitudes, float)
    bvi = float(amp[T2 <= cutoff].sum())
    bvm = float(amp[T2 > cutoff].sum())
    return bvi, bvm


def saturation_from_t2(T2_ms, amplitudes, phi_total, cutoff=T2_CUTOFF_SANDSTONE):
    """Total water saturation = (water amplitude) / total porosity."""
    return total_porosity(amplitudes) / phi_total


# ---------------------------------------------- fluid substitution ------

def reconstruct_full_saturation(amp_partial, movable_mask, s_w_eff, shift_bins):
    """Reconstruct the Sw=1 distribution from a partially-saturated one.

    The movable-water bins (where hydrocarbon displaced water) are shifted
    back by `shift_bins` toward larger pores and re-amplified by 1/s_w_eff to
    restore the porosity that the displacing fluid had occupied.  Irreducible
    bins are unchanged.  Returns the reconstructed amplitude array.
    """
    amp = np.asarray(amp_partial, float)
    mask = np.asarray(movable_mask, bool)
    out = amp.copy()
    # remove the (suppressed) partial movable water, then add it back shifted
    out[mask] = 0.0
    movable = amp[mask] / s_w_eff                 # re-amplify to full water
    idx = np.flatnonzero(mask) + shift_bins       # shift toward larger pores
    idx = np.clip(idx, 0, len(amp) - 1)
    np.add.at(out, idx, movable)
    return out


# ---------------------------------------------- tests --------------

def test_all():
    print("=" * 60)
    print("Article 3: NMR Fluid Substitution of T2 Distributions")
    print("=" * 60)

    # T2 -> pore radius is linear and increasing
    r = t2_to_pore_radius([1.0, 10.0, 100.0], rho_um_s=10.0, shape_factor=3.0)
    print(f"  pore radii (1/10/100 ms) = {r.round(4)} um")
    assert np.all(np.diff(r) > 0)
    assert abs(r[1] - 3.0 * (10e-3) * 10.0) < 1e-9

    # Build a full-saturation (Sw=1) truth distribution: bound + movable peaks
    T2 = np.logspace(-1, 3, 64)                   # 0.1 .. 1000 ms
    def lognorm(center, width, amp):
        return amp * np.exp(-((np.log(T2) - np.log(center)) ** 2) / (2 * width ** 2))
    bound = lognorm(3.0, 0.5, 0.03)               # clay/capillary-bound water
    movable = lognorm(150.0, 0.5, 0.12)           # large-pore movable water
    truth = bound + movable
    phi_T = total_porosity(truth)
    print(f"  total porosity (Sw=1)  = {phi_T:.4f}")

    # Forward operator: hydrocarbon displaces movable water, suppressing the
    # movable-water amplitude by the pore-scale effective water saturation.
    movable_mask = T2 > T2_CUTOFF_SANDSTONE
    s_w_eff = 0.30
    partial = truth.copy()
    partial[movable_mask] = truth[movable_mask] * s_w_eff
    bvi, bvm = bvi_bvm(T2, partial)
    print(f"  partial BVI / BVM      = {bvi:.4f} / {bvm:.4f}")
    assert total_porosity(partial) < phi_T        # water was displaced

    # Fluid substitution: re-amplify the movable-water bins by 1/s_w_eff to
    # restore the porosity the hydrocarbon had occupied (porosity conserved).
    recon = reconstruct_full_saturation(partial, movable_mask, s_w_eff,
                                        shift_bins=0)
    phi_recon = total_porosity(recon)
    print(f"  reconstructed porosity = {phi_recon:.4f}  (target {phi_T:.4f})")
    assert abs(phi_recon - phi_T) < 1e-9, "reconstruction must conserve porosity"
    assert np.allclose(recon, truth)              # exact recovery of the Sw=1 dist

    # The shift option moves the movable peak toward larger pores
    shifted = reconstruct_full_saturation(partial, movable_mask, s_w_eff,
                                          shift_bins=3)
    assert np.all(shifted >= -1e-12)
    print("  PASS")
    return {"phi_T": phi_T, "phi_recon": phi_recon, "bvi": bvi, "bvm": bvm}


if __name__ == "__main__":
    test_all()
