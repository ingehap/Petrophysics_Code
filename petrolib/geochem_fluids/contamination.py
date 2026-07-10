"""OBM/filtrate contamination mixing and cleanup.

Linear property mixing between virgin and filtrate fluid, the contamination
fraction (its inverse), power-law cleanup vs pumped volume, and the pumped volume
needed to reach a contamination target.

Unit-neutral; ``eta`` is the filtrate (contamination) fraction.  Sources:
src2015_06/article2, src2017_08/article5, src2018_10/article6, src2021_02/article5,
src2024_02/article2.

References
----------
Complete citations for the source tags used in this module (SPWLA journal
*Petrophysics*):

src2015_06/article2 -- Article 2: A Breakthrough in Accurate Downhole Fluid Sample Contamination
  Prediction in Real Time. Zuo, Gisolf, Dumont, Dubost, Pfeiffer, Wang, Mishra, Chen, Mullins,
  Biagi, Gemelli (2015). Petrophysics Vol. 56, No. 3 (June 2015), pp. 251-265. DOI: none assigned
  (this issue predates SPWLA DOI assignment).
src2015_06/article2_fluid_contamination_prediction -- Article 2: A Breakthrough in Accurate
  Downhole Fluid Sample Contamination Prediction in Real Time. Zuo, Gisolf, Dumont, Dubost,
  Pfeiffer, Wang, Mishra, Chen, Mullins, Biagi, Gemelli (2015). Petrophysics Vol. 56, No. 3 (June
  2015), pp. 251-265. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2017_08/article5 -- Article 5: Advances in Quantification of Miscible Contamination in
  Hydrocarbon and Water Samples From Downhole to Surface Laboratories. Zuo, Gisolf, Pfeiffer,
  Achourov, Chen, Mullins, Edmundson, Partouche (2017). Petrophysics Vol. 58, No. 4 (August 2017),
  pp. 397-410. DOI: none assigned (this issue predates SPWLA DOI assignment).
src2017_08/article5_contamination_quantification -- Article 5: Advances in Quantification of
  Miscible Contamination in Hydrocarbon and Water Samples From Downhole to Surface Laboratories.
  Zuo, Gisolf, Pfeiffer, Achourov, Chen, Mullins, Edmundson, Partouche (2017). Petrophysics Vol.
  58, No. 4 (August 2017), pp. 397-410. DOI: none assigned (this issue predates SPWLA DOI
  assignment).
src2018_10/article6 -- Article 6: Proxy-Enabled Stochastic Interpretation of Downhole Fluid
  Sampling Under Immiscible Flow Conditions. Kristensen, Chugunov, Cig, Jackson (2018). DOI:
  10.30632/PJV59N5-2018a5. Petrophysics Vol. 59 No. 5 (Oct 2018) - "Best of 2018 SPWLA Symposium"
  issue.
src2018_10/article6_proxy_stochastic_fluid_sampling -- Article 6: Proxy-Enabled Stochastic
  Interpretation of Downhole Fluid Sampling Under Immiscible Flow Conditions. Kristensen, Chugunov,
  Cig, Jackson (2018). DOI: 10.30632/PJV59N5-2018a5. Petrophysics Vol. 59 No. 5 (Oct 2018) - "Best
  of 2018 SPWLA Symposium" issue.
src2021_02/article5 -- Article 5: Innovative Formation Tester Sampling Procedures for Carbon
  Dioxide and Other Reactive Components. Piazza, Vieira, Sacorague, Jones, Dai, Pearl, Aguiar
  (2021). DOI: 10.30632/PJV62N1-2021a4. Petrophysics Vol. 62 No. 1 (Feb 2021).
src2021_02/article5_formation_tester_co2_sampling -- Article 5: Innovative Formation Tester
  Sampling Procedures for Carbon Dioxide and Other Reactive Components. Piazza, Vieira, Sacorague,
  Jones, Dai, Pearl, Aguiar (2021). DOI: 10.30632/PJV62N1-2021a4. Petrophysics Vol. 62 No. 1 (Feb
  2021).
src2024_02/article2 -- Article 2: Fluid Contamination Transient Analysis Gelvez and Torres-Verdín
  (Petrophysics, Vol. 65, No. 1, Feb 2024, pp. 32-50).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Float = NDArray[np.float64]


def mix_linear(p_v: ArrayLike, p_f: ArrayLike, eta: ArrayLike) -> _Float:
    """Linear mixing ``P = (1 - eta)*P_virgin + eta*P_filtrate``.

    Sources: src2015_06/article2_fluid_contamination_prediction,
    src2017_08/article5_contamination_quantification.
    """
    e = np.asarray(eta, np.float64)
    return np.asarray((1.0 - e) * np.asarray(p_v, np.float64) + e * np.asarray(p_f, np.float64))


def contamination_fraction(p: ArrayLike, p_v: ArrayLike, p_f: ArrayLike) -> _Float:
    """Contamination fraction ``eta = (P - P_v)/(P_f - P_v)`` (inverse of the mixing rule).

    Sources: src2015_06/article2_fluid_contamination_prediction.
    """
    pv = np.asarray(p_v, np.float64)
    return np.asarray((np.asarray(p, np.float64) - pv) / (np.asarray(p_f, np.float64) - pv))


def cleanup_powerlaw(
    v: ArrayLike, eta0: ArrayLike, v_star: ArrayLike, *, exponent: float = 5.0 / 12.0
) -> _Float:
    """Power-law cleanup ``eta(V) = eta0*(1 + V/V_star)**(-exponent)``.

    The default exponent 5/12 is the wireline-cleanup value (2/3 for radial).

    Sources: src2018_10/article6_proxy_stochastic_fluid_sampling,
    src2021_02/article5_formation_tester_co2_sampling.
    """
    ratio = np.asarray(v, np.float64) / np.asarray(v_star, np.float64)
    return np.asarray(np.asarray(eta0, np.float64) * (1.0 + ratio) ** (-exponent))


def volume_to_target(
    eta0: ArrayLike, v_star: ArrayLike, eta_t: ArrayLike, *, exponent: float = 5.0 / 12.0
) -> _Float:
    """Pumped volume to reach contamination ``eta_t`` -- inverse of :func:`cleanup_powerlaw`.

    ``V = V_star*((eta0/eta_t)**(1/exponent) - 1)``.

    Sources: src2018_10/article6_proxy_stochastic_fluid_sampling,
    src2021_02/article5_formation_tester_co2_sampling.
    """
    ratio = np.asarray(eta0, np.float64) / np.asarray(eta_t, np.float64)
    return np.asarray(np.asarray(v_star, np.float64) * (ratio ** (1.0 / exponent) - 1.0))
