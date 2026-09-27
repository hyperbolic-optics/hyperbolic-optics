"""Grating couplers: waveguide mode → fibre, from the grating's leaky mode.

A uniform grating on a waveguide turns its guided mode into a leaky one that
radiates as it goes (:func:`hyperbolic_optics.resonances.grating_leaky_modes`,
from the full-vector RCWA, anisotropic layers included). Its complex Bloch
wavevector and its field give, per unit guided power entering the grating:

- the decay rate ``α`` (field), so the emitted field along the surface is
  ``exp(−α x)`` over the grating's length ``L``;
- the emission angle, where the −1st order leaves;
- the directionality ``D``, the share of the radiated power going up, and its
  polarization.

A fibre tilted to the emission angle receives the overlap ``η`` of that
exponential with its Gaussian mode (footprint ``w₀/cos θ`` on the surface),
maximised over the fibre's position:

    efficiency = D · (1 − e^{−2αL}) · η.

This is the standard leaky-mode model of a uniform grating coupler (e.g.
D. Taillaert et al., Jpn. J. Appl. Phys. 45, 6071 (2006)). It neglects the
back-reflection at the grating's start and the grating's finite-length edge
effects; an FDTD simulation of the finite device checks both.
:func:`grating_coupler_model` wraps it as a SAX model.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf

from hyperbolic_optics import resonances

__all__ = [
    "ApodizedCoupler",
    "GratingCoupler",
    "apodized_grating_coupler",
    "coupling_from_profile",
    "exponential_gaussian_overlap",
    "grating_coupler",
    "grating_coupler_model",
    "optimal_overlap",
]


def exponential_gaussian_overlap(
    alpha_per_um, waist_um, offset_um, length_um=np.inf, angle_deg=0.0
):
    """Power overlap of ``exp(−α x)`` on ``[0, L]`` with a tilted Gaussian fibre mode.

    The fibre mode's field on the surface is ``exp(−((x − x₀)·cos θ)²/w₀²)``,
    phase-matched to the emission (the fibre tilted to its angle ``θ``).

    Args:
        alpha_per_um: Field decay rate ``α``.
        waist_um: The fibre mode's waist ``w₀`` (10.4 µm MFD → 5.2 µm).
        offset_um: The fibre's centre ``x₀`` from the grating's start.
        length_um: The grating's length ``L``.
        angle_deg: The emission (and fibre) angle ``θ``.

    Returns:
        ``|∫ E·G|² / (∫|E|² · ∫|G|²)``, the field overlaps integrated over the
        grating (E) and the whole surface (G).
    """
    a = float(alpha_per_um)
    w = float(waist_um) / np.cos(np.radians(angle_deg))  # footprint
    x0, length = float(offset_um), float(length_um)
    shift = a * w / 2
    upper = (length - x0) / w + shift if np.isfinite(length) else np.inf
    integral = (
        w * np.sqrt(np.pi) / 2 * np.exp(shift**2 - a * x0)
        * (erf(upper) - erf(-x0 / w + shift))
    )  # fmt: skip
    emitted = (1 - np.exp(-2 * a * length)) / (2 * a)
    gaussian = w * np.sqrt(np.pi / 2)
    return float(integral**2 / (emitted * gaussian))


def optimal_overlap(alpha_per_um, waist_um, length_um=np.inf, angle_deg=0.0):
    """``(overlap, offset_um)``: the fibre position that maximises the overlap."""
    footprint = float(waist_um) / np.cos(np.radians(angle_deg))
    span = min(float(length_um), 5 / float(alpha_per_um) + 3 * footprint)
    result = minimize_scalar(
        lambda x0: -exponential_gaussian_overlap(alpha_per_um, waist_um, x0, length_um, angle_deg),
        bounds=(-footprint, span),
        method="bounded",
        options={"xatol": 1e-6 * footprint},
    )
    return -float(result.fun), float(result.x)


class GratingCoupler(NamedTuple):
    """A uniform grating coupler's performance at one wavelength.

    Attributes:
        efficiency: Guided power into the fibre mode, both polarizations.
        efficiency_te, efficiency_tm: Into the fibre's s (TE, E along the
            grating lines) and p (TM) polarizations.
        directionality: Share of the radiated power going up.
        radiated: ``1 − e^{−2αL}``: share of the guided power radiated.
        overlap: With the fibre mode, at the best fibre position.
        fiber_offset_um: That position, from the grating's start.
        emission_deg: Emission angle of the coupling order, from the normal.
        mode: The :class:`~hyperbolic_optics.resonances.LeakyMode`.
    """

    efficiency: float
    efficiency_te: float
    efficiency_tm: float
    directionality: float
    radiated: float
    overlap: float
    fiber_offset_um: float
    emission_deg: float
    mode: resonances.LeakyMode


def grating_coupler(
    layers,
    period_um,
    wavelength_um,
    eps_top,
    eps_bottom,
    index_range,
    waist_um: float = 5.2,
    length_um: float = np.inf,
    max_order: int = 10,
    **kwargs,
) -> GratingCoupler:
    """The coupling efficiency of a uniform grating coupler, waveguide to fibre.

    Args:
        layers, period_um, wavelength_um, eps_top, eps_bottom, index_range,
        max_order: As in :func:`hyperbolic_optics.resonances.grating_leaky_modes`
            (``index_range`` brackets the leaky mode's ``Re kx₀/k₀``, near the
            waveguide's effective index).
        waist_um: The fibre mode's waist (5.2 µm: SMF-28 at 1550 nm).
        length_um: The grating's length.
        **kwargs: Passed on to ``grating_leaky_modes``.

    Returns:
        :class:`GratingCoupler` for the leaky mode with the highest ``Re kx₀``
        in the range.
    """
    modes = resonances.grating_leaky_modes(
        layers, period_um, wavelength_um, eps_top, eps_bottom, index_range, max_order, **kwargs
    )
    if not modes:
        raise ValueError(f"no leaky mode with Re kx₀/k₀ in {index_range}")
    mode = modes[0]
    if not mode.emission_deg:
        raise ValueError("the leaky mode radiates no order into the top half-space")
    # the top order that carries the emission: the −1st if it radiates
    order = -1 if -1 in mode.emission_deg else min(mode.emission_deg, key=abs)
    angle = mode.emission_deg[order]
    overlap, offset = optimal_overlap(mode.alpha_per_um, waist_um, length_um, angle)
    radiated = 1 - np.exp(-2 * mode.alpha_per_um * length_um) if np.isfinite(length_um) else 1.0
    total = mode.directionality * radiated * overlap
    p, s = mode.top_polarization[order]
    return GratingCoupler(
        efficiency=float(total),
        efficiency_te=float(total * abs(s) ** 2),
        efficiency_tm=float(total * abs(p) ** 2),
        directionality=mode.directionality,
        radiated=float(radiated),
        overlap=overlap,
        fiber_offset_um=offset,
        emission_deg=angle,
        mode=mode,
    )


def grating_coupler_model(
    layers_at,
    period_um,
    eps_top,
    eps_bottom,
    index_range,
    waist_um: float = 5.2,
    length_um: float = np.inf,
    max_order: int = 10,
    track: float = 0.02,
    **kwargs,
):
    """A SAX model of a grating coupler, waveguide ↔ fibre, over wavelength.

    Args:
        layers_at: ``wavelength_um → layers`` (so materials can disperse), or
            the layers themselves.
        index_range: Where to look for the leaky mode at the first wavelength;
            at each next one the search follows it, ``±track`` about the last.
        Others: As in :func:`grating_coupler`.

    Returns:
        ``model(wl=1.55) → SDict`` over ports ``waveguide@TE`` and ``fiber@TE``,
        ``fiber@TM`` (SAX's default mode names; fibre TE is s, E along the
        grating lines), reciprocal, with amplitudes ``√efficiency``. It is a
        lumped, magnitude-accurate model: the phases are zero, and the
        back-reflection into the waveguide is neglected. Not differentiable
        (each wavelength is a root search); evaluate it on the wavelengths
        your circuit uses.
    """

    requested_loss = kwargs.pop("max_loss", None)

    def model(wl=1.55):
        wavelengths = np.atleast_1d(np.asarray(wl, dtype=float))
        te, tm = [], []
        window, loss = tuple(index_range), requested_loss
        for wavelength in wavelengths:
            layers = layers_at(wavelength) if callable(layers_at) else layers_at
            result = grating_coupler(
                layers, period_um, wavelength, eps_top, eps_bottom, window, waist_um,
                length_um, max_order, max_loss=loss, **kwargs,
            )  # fmt: skip
            te.append(np.sqrt(result.efficiency_te))
            tm.append(np.sqrt(result.efficiency_tm))
            # follow the mode: a narrower window about it, deep enough for its loss
            centre, decay = result.mode.bloch_index.real, result.mode.bloch_index.imag
            window = (centre - track, centre + track)
            loss = max(requested_loss or 0.0, 3 * decay)
        te = np.reshape(np.asarray(te, dtype=complex), np.shape(wl))
        tm = np.reshape(np.asarray(tm, dtype=complex), np.shape(wl))
        return {
            ("waveguide@TE", "fiber@TE"): te,
            ("fiber@TE", "waveguide@TE"): te,
            ("waveguide@TE", "fiber@TM"): tm,
            ("fiber@TM", "waveguide@TE"): tm,
        }

    return model


# ---------------------------------------------------------------------------
# Apodized, chirped and focusing gratings: the local leaky-mode model
# ---------------------------------------------------------------------------


def _fiber_mode(x, waist_um, angle_rad, n_top, k0, offset):
    """A tilted Gaussian fibre mode on the surface, normalised: ``∫|G|² dx = 1``."""
    footprint = waist_um / np.cos(angle_rad)
    envelope = np.exp(-(((x - offset) / footprint) ** 2))
    phase = np.exp(1j * k0 * n_top * np.sin(angle_rad) * (x - offset))
    return envelope * phase / np.sqrt(footprint * np.sqrt(np.pi / 2))


def coupling_from_profile(
    x_um,
    alpha_per_um,
    directionality,
    emission_index,
    jones,
    wavelength_um,
    waist_um: float = 5.2,
    fiber_angle_deg: float | None = None,
    n_top: float = 1.0,
):
    """Waveguide-to-fibre coupling from a grating's local leaky-mode parameters along x.

    The guided power decays as ``P(x) = exp(−2∫α dx)``; the field emitted
    upward is ``√(2αD·P) · exp(i k₀ ∫ q dx)`` in each polarization of ``jones``,
    with ``q = Re kx/k₀`` of the radiating order. Its overlap with a Gaussian
    fibre mode tilted to ``fiber_angle_deg`` is maximised over the fibre's
    position (and its angle, if not given).

    Args:
        x_um: Sample positions along the grating ``[X]``, increasing.
        alpha_per_um, directionality, emission_index: ``α``, ``D`` and ``q`` there ``[X]``.
        jones: The emission's unit Jones vector ``(p, s)`` there ``[X, 2]``.
        wavelength_um, waist_um, n_top: As in :func:`grating_coupler`.
        fiber_angle_deg: The fibre's angle in the top medium; by default the
            best one.

    Returns:
        ``(efficiency_te, efficiency_tm, offset_um, angle_deg)``: power into
        the fibre's s (TE) and p (TM) polarizations.
    """
    x = np.asarray(x_um, dtype=float)
    k0 = 2 * np.pi / float(wavelength_um)
    alpha = np.asarray(alpha_per_um, dtype=float)
    dx = np.gradient(x)
    decayed = np.cumsum(alpha * dx) - 0.5 * alpha * dx  # ∫α to each sample's centre
    phase = np.cumsum(np.asarray(emission_index) * dx) - 0.5 * np.asarray(emission_index) * dx
    amplitude = np.sqrt(np.clip(2 * alpha * np.asarray(directionality), 0, None)) * np.exp(-decayed)
    field = (amplitude * np.exp(1j * k0 * phase))[:, None] * np.asarray(jones)  # [X, (p, s)]

    def efficiencies(offset, angle_rad):
        mode = np.conj(_fiber_mode(x, waist_um, angle_rad, n_top, k0, offset))
        coupled = np.sum(field * (mode * dx)[:, None], axis=0)
        return np.abs(coupled[1]) ** 2, np.abs(coupled[0]) ** 2  # TE = s, TM = p

    from scipy.optimize import minimize

    q_mean = float(np.average(np.asarray(emission_index), weights=amplitude**2 + 1e-300))
    angle0 = np.arcsin(np.clip(q_mean / n_top, -0.99, 0.99))
    start = [float(x[np.argmax(amplitude)] + 0.5 * waist_um)]
    options = {"xatol": 1e-7, "fatol": 1e-12}
    if fiber_angle_deg is None:
        result = minimize(lambda v: -sum(efficiencies(v[0], v[1])), start + [angle0],
                          method="Nelder-Mead", options=options)  # fmt: skip
        offset, angle = result.x
    else:
        angle = np.radians(fiber_angle_deg)
        result = minimize(lambda v: -sum(efficiencies(v[0], angle)), start,
                          method="Nelder-Mead", options=options)  # fmt: skip
        offset = result.x[0]
    te, tm = efficiencies(offset, angle)
    return float(te), float(tm), float(offset), float(np.degrees(angle))


def _follow(layers, period, wavelength_um, eps_top, eps_bottom, previous, max_order):
    """The leaky mode continuing ``previous`` into a neighbouring grating.

    The secant method from the neighbour's root first; if the step was too
    big for it, a small contour search about that root, keeping the nearest.
    """
    guess = previous.bloch_index
    try:
        return resonances.refine_leaky_mode(
            layers, period, wavelength_um, eps_top, eps_bottom, guess, max_order
        )
    except ValueError:
        window = (guess.real - 0.05, guess.real + 0.05)
        found = resonances.grating_leaky_modes(
            layers, period, wavelength_um, eps_top, eps_bottom, window, max_order,
            max_loss=max(4 * guess.imag, 0.01),
        )  # fmt: skip
        if not found:
            raise
        return min(found, key=lambda mode: abs(mode.bloch_index - guess))


class ApodizedCoupler(NamedTuple):
    """A grating coupler's performance from its teeth, one by one.

    Attributes:
        efficiency, efficiency_te, efficiency_tm: As in :class:`GratingCoupler`.
        fiber_offset_um: The fibre's best position along x.
        fiber_angle_deg: Its angle (given, or the best one).
        modes: The local leaky mode of every tooth.
    """

    efficiency: float
    efficiency_te: float
    efficiency_tm: float
    fiber_offset_um: float
    fiber_angle_deg: float
    modes: tuple


def apodized_grating_coupler(
    layers_for,
    starts_um,
    periods_um,
    wavelength_um,
    eps_top,
    eps_bottom,
    index_range,
    waist_um: float = 5.2,
    fiber_angle_deg: float | None = None,
    max_order: int = 10,
    samples_per_tooth: int = 12,
    expected_index: float | None = None,
    **kwargs,
) -> ApodizedCoupler:
    """Coupling efficiency of an apodized, chirped or focusing grating coupler.

    Each tooth is treated as a period of a uniform grating of its own pitch and
    fill (the local leaky-mode model): the first tooth's leaky mode is found
    by the contour search, each next one by refining its neighbour's
    (:func:`hyperbolic_optics.resonances.refine_leaky_mode`), and the emission
    of the whole grating is overlapped with the fibre mode
    (:func:`coupling_from_profile`). For identical teeth this is
    :func:`grating_coupler`.

    Args:
        layers_for: ``i → layers`` of tooth ``i``'s uniform grating (e.g. from
            :func:`hyperbolic_optics.integrations.gdsfactory.grating_layers`).
        starts_um, periods_um: Each tooth's start and pitch ``[N]``.
        wavelength_um, eps_top, eps_bottom, index_range, waist_um, max_order,
        **kwargs: As in :func:`grating_coupler`.
        fiber_angle_deg: The fibre's angle in the top medium; by default the best.
        samples_per_tooth: Integration samples per tooth.
        expected_index: If several leaky modes lie in ``index_range``, the one
            with ``Re kx₀/k₀`` nearest this (e.g. the average of the waveguide's
            index under the teeth and under the gaps); by default the highest.
    """
    starts = np.asarray(starts_um, dtype=float)
    periods = np.asarray(periods_um, dtype=float)
    first = resonances.grating_leaky_modes(
        layers_for(0), periods[0], wavelength_um, eps_top, eps_bottom, index_range, max_order,
        **kwargs,
    )  # fmt: skip
    if not first:
        raise ValueError(f"no leaky mode with Re kx₀/k₀ in {index_range} for the first tooth")
    if expected_index is not None:
        first = sorted(first, key=lambda mode: abs(mode.bloch_index.real - expected_index))
    modes = [first[0]]
    for i in range(1, len(starts)):
        same = np.isclose(periods[i], periods[i - 1]) and layers_for(i) is layers_for(i - 1)
        modes.append(
            modes[-1] if same else _follow(
                layers_for(i), periods[i], wavelength_um, eps_top, eps_bottom, modes[-1],
                max_order,
            )
        )  # fmt: skip

    order = -1
    xs, alpha, directionality, q, jones = [], [], [], [], []
    for start, period, mode in zip(starts, periods, modes, strict=True):
        if order not in mode.emission_deg:
            raise ValueError("a tooth's leaky mode does not radiate its −1st order upward")
        local = (np.arange(samples_per_tooth) + 0.5) / samples_per_tooth * period + start
        xs.append(local)
        alpha.append(np.full(samples_per_tooth, mode.alpha_per_um))
        directionality.append(np.full(samples_per_tooth, mode.directionality))
        q.append(np.full(samples_per_tooth, mode.bloch_index.real - wavelength_um / period))
        jones.append(np.tile(np.asarray(mode.top_polarization[order]), (samples_per_tooth, 1)))
    n_top = float(np.sqrt(np.real(eps_top)))
    te, tm, offset, angle = coupling_from_profile(
        np.concatenate(xs), np.concatenate(alpha), np.concatenate(directionality),
        np.concatenate(q), np.concatenate(jones), wavelength_um, waist_um, fiber_angle_deg, n_top,
    )  # fmt: skip
    return ApodizedCoupler(
        efficiency=te + tm, efficiency_te=te, efficiency_tm=tm, fiber_offset_um=offset,
        fiber_angle_deg=angle, modes=tuple(modes),
    )  # fmt: skip
