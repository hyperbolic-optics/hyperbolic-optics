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
    "GratingCoupler",
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
