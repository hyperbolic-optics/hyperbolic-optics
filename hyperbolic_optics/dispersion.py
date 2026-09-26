"""Differentiable permittivity of the built-in materials, in JAX.

The materials of :mod:`hyperbolic_optics.materials` -- hBN, α-MoO₃, calcite,
sapphire, quartz, SiC, AlN, GaN and β-Ga₂O₃ -- are phonon-polariton media whose
Reststrahlen bands are where they turn hyperbolic. Their NumPy classes evaluate
the permittivity on the host, which breaks the chain of derivatives: a tensor
taken from them carries no gradient with respect to frequency. This module
evaluates the same oscillator models in JAX, so ``jax.grad`` of a
:mod:`~hyperbolic_optics.bcm` or :mod:`~hyperbolic_optics.rcwa` result with
respect to wavelength includes the material's dispersion.

The parameters are read from each material's own class (and so from
``material_params.json``), so the two implementations cannot drift apart; the
test suite checks they agree to rounding.

Models:

- transparency-window crystals (LiNbO₃, TiO₂, quartz, calcite, sapphire) and
  PIC materials (SiO₂, Si₃N₄, Si): lossless Sellmeier fits per principal axis;
- uniaxial and biaxial: the factorised TO–LO form
  ``ε(ω) = ε∞ ∏ᵢ (ω_LO,i² − ω² − iωγ_LO,i) / (ω_TO,i² − ω² − iωγ_TO,i)`` per
  principal axis, with a uniaxial tensor ``diag(ε_o, ε_o, ε_e)`` (optic axis
  along z);
- monoclinic (β-Ga₂O₃): Lorentz Bu modes whose dipoles lie at angles αᵢ in the
  x-y plane, which gives the xy component, plus Au modes along z.

Frequencies are in cm⁻¹, as elsewhere in the package; :func:`at_wavelength`
takes µm. The tensors are in the crystal frame; rotate them with
:func:`hyperbolic_optics.bcm.rotate_tensor`.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from hyperbolic_optics.materials import (
    _MATERIAL_REGISTRY,
    BiaxialMaterial,
    MonoclinicMaterial,
    SellmeierMaterial,
    UniaxialMaterial,
)

__all__ = ["at_wavelength", "frequency_range", "materials", "permittivity"]


def materials() -> list[str]:
    """Names accepted by :func:`permittivity` (the payload API's material names)."""
    return sorted(_MATERIAL_REGISTRY)


@lru_cache(maxsize=None)
def _model(name: str):
    if name not in _MATERIAL_REGISTRY:
        raise ValueError(f"Unknown material {name!r}; choose from {materials()}")
    material = _MATERIAL_REGISTRY[name]()
    params = material.permittivity_parameters()
    if isinstance(material, SellmeierMaterial):  # before UniaxialMaterial: a subclass
        return "sellmeier", params
    if isinstance(material, BiaxialMaterial):
        return "biaxial", {axis: _oscillators(params[axis]) for axis in ("x", "y", "z")}
    if isinstance(material, UniaxialMaterial):
        return "uniaxial", {
            "ordinary": _oscillators(params["ordinary"]),
            "extraordinary": _oscillators(params["extraordinary"]),
        }
    if isinstance(material, MonoclinicMaterial):
        bu, au = params["Bu"], params["Au"]
        return "monoclinic", {
            "bu": {k: np.asarray(bu[k]) for k in ("amplitude", "omega_tn", "gamma_tn", "alpha_tn")},
            "bu_high": {k: complex(bu["high_freq"][k]) for k in ("xx", "xy", "yy")},
            "au": {k: np.asarray(au[k]) for k in ("amplitude", "omega_tn", "gamma_tn")},
            "au_high": complex(au["high_freq"]),
        }
    raise NotImplementedError(f"No JAX model for {name!r} ({type(material).__name__})")


def _oscillators(axis: dict) -> dict:
    return {
        "high_freq": complex(np.asarray(axis["high_freq"])),
        **{
            k: np.asarray(axis[k], dtype=complex)
            for k in ("omega_tn", "gamma_tn", "omega_ln", "gamma_ln")
        },
    }


def _factorised(axis: dict, w: jax.Array) -> jax.Array:
    """``ε∞ ∏ (ω_LO² − ω² − iωγ_LO) / (ω_TO² − ω² − iωγ_TO)``, broadcasting over ``w``."""
    w = w[..., None]
    top = axis["omega_ln"] ** 2 - w**2 - 1j * w * axis["gamma_ln"]
    bottom = axis["omega_tn"] ** 2 - w**2 - 1j * w * axis["gamma_tn"]
    return axis["high_freq"] * jnp.prod(top / bottom, axis=-1)


def _lorentz(modes: dict, w: jax.Array) -> jax.Array:
    """Per-mode ``A² / (ω_TO² − ω² − iωγ)``, ``[..., modes]``."""
    w = w[..., None]
    return modes["amplitude"] ** 2 / (modes["omega_tn"] ** 2 - w**2 - 1j * w * modes["gamma_tn"])


def permittivity(material: str, wavenumber_cm: jax.Array) -> jax.Array:
    """Crystal-frame permittivity tensor of a built-in material, ``[..., 3, 3]``.

    Args:
        material: A name from :func:`materials`, e.g. ``"hBN"``.
        wavenumber_cm: Frequency in cm⁻¹, any shape; differentiable.
    """
    kind, model = _model(material)
    w = jnp.asarray(wavenumber_cm, dtype=float)
    zero = jnp.zeros(w.shape, dtype=complex)
    if kind == "sellmeier":
        lam_sq = (1e4 / w) ** 2

        def sellmeier(axis):
            eps = axis["constant"] + zero
            for a, b in axis["resonances"]:
                eps = eps + a * lam_sq / (lam_sq - b)
            for d, e in axis["poles"]:
                eps = eps + d / (lam_sq - e)
            return eps

        ordinary = sellmeier(model["ordinary"])
        return _diagonal(ordinary, ordinary, sellmeier(model["extraordinary"]))
    if kind == "uniaxial":
        ordinary = _factorised(model["ordinary"], w)
        diagonal = (ordinary, ordinary, _factorised(model["extraordinary"], w))
        return _diagonal(*diagonal)
    if kind == "biaxial":
        return _diagonal(*(_factorised(model[axis], w) for axis in ("x", "y", "z")))
    bu = _lorentz(model["bu"], w)
    alpha = model["bu"]["alpha_tn"] * np.pi / 180.0  # stored complex, as the NumPy model uses it
    xx = model["bu_high"]["xx"] + jnp.sum(bu * np.cos(alpha) ** 2, axis=-1)
    xy = model["bu_high"]["xy"] + jnp.sum(bu * np.sin(alpha) * np.cos(alpha), axis=-1)
    yy = model["bu_high"]["yy"] + jnp.sum(bu * np.sin(alpha) ** 2, axis=-1)
    zz = model["au_high"] + jnp.sum(_lorentz(model["au"], w), axis=-1)
    rows = [[xx, xy, zero], [xy, yy, zero], [zero, zero, zz]]
    return jnp.stack([jnp.stack(row, axis=-1) for row in rows], axis=-2)


def _diagonal(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    return jnp.stack([a, b, c], axis=-1)[..., None, :] * jnp.eye(3, dtype=complex)


def at_wavelength(material: str, wavelength_um: jax.Array) -> jax.Array:
    """:func:`permittivity` at a vacuum wavelength in µm (``1e4 / λ`` cm⁻¹)."""
    return permittivity(material, 1e4 / jnp.asarray(wavelength_um, dtype=float))


def frequency_range(material: str) -> tuple[float, float]:
    """The material's frequency window in cm⁻¹ -- its Reststrahlen fit window for the
    phonon models, its published validity for the Sellmeier ones."""
    material_obj = _MATERIAL_REGISTRY[material]()
    return float(material_obj.frequency[0]), float(material_obj.frequency[-1])
