"""Differentiable permittivity of the built-in materials, in JAX.

The models of :mod:`hyperbolic_optics.materials`, evaluated in JAX so that
wavelength gradients include the dispersion. Parameters come from the same
``material_params.json``; the tests check the two agree.

Models:

- transparency-window crystals (LiNbO₃, TiO₂, quartz, calcite, sapphire) and
  PIC materials (SiO₂, Si₃N₄, Si): lossless Sellmeier fits per principal axis;
- uniaxial and biaxial: the factorised TO–LO form
  ``ε(ω) = ε∞ ∏ᵢ (ω_LO,i² − ω² − iωγ_LO,i) / (ω_TO,i² − ω² − iωγ_TO,i)`` per
  principal axis, with a uniaxial tensor ``diag(ε_o, ε_o, ε_e)`` (optic axis
  along z);
- monoclinic (β-Ga₂O₃): Lorentz Bu modes whose dipoles lie at angles αᵢ in the
  x-y plane, which gives the xy component, plus Au modes along z.

Frequencies are in cm⁻¹ (:func:`at_wavelength` takes µm); tensors are in the
crystal frame (rotate with :func:`hyperbolic_optics.bcm.rotate_tensor`).
:func:`pockels_tensor` adds the Pockels effect of an applied field.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from hyperbolic_optics import bcm
from hyperbolic_optics.materials import (
    _MATERIAL_REGISTRY,
    BiaxialMaterial,
    MonoclinicMaterial,
    SellmeierMaterial,
    UniaxialMaterial,
)

__all__ = [
    "LITHIUM_NIOBATE",
    "at_wavelength",
    "frequency_range",
    "materials",
    "permittivity",
    "pockels_tensor",
    "trigonal_3m",
]


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


# contracted (Voigt) index I → (i, j): 1 xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def trigonal_3m(r13: float, r22: float, r33: float, r51: float) -> jax.Array:
    """The electro-optic tensor ``rᵢⱼₖ`` ``[3, 3, 3]`` (pm/V) of a point-group-3m crystal.

    In the crystal frame: z the optic (c) axis, x perpendicular to a mirror
    plane. In contracted notation ``r = [[0, −r22, r13], [0, r22, r13],
    [0, 0, r33], [0, r51, 0], [r51, 0, 0], [−r22, 0, 0]]``.
    """
    contracted = jnp.array(
        [[0.0, -r22, r13], [0.0, r22, r13], [0.0, 0.0, r33],
         [0.0, r51, 0.0], [r51, 0.0, 0.0], [-r22, 0.0, 0.0]]
    )  # fmt: skip
    r = jnp.zeros((3, 3, 3))
    for index, (i, j) in enumerate(_VOIGT):
        r = r.at[i, j, :].set(contracted[index])
        r = r.at[j, i, :].set(contracted[index])
    return r


#: Congruent LiNbO₃, unclamped, measured at 633 nm (R. S. Weis and T. K. Gaylord,
#: Appl. Phys. A 37, 191 (1985)), pm/V. Pass your own tensor for other wavelengths
#: or for clamped (high-frequency) values.
LITHIUM_NIOBATE = trigonal_3m(r13=8.6, r22=3.4, r33=30.8, r51=28.0)


def pockels_tensor(
    eps_crystal: jax.Array,
    r_pm_per_v: jax.Array,
    field_v_per_um: jax.Array,
    rotation: jax.Array | None = None,
) -> jax.Array:
    """The permittivity under an applied field, in the lab frame.

    Args:
        eps_crystal: The unperturbed tensor in the crystal frame ``[..., 3, 3]``
            (e.g. ``dispersion.at_wavelength("LiNbO3", 1.55)``).
        r_pm_per_v: The electro-optic tensor ``[3, 3, 3]`` in the crystal
            frame (e.g. :data:`LITHIUM_NIOBATE`).
        field_v_per_um: The applied field in the lab frame ``[..., 3]``, V/µm.
        rotation: The crystal's orientation, ``R`` with ``ε_lab = R·ε_crystal·Rᵀ``
            (:func:`hyperbolic_optics.bcm.rotation_matrix`); identity if omitted.

    The Pockels effect: the impermeability ``η = ε⁻¹`` changes by
    ``Δηᵢⱼ = Σₖ rᵢⱼₖ Eₖ`` (``1e-6`` per pm/V · V/µm).

    Returns:
        ``ε(E)`` in the lab frame ``[..., 3, 3]``; differentiable in every argument.
    """
    eps_crystal = jnp.asarray(eps_crystal, dtype=complex)
    rotation = jnp.eye(3) if rotation is None else jnp.asarray(rotation)
    field_crystal = jnp.einsum("...ji,...j->...i", rotation, jnp.asarray(field_v_per_um))
    delta_eta = 1e-6 * jnp.einsum("ijk,...k->...ij", jnp.asarray(r_pm_per_v), field_crystal)
    eta = jnp.linalg.inv(eps_crystal) + delta_eta
    return bcm.rotate_tensor(jnp.linalg.inv(eta), rotation)
