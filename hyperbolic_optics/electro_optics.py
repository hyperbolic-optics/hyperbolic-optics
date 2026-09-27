"""Electro-optics: the Pockels effect in anisotropic crystals, and modulator figures.

An applied field ``E`` (V/µm) changes a crystal's impermeability
``η = ε⁻¹`` linearly (the Pockels effect):

    Δηᵢⱼ = Σₖ rᵢⱼₖ Eₖ,

with ``r`` the electro-optic tensor in pm/V (so ``r·E`` is dimensionless at
``1e-6`` per pm/V·V/µm). :func:`pockels_tensor` gives ``ε(E) = (η + Δη)⁻¹`` in the
lab frame, for a crystal in any orientation and a field in any direction, and
is differentiable in the field -- a voltage is then an ordinary design
parameter for every solver in the package.

:func:`mode_index_shift` gives a guided mode's effective-index change from its
profile (:func:`hyperbolic_optics.resonances.slab_mode_fields`), to first order,

    Δn_eff = ¼ ∫ E†·Δε·E dz / ∫ S_x dz,

and :func:`vpi_length` the half-wave voltage–length product of a phase
shifter from it.

Coefficients: :data:`LITHIUM_NIOBATE` (point group 3m, congruent, unclamped at
633 nm: r₃₃ = 30.8, r₁₃ = 8.6, r₂₂ = 3.4, r₅₁ = 28 pm/V, from R. S. Weis and
T. K. Gaylord, Appl. Phys. A 37, 191 (1985)). Pass your own tensor for other
crystals (:func:`trigonal_3m` builds one from its four coefficients), or for
LiNbO₃ at other wavelengths and clamping conditions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hyperbolic_optics import bcm

__all__ = [
    "LITHIUM_NIOBATE",
    "mode_index_shift",
    "pockels_tensor",
    "trigonal_3m",
    "vpi_length",
]

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


#: Congruent LiNbO₃, unclamped, 633 nm (Weis & Gaylord 1985), pm/V.
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

    Returns:
        ``ε(E)`` in the lab frame ``[..., 3, 3]``; differentiable in every argument.
    """
    eps_crystal = jnp.asarray(eps_crystal, dtype=complex)
    rotation = jnp.eye(3) if rotation is None else jnp.asarray(rotation)
    field_crystal = jnp.einsum("...ji,...j->...i", rotation, jnp.asarray(field_v_per_um))
    delta_eta = 1e-6 * jnp.einsum("ijk,...k->...ij", jnp.asarray(r_pm_per_v), field_crystal)
    eta = jnp.linalg.inv(eps_crystal) + delta_eta
    return bcm.rotate_tensor(jnp.linalg.inv(eta), rotation)


def mode_index_shift(z_um, fields, thickness_um, delta_eps) -> jax.Array:
    """First-order effective-index change of a guided mode, ``¼∫E†·Δε·E / ∫S_x``.

    Args:
        z_um: Depths of the mode's samples ``[Z]`` (from
            :func:`hyperbolic_optics.resonances.slab_mode_fields`).
        fields: ``(Ex, Ey, Ez, Hx, Hy, Hz)`` there ``[Z, 6]``.
        thickness_um: The layers' thicknesses ``[L]``.
        delta_eps: The perturbation in each layer ``[L, 3, 3]`` (uniform in
            it; zero where there is no field, and in the claddings).

    Returns:
        ``Δn_eff`` (complex: its imaginary part is a change of loss).

    Each layer's term is integrated over its own samples only (they include
    both faces), so the step in ``Δε`` at a layer's edge costs no accuracy.
    """
    z = jnp.asarray(z_um)
    order = jnp.argsort(z, stable=True)
    z, fields = z[order], jnp.asarray(fields)[order]
    e = fields[:, :3]
    flux = 0.5 * jnp.real(
        fields[:, 1] * jnp.conj(fields[:, 5]) - fields[:, 2] * jnp.conj(fields[:, 4])
    )
    thickness = jnp.asarray(thickness_um)
    tops = jnp.concatenate([jnp.zeros(1), jnp.cumsum(thickness)])
    total = 0.0
    for i in range(thickness.shape[0]):
        inside = (z >= tops[i]) & (z <= tops[i + 1])
        density = jnp.einsum("zi,ij,zj->z", jnp.conj(e), jnp.asarray(delta_eps)[i], e)
        # a trapezoid over the layer's samples: segments with both ends inside
        weights = jnp.where(inside[1:] & inside[:-1], jnp.diff(z), 0.0)
        total = total + jnp.sum(0.5 * weights * (density[1:] + density[:-1]))
    return 0.25 * total / jnp.trapezoid(flux, z)


def vpi_length(index_per_volt: float, wavelength_um: float) -> float:
    """Half-wave voltage–length product ``Vπ·L`` in V·cm of a phase shifter.

    Args:
        index_per_volt: ``dn_eff/dV`` (1/V): e.g. :func:`mode_index_shift` for a
            field ``V/g·Γ`` per volt, with ``g`` the electrode gap and ``Γ`` the
            RF field's overlap factor.
        wavelength_um: Vacuum wavelength.
    """
    return float(wavelength_um * 1e-4 / (2 * abs(index_per_volt)))
