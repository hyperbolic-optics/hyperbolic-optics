# Guided Modes, Pockels Effect and Beam Shifts

## Guided modes of anisotropic stacks

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from hyperbolic_optics import bcm, dispersion, modes

# x-cut LiNbO3, 0.6 µm, optic axis in the plane at 45° to the propagation direction
rotation = bcm.rotation_matrix(0.0, np.radians(90.0), np.radians(45.0))
eps = bcm.rotate_tensor(dispersion.at_wavelength("LiNbO3", 1.55), rotation)[None]
thickness_um = jnp.array([0.6])
found = modes.slab_modes(eps, thickness_um, wavelength_um=1.55, eps_top=1.0,
                         eps_bottom=1.444**2)
z_um, fields = modes.slab_mode_fields(eps, thickness_um, 1.55, 1.0, 1.444**2,
                                      found[0].effective_index)
```

- Any 3 × 3 ε and μ in any layer. In a rotated crystal the modes are hybrid,
  neither TE nor TM. Lossy layers give a complex `n_eff`.
- Modes propagate along x; rotate the tensors to change direction.
- Indefinite tensors need an explicit `index_range` above both cladding
  indices. Unresolved contours raise rather than return a partial list, and
  modes just above cutoff are found (the default search starts 1e-6 above it).
- `slab_mode_fields` returns E and H through the stack and into the
  claddings, normalised to unit power along x.

Tests: `tests/test_modes.py` (analytic slab modes, absorbed power, prism
coupling in the NumPy engine).

## The Pockels effect

`dispersion.pockels_tensor` gives ε under an applied field, for a crystal in
any orientation and a field in any direction. It is differentiable, so a
voltage is a design parameter for every solver here. A mode's index shift is
the change in its root:

```python
ln = dispersion.at_wavelength("LiNbO3", 1.55)
x_cut = bcm.rotation_matrix(0.0, np.radians(90.0), np.radians(90.0))   # axis along y


def index(volts_per_um):
    field = jnp.array([0.0, volts_per_um, 0.0])
    film = dispersion.pockels_tensor(ln, dispersion.LITHIUM_NIOBATE, field, x_cut)[None]
    return modes.slab_modes(film, thickness_um, 1.55, 1.0, 1.444**2)[0].effective_index


dn_per_v_per_um = (index(0.5) - index(-0.5)).real
```

`LITHIUM_NIOBATE` holds the unclamped coefficients measured at 633 nm (Weis &
Gaylord 1985). `trigonal_3m(r13, r22, r33, r51)` builds a tensor from other
data, and any 3 × 3 × 3 tensor is accepted.

## Beam shifts

```python
from hyperbolic_optics import beams

x, y, field = beams.reflected_beam(eps, thickness_um, 1.55, np.radians(40.0), 1.5**2, 1.0,
                                   waist_um=10.0, polarization=np.array([1, 1j]) / np.sqrt(2))
dx, dy = beams.beam_shift(x, y, field)      # Goos–Hänchen and Imbert–Fedorov shifts, µm
```

A Gaussian beam is a superposition of plane waves, each passing through the
stack with its own S-matrix. At total internal reflection the Goos–Hänchen
shift matches Artmann's `−dφ/dq_x`. The Imbert–Fedorov shift flips sign with
the helicity (`tests/test_beams.py`).
