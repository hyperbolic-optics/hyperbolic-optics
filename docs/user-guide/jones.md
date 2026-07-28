# Jones Calculus

[Mueller matrices](mueller.md) work at the intensity level: they handle
depolarization but discard absolute phase. `Jones` is the amplitude-level
counterpart — 2×2 complex matrices and 2-component vectors that **preserve
phase**, which is what you want for fully-polarized coherent light.

A structure's reflection coefficients already *are* a Jones matrix:

```
J = [[r_pp, r_ps],
     [r_sp, r_ss]]     rows = output p/s, columns = input p/s
```

## Basic usage

Every snippet on this page follows on from:

```python
payload = {
    "ScenarioData": {"type": "Incident"},
    "Layers": [
        {"type": "Ambient Incident Layer", "permittivity": 50.0},
        {
            "type": "Semi Infinite Anisotropic Layer",
            "material": "Calcite",
            "rotationY": 90,
        },
    ],
}
```

```python
from hyperbolic_optics.structure import Structure
from hyperbolic_optics.jones import Jones

structure = Structure()
structure.execute(payload)

jones = Jones(structure)
jones.set_incident_polarization("linear", angle=0)    # p-polarized
jones.add_optical_component("sample")
jones.add_optical_component("linear_polarizer", 90)   # crossed analyzer
extinction = jones.get_intensity()
```

Ideal components (`linear_polarizer`, `quarter_wave_plate`, `half_wave_plate`,
`rotator`) are angle-independent and broadcast over any scenario sweep.

## Eigenpolarizations and exceptional points

The eigenvectors of `J` are the states that reflect **without polarization
conversion** — `J·v = λ·v`, so the state comes back scaled by a complex `λ` but
otherwise unchanged.

```python
data = jones.eigenpolarizations()
data["eigenvalues"]          # [..., 2]
data["eigenpolarizations"]   # [..., 2, 2], columns are the eigenvectors
data["discriminant"]         # -> 0 at an exceptional point
data["eigenvector_overlap"]  # -> 1 at an exceptional point
```

Index 0 is the **more p-like** eigenvector, index 1 the more s-like. That
labelling matters: `np.linalg.eig` returns them in no particular order, so
without it the two swap arbitrarily across a sweep and any per-channel map
inherits seams that are not physical. Ordering by `|λ|` — the obvious
alternative — is much worse, seaming along every `|λ₀| = |λ₁|` contour.

A reflection Jones matrix from a lossy anisotropic sample is generally
**non-normal**: its eigenvectors are not orthogonal and can *coalesce* at an
exceptional point, where the matrix stops being diagonalizable.

```python
found = jones.find_exceptional_points()
found["ep_index"]       # strongest candidate on the grid
found["near_ep"]        # boolean mask on eigenvector overlap
found["defectiveness"]  # scale-free, 0 where the eigenvectors truly coalesce
```

!!! note "No labelling is globally continuous near an EP"
    Encircling an exceptional point exchanges the two eigenvalue sheets, so
    every labelling scheme owns at least one branch cut per EP. Ordering by
    polarization character puts its cuts on the EP chains — where a
    discontinuity is physically correct — and nowhere else.

## Ellipsometry

```python
jones.ellipsometric_parameters()    # psi and delta
```

## Composing elements

```python
from hyperbolic_optics.jones import compose_jones

analyzer = jones.linear_polarizer(90)
total = compose_jones(structure, analyzer)   # beam order
```

Two `Structure` elements must share the same `kx` and frequency grids — the
in-plane wavevector is conserved, so composing samples evaluated at different
angles is not a meaningful product, and the guard rejects it.

## Bridging to Mueller

```python
jones.to_mueller()
```

Reuses the same Jones-to-Mueller transform `Mueller` does, so the two
formalisms stay consistent — including the handedness convention for circular
polarization.
