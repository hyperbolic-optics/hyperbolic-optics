# Resonances, Modes, Beams and Couplers

These tools go beyond plane-wave reflection and transmission to the objects
devices are built from:

- **band-edge modes** of photonic-crystal slabs (PCSELs);
- **guided modes** of anisotropic waveguides;
- **finite Gaussian beams**;
- **grating couplers** from a waveguide to a fibre.

All of them work with full 3×3 tensors in any layer.

## How resonances are found

A resonance is a field that exists with nothing incident: a pole of the
scattering matrix. Pick a plane inside the structure, and let `A` be the
scattering matrix of everything above it and `B` of everything below. The
downward and upward amplitudes there satisfy `f = A₀₁·b` and `b = B₁₀·f`, so
at a resonance an eigenvalue of `A₀₁·B₁₀` equals 1.

- **Outgoing branch.** The half-spaces' waves are continued analytically to
  their outgoing branch. That way, leaky resonances are found as well as
  bound ones: those at complex frequency, radiating into the open
  diffraction channels, and those at complex propagation constant, radiating
  as they go.
- **Contour search.** Roots are found in the complex plane by Beyn's
  contour-integral method (W.-J. Beyn, Linear Algebra Appl. 436, 3839
  (2012)). It finds every root inside a contour at once, and degenerate pairs
  appear as two roots. Each root is then refined by the secant method.
- **Tiling.** The contour is a chain of ellipses along the real range. None
  reaches past the range's ends, because an end may be a branch point (a
  cladding's index, for a guided mode).

## Band-edge modes of photonic-crystal slabs

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from hyperbolic_optics import rcwa2d, resonances

eye = jnp.eye(3, dtype=complex)
holes = rcwa2d.PatternedLayer(
    0.5, 3.5**2 * eye, (rcwa2d.Shape(rcwa2d.ellipse((0.5, 0.5), (0.15, 0.15), 64), eye),)
)  # a membrane with air holes, lengths in units of the lattice constant
modes = resonances.band_edge_modes(
    [holes], rcwa2d.Lattice.square(1.0), frequency_range=(0.33, 0.37),
    eps_top=1.0, eps_bottom=1.0, max_order=(4, 4), truncation="circular",
    max_loss=0.01, probes=24,
)
for m in modes:
    print(m.frequency, m.q_factor, m.multiplicity)   # a/λ (complex), Q, degeneracy
```

These are the modes a PCSEL lases on. M. N. Robinson's thesis computes them
with 3D coupled-wave theory for square and triangular lattices (*Developing
coupled wave theory for photonic crystal surface emitting lasers*, Queen's
University Belfast, 2026). Here they come from the full-vector RCWA:

- on any lattice;
- with any stack above and below;
- with any tensor in any layer;
- at any Bloch wavevector (`bloch`, in units of 2π/a; Γ by default).

For the square-lattice membrane above (r = 0.15a, t = 0.5a, n = 3.5), the
TE-like band edge at Γ is:

| a/λ | Q | multiplicity | |
|---|---|---|---|
| 0.3405112 | > 10¹⁵ | 1 | dark |
| 0.3445895 | > 10¹⁵ | 1 | dark |
| 0.363832 − 0.000394i | 462 | 2 | bright |

Two checks on this result:

- **The spectrum sees only the bright pair.** The AAA rational fit
  (`resonances.spectrum_poles`) of the real-frequency normal-incidence
  reflection has a pole at the bright pair's frequency, to all seven digits,
  and none at the dark modes. Symmetry forbids the dark modes from coupling
  to a normally incident wave.
- **The triangular lattice has the C6v pattern at Γ.** With Laurent's rule
  (whose symmetry is exact) it gives a dark singlet, a dark pair and a
  bright pair. The normal-vector rule splits the pairs by up to 2e-4 in a/λ,
  its sampling error on a hexagonal lattice.
- **Off Γ** (`bloch=(0.02, 0)`) the pairs split and the dark modes leak, with
  Q between 10⁴ and 10⁵.

## Guided modes of anisotropic waveguides

```python
from hyperbolic_optics import bcm, dispersion

# x-cut LiNbO3, 0.6 µm, optic axis in the plane at 45° to the propagation direction
rotation = bcm.rotation_matrix(0.0, np.radians(90.0), np.radians(45.0))
eps = bcm.rotate_tensor(dispersion.at_wavelength("LiNbO3", 1.55), rotation)[None]
thickness_um = jnp.array([0.6])
modes = resonances.slab_modes(eps, thickness_um, wavelength_um=1.55,
                              eps_top=1.0, eps_bottom=1.444**2)
modes[0].effective_index, modes[0].loss_db_per_cm
z_um, fields = resonances.slab_mode_fields(eps, thickness_um, 1.55, 1.0, 1.444**2,
                                           modes[0].effective_index)
```

- **Full-vector.** In a rotated crystal (an x-cut LiNbO₃ film with its axis
  at 45°, say) the modes are hybrid, neither TE nor TM.
- **Lossy layers.** These give a complex `n_eff`.
- **Propagation direction.** Modes propagate along x; rotate the tensors to
  change direction.
- **Profiles.** `slab_mode_fields` returns E and H through the stack and
  into the claddings, normalised to unit power along x. It marches from an
  inner plane with bounded scattering matrices, so thick or strongly
  evanescent claddings stay stable.

Validation (`tests/test_resonances.py`):

- **Closed forms.** Even and odd TE and TM modes of an isotropic film, and
  the uniaxial TM case (ε_x ≠ ε_z), agree with the slab dispersion relations
  to 1e-9.
- **Loss.** A lossy mode's Im n_eff equals the absorbed power integrated
  over its profile, `¼∫E†ε_aE dz / ∫S_x dz`, to 1e-6.
- **Hybrid modes.** A hybrid x-cut LiNbO₃ mode is confirmed by the NumPy
  payload engine. Prism-coupled through a 1.4 µm air gap, p→s conversion
  peaks at the mode's effective index, to 5e-9.

## Gaussian beams

```python
from hyperbolic_optics import beams

# a 10 µm beam at 40° from glass onto that LiNbO3 film, air below
s_beam = beams.beam_smatrix(eps, thickness_um, 1.55, np.radians(40.0), 1.5**2, 1.0,
                            waist_um=10.0)                        # [4, 4] over beam modes
x, y, field = beams.reflected_beam(eps, thickness_um, 1.55, np.radians(40.0), 1.5**2, 1.0,
                                   10.0, polarization=np.array([1, 1j]) / np.sqrt(2))
dx, dy = beams.beam_shift(x, y, field)      # Goos–Hänchen and Imbert–Fedorov shifts, µm
```

- **Construction.** A beam is a superposition of plane waves. Each passes
  through the stack with its own S-matrix, at its own in-plane wavevector,
  in its own p and s.
- **The beam S-matrix.** `beam_smatrix` couples Gaussian beam modes (waist
  w₀) at the four ports, so |S|² is the power carried from one beam mode
  into another. It tends to the plane-wave S-matrix as 1/w₀².
- **Beam effects.** Its departures from the plane-wave result are the beam
  effects: shifts, the angular spread of polarization conversion, and a
  resonance's finite angular width. Anisotropic media make them large and
  asymmetric.

Validation (`tests/test_beams.py`):

- **Goos–Hänchen.** At total internal reflection, the reflected centroid
  moves by Artmann's shift `−dφ/dq_x`, to within the finite-beam correction
  (0.06% at w₀ = 20λ).
- **Coupling loss.** The specular beam mode is reached with the overlap of a
  Gaussian shifted by that amount.
- **Imbert–Fedorov.** The transverse shift vanishes for linear polarization
  and flips sign with the helicity of circular polarization.

## Grating couplers

```python
from hyperbolic_optics import couplers, rcwa

layers = [rcwa.lamellar_layer(2.0**2 * eye, eye, 0.5, 0.2),   # half-etched SiN, in air
          rcwa.uniform_layer(2.0**2 * eye, 0.2)]
coupler = couplers.grating_coupler(layers, period_um=1.0, wavelength_um=1.55,
                                   eps_top=1.0, eps_bottom=1.45**2,
                                   index_range=(1.52, 1.72), length_um=15.0)
coupler.efficiency, coupler.directionality, coupler.emission_deg
model = couplers.grating_coupler_model(layers, 1.0, 1.0, 1.45**2, (1.52, 1.72))
model(wl=np.array([1.55, 1.56]))     # SDict: waveguide@TE ↔ fiber@TE / fiber@TM
```

- **The leaky mode.** A uniform grating makes the waveguide's guided mode
  leaky. `resonances.grating_leaky_modes` finds its complex Bloch wavevector
  from the full-vector RCWA, anisotropic layers included. It gives:
  - the decay rate α;
  - the emission angle of each radiating order;
  - the directionality D (the share of the radiated power going up);
  - the polarization of the emission.
- **Efficiency.** The coupling efficiency is
  `D · (1 − e^{−2αL}) · η`, where η is the overlap of the emitted
  `exp(−αx)` with a fibre mode tilted to the emission angle, at its best
  position. This is the standard leaky-mode model of a uniform grating
  coupler.
- **SAX model.** `grating_coupler_model` is a lumped, magnitude-accurate SAX
  model over wavelength, following the leaky mode from one wavelength to the
  next.
- **Neglected.** The back-reflection at the grating's start and finite-length
  effects are neglected; an FDTD simulation of the finite device is the check
  on those.

For the SiN grating above:

- α = 0.059/µm, emission at 2.5°, and D = 0.455.
- The same leaky mode is a pole of the infinite grating's own reflection and
  transmission spectra at real incidence angles, at the −1st order's complex
  kx, to 2e-6.
- The ratio of the spectrum's transmission and reflection residues there
  gives the same directionality, to 3e-5.
- The overlap's closed form matches quadrature, and its maximum is the
  uniform grating's 80.1%, at 2αw₀ = 1.37.
- An x-cut LiNbO₃ grating emits pure TE with its axis along the grating
  lines, and both polarizations with the axis at 45°.
