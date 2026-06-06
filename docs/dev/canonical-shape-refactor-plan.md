# Refactor & Roadmap Plan: Canonical Shapes → Stable Transmission

> **Status:** Planning / handoff document.
> **Audience:** A developer or coding agent picking this up cold.
> **Branch:** All work belongs on `claude-havoc` (do **not** push to `main`).
> **Verified against:** HEAD `c011b5b` (v0.2.3). Every `file:line` reference and
> shape claim below was checked against the source on this commit.
> **Why this doc exists:** It captures the strategic rationale, the architectural
> diagnosis (with `file:line` references), the target design, and a phased,
> test-guarded execution plan so the work can be carried out incrementally and
> safely.

---

## 0. How to use this document

Read sections 1–4 to understand *what* and *why*. Execute sections 7–11 in
order — they are phased deliberately so each step is guarded by tests written in
the step before it. Section 12 records decisions already made; section 13 lists
the questions still open for the maintainer (Mark).

**Status of work in this clone:** *Phases 0 and 1 are DONE.* The golden-master
harness (`tests/golden/`, `tests/test_golden.py`, 12 snapshots) is committed and
green, and the canonical `[A, B, F]` refactor has landed: `Wave` is mode-free,
`hyperbolic_optics/axes.py` enforces the convention, and all 12 goldens remain
bit-identical. See [`phase-0-1-execution-plan.md`](phase-0-1-execution-plan.md)
for as-built detail. Next up is **Phase 2** (mode-sorting hardening).

**Golden rule of this whole effort:** Phases 1–2 are *pure refactors*. They must
not change a single reflection-coefficient value. Phase 0 builds the safety net
that proves this. Do not start Phase 1 until Phase 0 is green.

**Phase status at a glance:**

| Phases | Nature | Status |
|---|---|---|
| **0–2** | Safety net + pure refactor + physics hardening | **Actionable now** |
| **3–4** | Scattering-matrix engine + new physics (transmission/absorption/fields) | **Roadmap** — new capability, approve separately |

---

## 1. Project context

`hyperbolic-optics` is a published, packaged Python implementation of the
**4×4 transfer-matrix (Berreman) method** for light in anisotropic / hyperbolic
stratified media. It computes reflection coefficients and Mueller-matrix
polarimetry for multilayer stacks driven by JSON payloads.

- ~5,100 lines across `hyperbolic_optics/`.
- On PyPI; Zenodo DOI; mkdocs docs; CI/CD; ~93 tests.
- Backs a real paper: *Optical footprint of ghost and leaky hyperbolic
  polaritons*, Photonics Research 13, 2291–2305 (2025).

**Core pipeline** (per layer): build Berreman Δ-matrix → eigen-decompose into 4
partial waves → sort into forward/backward → compute fields + Poynting vectors →
classify p vs s → assemble layer transfer matrix. The structure multiplies all
layer matrices and extracts `r_pp, r_ps, r_sp, r_ss`.

**Key files:**

| File | Role |
|---|---|
| `hyperbolic_optics/scenario.py` | Builds angle/frequency arrays per scenario type |
| `hyperbolic_optics/materials.py` | Permittivity/permeability tensors (Lorentz model) |
| `hyperbolic_optics/anisotropy_utils.py` | Euler-angle tensor rotation helpers |
| `hyperbolic_optics/layers.py` | Layer classes + `LayerFactory`; tensor rotation orchestration |
| `hyperbolic_optics/waves.py` | The Berreman/eigen/transfer-matrix core |
| `hyperbolic_optics/structure.py` | Orchestration; reflection extraction |
| `hyperbolic_optics/mueller.py` | Jones → Mueller polarimetry |

**Scenario types** (`scenario.py`): `Simple` (single point), `Incident`
(kx–ω), `Azimuthal` (β–ω), `Dispersion` (kx–ky), `FullSweep` (3D).

**Layer types** (`LayerFactory`, `layers.py:706`):
`Ambient Incident Layer` (prism), `Isotropic Middle-Stack Layer` (air gap),
`Crystal Layer` (finite anisotropic), `Semi Infinite Anisotropic Layer`,
`Semi Infinite Isotropic Layer` (exit).

---

## 2. Strategic rationale — why this work, and why now

The repository feels stagnant because the **next scientifically valuable
feature — transmission, layer-resolved absorption, and field profiling — was
attempted and abandoned** (GitHub issue #30, priority-high). The maintainer
could not get transmission to behave for surface-polariton / `kx/k0 > 1` cases.

The strategic thesis:

1. **The differentiator is the niche, not the method.** There are many
   transfer-matrix codes (pyGTM, PyLlama, py_matrix, RCWA family). What makes
   *this* package distinct is **hyperbolic-material focus + full Mueller
   polarimetry + the specific sweep geometries** polariton researchers use.
   To *own* that niche it must answer "what happens *inside* the stack"
   (absorption, fields), not just "what reflects off it".

2. **Transmission/absorption is the highest-leverage direction** — it is the
   genuine frontier, it differentiates from generic TMM codes, it has a
   validation target (Passler 2020, arXiv:2002.03832), and it could justify a
   methods/software-paper update.

3. **But it cannot be built on the current internals.** The `waves.py` array
   handling is implicit and scenario-coupled (section 3). You cannot correctly
   carry per-layer forward/backward field amplitudes — which absorption and
   field profiling require — on top of five different implicit array layouts.
   **So cleanup is on the critical path, not a detour.**

4. **The transfer matrix is the wrong numerical engine for this regime.** It
   carries growing exponentials `exp(+i·kz·d)` that overflow for thick or
   strongly evanescent layers. Hyperbolic media with `kx/k0 > 1` give large
   `Im(kz)` — exactly where T-matrices lose precision. Reflection often survives
   (ratios cancel); transmission and internal fields do not. This is almost
   certainly *why* issue #30 stalled. The **scattering-matrix (S-matrix)
   reformulation** only ever propagates *decaying* exponentials →
   unconditionally stable, and is the natural home for layer-resolved
   amplitudes. This decision is made up front because it changes which T-matrix
   internals are worth polishing (some get replaced).

**Net plan:** lock behaviour with golden tests → make the array handling
canonical and scenario-agnostic → harden the physics of mode sorting → swap the
engine to a scattering matrix → then implement transmission / absorption /
field profiling on the stable substrate.

---

## 3. Root-cause diagnosis — the implicit, conflated array layout

The disorder in `waves.py` has a **single root cause**: the `mode` string
passed to `Wave` conflates two orthogonal concerns —

- **scenario shape** (`Simple` / `Incident` / `Azimuthal` / `Dispersion` /
  `FullSweep`), and
- **layer kind** (the air-gap variants: `airgap`, `simple_airgap`,
  `azimuthal_airgap`, `simple_scalar_airgap`, `full_sweep_airgap`).

That is ~10 string values (`Wave.__init__` docstring lists them, `waves.py:92`;
the air-gap variants are assigned in `AirGapLayer.calculate_mode`,
`layers.py:536–551`), and each triggers bespoke `np.newaxis` surgery in several
places. There is **no single source of truth for what each array axis means**,
so every site that touches the arrays re-derives it and can drift.

**Sites that special-case axis layout today:**

| Location | What it does |
|---|---|
| `waves.py:222–230` (`delta_matrix_calc`) | Reshapes `kx` per mode for broadcasting |
| `waves.py:374–408` (`get_matrix`) | Six branches reshaping `k0` per mode |
| `waves.py:290–306` (`delta_permutations`) | Sets `batch_dims` per mode (no longer permutes — vestigial) |
| `layers.py:445–473` (`PrismLayer.create`) | Per-scenario `newaxis` to align prism matrix |
| `layers.py:536–551` (`AirGapLayer.calculate_mode`) | Maps scenario → air-gap mode string |
| `layers.py:553–578` (`AirGapLayer.create`) | Per-scenario `newaxis` after `Wave` |
| `layers.py:409–412` (`rotate_tensors`) | `swapaxes(0,1)` to fix Azimuthal axis order |
| `layers.py:687–703` (`IsotropicSemiInfiniteLayer.create`) | Per-scenario `newaxis`; **no FullSweep branch** |
| `structure.py:263–275` (`calculate_reflectivity`) | `swapaxes` / `moveaxis` per scenario for output ordering |

**Vestigial scar tissue** (leftovers from an abandoned TensorFlow/GPU era,
issue #35) — these now just return their inputs unchanged or do nothing useful,
and should be deleted:

- `waves.py:151–159` `_get_tensor_shapes_for_mode` — returns inputs unchanged.
- `waves.py:161–173` `_get_poynting_tensor_shapes` — returns inputs unchanged.
- `waves.py:196–205` `mode_reshaping` — thin wrapper over the above.
- `waves.py:435–442` `poynting_reshaping` — thin wrapper over the above.
- `waves.py:290–306` `delta_permutations` — "no longer does permutations"; only
  sets `self.batch_dims`, which **is never read anywhere** (verified by grep).
  `Wave.batch_size` (`waves.py:108`) is likewise set and never read. Both go,
  along with the `delta_permutations()` call at `waves.py:738`.
- `anisotropy_utils.py:149–217` `anisotropy_rotation_all_axes` — **dead code**,
  never called from anywhere. (FullSweep uses `anisotropy_rotation_two_axes`,
  Azimuthal uses `anisotropy_rotation_one_axis`, the rest use
  `anisotropy_rotation_one_value`.) Delete it.

**A concrete latent inefficiency** in `get_matrix` (`waves.py:410–427`):
`eigenvalues_diag` is constructed diagonal by definition
(`waves.py:188–191`), yet every call runs
`np.allclose(exponent[..., off_diag_mask], 0)` over the *entire batched array*
to "check if it's diagonal" — it always is. `scipy.linalg.expm` is effectively
dead code on this path. After the refactor this collapses to `exp` of the
diagonal.

---

## 4. The target design — one canonical shape convention

### 4.1 The insight

The code **already has** a consistent batch ordering hidden underneath the
per-mode reshaping. Every scenario is that one ordering with **different axes
collapsed**, then reordered for plotting at the very end. Make it explicit.

**Canonical layout** (matrix axes are always trailing):

```
[ A , B , F , <matrix axes 3×3 or 4×4> ]
  │   │   │
  │   │   └─ F = frequency points        (material dispersion)
  │   └───── B = azimuthal angle points   (sample rotation)
  └───────── A = incident angle points    (kx)
```

### 4.2 The rule

**Every batched quantity is broadcast to a fixed rank-3 batch `[A, B, F]`
(+ trailing matrix axes). Any axis a scenario does not sweep is size 1, not
absent.** NumPy never materialises size-1 broadcast axes, so this is free in
time and memory.

### 4.3 Canonical shapes per quantity (invariant across all scenarios)

| Quantity | Canonical shape | Varies over |
|---|---|---|
| `kx` | `[A, 1, 1]` | incident angle only |
| `k0` | `[1, 1, F]` | frequency only |
| `eps` / `mu` (rotating crystal) | `[A, B, F, 3, 3]` | whatever it varies over; rest size 1 |
| `eps` / `mu` (static crystal) | `[1, 1, F, 3, 3]` | frequency only |
| `eps` / `mu` (air gap, dispersionless) | `[1, 1, 1, 3, 3]` | nothing |
| every layer transfer matrix | `[A, B, F, 4, 4]` | — |

### 4.4 Per-scenario axis sizes (current behaviour, to be preserved)

Frequency length defaults to **410** (`materials.py:53`, `frequency_length=410`).
Angle counts come from `scenario.py`. The current `eps` shapes below were traced
through `materials.fetch_*` → `rotate_tensors`; they are what the refactor must
reproduce *as the squeezed view of* the canonical `[A,B,F,3,3]` form.

| Scenario | A (incident) | B (azimuth) | F (freq) | current `eps` shape | current `r_pp` shape |
|---|---|---|---|---|---|
| `Simple` | 1 | 1 | 1 | `[3,3]` | scalar |
| `Incident` | 360 | 1 | 410 | `[410,3,3]` | `(410, 360)` |
| `Azimuthal` | 1 | 360 | 410 | `[360,410,3,3]`* | `(410, 360)` |
| `Dispersion` | 180 | 480 | 1 | `[480,3,3]`† | `(180, 480)` |
| `FullSweep` | 180 | 120 | 410 | `[180,120,410,3,3]` | `(410, 180, 120)` |

\* Azimuthal currently needs `swapaxes(0,1)` (`layers.py:409–412`) to fix an
axis order that came out transposed — a direct symptom of the missing
convention.

† Dispersion folds the azimuth (B=480) into `eps` via `rotationZ` (relative
z-rotation = azimuthal array, `calculate_z_rotation`, `layers.py:333–341`) and
the incident sweep (A=180) in via `kx`; the two combine by broadcasting inside
`delta_matrix_calc` to give `r_pp = (180, 480)`.

### 4.5 Why this deletes so much

Because `np.matmul` and `np.linalg.eig` broadcast over leading axes natively,
**the physics code stops caring about scenarios entirely**:

- `Wave` **loses its `mode` parameter completely.** All ~10 mode strings gone.
  It takes `eps[...,3,3]`, `mu[...,3,3]`, `kx`, `k0` and does pure
  trailing-axis physics.
- `AirGapLayer.calculate_mode` (`layers.py:536`) deletes entirely — an air gap
  is just an isotropic finite layer; nothing special.
- All per-mode `kx`/`k0` reshaping branches in `delta_matrix_calc` and
  `get_matrix` delete — `kx=[A,1,1]` and `k0=[1,1,F]` broadcast against
  `[A,B,F,3,3]` automatically.
- All per-scenario `newaxis` in `PrismLayer.create`, `AirGapLayer.create`,
  `IsotropicSemiInfiniteLayer.create` delete — `prism[A,1,1,4,4]` ⊗
  `airgap[1,1,F,4,4]` ⊗ `crystal[A,B,F,4,4]` broadcast-matmul to `[A,B,F,4,4]`
  in the existing `functools.reduce` (`structure.py:229`).
- The `swapaxes` in `rotate_tensors` deletes — build tensors in canonical order.
- The vestigial methods + dead rotation helper (section 3) delete.
- The `allclose` diagonal check in `get_matrix` deletes.

### 4.6 Where scenario knowledge concentrates (exactly two places)

1. **Boundary-in** (in `scenario.py` / layer setup): construct
   `incident_angle → [A,1,1]`, `azimuthal → [1,B,1]`, `frequency`/`k0 → [1,1,F]`,
   and feed rotations/materials so tensors emerge `[A?, B?, F?, 3, 3]`.
2. **Boundary-out** (one presentation function, replacing
   `structure.py:263–275`): **squeeze size-1 axes, then order the survivors as
   `(F, A, B)`.** This single rule reproduces every current output:
   - Incident → `(F, A)` = `(410, 360)` ✓
   - Azimuthal → `(F, B)` = `(410, 360)` ✓
   - Dispersion → `(A, B)` = `(180, 480)` ✓
   - FullSweep → `(F, A, B)` = `(410, 180, 120)` ✓
   - Simple → scalar ✓

### 4.7 Enforcement (decided)

Use **named axis constants + cheap runtime assertions** — no new dependencies.

Create `hyperbolic_optics/axes.py`:

```python
"""Canonical batch-axis convention for all scenario arrays.

Every batched quantity in the pipeline has shape [A, B, F, *matrix_axes],
where un-swept axes are size 1 (never absent).
See docs/dev/canonical-shape-refactor-plan.md.
"""
A, B, F = 0, 1, 2          # incident angle, azimuthal angle, frequency
N_BATCH = 3                # number of leading batch axes

def assert_canonical(arr, matrix_ndim=2, name="array"):
    """Assert `arr` has N_BATCH leading batch axes + `matrix_ndim` trailing."""
    expected = N_BATCH + matrix_ndim
    if arr.ndim != expected:
        raise AssertionError(
            f"{name}: expected {expected}D [A,B,F,...], got {arr.ndim}D {arr.shape}"
        )
```

Call `assert_canonical` at the two boundaries and at `Wave`'s entry. Rationale:
catches layout drift instantly with near-zero overhead; richer options (einops,
jaxtyping shape hints) were considered and rejected as not worth the
dependency/tooling cost at this package size. Revisit later if desired.

---

## 5. Concrete deletion / change hit-list (for Phase 1)

> Guarded by Phase 0 golden tests — every change below must keep them green.

**`hyperbolic_optics/waves.py`**
- Remove `mode` (and dead `magnet`) params from `Wave.__init__`; delete the
  `_setup_tensor_shapes` mode branching and the unused `self.batch_size`.
- Delete `_get_tensor_shapes_for_mode`, `_get_poynting_tensor_shapes`,
  `mode_reshaping`, `poynting_reshaping`, `delta_permutations` (and its call in
  `execute`, `waves.py:738`).
- `delta_matrix_calc`: delete the per-mode `kx` reshaping (222–230); rely on
  canonical `kx=[A,1,1]`.
- `get_matrix`: delete the six per-mode `k0` branches (374–408) and the
  `allclose` diagonal check (410–427); compute `exp` of the diagonal directly,
  then `V · diag · V⁻¹`.
- Keep `wave_sorting`, `get_poynting`, `sort_poynting_indices`,
  `sort_profile_back_to_matrix` (already trailing-axis; scenario-agnostic) — but
  see Phase 2 for hardening `wave_sorting`.

**`hyperbolic_optics/layers.py`**
- `rotate_tensors`: produce canonical `[A?,B?,F,3,3]` directly; delete the
  Azimuthal `swapaxes` (409–412).
- `PrismLayer.create`: emit `[A,1,1,4,4]`; delete per-scenario `newaxis`.
- `AirGapLayer`: delete `calculate_mode`; `create` emits `[1,1,F,4,4]`; delete
  per-scenario `newaxis`. Call `Wave` without a mode.
- `CrystalLayer.create` / `SemiInfiniteCrystalLayer.create`: call `Wave` without
  a mode (they currently pass `self.scenario` as the mode, `layers.py:615,656`);
  ensure tensors are canonical.
- `IsotropicSemiInfiniteLayer.create`: emit canonical shape; **add the missing
  FullSweep path** (currently absent — only Incident/Azimuthal/Dispersion/Simple
  branches exist, 687–703).

**`hyperbolic_optics/anisotropy_utils.py`**
- Delete the unused `anisotropy_rotation_all_axes` (149–217). Ensure the
  surviving helpers emit canonical-order axes (this removes the need for the
  Azimuthal `swapaxes`).

**`hyperbolic_optics/structure.py`**
- `calculate_reflectivity`: replace the per-scenario `swapaxes`/`moveaxis`
  (263–275) with the single squeeze-then-`(F,A,B)` presentation rule (4.6).

**`hyperbolic_optics/scenario.py`**
- Centralise construction of `[A,1,1]` / `[1,B,1]` / `[1,1,F]` arrays here (the
  "boundary-in" point), documented against `axes.py`.

**New:** `hyperbolic_optics/axes.py` (section 4.7).

---

## 6. Phased execution plan (overview)

| Phase | Goal | Risk | Gate to next phase |
|---|---|---|---|
| **0** | Golden numeric safety net | None (additive) | Golden tests pass on current HEAD |
| **1** | Canonical shapes; delete scenario-coupling | Pure refactor | Golden tests still bit-identical |
| **2** | Harden forward/backward mode sorting | Physics | Golden green + new sorting unit tests |
| **3** | Scattering-matrix engine | New capability | R matches T-matrix in stable regime; S survives where T diverges |
| **4** | Transmission, absorption, field profiling | New capability | Energy conservation `R+T+A=1` within tolerance |

---

## 7. Phase 0 — Golden-master regression harness (DO THIS FIRST)

**Purpose:** the existing suite only checks shapes, `iscomplexobj`, physical
bounds (`0 ≤ R ≤ 1`), and non-NaN-ness (e.g. `tests/test_structure.py:41–114`) —
it would *not* catch a refactor that changed `r_pp` from 0.42 to 0.43. Phase 0
freezes the actual complex coefficient arrays so Phases 1–2 can be proven
behaviour-preserving.

**Layout** (under `tests/golden/`):
- `__init__.py` — documents the workflow.
- `payloads.py` — a `PAYLOADS: dict[str, dict]` battery (see below).
- `generate.py` — runs each payload through `Structure().execute(...)`, saves
  `r_pp, r_ps, r_sp, r_ss` to `tests/golden/data/<name>.npz`. Invoked as
  `python -m tests.golden.generate`. **Must be run on a known-good commit
  (current HEAD before Phase 1).**
- `tests/test_golden.py` — parametrized over `PAYLOADS`; loads each `.npz`,
  re-runs the payload, asserts
  `np.testing.assert_allclose(actual, golden, rtol=1e-7, atol=1e-9)` for each
  coefficient.

**Battery coverage** (aim ~12 payloads). Must include every scenario type, the
arbitrary-material path, multi-layer stacks (prism + air gap + crystal +
semi-infinite), `rotationZType` relative *and* static, an isotropic exit layer,
and **at least one hyperbolic `kx/k0 > 1` case** (e.g. `Incident` with a
high-permittivity prism, `permittivity: 50`, into a hyperbolic material like
Quartz/Calcite in its reststrahlen band — large `kx` probes the
evanescent/leaky regime that Phase 3/4 must get right).

Reuse the fixtures in `tests/conftest.py` as starting points (they already
define valid `simple`, `incident`, `azimuthal`, `dispersion`, and
arbitrary-material payloads). Valid layer-type strings and required keys:

- `"Ambient Incident Layer"` → `permittivity`.
- `"Isotropic Middle-Stack Layer"` → `thickness` (µm), `permittivity`,
  optional `permeability` (scalar or `{real, imag}`).
- `"Crystal Layer"` → `material`, `thickness` (µm), `rotationX/Y/Z`,
  optional `rotationZType` (`"relative"` default / `"static"`).
- `"Semi Infinite Anisotropic Layer"` → `material`, `rotationX/Y/Z`.
- `"Semi Infinite Isotropic Layer"` → `permittivity` (exit; note: **no
  FullSweep support yet** — don't use it in a FullSweep golden until Phase 1
  adds it).

**Isotropic-exit caveat for the battery:** when the last layer is
`Semi Infinite Isotropic Layer`, `Structure.get_layers` derives the frequency
range from the **second-to-last** layer's material (`structure.py:189–194`), so
that second-to-last layer must be a real material (e.g. a `Crystal Layer`), not
another isotropic one.

Thickness in payloads is microns and converted to cm via `×1e-4`
(`layers.py:289–291`). Materials available:
`Quartz, Sapphire, Calcite (=CalciteUpper), CalciteLower, GalliumOxide`, or an
arbitrary `eps_*`/`mu_*` dict.

**Caveat to document in the harness:** golden values depend on NumPy's
eigensolver and are environment-pinned. Regenerate deliberately, never casually.
Pin `numpy`/`scipy` versions in the dev environment if drift appears.

**Phase 0 exit criterion:** `pytest tests/test_golden.py` passes on the commit
that generated the snapshots (sanity-check of the harness wiring), and the
snapshots are committed.

---

## 8. Phase 1 — Canonical shapes (pure refactor)

Implement sections 4–5. Work in small commits, running `pytest` (especially
`tests/test_golden.py`) after each. Suggested order:

1. Add `hyperbolic_optics/axes.py` (constants + `assert_canonical`).
2. Boundary-in: make `scenario.py` emit canonical `[A,1,1]/[1,B,1]/[1,1,F]`
   arrays; make `materials`/`rotate_tensors` emit `[A?,B?,F?,3,3]`.
3. Strip `mode` (and dead `magnet`) from `Wave`; delete vestigial methods +
   `anisotropy_rotation_all_axes`; simplify `delta_matrix_calc` and `get_matrix`.
4. Simplify the layer `create` methods (drop all per-scenario `newaxis`); add
   the missing FullSweep branch to `IsotropicSemiInfiniteLayer`.
5. Boundary-out: replace `calculate_reflectivity` reshaping with the
   squeeze-then-`(F,A,B)` rule.
6. Add `assert_canonical` calls at the two boundaries + `Wave` entry.

**Exit criterion:** golden tests bit-identical; the words `Incident`,
`Azimuthal`, `airgap`, etc. no longer appear inside `waves.py`.

---

## 9. Phase 2 — Harden forward/backward mode sorting

The fragile heart is `wave_sorting` (`waves.py:308–348`), specifically the
`is_complex ? argsort(Im) : argsort(Re)` heuristic (324–328) that classifies the
4 eigenmodes into 2 transmitted (forward) + 2 reflected (backward). This is
**exactly where `kx/k0 > 1` correctness lives** and where issue #30 says things
break. `sort_poynting_indices` (628–664) then classifies p vs s.

Tasks:
- Replace the heuristic with a principled, documented criterion:
  **propagating modes** sorted by sign of time-averaged Poynting-z (energy flow
  direction); **evanescent modes** by sign of `Im(kz)` (decay direction), with a
  well-defined tie-break for the ghost/leaky mixed cases. (Cross-check against
  Passler & Paarmann 2017 §sorting, and the layer-resolved-absorption paper.)
- Add unit tests on known analytic cases (isotropic interface; uniaxial at
  normal incidence; a deliberately evanescent `kx/k0 > 1` case) asserting the
  correct mode partition and that R is unchanged from golden.

**Exit criterion:** golden green; new sorting tests pass; the criterion is
documented in the module.

---

## 10. Phase 3 — Scattering-matrix engine (ROADMAP)

**Motivation (section 2.4):** the T-matrix product (`structure.py:218–229`,
`functools.reduce(matmul, ...)`) carries growing exponentials and is unstable
for thick/evanescent layers — the regime hyperbolic physics lives in. The
S-matrix only propagates decaying exponentials → unconditionally stable, and
keeps per-layer forward/backward amplitudes that Phase 4 needs.

Tasks:
- Implement an S-matrix layer-stacking recursion (Redheffer star product)
  alongside the existing T-matrix path, operating on the canonical `[A,B,F,...]`
  arrays from Phase 1. The per-layer eigenmode decomposition from `Wave` already
  provides the interface/propagation ingredients.
- Validate: in the numerically *stable* regime, S-matrix R must match the golden
  T-matrix R (`assert_allclose`). Then construct a thick / strongly evanescent
  stack where the T-matrix overflows or loses precision and show the S-matrix
  stays finite and physical.
- References: Li, JOSA A 13, 1024 (1996); Whittaker & Culshaw, PRB 60, 2610
  (1999); Rumpf, PIER B 35, 241 (2011). Prior art: PyLlama uses an S-matrix;
  pyGTM uses a T-matrix and exhibits the instability.

**Note:** some current T-matrix assembly (`sort_profile_back_to_matrix`,
`waves.py:666–716`; the `V·exp·V⁻¹` in `get_matrix`) is superseded here. Don't
over-polish it in Phase 1 beyond what canonicalisation requires.

**Exit criterion:** R parity with golden in the stable regime; demonstrated
stability where the T-matrix fails.

---

## 11. Phase 4 — Transmission, layer-resolved absorption, field profiling (ROADMAP)

This is the issue #30 payoff and the paper-worthy deliverable. Built natively on
the stable S-matrix amplitudes from Phase 3.

Tasks:
- **Transmission coefficients** `t_pp, t_ps, t_sp, t_ss` computed from the
  S-matrix (replaces the unvalidated `calculate_transmissivity`,
  `structure.py:277–294`, which `execute` never even calls,
  `structure.py:304–334`).
- **Layer-resolved absorption** via the divergence of the time-averaged Poynting
  vector across each layer (the `get_poynting` machinery already computes the
  needed fields). Follow Passler 2020, *Layer-Resolved Absorption of Light in
  Arbitrarily Anisotropic Heterostructures*, arXiv:2002.03832.
- **Field profiling** `E(z), H(z)` through the stack from the stable per-layer
  forward/backward amplitudes.
- **Validation oracle:** energy conservation `R + T + A = 1` within tolerance —
  the maintainer explicitly flagged this in issue #30 as the physical sanity
  check. Add it as a test across representative stacks. (Note the subtlety raised
  in #30: for a symmetric structure where the exit medium matches the prism
  permittivity, conservation should hold exactly; use such a case as a clean
  check.)

**Exit criterion:** `R+T+A=1` tests pass; field-profile and absorption examples
added under `examples/`.

---

## 12. Decisions log

| # | Decision | Rationale |
|---|---|---|
| D1 | Cleanup *before* transmission | Transmission/absorption can't be built on 5 implicit array layouts; cleanup is on the critical path |
| D2 | Canonical `[A,B,F,...]` layout, size-1 for un-swept axes | Makes axis meaning unambiguous everywhere; enables scenario-agnostic physics via pure broadcasting |
| D3 | Enforcement = named constants + runtime asserts (`axes.py`); no new deps | Catches drift instantly, near-zero cost; einops/jaxtyping not worth it at this size |
| D4 | Commit to the scattering matrix as the engine | T-matrix is unstable exactly in the hyperbolic `kx/k0>1` regime; S-matrix is the natural substrate for absorption/field profiling |
| D5 | Golden numeric tests are a hard prerequisite (Phase 0) | Existing tests only check shapes/bounds; refactor needs a behaviour lock |
| D6 | Presentation rule = squeeze size-1, order survivors `(F, A, B)` | Single rule reproduces all current output shapes |
| D7 | Keep `mu` generality through the pipeline; drop the dead `magnet` flag | `mu_tensor` already flows through Berreman build, `Ez/Hz`, and Poynting; only the unused `magnet` flag (`waves.py:113`) is dead |

---

## 13. Open questions for the maintainer

Where the code answers a question, the answer is recorded inline; the genuinely
open ones are marked **[OPEN]**.

1. **`rotationZType` default — resolved.** Defaults to `"relative"`
   (`layers.py:276`); `calculate_z_rotation` (`layers.py:322–355`) folds the
   azimuth into `rotationZ` for relative and broadcasts a constant for static.
   The refactor changes only axis *placement*, not the rotation math, so
   relative/static semantics are preserved verbatim. **Confirm the explicit
   nod** since rotation feeds the `A`/`B` axes.
2. **FullSweep axis sizes — flagged.** Currently incident=180, azimuth=120,
   freq=410 (`scenario.py:139–144`). **[OPEN]** Keep hard-coded, or parameterise?
   (Out of scope for the refactor; flag only.)
3. **Magnetic media — resolved (see D7).** `mu` flows through the whole
   pipeline; the standalone `magnet` flag is unused (`waves.py:113`). Plan keeps
   `mu` generality through the S-matrix work and deletes the dead flag.
   **Confirm** that full `mu` support should be carried into Phase 3/4 v1.
4. **Golden tolerance / version pinning — [OPEN].** Accept environment-pinned
   goldens with `rtol=1e-7`? If cross-machine stability is needed, pin
   numpy/scipy in `pyproject.toml` dev extras.
5. **Scope of Phase 4 v1 — [OPEN].** Ship transmission + absorption + field
   profiling together, or land transmission first (smaller PR) then
   absorption/profiling?

---

## 14. References

**Method / physics**
- Berreman, *JOSA* 62, 502 (1972) — 4×4 formalism.
- Yeh, *JOSA* 69, 742 (1979) — birefringent layered media.
- Passler & Paarmann, *JOSA B* 34, 2128 (2017) — generalized 4×4; mode sorting.
- Passler et al., *PRB* 107, 235426 (2023) — layer-resolved resonance intensity.
- Passler et al., *Layer-Resolved Absorption…*, arXiv:2002.03832 (2020) — the
  Phase 4 target.
- Cunningham et al., *Photonics Research* 13, 2291 (2025) — the package's paper.

**Scattering-matrix stability**
- Li, *JOSA A* 13, 1024 (1996).
- Whittaker & Culshaw, *PRB* 60, 2610 (1999).
- Rumpf, *PIER B* 35, 241 (2011).

**Prior-art codes**
- PyLlama — S-matrix; numerically stable.
- pyGTM — T-matrix; exhibits the instability this plan avoids.

**Relevant GitHub issues**
- #30 — transmission / layer-resolved absorption / field profiling (the goal).
- #35 — GPU/TensorFlow technical debt (origin of the vestigial code).
- #43 — "simplify tensor operations in waves.py" (this refactor closes it).
- #47 — 2D/metamaterial parameters (adjacent; not in this plan).

---

*End of plan. Work on `claude-havoc`; never push to `main`.*
