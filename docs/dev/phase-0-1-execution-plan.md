# Execution Plan: Phase 0 (Golden Net) + Phase 1 (Canonical Shapes)

> **Scope:** The two *actionable-now* phases only. This is the granular,
> execution-ready companion to
> [`canonical-shape-refactor-plan.md`](canonical-shape-refactor-plan.md) — read
> that for the *why* (root-cause diagnosis §3, target design §4). This doc is
> the *how*.
> **Branch:** `claude-havoc`. Never push to `main`.
> **Verified against:** HEAD `c011b5b` (v0.2.3).
> **Hard rule:** Phase 1 is a *pure refactor*. It must not change a single
> reflection-coefficient (or Mueller) value. Phase 0 builds the proof of that.
> Do not start Phase 1 until Phase 0 is green.

---

## Why these two phases stand alone

Phase 0 + Phase 1 deliver real value on their own and carry **no physics risk**:

- Phase 0 is purely additive (new tests, no source changes).
- Phase 1 is a behaviour-preserving cleanup that deletes ~10 mode strings and
  every per-scenario `np.newaxis` site, gated bit-for-bit by Phase 0.

Everything past this (mode-sorting hardening, S-matrix, transmission) is new
physics/capability and is deferred to the roadmap in the full plan. You can ship
0+1, stop, and the package is strictly better and unchanged in output.

---

## Phase 0 — Golden-master regression harness

### Goal

Freeze the *actual complex output arrays* of the current code so Phase 1 can be
proven behaviour-preserving. The existing suite only checks shapes,
`iscomplexobj`, and `0 ≤ R ≤ 1` ([`tests/test_structure.py:41–114`](../../tests/test_structure.py)) —
it would not catch `r_pp` drifting 0.42 → 0.43.

> **Status: DONE (as-built).** The harness exists and all 12 payloads pass
> against committed snapshots. The subsections below describe what was actually
> built; where it differs from the original sketch, the difference is called out.

### Layout (as built)

```
tests/
  golden/
    __init__.py        # documents the regenerate workflow + env-pinning caveat
    payloads.py        # PAYLOADS: dict[str, dict] + SLOW_PAYLOADS  (battery below)
    generate.py        # _subsample + compute_outputs; python -m tests.golden.generate
    data/
      <name>.npz       # one per payload: r_pp, r_ps, r_sp, r_ss [, mueller]
  test_golden.py       # parametrized over PAYLOADS; assert_allclose vs .npz
```

### What to snapshot (per payload)

Save **both** levels of the user-facing output, not just the raw coefficients:

1. `r_pp, r_ps, r_sp, r_ss` from `Structure.execute(payload)`.
2. The **4×4 Mueller matrix** from `Mueller(structure).calculate_mueller_matrix()`.

> **Why Mueller too (critique fold-in):** `mueller.py` consumes the coefficients
> *and* carries its own `scenario.type == "Simple"` branches
> ([`mueller.py:284–313, 320, 333, 389`](../../hyperbolic_optics/mueller.py)).
> It is a headline feature. Freezing only `r_*` leaves the actual polarimetry
> unguarded. Snapshotting the Mueller matrix is cheap insurance and locks the
> full pipeline. (It also exposes that `mueller.py` has the *same*
> scenario-coupling disease — a candidate for a later cleanup pass, out of scope
> here.)

> **As-built — two realities discovered while building:**
> 1. **Strided fingerprint, not full resolution.** Full-res snapshots are
>    ~215 MB across the battery (the `(410, 360, 4, 4)` Mueller arrays dominate),
>    and FullSweep alone would add ~500 MB — unacceptable in git. `compute_outputs`
>    applies `_subsample`, striding the *batch* axes (capped at 64 points each;
>    16 for 3D FullSweep) while keeping the 4×4 Mueller block intact and full
>    float precision. Total drops to **~5.7 MB**. A strided fingerprint still
>    catches any value drift from the (uniform) layout refactor. Subsampling
>    lives inside `compute_outputs`, so generation and the test apply it
>    identically and the test stays a pure equality check.
> 2. **Mueller does not support FullSweep.** `calculate_mueller_matrix`'s
>    `np.transpose(..., axes=[2,3,0,1])` ([`mueller.py:325`](../../hyperbolic_optics/mueller.py))
>    assumes a 2D coefficient array; FullSweep's 3D output makes it raise. The
>    snapshot therefore omits `mueller` for FullSweep (deterministically, so the
>    test agrees). This is a pre-existing limitation and a candidate for the
>    later `mueller.py` cleanup — **not** something Phase 1 changes.

### The battery (12 payloads, as built)

Built from the fixtures in [`tests/conftest.py`](../../tests/conftest.py) and
extended. Covers every scenario type, both `rotationZType` modes, the
arbitrary-material path, a magnetic air gap, an isotropic exit, the monoclinic
material path, and a hyperbolic `kx/k0 > 1` case. Defined in
[`tests/golden/payloads.py`](../../tests/golden/payloads.py).

| Key | Scenario | What it exercises |
|---|---|---|
| `simple_calcite` | Simple | scalar path; baseline |
| `incident_calcite` | Incident | kx–ω sweep, `(410, 360)` output |
| `azimuthal_calcite` | Azimuthal | the `swapaxes` axis-order fix point; `rotationZType` relative (default) |
| `dispersion_calcite` | Dispersion | azimuth-into-`rotationZ`, `(180, 480)` |
| `fullsweep_quartz` | FullSweep | 3D `(410, 180, 120)`; **anisotropic exit only**; `mueller` omitted; **slow-marked** |
| `arbitrary_material` | Simple | `ArbitraryMaterial` eps dict path |
| `multilayer_incident` | Incident | prism + air gap + finite crystal + semi-infinite, same material (Quartz) both sides |
| `rotz_static_azimuthal` | Azimuthal | `rotationZType: "static"` |
| `isotropic_exit_simple` | Simple | `Semi Infinite Isotropic Layer` exit (composes only at scalar frequency) |
| `magnetic_gap_incident` | Incident | `Isotropic Middle-Stack Layer` with complex `permeability` |
| `hyperbolic_evanescent` | Incident | `permittivity: 50` prism into Calcite reststrahlen → large `kx`, evanescent/leaky regime |
| `gallium_oxide_incident` | Incident | monoclinic material (off-diagonal `eps`) path |

**Battery gotchas (verified while building):**
- **Materials don't share a frequency axis.** `fetch_permittivity_tensor()` uses
  each material's *own* default range, and for Incident/Azimuthal/FullSweep the
  stack's `k0` range comes from the *last* layer. Combining a finite `Crystal
  Layer` with a semi-infinite crystal of a *different* material yields two
  length-410 axes over *different* physical frequencies — incoherent. The
  multilayer payload uses **Quartz on both sides** to avoid this.
- **Isotropic exit only composes at scalar frequency.** A `Semi Infinite
  Isotropic Layer` returns a `[..., 4, 4]` matrix with no frequency axis, so it
  only broadcasts against the rest of the stack in `Simple` / `Dispersion`
  (single-frequency) scenarios — not multi-frequency `Incident`/`Azimuthal`/
  `FullSweep`. Hence `isotropic_exit_simple` is a `Simple` stack. (The
  `structure.py:189–194` second-to-last-layer frequency lookup is therefore only
  reachable for these scalar-frequency cases.)
- Thickness in payloads is **microns**, converted to cm via `×1e-4`
  ([`layers.py:289–291`](../../hyperbolic_optics/layers.py)).
- Materials: `Quartz, Sapphire, Calcite (=CalciteUpper), CalciteLower,
  GalliumOxide`, or an arbitrary `eps_*`/`mu_*` dict.

### test_golden.py

Parametrized over `PAYLOADS` (slow payloads carry `pytest.mark.slow` via
`pytest.param(..., marks=...)`). For each: load `data/<name>.npz`, re-run the
payload through the same `compute_outputs` (so subsampling matches), assert the
stored key set equals the computed key set, and for every saved array:

```python
np.testing.assert_allclose(actual, golden, rtol=1e-7, atol=1e-9)
```

**Tolerance guidance (critique fold-in):** `rtol=1e-7` is right for a pure
refactor (the numbers fed to `np.linalg.eig` don't change, so output should
match to round-off). On the snapshot-generating commit the test trivially passes
(identical code). The tolerance matters in Phase 1: in the `kx/k0 > 1`
evanescent regime, field components pass through nulls where relative error
explodes — **expect to relax `atol`** (e.g. `1e-8`–`1e-7`) for the hyperbolic and
Mueller arrays before suspecting a real change. Relaxing `atol` near zeros is
*not* a behaviour change; tightening `rtol` below `1e-7` invites false failures
from eigensolver round-off. Tune per-array if needed.

### Caveat documented in `__init__.py`

Golden values depend on the NumPy/LAPACK eigensolver and are
**environment-pinned**. Generated with **Python 3.12.11, numpy 2.3.3, scipy
1.16.2**. Regenerate deliberately, never casually. If drift appears across
machines, pin `numpy`/`scipy` in `pyproject.toml` dev extras and regenerate.

### Phase 0 exit criterion — MET

Snapshots generated on HEAD `c011b5b` and committed (12 files, ~5.7 MB total).
Verified:

```
python -m tests.golden.generate                       # 12/12 snapshots written
pytest tests/test_golden.py -o addopts=""             # 12 passed (87s, incl. slow FullSweep)
pytest tests/test_golden.py -m "not slow" -o addopts="" # 11 passed, 1 deselected (10s)
pytest -m "not slow" -o addopts=""                    # 102 passed, 3 deselected (no regressions)
```

> Everyday use: `pytest tests/test_golden.py -m "not slow"` (~10s). Run the full
> set (incl. FullSweep, ~90s) before declaring Phase 1 done.

---

## Phase 1 — Canonical shapes (pure refactor)

> **Status: DONE.** Landed in two commits — `Phase 1a` (add `axes.py` + remove
> vestigial code, behaviour-preserving) and `Phase 1b` (the coupled canonical
> shape change). All 12 goldens stay bit-identical; the full suite (102 tests)
> passes; `uv run ruff check` is clean. As-built notes are inline below.

### Goal

Make `[A, B, F, *matrix]` (un-swept axes size 1) the *only* array convention,
so the physics code stops knowing about scenarios. See full plan §4 for the
convention and §5 for the complete hit-list. Work in **small commits, running
`pytest tests/test_golden.py` after each.**

> **As-built deltas from the sketch below:**
> - The coupled change had to land atomically (boundary-in + `Wave` + layers +
>   boundary-out all change the data-flow contract together), not as six
>   independently-green commits. It was split into the two commits above.
> - The air-gap matrix is canonical `[A, 1, F, 4, 4]` (it depends on incident
>   angle *and* frequency via kx/k0), not `[1, 1, F, 4, 4]` as §4.5 sketched.
> - Isotropic exit became fully scenario-agnostic: with size-1 axes it now
>   broadcasts in FullSweep too, so no explicit FullSweep branch was needed.
> - Extra dead code removed: `AmbientIncidentMedium.construct_tensor_singular`
>   (the prism now always uses `construct_tensor`).
> - `tests/test_layers.py` asserted the old internal shapes and passed scalar
>   kx/k0; it was updated to feed canonical inputs and assert canonical shapes.

### Commit-by-commit order

1. **`hyperbolic_optics/axes.py`** — add `A, B, F = 0, 1, 2`, `N_BATCH = 3`,
   and `assert_canonical(arr, matrix_ndim=2, name=...)` (full plan §4.7). No
   behaviour change; nothing calls it yet.

2. **Boundary-in** — in `scenario.py`, construct angle/frequency arrays already
   shaped `incident → [A,1,1]`, `azimuthal → [1,B,1]`, `k0 → [1,1,F]`. Make
   `materials` / `rotate_tensors` emit `[A?,B?,F?,3,3]` in canonical order.
   Delete the Azimuthal `swapaxes(0,1)` ([`layers.py:409–412`](../../hyperbolic_optics/layers.py))
   by building the tensors in the right order to begin with.

3. **Strip the engine** — in `waves.py`:
   - Remove `mode` (and the dead `magnet`) params from `Wave.__init__`; delete
     the `_setup_tensor_shapes` mode branching and the unused `self.batch_size`.
   - Delete `_get_tensor_shapes_for_mode`, `_get_poynting_tensor_shapes`,
     `mode_reshaping`, `poynting_reshaping`, `delta_permutations` (and its call
     in `execute`, [`waves.py:738`](../../hyperbolic_optics/waves.py)).
   - `delta_matrix_calc`: delete the per-mode `kx` reshape (222–230); rely on
     `kx=[A,1,1]`.
   - `get_matrix`: delete the six per-mode `k0` branches (374–408) and the
     `allclose` diagonal check (410–427); compute `exp` of the diagonal directly,
     then `V · diag · V⁻¹`.
   - In `anisotropy_utils.py`, delete the unused
     `anisotropy_rotation_all_axes` (149–217).

4. **Simplify the layers** — in `layers.py`:
   - `PrismLayer.create` → emit `[A,1,1,4,4]`; drop per-scenario `newaxis`.
   - `AirGapLayer` → delete `calculate_mode`; `create` emits `[1,1,F,4,4]`;
     call `Wave` without a mode; drop per-scenario `newaxis`.
   - `CrystalLayer` / `SemiInfiniteCrystalLayer` → call `Wave` without a mode
     (they currently pass `self.scenario` as the mode, 615/656).
   - `IsotropicSemiInfiniteLayer.create` → emit canonical shape **and add the
     missing FullSweep branch** (687–703 currently has no FullSweep path).

5. **Boundary-out** — replace the per-scenario `swapaxes`/`moveaxis` in
   `calculate_reflectivity` ([`structure.py:263–275`](../../hyperbolic_optics/structure.py))
   with the single rule: **squeeze size-1 axes, then order survivors `(F, A, B)`**
   (full plan §4.6). This reproduces all five output shapes.

6. **Wire the asserts** — add `assert_canonical` at the two boundaries and at
   `Wave`'s entry.

### Watch-outs

- Re-run the golden suite after *every* commit, not just at the end — it
  localises any regression to one small change.
- The matmul chain in `calculate` ([`structure.py:229`](../../hyperbolic_optics/structure.py))
  must broadcast cleanly once every layer matrix is `[A?,B?,F?,4,4]`; a shape
  mismatch here is the most likely failure and is easy to read off.
- If a golden array trips only by a hair, suspect `atol` near nulls (see Phase 0
  tolerance note) before suspecting a real change — but confirm by eye.

### Phase 1 exit criterion

- `pytest tests/test_golden.py` is **bit-identical** (within tolerance) for
  every payload, including the Mueller snapshots.
- The strings `Incident`, `Azimuthal`, `airgap`, `Dispersion`, `FullSweep` no
  longer appear inside `waves.py`.
- Full `pytest` suite still green.

---

## After 0+1

Stop here for a clean, shippable result, or proceed to the roadmap (Phase 2
mode-sorting hardening → Phase 3 S-matrix → Phase 4 transmission/absorption) in
[`canonical-shape-refactor-plan.md`](canonical-shape-refactor-plan.md). Note the
full plan's caveat: **reproduce the T-matrix instability on a concrete stack
before committing to the Phase 3 S-matrix rewrite** — that bet is currently a
hypothesis, not a measured fact.
