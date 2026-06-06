"""Generate golden-master snapshots for the payload battery.

Run on a known-good commit::

    python -m tests.golden.generate            # all payloads
    python -m tests.golden.generate simple_calcite incident_calcite   # a subset

For each payload this runs ``Structure().execute(...)``, captures the four
reflection coefficients and the 4x4 Mueller matrix, and writes
``tests/golden/data/<name>.npz``. Failures are reported but do not abort the run,
so a single non-composing payload never blocks the rest of the battery.
"""

import sys
import traceback
from pathlib import Path

import numpy as np

from hyperbolic_optics.mueller import Mueller
from hyperbolic_optics.structure import Structure

from tests.golden.payloads import PAYLOADS

DATA_DIR = Path(__file__).parent / "data"


def _subsample(arr: np.ndarray, matrix_ndim: int) -> np.ndarray:
    """Stride the *batch* (leading) axes so the snapshot stays git-friendly.

    Full-resolution outputs (e.g. (410, 360, 4, 4) Mueller) are ~200 MB across
    the battery -- far too heavy to commit. A deterministic strided fingerprint
    at full float precision still catches any value drift from the (uniform)
    canonical-shape refactor. The trailing ``matrix_ndim`` axes (the 4x4 Mueller
    block) are kept intact; batch axes are capped at ``cap`` points each.
    """
    batch_ndim = arr.ndim - matrix_ndim
    if batch_ndim <= 0:
        return arr  # scalar (Simple) or pure matrix -- already tiny
    cap = 64 if batch_ndim <= 2 else 16  # 3D FullSweep gets a tighter cap
    slices = []
    for axis in range(batch_ndim):
        n = arr.shape[axis]
        step = max(1, -(-n // cap))  # ceil(n / cap)
        slices.append(slice(None, None, step))
    slices.extend([slice(None)] * matrix_ndim)
    return arr[tuple(slices)]


def compute_outputs(payload: dict) -> dict[str, np.ndarray]:
    """Run a payload and return the (subsampled) arrays to snapshot.

    Always returns ``r_pp, r_ps, r_sp, r_ss`` (complex reflection coefficients).
    Adds ``mueller`` (the real 4x4 sample Mueller matrix, batched per scenario)
    when it can be computed -- ``Mueller`` does not support ``FullSweep`` (its
    transpose assumes a 2D coefficient array), so that key is omitted there. The
    omission is deterministic per payload, so generation and the test agree.
    """
    structure = Structure()
    structure.execute(payload)

    outputs = {
        "r_pp": _subsample(np.asarray(structure.r_pp), matrix_ndim=0),
        "r_ps": _subsample(np.asarray(structure.r_ps), matrix_ndim=0),
        "r_sp": _subsample(np.asarray(structure.r_sp), matrix_ndim=0),
        "r_ss": _subsample(np.asarray(structure.r_ss), matrix_ndim=0),
    }

    try:
        mueller = Mueller(structure)
        mueller.calculate_mueller_matrix()
        outputs["mueller"] = _subsample(np.asarray(mueller.mueller_matrix), matrix_ndim=2)
    except Exception as exc:  # noqa: BLE001 - Mueller lacks FullSweep support
        print(f"  (mueller skipped: {type(exc).__name__}: {exc})")

    return outputs


def main(names: list[str]) -> int:
    """Generate snapshots for ``names`` (or all payloads if empty)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    selected = names or list(PAYLOADS)
    unknown = [n for n in selected if n not in PAYLOADS]
    if unknown:
        print(f"Unknown payload(s): {unknown}\nAvailable: {list(PAYLOADS)}")
        return 2

    failures = []
    for name in selected:
        try:
            outputs = compute_outputs(PAYLOADS[name])
            np.savez(DATA_DIR / f"{name}.npz", **outputs)
            shapes = {k: v.shape for k, v in outputs.items()}
            print(f"OK    {name}: {shapes}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append(name)
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print(
        f"\nGenerated {len(selected) - len(failures)}/{len(selected)} "
        f"snapshots into {DATA_DIR}"
    )
    if failures:
        print(f"Failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
