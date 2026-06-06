"""Golden-master regression tests.

Re-runs every payload in :mod:`tests.golden.payloads` and asserts the reflection
coefficients and Mueller matrix are unchanged versus the committed snapshots in
``tests/golden/data/``. This is the behaviour lock that the canonical-shape
refactor (Phases 1-2) must keep green.

Regenerate snapshots with ``python -m tests.golden.generate`` on a known-good
commit. See ``docs/dev/phase-0-1-execution-plan.md``.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.golden.generate import compute_outputs
from tests.golden.payloads import PAYLOADS, SLOW_PAYLOADS

DATA_DIR = Path(__file__).parent / "golden" / "data"

# Baseline tolerances for a pure refactor: the numbers fed to np.linalg.eig do
# not change, so output should match to round-off. atol is deliberately a touch
# loose to absorb field nulls in the evanescent / hyperbolic regime, where
# relative error explodes near zero (relaxing atol there is not a behaviour
# change). Tighten rtol below 1e-7 only if you enjoy false failures.
RTOL = 1e-7
ATOL = 1e-9

# Goldens are an environment-pinned refactor lock (see tests/golden/__init__.py),
# not a cross-platform gate; CI deselects them via `-m "not golden"`.
pytestmark = pytest.mark.golden


def _params():
    """Build parametrization, marking expensive payloads as slow."""
    params = []
    for name in PAYLOADS:
        marks = [pytest.mark.slow] if name in SLOW_PAYLOADS else []
        params.append(pytest.param(name, marks=marks, id=name))
    return params


@pytest.mark.parametrize("name", _params())
def test_golden(name):
    """Outputs for ``name`` match the committed snapshot."""
    snapshot_path = DATA_DIR / f"{name}.npz"
    if not snapshot_path.exists():
        pytest.skip(
            f"No golden snapshot for '{name}'. "
            f"Run `python -m tests.golden.generate {name}` on a known-good commit."
        )

    golden = np.load(snapshot_path)
    actual = compute_outputs(PAYLOADS[name])

    # Compare exactly the arrays the snapshot stored (FullSweep has no Mueller).
    assert set(golden.files) == set(actual), (
        f"{name}: snapshot keys {set(golden.files)} != computed keys {set(actual)}"
    )
    for key in golden.files:
        np.testing.assert_allclose(
            actual[key],
            golden[key],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{name}: '{key}' drifted from golden",
        )
