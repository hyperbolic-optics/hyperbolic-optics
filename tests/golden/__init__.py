"""Golden-master regression harness for reflection / Mueller outputs.

This package freezes the *actual complex output arrays* of the current code so
that the canonical-shape refactor (see
``docs/dev/phase-0-1-execution-plan.md``) can be proven behaviour-preserving.
The pre-existing test suite only checks shapes, ``iscomplexobj`` and
``0 <= R <= 1`` -- it would not notice ``r_pp`` drifting 0.42 -> 0.43. These
goldens close that gap.

Workflow
--------
1. On a known-good commit, generate the snapshots::

       python -m tests.golden.generate

   This runs every payload in :mod:`tests.golden.payloads` through
   ``Structure().execute(...)`` plus a ``Mueller`` matrix calculation and writes
   ``tests/golden/data/<name>.npz`` containing ``r_pp, r_ps, r_sp, r_ss`` and the
   4x4 ``mueller`` matrix.

2. ``pytest tests/test_golden.py`` re-runs each payload and asserts the output is
   unchanged (``np.testing.assert_allclose``).

Environment caveat
------------------
Golden values depend on the NumPy/LAPACK eigensolver and are **environment
pinned**. Last regenerated with:

    Python 3.12.11, numpy 2.4.6

(The note here previously also cited scipy, which stopped being a dependency in
0.3.0.)

This is why the battery is **not** run in CI: a runner's LAPACK differs enough
from the generating machine's to exceed ``rtol=1e-7 / atol=1e-9``, so a CI job
would fail on every pull request while telling you nothing about the code. It
is a local refactor lock -- run it before and after a change on one machine.

What guards behaviour in CI instead are the tests that assert *physics* rather
than bit-values, and so hold on any platform: ``test_physical_invariants.py``
(cross-validation against the closed-form isotropic path, passivity),
``test_material_literature.py`` (published band edges), ``test_general_tensors.py``
(mode-partition invariants, gamma -> 0 continuity) and the cross-backend
comparisons in ``test_scattering.py``.

Regenerate deliberately, never casually.
"""
