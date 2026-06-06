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
pinned**. They were generated with:

    Python 3.12.11, numpy 2.3.3, scipy 1.16.2

Regenerate deliberately, never casually. If values drift across machines, pin
``numpy``/``scipy`` in the dev environment (``pyproject.toml`` dev extras) and
regenerate on the pinned versions.
"""
