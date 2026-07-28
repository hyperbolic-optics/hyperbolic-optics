"""
Every ```python block in the README and docs must actually run.

Documentation drifts silently: a renamed argument or a removed helper leaves the
prose looking right and the snippet dead. Several were broken when this was
first run -- pages opening with ``structure.execute(payload)`` where ``payload``
was never defined, a plotting example using an undefined name, and config
fragments tagged as Python that are not Python.

Each page is executed cumulatively in one namespace, which is how a reader
follows it top to bottom: later blocks may use names the earlier ones bound.
"""

import pathlib
import re

import matplotlib
import pytest

matplotlib.use("Agg")

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def _pages():
    pages = [REPOSITORY / "README.md"] + sorted((REPOSITORY / "docs").rglob("*.md"))
    return [page for page in pages if BLOCK.search(page.read_text(encoding="utf-8"))]


def _is_program(code: str) -> bool:
    """Skip data literals and dangling fragments -- they are not programs."""
    stripped = code.strip()
    return not (
        stripped.startswith(("{", "custom_material", "gyrotropic")) or stripped.endswith(",")
    )


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.name)
def test_python_blocks_execute(page, tmp_path, monkeypatch):
    """Run one page's blocks in a shared namespace, in order."""
    monkeypatch.chdir(tmp_path)  # snippets that save figures write here
    namespace = {"__name__": "__main__"}

    for index, code in enumerate(BLOCK.findall(page.read_text(encoding="utf-8"))):
        if not _is_program(code):
            continue
        try:
            exec(compile(code, f"{page.name}:{index}", "exec"), namespace)  # noqa: S102
        except Exception as error:  # noqa: BLE001
            pytest.fail(
                f"{page.relative_to(REPOSITORY)} block {index} failed with "
                f"{type(error).__name__}: {error}\n\n{code}"
            )
