#!/usr/bin/env python3
"""
Synchronize version numbers across all project files.

``__init__.py`` is the source of truth. Run with no arguments to rewrite every
other file to match it; run with ``--check`` to verify they already agree and
exit non-zero if not, which is what the release workflow does so a stale
CITATION.cff or docs citation fails the release rather than shipping.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

#: Only the *software* citation tracks the release. The related-publication
#: entries are @article blocks with their own, fixed, years -- rewriting those
#: to the current year (as this script used to) silently falsifies a reference.
SOFTWARE_BLOCK = re.compile(r"@software\{[^@]*?\n\}", re.S)


def get_version_from_init() -> str:
    """Extract version from __init__.py."""
    content = Path("hyperbolic_optics/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find __version__ in __init__.py")
    return match.group(1)


def _retag_software_citations(content: str, version: str, year: int) -> str:
    """Rewrite version/year inside @software blocks only."""

    def fix(match: re.Match[str]) -> str:
        block = match.group(0)
        block = re.sub(r"version=\{.*?\}", f"version={{{version}}}", block)
        return re.sub(r"year=\{.*?\}", f"year={{{year}}}", block)

    return SOFTWARE_BLOCK.sub(fix, content)


def _citation_cff(content: str, version: str, year: int) -> str:
    content = re.sub(r'version: ".*"', f'version: "{version}"', content)
    return re.sub(
        r'date-released: ".*"',
        f'date-released: "{datetime.now().strftime("%Y-%m-%d")}"',
        content,
    )


TARGETS = {
    Path("CITATION.cff"): _citation_cff,
    Path("README.md"): _retag_software_citations,
    Path("docs/index.md"): _retag_software_citations,
    Path("docs/citation.md"): _retag_software_citations,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report disagreements and exit 1 without writing anything",
    )
    args = parser.parse_args()

    version = get_version_from_init()
    year = datetime.now().year
    stale = []

    for path, transform in TARGETS.items():
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        updated = transform(current, version, year)
        if current == updated:
            continue
        if args.check:
            # CITATION.cff carries a release date that moves every day, so a
            # date-only difference is not staleness.
            if transform is _citation_cff and re.sub(
                r'date-released: ".*"', "", current
            ) == re.sub(r'date-released: ".*"', "", updated):
                continue
            stale.append(path)
        else:
            path.write_text(updated, encoding="utf-8")
            print(f"updated {path} to version {version}")

    if args.check:
        if stale:
            print(f"version {version} (from __init__.py) is not reflected in:")
            for path in stale:
                print(f"  {path}")
            print("run `python scripts/sync_versions.py` to fix")
            return 1
        print(f"all version references agree with __init__.py ({version})")
        return 0

    print(f"all version numbers synchronized to {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
