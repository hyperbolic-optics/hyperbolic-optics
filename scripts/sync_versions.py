#!/usr/bin/env python3
"""
Synchronize version numbers across all project files.
"""

import re
from datetime import datetime
from pathlib import Path


def get_version_from_init():
    """Extract version from __init__.py"""
    init_path = Path("hyperbolic_optics/__init__.py")
    with open(init_path) as f:
        content = f.read()

    # Find __version__ = "x.y.z"
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    else:
        raise ValueError("Could not find __version__ in __init__.py")


def update_citation_cff(version):
    """Update CITATION.cff file"""
    cff_path = Path("CITATION.cff")
    if not cff_path.exists():
        return

    with open(cff_path) as f:
        content = f.read()

    # Update version and date
    content = re.sub(r'version: ".*"', f'version: "{version}"', content)
    content = re.sub(
        r'date-released: ".*"',
        f'date-released: "{datetime.now().strftime("%Y-%m-%d")}"',
        content,
    )

    with open(cff_path, "w") as f:
        f.write(content)

    print(f"✅ Updated CITATION.cff to version {version}")


def update_readme_citation(version):
    """Update README.md software citation"""
    readme_path = Path("README.md")
    if not readme_path.exists():
        return

    with open(readme_path) as f:
        content = f.read()

    # Update BibTeX citation
    content = re.sub(r"version={.*?}", f"version={{{version}}}", content)
    content = re.sub(r"year={.*?}", f"year={{{datetime.now().year}}}", content)

    with open(readme_path, "w") as f:
        f.write(content)

    print(f"✅ Updated README.md citation to version {version}")


def update_docs_index(version):
    """Update version in docs/index.md if it exists"""
    docs_index_path = Path("docs/index.md")
    if not docs_index_path.exists():
        return

    with open(docs_index_path) as f:
        content = f.read()

    # Update version in BibTeX citation
    content = re.sub(r"version={.*?}", f"version={{{version}}}", content)
    content = re.sub(r"year={.*?}", f"year={{{datetime.now().year}}}", content)

    with open(docs_index_path, "w") as f:
        f.write(content)

    print(f"✅ Updated docs/index.md to version {version}")


def update_docs_citation(version):
    """Update version in docs/citation.md if it exists"""
    docs_citation_path = Path("docs/citation.md")
    if not docs_citation_path.exists():
        return

    with open(docs_citation_path) as f:
        content = f.read()

    # Update version in all BibTeX citations
    content = re.sub(r"version={.*?}", f"version={{{version}}}", content)
    content = re.sub(r"year={.*?}", f"year={{{datetime.now().year}}}", content)

    with open(docs_citation_path, "w") as f:
        f.write(content)

    print(f"✅ Updated docs/citation.md to version {version}")


def main():
    """Sync all version numbers"""
    version = get_version_from_init()
    print(f"Syncing all files to version {version} from __init__.py")

    update_citation_cff(version)
    update_readme_citation(version)
    update_docs_index(version)
    update_docs_citation(version)

    print(f"🎉 All version numbers synchronized to {version}")


if __name__ == "__main__":
    main()
