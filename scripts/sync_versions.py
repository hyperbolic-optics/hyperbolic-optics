#!/usr/bin/env python3
"""
Synchronize version numbers across all project files.
"""

import re
import toml
from datetime import datetime
from pathlib import Path

def get_version_from_pyproject():
    """Extract version from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path) as f:
        data = toml.load(f)
    return data["project"]["version"]

def update_citation_cff(version):
    """Update CITATION.cff file"""
    cff_path = Path("CITATION.cff")
    if not cff_path.exists():
        return
    
    with open(cff_path) as f:
        content = f.read()
    
    # Update version and date
    content = re.sub(r'version: ".*"', f'version: "{version}"', content)
    content = re.sub(r'date-released: ".*"', 
                    f'date-released: "{datetime.now().strftime("%Y-%m-%d")}"', content)
    
    with open(cff_path, 'w') as f:
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
    content = re.sub(r'version={.*?}', f'version={{{version}}}', content)
    content = re.sub(r'year={.*?}', f'year={{{datetime.now().year}}}', content)
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated README.md citation to version {version}")

def update_init_version(version):
    """Update __version__ in __init__.py if it exists"""
    init_path = Path("hyperbolic_optics/__init__.py")
    if not init_path.exists():
        return
    
    with open(init_path) as f:
        content = f.read()
    
    # Add or update __version__
    if '__version__' in content:
        content = re.sub(r'__version__ = ["\'].*?["\']', 
                        f'__version__ = "{version}"', content)
    else:
        content = f'__version__ = "{version}"\n\n' + content
    
    with open(init_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated __init__.py to version {version}")

def main():
    """Sync all version numbers"""
    version = get_version_from_pyproject()
    print(f"Syncing all files to version {version}")
    
    update_citation_cff(version)
    update_readme_citation(version)
    update_init_version(version)
    
    print(f"🎉 All version numbers synchronized to {version}")

if __name__ == "__main__":
    main()