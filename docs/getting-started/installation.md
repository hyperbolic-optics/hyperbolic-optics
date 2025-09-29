# Installation

## Requirements

- Python 3.12 or higher
- pip or uv package manager

## Install from PyPI

### Using pip

```bash
pip install hyperbolic-optics
```

### Using uv (recommended)

```bash
uv add hyperbolic-optics
```

## Development Installation

If you want to contribute or modify the source code:

```bash
# Clone the repository
git clone https://github.com/MarkCunningham0410/hyperbolic_optics.git
cd hyperbolic_optics

# Install with all extras
uv sync --all-extras
```

## Verify Installation

Test your installation:

```python
from hyperbolic_optics.structure import Structure
print("Installation successful!")
```

## Dependencies

The package automatically installs:

- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0

## Optional Dependencies

For development:

```bash
uv sync --extra dev
```

This includes pytest, black, isort, and flake8 for testing and code quality.