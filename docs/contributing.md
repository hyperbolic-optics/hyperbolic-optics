# Contributing

We welcome contributions to the Hyperbolic Optics package! This guide will help you get started.

## Ways to Contribute

- **Report bugs** via [GitHub Issues](https://github.com/MarkCunningham0410/hyperbolic_optics/issues)
- **Request features** or enhancements
- **Improve documentation** (fix typos, add examples, clarify explanations)
- **Submit bug fixes** or new features via Pull Requests
- **Add new materials** to the materials library
- **Share your research** that uses this package

## Development Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Git

### Getting Started

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/hyperbolic_optics.git
cd hyperbolic_optics
```

2. **Install dependencies**

```bash
# Using uv (recommended)
uv sync --all-extras

# Or using pip
pip install -e ".[dev]"
```

3. **Verify installation**

```bash
# Run tests
uv run pytest

# Check code works
python examples/calcite.py
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

Edit the code in your favorite editor. The main package is in `hyperbolic_optics/`.

### 3. Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=hyperbolic_optics --cov-report=html

# Run specific test file
uv run pytest tests/test_structure.py

# Run specific test
uv run pytest tests/test_structure.py::TestStructureBasicFunctionality::test_simple_scenario_execution
```

### 4. Check Code Quality

```bash
# Format code with black
uv run black hyperbolic_optics tests

# Sort imports with isort
uv run isort hyperbolic_optics tests

# Lint with flake8
uv run flake8 hyperbolic_optics tests --max-line-length=100 --extend-ignore=E203,W503
```

### 5. Update Documentation

If you added new features:

- Add docstrings to your functions/classes (they auto-generate API docs!)
- Update relevant user guide pages in `docs/`
- Add examples if appropriate

Build docs locally to preview:

```bash
uv run mkdocs serve
# Visit http://127.0.0.1:8000
```

### 6. Commit Your Changes

```bash
git add .
git commit -m "Brief description of your changes"
```

Write clear commit messages:
- Use present tense ("Add feature" not "Added feature")
- Be descriptive but concise
- Reference issue numbers if applicable (#123)

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with:
- Clear title and description
- Reference to related issues
- Summary of changes made
- Any breaking changes noted

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1, param2):
    """Brief description of function.
    
    More detailed description if needed, explaining what the
    function does and any important details.
    
    Args:
        param1 (float): Description of param1.
        param2 (np.ndarray): Description of param2.
        
    Returns:
        dict: Description of return value.
        
    Raises:
        ValueError: When param1 is negative.
        
    Example:
        >>> result = example_function(1.0, np.array([1, 2, 3]))
    """
    pass
```

### Test Guidelines

- Write tests for all new features
- Aim for >90% code coverage
- Use descriptive test names
- Group related tests in classes
- Use fixtures for common setup

```python
def test_structure_executes_simple_scenario(simple_payload):
    """Test that structure executes simple scenario correctly."""
    structure = Structure()
    structure.execute(simple_payload)
    
    assert structure.r_pp is not None
    assert structure.r_ss is not None
```

## Project Structure

```
hyperbolic_optics/
├── hyperbolic_optics/          # Main package
│   ├── __init__.py
│   ├── structure.py            # Structure class
│   ├── materials.py            # Material definitions
│   ├── layers.py               # Layer classes
│   ├── mueller.py              # Mueller matrix calculations
│   ├── scenario.py             # Scenario types
│   ├── waves.py                # Wave calculations
│   ├── plots.py                # Plotting functions
│   └── material_params.json    # Material parameters
├── tests/                      # Test suite
│   ├── conftest.py            # Pytest fixtures
│   ├── test_structure.py
│   ├── test_materials.py
│   └── ...
├── examples/                   # Example scripts
├── docs/                       # Documentation
├── .github/workflows/          # CI/CD pipelines
├── pyproject.toml             # Project configuration
└── README.md
```

## Adding New Features

### Adding a New Material

1. Add material parameters to `hyperbolic_optics/material_params.json`
2. Create material class in `hyperbolic_optics/materials.py`
3. Add tests in `tests/test_materials.py`
4. Update documentation in `docs/user-guide/materials.md`

### Adding New Layer Types

1. Create layer class in `hyperbolic_optics/layers.py`
2. Register in `LayerFactory.layer_classes`
3. Add tests in `tests/test_layers.py`
4. Update documentation

## Running CI/CD Locally

The project uses GitHub Actions for CI/CD. You can test similar checks locally:

```bash
# Run tests (like CI does)
uv run pytest --cov=hyperbolic_optics --cov-report=xml --cov-report=term

# Check formatting
uv run black --check hyperbolic_optics tests
uv run isort --check-only hyperbolic_optics tests
uv run flake8 hyperbolic_optics tests --max-line-length=100 --extend-ignore=E203,W503
```

## Reporting Issues

When reporting bugs, please include:

- Python version (`python --version`)
- Package version (`pip show hyperbolic-optics`)
- Operating system
- Minimal code to reproduce the issue
- Full error traceback
- Expected vs actual behavior

Example:

```markdown
## Bug Description
Brief description of the bug

## To Reproduce
\```python
# Minimal code to reproduce
\```

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: macOS 14.2
- Python: 3.12.1
- Package version: 0.1.8
```

## Code Review Process

1. All submissions require review
2. Maintainers will provide feedback
3. Address review comments by pushing new commits
4. Once approved, maintainers will merge your PR

## Questions?

- Open a [Discussion](https://github.com/MarkCunningham0410/hyperbolic_optics/discussions) for questions
- Check existing [Issues](https://github.com/MarkCunningham0410/hyperbolic_optics/issues)
- Email: m.cunningham.2@research.gla.ac.uk

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Thank you for considering contributing to Hyperbolic Optics! Every contribution helps make the package better for everyone.