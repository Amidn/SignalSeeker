# Contributing to SignalSeeker

Thank you for your interest in contributing to SignalSeeker! This document provides guidelines and instructions for contributing.

## 🤝 Code of Conduct

Be respectful, inclusive, and constructive in all interactions. We welcome contributions from everyone.

## 💡 How to Contribute

### Reporting Bugs

1. **Check if the bug exists**: Search [existing issues](https://github.com/yourusername/SignalSeeker/issues)
2. **Include details**:
   - Python version (`python --version`)
   - Package version (`pip show signalseeker`)
   - Operating system
   - Minimal code to reproduce the issue
   - Full error traceback
3. **Create an issue** with clear title and description

### Suggesting Enhancements

1. Use the issue tracker with a clear title
2. Describe the motivation and use case
3. Provide examples of how it would be used
4. Discuss pros/cons if applicable

### Submitting Code Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/SignalSeeker.git
   cd SignalSeeker
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # OR for bug fixes:
   git checkout -b fix/issue-number-description
   ```

5. **Make your changes**
   - Write clean, well-commented code
   - Follow the code style (see below)
   - Add/update tests for your changes
   - Update documentation if needed

6. **Run tests locally**
   ```bash
   pytest tests/
   pytest --cov=src/signalseeker
   ```

7. **Format your code**
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

8. **Commit your changes**
   ```bash
   git add .
   git commit -m "Descriptive commit message"
   ```

9. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

10. **Create a Pull Request**
    - Go to the main repository
    - Click "New Pull Request"
    - Select your branch
    - Fill in the PR template
    - Reference related issues

## 📝 Code Style

### Formatting

We use **Black** for code formatting (line length: 100 characters):

```bash
black src/ tests/
```

### Import Sorting

We use **isort** for import organization:

```bash
isort src/ tests/
```

### Linting

Run **flake8** to check for style issues:

```bash
flake8 src/ tests/
```

### Type Checking

We use **mypy** for optional type checking:

```bash
mypy src/
```

### Code Style Guidelines

```python
# ✅ Good
def calculate_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate evaluation metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred_proba: Predicted probabilities.
        threshold: Classification threshold.
    
    Returns:
        Dictionary with metric values.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
    }

# ❌ Bad
def calc_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
    }
```

### Documentation

- Use docstrings for all public modules, classes, and functions
- Follow Google-style docstrings
- Include type hints for all parameters and return values
- Explain the "why", not the "what"

```python
def compute_class_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for handling class imbalance.
    
    In imbalanced binary classification, we weight the positive class
    to compensate for its rarity. This prevents the model from simply
    predicting the majority class and getting high accuracy.
    
    Formula: scale_pos_weight = (n_negative / n_positive)
    
    Args:
        y: Array of binary labels (0 or 1).
    
    Returns:
        Computed weight for the positive class.
    
    Raises:
        ValueError: If no positive samples found.
    """
    n_positive = np.sum(y == 1)
    if n_positive == 0:
        raise ValueError("No positive samples found in y")
    
    n_negative = np.sum(y == 0)
    return float(n_negative / n_positive)
```

## 🧪 Testing

All contributions must include tests. We aim for >80% code coverage.

### Writing Tests

```python
import pytest
import numpy as np
from signalseeker import DataLoader
from signalseeker.config import DataConfig

class TestDataLoader:
    """Test suite for DataLoader."""
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        config = DataConfig(n_samples=1000, signal_fraction=0.1)
        loader = DataLoader(config)
        assert loader.config.n_samples == 1000
    
    def test_generate_data(self):
        """Test data generation."""
        config = DataConfig(n_samples=1000, signal_fraction=0.1)
        loader = DataLoader(config)
        X, y = loader.generate_data()
        
        assert X.shape[0] == 1000
        assert y.shape[0] == 1000
        assert np.sum(y == 1) == pytest.approx(100, abs=10)
    
    @pytest.mark.slow
    def test_large_dataset(self):
        """Test with large dataset (slow)."""
        config = DataConfig(n_samples=100000)
        loader = DataLoader(config)
        X, y = loader.generate_data()
        assert X.shape[0] == 100000
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_data_loader.py

# Specific test
pytest tests/test_data_loader.py::TestDataLoader::test_generate_data

# With coverage
pytest --cov=src/signalseeker --cov-report=html

# Parallel execution
pytest -n auto

# Exclude slow tests
pytest -m "not slow"
```

## 📚 Documentation Updates

If your change affects documentation:

1. Update relevant docstrings
2. Update README.md if adding features
3. Add examples for new functionality

## 🔄 Pre-commit Hooks

To automatically check your code before committing:

```bash
pre-commit install
pre-commit run --all-files
```

This runs:
- Black (formatting)
- isort (import sorting)
- flake8 (linting)
- mypy (type checking)

## 📦 Making a Release

(For maintainers)

1. Update version in `src/signalseeker/__init__.py` and `pyproject.toml`
2. Create a git tag: `git tag v0.2.0`
3. Push tag: `git push origin v0.2.0`
4. GitHub Actions automatically publishes to PyPI

## 🐛 Debugging

### Enable Debug Logging

```python
import logging
from signalseeker import setup_logging

logger = setup_logging(verbose=True)
logger.setLevel(logging.DEBUG)
```

### Common Issues

**Import Errors**
```bash
# Reinstall in development mode
pip install -e .
```

**Test Failures**
```bash
# Check environment
python --version
pip show xgboost  # Check if installed
```

**GPU Issues**
```bash
# Verify CUDA
nvcc --version
python -c "import xgboost as xgb; print(xgb.get_config())"
```

## ✅ Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guide (black, isort, flake8)
- [ ] All tests pass (`pytest`)
- [ ] Coverage maintained/improved
- [ ] Type hints added/updated
- [ ] Docstrings added/updated
- [ ] No debug code or print statements
- [ ] Commits are clear and descriptive
- [ ] Related issues referenced
- [ ] No breaking changes (or clearly documented)

## 📞 Getting Help

- **Questions**: Open an issue with the "question" label
- **Discussions**: Use GitHub Discussions if available
- **Slack/Chat**: (if applicable) Join our community chat
- **Email**: Create an issue for visibility

## 🎓 Learning Resources

To better understand the codebase:

1. Read the [README.md](README.md)
2. Review module docstrings: `python -c "import signalseeker; help(signalseeker)"`
3. Check existing tests for usage examples
4. Review related papers in [References](README.md#-references)

## 🙏 Recognition

Contributors will be recognized in:
- Commit history
- Release notes
- CONTRIBUTORS.md (coming soon)

Thank you for contributing to make SignalSeeker better! 🎉
