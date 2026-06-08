# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SignalSeeker** is a production-quality educational Python package for binary classification using XGBoost in a signal/background separation context (inspired by particle physics). It's published to PyPI and includes a complete ML pipeline with tests, CI/CD, and professional code quality.

**Key distinction**: This is an educational package, so code should prioritize clarity and explanatory docstrings over brevity. The package is already mature (v0.2.1 on PyPI), so changes should be minimal and well-tested.

## Common Commands

### Development Setup
```bash
pip install -e ".[dev]"              # Install with all dev tools
pip install -r requirements-dev.txt  # Alternative method
```

### Testing
```bash
pytest tests/                                    # Run all tests
pytest tests/test_config.py                     # Single test file
pytest tests/test_config.py::TestDataConfig     # Single test class
pytest --cov=src/signalseeker --cov-report=html  # Coverage report
pytest -n auto                                  # Parallel execution
pytest -m "not slow"                           # Skip slow tests
```

### Code Quality
```bash
black src/ tests/                  # Format code
isort src/ tests/                  # Sort imports
flake8 src/ tests/                 # Lint
mypy src/                          # Type check
```

### Running the Package
```bash
signal-seeker                      # Full pipeline with default config
signal-seeker --tune               # With hyperparameter tuning
signal-seeker --n-samples 50000    # Custom parameters
python -m signalseeker.main        # Alternative invocation
```

### Publishing
```bash
git tag -a v0.2.X -m "Release message"   # Create version tag
git push origin v0.2.X                   # Push tag (triggers publish)
# GitHub Actions auto-publishes to PyPI via trusted publisher
```

## Architecture

### Pipeline Structure
The core workflow flows through these modules:

1. **config.py**: Central configuration (PipelineConfig, DataConfig, XGBoostConfig, etc.)
   - All hyperparameters defined as dataclasses with sensible defaults
   - Key: `use_gpu` flag for CUDA acceleration, `scale_pos_weight` for class imbalance

2. **data_loader.py**: Synthetic data generation and splitting
   - DataLoader: generates imbalanced classification data (configurable signal/background ratio)
   - DataSplitter: stratified train/val/test splits to maintain class balance
   - Factory: `create_data_pipeline()` combines both

3. **preprocessor.py**: Feature scaling and preprocessing
   - FeaturePreprocessor: StandardScaler with save/load for reproducibility
   - MissingValueHandler: SimpleImputer for NaN values
   - DataPreprocessingPipeline: orchestrates both, persists scalers to disk

4. **model_builder.py**: XGBoost model initialization
   - XGBoostModelBuilder: wraps XGBClassifier with class weight computation
   - Handles imbalance via `scale_pos_weight` (n_negative / n_positive)
   - Key method: `build_with_class_weights(y_train)` for automatic weighting

5. **trainer.py**: Training with early stopping
   - ModelTrainer: fits model on training data, monitors validation loss
   - Returns trained model and training history/metrics
   - Early stopping prevents overfitting after N rounds without improvement

6. **tuner.py**: Bayesian hyperparameter optimization
   - HyperparameterTuner: uses Optuna for efficient search
   - Objective function: cross-validated AUC score
   - Prunes unpromising trials early for speed

7. **metrics.py**: Comprehensive evaluation
   - MetricsCalculator: computes accuracy, precision, recall, F1, AUC-ROC, AUC-PR
   - PerformanceReporter: formatted reporting
   - Key insight: AUC-PR preferred for imbalanced data (focuses on minority class)

8. **visualizer.py**: Publication-quality plots
   - Core plot: `plot_probability_score_distribution()` — shows signal/background separation
   - Others: ROC curve, PR curve, confusion matrix, feature importance, summary dashboard
   - Saves to configured output directory with high DPI (300)

9. **main.py**: Pipeline orchestrator (SignalSeekerPipeline)
   - Runs full pipeline: generate → preprocess → build → train → evaluate → visualize
   - Entry point for CLI (`signal-seeker` command)
   - Supports `--tune` flag for hyperparameter optimization

10. **utils.py**: Logging and utilities
    - setup_logging(): configurable logger with file and console output
    - save_results()/load_results(): JSON persistence with numpy type serialization
    - ExperimentTracker: logs metrics and prints summaries

### Key Design Patterns

- **Configuration objects** (dataclasses): PipelineConfig contains all sub-configs; passed through pipeline for reproducibility
- **Factory functions**: `create_data_pipeline()`, `create_model_for_data()` — simplify common operations
- **Fit/transform pattern**: Preprocessors fit on training data, transform all splits — prevents data leakage
- **Save/load methods**: Preprocessing state saved to `scaler.joblib` + `imputer.joblib` for later inference
- **Automatic class weighting**: Model builder computes `scale_pos_weight` from training distribution

## Important Codebase Notes

### Imbalanced Classification
- Default is 10% signal, 90% background (configurable)
- XGBoost handles this via `scale_pos_weight` parameter (automatically computed)
- Metrics: prefer AUC-PR over AUC-ROC for imbalanced data (focuses on minority class)
- The "probability score distribution" plot is the core educational visualization

### Testing
- 69 tests across 5 modules (config, data_loader, model_builder, preprocessor, metrics)
- Uses pytest with fixtures for reusable test data
- Slow tests marked with `@pytest.mark.slow` for selective running
- Coverage target: 80%+ via `--cov` flag

### CI/CD
- Tests run on Python 3.9, 3.10, 3.11 on Ubuntu (simplified matrix for speed)
- Tests triggered on every push to main and all PRs
- Publish workflow auto-triggers on git tags (`v*` pattern)
- Uses GitHub trusted publisher for PyPI (no API tokens stored)
- Failing deployments in GitHub UI are benign — they're from earlier failed attempts before package was live

### GPU Support
- XGBoost GPU mode enabled via `config.use_gpu = True`
- Requires `pip install -e ".[gpu]"` and CUDA Toolkit installed
- Sets `tree_method='gpu_hist'` for tree building

### Documentation Quality
- Every public class/function has detailed docstring explaining the *why* (educational focus)
- Type hints on all functions
- Pedagogical comments explain boosting, class imbalance, early stopping, etc.

## File Locations Quick Reference

- **Configuration**: src/signalseeker/config.py (10 lines per config class is typical)
- **Tests**: tests/test_*.py (pytest conventions; one test file per main module)
- **Entry point**: src/signalseeker/main.py (SignalSeekerPipeline class)
- **Package exports**: src/signalseeker/__init__.py (public API listed in `__all__`)
- **CI/CD**: .github/workflows/ (tests.yml for testing, publish.yml for PyPI)
- **Development**: pyproject.toml (modern Python packaging, replaces most of setup.py)

## Debugging Tips

- **Import errors**: After modifying imports, reinstall: `pip install -e .`
- **Test failures**: Check if recent config changes broke assumptions (e.g., new field added to XGBoostConfig)
- **GPU errors**: Verify CUDA with `nvcc --version`; fall back to CPU testing if not available
- **Type checking**: Run `mypy src/` to find type mismatches before runtime
