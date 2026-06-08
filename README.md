# SignalSeeker: Educational Binary Classification with XGBoost

**SignalSeeker** is a comprehensive, production-quality Python package that teaches how to build machine learning pipelines for **binary classification using Boosted Decision Trees (XGBoost)** in a **signal/background separation context**, inspired by particle physics applications.

[![GitHub](https://img.shields.io/badge/GitHub-SignalSeeker-blue)](https://github.com/Amidn/SignalSeeker)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

In particle physics experiments, **signal** refers to the desired particle interaction you're searching for, while **background** refers to all other processes that mimic the signal. The challenge is to build a classifier that:

1. **Identifies signal events** with high efficiency (recall)
2. **Rejects background events** with high purity (precision)
3. **Provides interpretable probability scores** (BDT scores) for decision-making

SignalSeeker teaches all these concepts through a **production-quality, fully-featured Python package**.

## ✨ Key Features

✅ **Complete ML Pipeline**: Data generation → Preprocessing → Training → Tuning → Evaluation → Visualization

✅ **Realistic Imbalanced Data**: Generates synthetic datasets with ~90% background, 10% signal (fully configurable)

✅ **XGBoost Implementation**: Industry-standard gradient boosted decision trees with full hyperparameter control

✅ **Hyperparameter Tuning**: Bayesian optimization using Optuna for automatic parameter discovery

✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, Confusion Matrix

✅ **Publication-Quality Visualizations**:
   - **Probability Score Distribution**: Core visualization showing signal/background separation
   - ROC and Precision-Recall curves
   - Feature importance analysis
   - Confusion matrix heatmap
   - Summary dashboard

✅ **GPU Acceleration**: Full CUDA support for XGBoost (optional)

✅ **Professional Package**: Installable via pip, PyPI-ready, proper src/ layout

✅ **Comprehensive Testing**: Full pytest coverage with CI/CD

✅ **Educational Comments**: Detailed docstrings explaining *why* each step is performed

✅ **Professional Code Quality**: Type hints, OOP design, proper error handling, Black-formatted

## 📦 Installation

### Option 1: Development Install (Recommended for Learning)

```bash
git clone https://github.com/Amidn/SignalSeeker.git
cd SignalSeeker
pip install -e .
```

This allows you to modify source files and see changes immediately.

### Option 2: Regular Install from GitHub

```bash
pip install git+https://github.com/Amidn/SignalSeeker.git
```

### Option 3: Install from PyPI (when published)

```bash
pip install signalseeker
```

### Option 4: Install with Development Tools

```bash
pip install -e ".[dev]"
```

Includes testing, linting, and documentation tools.

### Option 5: Install with GPU Support

```bash
pip install -e ".[gpu]"
```

Requires CUDA Toolkit to be installed. See [GPU Setup Guide](#gpu-acceleration-setup).

### Option 6: Install from Requirements Files

```bash
# Core dependencies only
pip install -r requirements.txt

# With development tools
pip install -r requirements-dev.txt

# With GPU support
pip install -r requirements-gpu.txt
```

## 🚀 Quick Start

### Run the Complete Pipeline

```bash
python -m signalseeker.main
```

Or after installation:

```bash
signal-seeker
```

This will:
1. Generate synthetic imbalanced data (10,000 samples, 10% signal)
2. Preprocess features (scaling, missing value handling)
3. Build and train an XGBoost model
4. Evaluate on validation and test sets
5. Generate publication-quality visualizations
6. Save all results to `./results/run_TIMESTAMP/`

### With Hyperparameter Tuning (Better Performance, Slower)

```bash
signal-seeker --tune
```

Uses Bayesian optimization to find optimal hyperparameters (takes ~2-5 minutes).

### Custom Configuration

```bash
signal-seeker --n-samples 50000 --signal-fraction 0.15 --output-dir ./my_results
```

**Available options:**
- `--tune`: Enable hyperparameter tuning
- `--output-dir`: Output directory for results
- `--n-samples`: Total number of samples to generate
- `--signal-fraction`: Fraction of signal samples (0-1)

### Python API Usage

```python
from signalseeker import SignalSeekerPipeline, DEFAULT_CONFIG

# Use default configuration
pipeline = SignalSeekerPipeline(DEFAULT_CONFIG)
results = pipeline.run(use_tuning=False)

# Or customize configuration
config = DEFAULT_CONFIG
config.data.n_samples = 50000
config.data.signal_fraction = 0.15
config.xgboost.max_depth = 8
config.xgboost.learning_rate = 0.05

pipeline = SignalSeekerPipeline(config)
results = pipeline.run(use_tuning=True)

# Access results
val_auc = results["validation_results"]["all_metrics"]["auc_roc"]
test_auc = results["test_results"]["all_metrics"]["auc_roc"]
model = results["model"]

print(f"Validation AUC: {val_auc:.4f}")
print(f"Test AUC: {test_auc:.4f}")
```

## 📊 Understanding the Output

### The Probability Score Distribution (Most Important Plot)

This is the fundamental visualization in signal/background separation:

```
Signal Distribution:   ████████████████░░░░░░░░░
Background Dist:      ░░░░░░░░████████████████████
                      0.0   Cut   1.0
                      (threshold)
```

- **Signal**: Concentrated near 1.0 (high probability of being signal)
- **Background**: Concentrated near 0.0 (low probability of being signal)  
- **Cut**: The threshold above which we classify events as signal

**Good Model**: Minimal overlap, clear separation  
**Poor Model**: Heavy overlap, hard to separate

This plot is analogous to the "BDT score" or "discriminant" in particle physics.

### Other Key Metrics

| Metric | Meaning | Ideal Value |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / Total | 1.0 (but misleading for imbalanced data) |
| **Precision** | TP / (TP + FP) - Of predicted signals, how many are true? | 1.0 |
| **Recall (TPR)** | TP / (TP + FN) - Of true signals, how many did we find? | 1.0 |
| **F1-Score** | Harmonic mean of precision and recall | 1.0 |
| **AUC-ROC** | Area under ROC curve (threshold-independent) | 1.0 |
| **AUC-PR** | Area under Precision-Recall curve (better for imbalanced) | 1.0 |

## 📁 Package Structure

```
SignalSeeker/
├── src/signalseeker/          # Main package code
│   ├── __init__.py            # Package initialization & exports
│   ├── config.py              # Configuration management
│   ├── data_loader.py         # Synthetic data generation
│   ├── preprocessor.py        # Feature scaling & normalization
│   ├── model_builder.py       # XGBoost model initialization
│   ├── trainer.py             # Training with early stopping
│   ├── tuner.py               # Bayesian hyperparameter optimization
│   ├── metrics.py             # Evaluation metrics
│   ├── visualizer.py          # Publication-quality plots
│   ├── utils.py               # Logging and utilities
│   └── main.py                # Pipeline orchestrator
├── tests/                      # Pytest test suite
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_model_builder.py
│   ├── test_trainer.py
│   ├── test_tuner.py
│   ├── test_metrics.py
│   └── test_visualizer.py
├── .github/workflows/          # CI/CD pipelines
│   ├── tests.yml              # Run tests on push/PR
│   └── publish.yml            # Auto-publish to PyPI on release
├── examples/                   # Example scripts and notebooks
│   ├── example_usage.py
│   └── custom_data_example.py
├── README.md                   # This file
├── CONTRIBUTING.md            # Contributing guidelines
├── LICENSE                     # MIT License
├── setup.py                    # Package installation script
├── pyproject.toml             # Modern Python packaging config
├── requirements.txt           # Core dependencies
├── requirements-dev.txt       # Development dependencies
├── requirements-gpu.txt       # GPU support dependencies
└── .pre-commit-config.yaml   # Code quality checks
```

## 🎓 Educational Concepts Covered

### 1. **Imbalanced Classification**
   - Why accuracy alone is misleading
   - Class weighting and scale_pos_weight
   - Precision vs Recall trade-offs
   - ROC and Precision-Recall curves

### 2. **Boosting & Decision Trees**
   - How gradient boosting works
   - Why boosting is effective for this problem
   - Feature importance in tree ensembles
   - Overfitting and regularization

### 3. **Cross-Validation & Early Stopping**
   - Preventing overfitting
   - Validation-based model selection
   - Learning curves and training dynamics

### 4. **Hyperparameter Tuning**
   - Grid vs Random vs Bayesian search
   - Optuna for efficient optimization
   - Interpreting tuning results
   - Trade-offs between performance and training time

### 5. **Model Evaluation**
   - Multiple metrics for imbalanced data
   - ROC curves and operating points
   - Precision-Recall analysis
   - Threshold optimization

### 6. **Signal/Background Separation**
   - The "cut" concept
   - Probability score interpretation
   - Acceptance vs Purity trade-off
   - Real-world applications in physics

## 🖥️ Advanced Usage

### Custom Data

```python
import numpy as np
from signalseeker import DataPreprocessingPipeline, DataSplitter
from signalseeker import XGBoostModelBuilder, ModelTrainer
from signalseeker.config import PreprocessConfig, XGBoostConfig

# Load your own data
X = np.load("features.npy")
y = np.load("labels.npy")

# Split the data
splitter = DataSplitter(PreprocessConfig())
X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

# Preprocess
preprocessor = DataPreprocessingPipeline(PreprocessConfig())
X_train, X_val, X_test = preprocessor.fit(X_train, X_val, X_test)

# Build and train model
builder = XGBoostModelBuilder(XGBoostConfig())
builder.build_with_class_weights(y_train)
trainer = ModelTrainer(builder)
model, results = trainer.train(X_train, y_train, X_val, y_val)

# Make predictions
predictions = model.predict_proba(X_test)[:, 1]
```

### Hyperparameter Tuning Only

```python
from signalseeker import HyperparameterTuner
from signalseeker.config import TunerConfig, XGBoostConfig

tuner = HyperparameterTuner(TunerConfig(), XGBoostConfig())
results = tuner.optimize(X_train, y_train, n_trials=50)

print(f"Best AUC: {results['best_score']:.4f}")
print(f"Best params: {results['best_params']}")

# Create model with best params
best_model_builder = tuner.create_model_from_best()
```

### Visualization Only

```python
from signalseeker import ModelVisualizer
from signalseeker.config import VisualizerConfig

viz = ModelVisualizer(VisualizerConfig())

# Probability distribution
viz.plot_probability_score_distribution(y_test, predictions)

# ROC curve
viz.plot_roc_curve(y_test, predictions)

# Feature importance
viz.plot_feature_importance(model, top_n=15)
```

## 📈 Performance Expectations

On the default 10,000 sample synthetic dataset:

| Metric | Without Tuning | With Tuning |
|--------|---|---|
| Test AUC-ROC | ~0.92 | ~0.95 |
| Test AUC-PR | ~0.88 | ~0.92 |
| Test F1-Score | ~0.80 | ~0.85 |
| Training Time | ~2 sec | ~3-5 min |

Performance improves with tuning, and improvements are typically more dramatic on real-world datasets.

## 🔧 GPU Acceleration Setup

### Prerequisites

1. **NVIDIA GPU** with CUDA Compute Capability 3.5+
2. **CUDA Toolkit** 10.2+ (check with `nvcc --version`)
3. **cuDNN** (optional, improves performance)

### Installation

```bash
# 1. Install CUDA Toolkit (from NVIDIA website for your OS)

# 2. Verify CUDA installation
nvcc --version

# 3. Install signalseeker with GPU support
pip install -e ".[gpu]"

# OR manually:
pip install xgboost[gpu]
```

### Enabling GPU in SignalSeeker

```python
from signalseeker import SignalSeekerPipeline, DEFAULT_CONFIG

config = DEFAULT_CONFIG
config.use_gpu = True  # Enable GPU acceleration
config.xgboost.tree_method = "gpu_hist"  # Use GPU for tree building

pipeline = SignalSeekerPipeline(config)
results = pipeline.run()
```

### Configuration Options

```python
# GPU-specific XGBoost parameters
config.xgboost.tree_method = "gpu_hist"  # GPU tree building
config.xgboost.gpu_id = 0  # Which GPU to use (if multiple)
config.xgboost.predictor = "gpu_predictor"  # GPU prediction
```

### Troubleshooting GPU

```bash
# Check if GPU is detected
python -c "import xgboost as xgb; print(xgb.get_config())"

# Test GPU acceleration
python -c "from xgboost import XGBClassifier; m = XGBClassifier(tree_method='gpu_hist'); print('GPU enabled!')"
```

## 🧪 Testing

Run the complete test suite:

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=src/signalseeker --cov-report=html

# Run specific test file
pytest tests/test_model_builder.py

# Run specific test
pytest tests/test_model_builder.py::TestXGBoostModelBuilder::test_initialization

# Run only fast tests
pytest -m "not slow"

# Run in parallel (faster)
pytest -n auto
```

## 🔄 CI/CD

This project uses **GitHub Actions** for continuous integration:

- **Tests**: Runs on Python 3.8-3.12 on every push/PR
- **Code Quality**: Black formatting, isort, flake8, mypy
- **Coverage**: Checks test coverage on push
- **Publishing**: Auto-publishes to PyPI on version tag

See `.github/workflows/` for configuration.

## 📚 Documentation

### Docstrings

Every module, class, and function includes comprehensive docstrings explaining:
- **What** the code does
- **Why** each step is necessary
- **How** to use it with examples

Read the docstrings in source files for detailed explanations.

### Type Hints

All functions include type hints for better IDE support and code clarity.

```python
def compute_class_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for class imbalance."""
```

## ❓ FAQ

**Q: Why is accuracy not a good metric for imbalanced data?**  
A: With 90% background, a model predicting "all background" gets 90% accuracy without detecting any signal!

**Q: What's the difference between AUC-ROC and AUC-PR?**  
A: For imbalanced data, AUC-PR is more informative because it focuses on the minority class (signal).

**Q: Why use early stopping?**  
A: After some boosting rounds, the model starts overfitting to noise. Early stopping detects when validation loss stops improving and halts training.

**Q: Can I use my own data?**  
A: Yes! Replace the `DataLoader` with code that loads your data, then use the rest of the pipeline unchanged.

**Q: Is GPU acceleration supported?**  
A: Yes! Set `config.use_gpu = True` if you have a CUDA-capable GPU.

**Q: How do I install this on Windows/Mac/Linux?**  
A: The installation process is identical across platforms. Use the standard pip commands.

**Q: Can I train on larger datasets?**  
A: Yes! Increase `n_samples` in the config. For very large datasets (>1M samples), consider using GPU acceleration.

**Q: How do I customize the configuration?**  
A: Edit `config.py` or pass a modified `PipelineConfig` object to `SignalSeekerPipeline`.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📖 References

- **XGBoost Paper**: Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System" - [Link](https://arxiv.org/abs/1603.02754)
- **Particle Physics ML**: Baldi et al. (2016) - "Searching for exotic particles in high-energy physics with deep learning" - [Link](https://arxiv.org/abs/1402.4735)
- **Imbalanced Learning**: He & Garcia (2009) - "Learning from Imbalanced Data" - [Link](https://ieeexplore.ieee.org/document/5128907)
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Scikit-learn**: https://scikit-learn.org/

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Amid Nayerhoda**  
Email: amid.nayerhoda@gmail.com  
GitHub: [@Amidn](https://github.com/Amidn)

## 🙏 Acknowledgments

This package was designed as an educational resource to teach:
- Machine learning best practices
- Professional Python package development
- Signal/background separation techniques used in particle physics
- Real-world ML pipeline design

## 📞 Support

For issues, questions, or suggestions:
1. Check the [FAQ](#-faq) above
2. Check existing [GitHub Issues](https://github.com/Amidn/SignalSeeker/issues)
3. Create a new issue with clear description and example code
4. Read the docstrings in the source code for detailed explanations

---

**Made with ❤️ for science and education**
