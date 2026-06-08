"""SignalSeeker: Educational binary classification with XGBoost for signal/background separation.

This package teaches students how to build production-quality ML pipelines using
Boosted Decision Trees (XGBoost) for binary classification in a signal/background
separation context, inspired by particle physics applications.

Main Components:
- DataLoader: Generate synthetic imbalanced datasets
- DataPreprocessingPipeline: Feature scaling and preprocessing
- XGBoostModelBuilder: Configure XGBoost classifiers
- ModelTrainer: Training with early stopping
- HyperparameterTuner: Bayesian optimization for hyperparameters
- MetricsCalculator: Comprehensive evaluation metrics
- ModelVisualizer: Publication-quality plots
- SignalSeekerPipeline: Complete end-to-end pipeline

Quick Start:
    from signalseeker import SignalSeekerPipeline, DEFAULT_CONFIG

    pipeline = SignalSeekerPipeline(DEFAULT_CONFIG)
    results = pipeline.run(use_tuning=False)

References:
- XGBoost Paper: Chen & Guestrin (2016)
- Particle Physics ML: Baldi et al. (2016)
"""

__version__ = "0.2.0"
__author__ = "SignalSeeker Team"
__email__ = "amid.nayerhoda@gmail.com"

from .config import (
    DataConfig,
    PreprocessConfig,
    XGBoostConfig,
    TunerConfig,
    VisualizerConfig,
    PipelineConfig,
    DEFAULT_CONFIG,
)
from .data_loader import DataLoader, DataSplitter, create_data_pipeline
from .preprocessor import DataPreprocessingPipeline, FeaturePreprocessor, MissingValueHandler
from .model_builder import XGBoostModelBuilder, create_model_for_data
from .trainer import ModelTrainer, AdaptiveTrainer, train_model_simple
from .tuner import HyperparameterTuner
from .metrics import MetricsCalculator, PerformanceReporter, evaluate_model
from .visualizer import ModelVisualizer
from .main import SignalSeekerPipeline
from .utils import (
    setup_logging,
    save_results,
    load_results,
    create_output_directory,
    ExperimentTracker,
)

__all__ = [
    # Config classes
    "DataConfig",
    "PreprocessConfig",
    "XGBoostConfig",
    "TunerConfig",
    "VisualizerConfig",
    "PipelineConfig",
    "DEFAULT_CONFIG",
    # Data handling
    "DataLoader",
    "DataSplitter",
    "create_data_pipeline",
    # Preprocessing
    "DataPreprocessingPipeline",
    "FeaturePreprocessor",
    "MissingValueHandler",
    # Model building
    "XGBoostModelBuilder",
    "create_model_for_data",
    # Training
    "ModelTrainer",
    "AdaptiveTrainer",
    "train_model_simple",
    # Tuning
    "HyperparameterTuner",
    # Evaluation
    "MetricsCalculator",
    "PerformanceReporter",
    "evaluate_model",
    # Visualization
    "ModelVisualizer",
    # Pipeline
    "SignalSeekerPipeline",
    # Utilities
    "setup_logging",
    "save_results",
    "load_results",
    "create_output_directory",
    "ExperimentTracker",
]
