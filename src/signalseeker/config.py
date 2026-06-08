"""
Configuration management for SignalSeeker package.

This module centralizes all hyperparameters, paths, and settings used throughout
the pipeline. Using a config object promotes reproducibility and makes it easy
to experiment with different parameter combinations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for synthetic data generation.

    Attributes:
        n_samples: Total number of samples to generate (signal + background).
        signal_fraction: Fraction of samples that are signal (0-1).
                         Typically small to simulate imbalance.
        n_features: Number of input features.
        n_informative: Number of features actually used for classification.
        random_state: Random seed for reproducibility.
    """
    n_samples: int = 10000
    signal_fraction: float = 0.1  # 10% signal, 90% background (imbalanced)
    n_features: int = 20
    n_informative: int = 15
    random_state: int = 42


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing steps.

    Attributes:
        test_size: Fraction of data used for testing (0-1).
        val_size: Fraction of training data used for validation (0-1).
        scale_features: Whether to standardize features (important for some models).
        handle_missing: Whether to handle missing values.
    """
    test_size: float = 0.2
    val_size: float = 0.2
    scale_features: bool = True
    handle_missing: bool = True


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model hyperparameters.

    Attributes:
        max_depth: Maximum tree depth (controls complexity).
        learning_rate: Shrinkage parameter (lower = more conservative).
        n_estimators: Number of boosting rounds (trees to build).
        subsample: Fraction of samples used per tree (reduces overfitting).
        colsample_bytree: Fraction of features used per tree.
        scale_pos_weight: Weight to balance classes (important for imbalanced data).
        objective: Loss function ('binary:logistic' for probability output).
        eval_metric: Evaluation metric during training.
    """
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: Optional[float] = None  # Will be computed based on class weights
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    random_state: int = 42


@dataclass
class TunerConfig:
    """Configuration for hyperparameter tuning.

    Attributes:
        n_trials: Number of optimization trials.
        cv_folds: Number of cross-validation folds.
        timeout: Maximum time per trial in seconds.
    """
    n_trials: int = 20
    cv_folds: int = 5
    timeout: Optional[int] = None


@dataclass
class VisualizerConfig:
    """Configuration for visualization settings.

    Attributes:
        figsize: Default figure size for plots (width, height).
        dpi: Resolution of saved figures.
        style: Matplotlib style name.
        save_plots: Whether to save plots to disk.
        output_dir: Directory for saving plots.
    """
    figsize: tuple = (10, 6)
    dpi: int = 300
    style: str = "seaborn-v0_8-darkgrid"
    save_plots: bool = True
    output_dir: Path = field(default_factory=lambda: Path("./outputs/plots"))


@dataclass
class PipelineConfig:
    """Master configuration object combining all sub-configs.

    This single object is passed through the pipeline, making it easy to
    track all settings and reproduce experiments.
    """
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    tuner: TunerConfig = field(default_factory=TunerConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)

    # General settings
    random_state: int = 42
    verbose: bool = True
    use_gpu: bool = False  # Set to True if you have GPU support

    def __post_init__(self) -> None:
        """Ensure output directories exist."""
        self.visualizer.output_dir.mkdir(parents=True, exist_ok=True)


# Global default configuration instance
DEFAULT_CONFIG = PipelineConfig()
