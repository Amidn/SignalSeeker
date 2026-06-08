"""Tests for configuration module."""

import pytest
from pathlib import Path
from signalseeker.config import (
    DataConfig,
    PreprocessConfig,
    XGBoostConfig,
    TunerConfig,
    VisualizerConfig,
    PipelineConfig,
    DEFAULT_CONFIG,
)


class TestDataConfig:
    """Test suite for DataConfig."""

    def test_initialization_defaults(self):
        """Test DataConfig with default values."""
        config = DataConfig()
        assert config.n_samples == 10000
        assert config.signal_fraction == 0.1
        assert config.n_features == 20
        assert config.n_informative == 15

    def test_initialization_custom(self):
        """Test DataConfig with custom values."""
        config = DataConfig(
            n_samples=50000,
            signal_fraction=0.2,
            n_features=30,
            n_informative=20,
        )
        assert config.n_samples == 50000
        assert config.signal_fraction == 0.2
        assert config.n_features == 30


class TestPreprocessConfig:
    """Test suite for PreprocessConfig."""

    def test_initialization_defaults(self):
        """Test PreprocessConfig with default values."""
        config = PreprocessConfig()
        assert config.test_size == 0.2
        assert config.val_size == 0.2
        assert config.scale_features is True
        assert config.handle_missing is True

    def test_initialization_custom(self):
        """Test PreprocessConfig with custom values."""
        config = PreprocessConfig(
            test_size=0.3,
            val_size=0.1,
            scale_features=False,
        )
        assert config.test_size == 0.3
        assert config.val_size == 0.1
        assert config.scale_features is False


class TestXGBoostConfig:
    """Test suite for XGBoostConfig."""

    def test_initialization_defaults(self):
        """Test XGBoostConfig with default values."""
        config = XGBoostConfig()
        assert config.max_depth == 6
        assert config.learning_rate == 0.1
        assert config.n_estimators == 100
        assert config.subsample == 0.8
        assert config.colsample_bytree == 0.8

    def test_initialization_custom(self):
        """Test XGBoostConfig with custom values."""
        config = XGBoostConfig(
            max_depth=8,
            learning_rate=0.05,
            n_estimators=200,
        )
        assert config.max_depth == 8
        assert config.learning_rate == 0.05
        assert config.n_estimators == 200


class TestTunerConfig:
    """Test suite for TunerConfig."""

    def test_initialization_defaults(self):
        """Test TunerConfig with default values."""
        config = TunerConfig()
        assert config.n_trials == 20
        assert config.cv_folds == 5

    def test_initialization_custom(self):
        """Test TunerConfig with custom values."""
        config = TunerConfig(n_trials=50, cv_folds=10)
        assert config.n_trials == 50
        assert config.cv_folds == 10


class TestVisualizerConfig:
    """Test suite for VisualizerConfig."""

    def test_initialization_defaults(self):
        """Test VisualizerConfig with default values."""
        config = VisualizerConfig()
        assert config.figsize == (10, 6)
        assert config.dpi == 300
        assert config.save_plots is True
        assert isinstance(config.output_dir, Path)

    def test_output_dir_creation(self):
        """Test that output directory is created."""
        config = VisualizerConfig()
        # Directory should exist after __post_init__
        assert config.output_dir.parent.exists()


class TestPipelineConfig:
    """Test suite for PipelineConfig."""

    def test_initialization_defaults(self):
        """Test PipelineConfig with default values."""
        config = PipelineConfig()
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.preprocess, PreprocessConfig)
        assert isinstance(config.xgboost, XGBoostConfig)
        assert isinstance(config.tuner, TunerConfig)
        assert isinstance(config.visualizer, VisualizerConfig)
        assert config.random_state == 42
        assert config.verbose is True
        assert config.use_gpu is False

    def test_initialization_custom(self):
        """Test PipelineConfig with custom sub-configs."""
        data_config = DataConfig(n_samples=5000)
        config = PipelineConfig(data=data_config)
        assert config.data.n_samples == 5000

    def test_gpu_flag(self):
        """Test GPU flag can be toggled."""
        config = PipelineConfig()
        assert config.use_gpu is False
        config.use_gpu = True
        assert config.use_gpu is True


class TestDefaultConfig:
    """Test suite for DEFAULT_CONFIG."""

    def test_default_config_exists(self):
        """Test that DEFAULT_CONFIG is properly initialized."""
        assert isinstance(DEFAULT_CONFIG, PipelineConfig)
        assert DEFAULT_CONFIG.data.n_samples == 10000
        assert DEFAULT_CONFIG.data.signal_fraction == 0.1
        assert DEFAULT_CONFIG.random_state == 42

    def test_default_config_is_mutable(self):
        """Test that DEFAULT_CONFIG can be modified."""
        original_value = DEFAULT_CONFIG.data.n_samples
        DEFAULT_CONFIG.data.n_samples = 20000
        assert DEFAULT_CONFIG.data.n_samples == 20000
        # Reset for other tests
        DEFAULT_CONFIG.data.n_samples = original_value
