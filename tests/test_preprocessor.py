"""Tests for preprocessor module."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from signalseeker.preprocessor import (
    DataPreprocessingPipeline,
    FeaturePreprocessor,
    MissingValueHandler,
)
from signalseeker.config import PreprocessConfig


@pytest.fixture
def sample_data():
    """Fixture providing sample feature matrices."""
    np.random.seed(42)
    X_train = np.random.randn(800, 20)
    X_val = np.random.randn(100, 20)
    X_test = np.random.randn(100, 20)
    return X_train, X_val, X_test


class TestFeaturePreprocessor:
    """Test suite for FeaturePreprocessor."""

    def test_initialization(self):
        """Test FeaturePreprocessor initialization."""
        preprocessor = FeaturePreprocessor()
        assert preprocessor.is_fitted is False

    def test_fit_and_transform(self, sample_data):
        """Test fitting and transforming data."""
        X_train, X_val, X_test = sample_data
        preprocessor = FeaturePreprocessor()

        preprocessor.fit(X_train)
        assert preprocessor.is_fitted is True

        X_train_scaled = preprocessor.transform(X_train)
        assert X_train_scaled.shape == X_train.shape

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X_train, _, _ = sample_data
        preprocessor = FeaturePreprocessor()

        X_transformed = preprocessor.fit_transform(X_train)
        assert X_transformed.shape == X_train.shape
        assert preprocessor.is_fitted is True

    def test_transform_before_fit_raises(self, sample_data):
        """Test that transform before fit raises error."""
        _, _, X_test = sample_data
        preprocessor = FeaturePreprocessor()

        with pytest.raises(RuntimeError):
            preprocessor.transform(X_test)

    def test_standardization(self, sample_data):
        """Test that standardization works correctly."""
        X_train, _, _ = sample_data
        preprocessor = FeaturePreprocessor()
        X_scaled = preprocessor.fit_transform(X_train)

        # Check mean close to 0 and std close to 1
        assert pytest.approx(X_scaled.mean(), abs=0.1) == 0
        assert pytest.approx(X_scaled.std(), abs=0.1) == 1

    def test_save_and_load(self, sample_data):
        """Test saving and loading fitted scaler."""
        X_train, X_val, _ = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = Path(tmpdir) / "scaler.joblib"

            # Fit and save
            preprocessor = FeaturePreprocessor()
            preprocessor.fit(X_train)
            preprocessor.save(scaler_path)
            assert scaler_path.exists()

            # Load and transform
            preprocessor2 = FeaturePreprocessor()
            preprocessor2.load(scaler_path)
            X_val_scaled = preprocessor2.transform(X_val)
            assert X_val_scaled.shape == X_val.shape


class TestMissingValueHandler:
    """Test suite for MissingValueHandler."""

    def test_initialization(self):
        """Test MissingValueHandler initialization."""
        handler = MissingValueHandler()
        assert handler.is_fitted is False

    def test_fit_and_transform(self):
        """Test fitting and transforming data with missing values."""
        X_train = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        handler = MissingValueHandler()

        handler.fit(X_train)
        assert handler.is_fitted is True

        X_imputed = handler.transform(X_train)
        assert not np.any(np.isnan(X_imputed))

    def test_different_strategies(self):
        """Test different imputation strategies."""
        X_train = np.array([[1, 2], [np.nan, 4], [7, 8]])

        # Mean imputation
        handler_mean = MissingValueHandler(strategy="mean")
        handler_mean.fit(X_train)
        X_mean = handler_mean.transform(X_train)
        assert not np.any(np.isnan(X_mean))

        # Median imputation
        handler_median = MissingValueHandler(strategy="median")
        handler_median.fit(X_train)
        X_median = handler_median.transform(X_train)
        assert not np.any(np.isnan(X_median))

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises error."""
        handler = MissingValueHandler()
        X = np.array([[1, 2, 3]])

        with pytest.raises(RuntimeError):
            handler.transform(X)


class TestDataPreprocessingPipeline:
    """Test suite for DataPreprocessingPipeline."""

    def test_initialization(self):
        """Test DataPreprocessingPipeline initialization."""
        config = PreprocessConfig()
        pipeline = DataPreprocessingPipeline(config)
        assert pipeline.config == config

    def test_fit_and_transform(self, sample_data):
        """Test full preprocessing pipeline."""
        X_train, X_val, X_test = sample_data
        config = PreprocessConfig(scale_features=True, handle_missing=True)
        pipeline = DataPreprocessingPipeline(config)

        X_train_proc, X_val_proc, X_test_proc = pipeline.fit(
            X_train, X_val, X_test
        )

        assert X_train_proc.shape == X_train.shape
        assert X_val_proc.shape == X_val.shape
        assert X_test_proc.shape == X_test.shape

    def test_save_and_load(self, sample_data):
        """Test saving and loading pipeline."""
        X_train, X_val, X_test = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_dir = Path(tmpdir)

            # Fit and save
            config = PreprocessConfig()
            pipeline = DataPreprocessingPipeline(config)
            pipeline.fit(X_train, X_val, X_test)
            pipeline.save(pipeline_dir)

            assert (pipeline_dir / "scaler.joblib").exists()
            assert (pipeline_dir / "imputer.joblib").exists()

            # Load
            pipeline2 = DataPreprocessingPipeline(config)
            pipeline2.load(pipeline_dir)
            assert pipeline2.is_fitted is True

    def test_scaling_disabled(self, sample_data):
        """Test preprocessing with scaling disabled."""
        X_train, X_val, X_test = sample_data
        config = PreprocessConfig(scale_features=False)
        pipeline = DataPreprocessingPipeline(config)

        X_train_proc, _, _ = pipeline.fit(X_train, X_val, X_test)

        # Should be unchanged
        np.testing.assert_array_almost_equal(X_train, X_train_proc)

    def test_consistent_transformation(self, sample_data):
        """Test that transformation is consistent across sets."""
        X_train, X_val, X_test = sample_data
        config = PreprocessConfig(scale_features=True)
        pipeline = DataPreprocessingPipeline(config)

        X_train_proc, X_val_proc, X_test_proc = pipeline.fit(
            X_train, X_val, X_test
        )

        # All should have similar statistics
        assert pytest.approx(X_train_proc.mean(), abs=0.2) == 0
        assert pytest.approx(X_val_proc.mean(), abs=0.2) == 0
        assert pytest.approx(X_test_proc.mean(), abs=0.2) == 0
