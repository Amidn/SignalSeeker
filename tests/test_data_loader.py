"""Tests for data_loader module."""

import pytest
import numpy as np
from signalseeker.data_loader import DataLoader, DataSplitter, create_data_pipeline
from signalseeker.config import DataConfig, PreprocessConfig


@pytest.fixture
def data_config():
    """Fixture providing DataConfig."""
    return DataConfig(
        n_samples=1000,
        signal_fraction=0.1,
        n_features=20,
        n_informative=15,
        random_state=42,
    )


@pytest.fixture
def preprocess_config():
    """Fixture providing PreprocessConfig."""
    return PreprocessConfig(test_size=0.2, val_size=0.2)


class TestDataLoader:
    """Test suite for DataLoader."""

    def test_initialization(self, data_config):
        """Test DataLoader initialization."""
        loader = DataLoader(data_config)
        assert loader.config == data_config

    def test_generate_data(self, data_config):
        """Test data generation."""
        loader = DataLoader(data_config)
        X, y = loader.generate_data()

        assert X.shape[0] == 1000
        assert X.shape[1] == 20
        assert y.shape[0] == 1000
        assert len(np.unique(y)) == 2
        assert np.all((y == 0) | (y == 1))

    def test_class_distribution(self, data_config):
        """Test class distribution calculation."""
        loader = DataLoader(data_config)
        X, y = loader.generate_data()
        dist = loader.get_class_distribution()

        assert "signal_count" in dist
        assert "background_count" in dist
        assert "signal_fraction" in dist
        assert "background_fraction" in dist
        assert "imbalance_ratio" in dist

        # Check signal fraction is approximately 0.1
        assert pytest.approx(dist["signal_fraction"], abs=0.05) == 0.1

    def test_data_reproducibility(self, data_config):
        """Test that same seed produces same data."""
        loader1 = DataLoader(data_config)
        X1, y1 = loader1.generate_data()

        loader2 = DataLoader(data_config)
        X2, y2 = loader2.generate_data()

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_property_access(self, data_config):
        """Test property-based data access."""
        loader = DataLoader(data_config)

        # Access via properties
        X = loader.X
        y = loader.y

        assert X.shape[0] == 1000
        assert y.shape[0] == 1000

        # Should be cached
        assert loader._X is not None
        assert loader._y is not None


class TestDataSplitter:
    """Test suite for DataSplitter."""

    def test_initialization(self, preprocess_config):
        """Test DataSplitter initialization."""
        splitter = DataSplitter(preprocess_config)
        assert splitter.config == preprocess_config

    def test_split_proportions(self, data_config, preprocess_config):
        """Test that split maintains correct proportions."""
        loader = DataLoader(data_config)
        X, y = loader.generate_data()

        splitter = DataSplitter(preprocess_config)
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            X, y, random_state=42
        )

        # Check sizes
        total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total == 1000

        # Check proportions (approximately)
        test_fraction = X_test.shape[0] / 1000
        assert pytest.approx(test_fraction, abs=0.05) == 0.2

    def test_stratified_split(self, data_config, preprocess_config):
        """Test that split is stratified (maintains class balance)."""
        loader = DataLoader(data_config)
        X, y = loader.generate_data()

        splitter = DataSplitter(preprocess_config)
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            X, y, random_state=42
        )

        # Check that signal fraction is similar in all splits
        original_signal_fraction = np.sum(y == 1) / len(y)
        train_signal_fraction = np.sum(y_train == 1) / len(y_train)
        val_signal_fraction = np.sum(y_val == 1) / len(y_val)
        test_signal_fraction = np.sum(y_test == 1) / len(y_test)

        # All fractions should be approximately 0.1
        assert pytest.approx(original_signal_fraction, abs=0.05) == 0.1
        assert pytest.approx(train_signal_fraction, abs=0.05) == 0.1
        assert pytest.approx(val_signal_fraction, abs=0.05) == 0.1
        assert pytest.approx(test_signal_fraction, abs=0.05) == 0.1

    def test_no_overlap(self, data_config, preprocess_config):
        """Test that train/val/test sets don't overlap."""
        loader = DataLoader(data_config)
        X, y = loader.generate_data()

        splitter = DataSplitter(preprocess_config)
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
            X, y, random_state=42
        )

        total_samples = (
            X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        )
        assert total_samples == X.shape[0]


class TestCreateDataPipeline:
    """Test suite for create_data_pipeline function."""

    def test_pipeline_output(self, data_config, preprocess_config):
        """Test that pipeline returns correct structure."""
        X_arrays, y_arrays = create_data_pipeline(data_config, preprocess_config)

        X_train, X_val, X_test = X_arrays
        y_train, y_val, y_test = y_arrays

        assert X_train.shape[0] > 0
        assert X_val.shape[0] > 0
        assert X_test.shape[0] > 0

        assert y_train.shape[0] == X_train.shape[0]
        assert y_val.shape[0] == X_val.shape[0]
        assert y_test.shape[0] == X_test.shape[0]

    def test_pipeline_consistency(self, data_config, preprocess_config):
        """Test that pipeline produces consistent results with same seed."""
        result1 = create_data_pipeline(data_config, preprocess_config)
        result2 = create_data_pipeline(data_config, preprocess_config)

        (X_train1, X_val1, X_test1), (y_train1, y_val1, y_test1) = result1
        (X_train2, X_val2, X_test2), (y_train2, y_val2, y_test2) = result2

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)


@pytest.mark.slow
class TestDataLoaderLarge:
    """Tests with large datasets (marked as slow)."""

    def test_large_dataset_generation(self):
        """Test generating a large dataset."""
        config = DataConfig(n_samples=100000, signal_fraction=0.1)
        loader = DataLoader(config)
        X, y = loader.generate_data()

        assert X.shape[0] == 100000
        assert y.shape[0] == 100000
