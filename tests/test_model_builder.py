"""Tests for model_builder module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from signalseeker.model_builder import XGBoostModelBuilder, create_model_for_data
from signalseeker.config import XGBoostConfig


@pytest.fixture
def xgboost_config():
    """Fixture providing XGBoostConfig."""
    return XGBoostConfig(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
    )


@pytest.fixture
def y_imbalanced():
    """Fixture providing imbalanced labels."""
    np.random.seed(42)
    y = np.concatenate([np.zeros(900), np.ones(100)])
    np.random.shuffle(y)
    return y


class TestXGBoostModelBuilder:
    """Test suite for XGBoostModelBuilder."""

    def test_initialization(self, xgboost_config):
        """Test XGBoostModelBuilder initialization."""
        builder = XGBoostModelBuilder(xgboost_config)
        assert builder.config == xgboost_config
        assert builder.model is not None
        assert builder.is_trained is False

    def test_model_creation(self, xgboost_config):
        """Test that model is created properly."""
        builder = XGBoostModelBuilder(xgboost_config)
        model = builder.get_model()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_compute_class_weight(self, xgboost_config, y_imbalanced):
        """Test class weight computation."""
        builder = XGBoostModelBuilder(xgboost_config)
        weight = builder.compute_class_weight(y_imbalanced)

        # With 90% background and 10% signal
        # weight should be approximately 9.0
        assert pytest.approx(weight, abs=0.5) == 9.0

    def test_build_with_class_weights(self, xgboost_config, y_imbalanced):
        """Test building model with automatic class weights."""
        builder = XGBoostModelBuilder(xgboost_config)
        builder.build_with_class_weights(y_imbalanced)

        assert builder.scale_pos_weight is not None
        assert builder.scale_pos_weight > 0

    def test_get_hyperparameters(self, xgboost_config):
        """Test getting hyperparameters."""
        builder = XGBoostModelBuilder(xgboost_config)
        params = builder.get_hyperparameters()

        assert "max_depth" in params
        assert "learning_rate" in params
        assert "n_estimators" in params
        assert params["max_depth"] == 6

    def test_save_and_load(self, xgboost_config, y_imbalanced):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"

            # Create and save
            builder = XGBoostModelBuilder(xgboost_config)
            builder.save(model_path)
            assert model_path.exists()

            # Load
            builder2 = XGBoostModelBuilder(xgboost_config)
            builder2.load(model_path)
            assert builder2.is_trained is True

    def test_is_fitted_before_training(self, xgboost_config):
        """Test is_fitted returns False before training."""
        builder = XGBoostModelBuilder(xgboost_config)
        assert builder.is_fitted() is False

    def test_method_chaining(self, xgboost_config, y_imbalanced):
        """Test method chaining."""
        builder = XGBoostModelBuilder(xgboost_config)
        result = builder.build_with_class_weights(y_imbalanced)
        assert result is builder


class TestCreateModelForData:
    """Test suite for create_model_for_data function."""

    def test_creation_with_weighting(self, xgboost_config, y_imbalanced):
        """Test model creation with automatic weighting."""
        builder = create_model_for_data(xgboost_config, y_imbalanced)

        assert builder.scale_pos_weight is not None
        assert builder.scale_pos_weight > 0

    def test_different_imbalance_ratios(self, xgboost_config):
        """Test model creation with different imbalance ratios."""
        # More imbalanced: 95% background, 5% signal
        y_95_5 = np.concatenate([np.zeros(950), np.ones(50)])
        builder_95_5 = create_model_for_data(xgboost_config, y_95_5)

        # Less imbalanced: 70% background, 30% signal
        y_70_30 = np.concatenate([np.zeros(700), np.ones(300)])
        builder_70_30 = create_model_for_data(xgboost_config, y_70_30)

        # More imbalanced should have higher weight
        assert builder_95_5.scale_pos_weight > builder_70_30.scale_pos_weight


class TestModelBuilderIntegration:
    """Integration tests for model building."""

    def test_model_training(self, xgboost_config, y_imbalanced):
        """Test that model can be trained."""
        np.random.seed(42)
        X = np.random.randn(1000, 20)

        builder = create_model_for_data(xgboost_config, y_imbalanced)
        model = builder.get_model()

        # Simple fit
        model.fit(X, y_imbalanced)
        builder.is_trained = True

        assert builder.is_fitted() is True

        # Test predictions
        predictions = model.predict(X)
        assert predictions.shape == y_imbalanced.shape

        # Test probabilities
        probas = model.predict_proba(X)
        assert probas.shape[0] == X.shape[0]
        assert probas.shape[1] == 2

    def test_model_predictions_valid(self, xgboost_config, y_imbalanced):
        """Test that predictions are valid probabilities."""
        np.random.seed(42)
        X = np.random.randn(1000, 20)

        builder = create_model_for_data(xgboost_config, y_imbalanced)
        model = builder.get_model()
        model.fit(X, y_imbalanced)

        probas = model.predict_proba(X)

        # All probabilities should be between 0 and 1
        assert np.all(probas >= 0)
        assert np.all(probas <= 1)

        # Each row should sum to approximately 1
        np.testing.assert_array_almost_equal(
            probas.sum(axis=1),
            np.ones(probas.shape[0]),
        )
