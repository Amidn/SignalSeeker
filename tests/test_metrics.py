"""Tests for metrics module."""

import pytest
import numpy as np
from signalseeker.metrics import MetricsCalculator


@pytest.fixture
def sample_predictions():
    """Fixture providing sample predictions."""
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_pred_proba = np.array(
        [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.15, 0.25, 0.85, 0.95]
    )
    return y_true, y_pred_proba


class TestMetricsCalculator:
    """Test suite for MetricsCalculator."""

    def test_initialization(self, sample_predictions):
        """Test MetricsCalculator initialization."""
        y_true, y_pred_proba = sample_predictions
        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        assert np.array_equal(calc.y_true, y_true)
        assert np.array_equal(calc.y_pred_proba, y_pred_proba)
        assert calc.threshold == 0.5

    def test_binary_classification_at_threshold(self, sample_predictions):
        """Test binary classification at threshold."""
        y_true, y_pred_proba = sample_predictions
        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        # At threshold 0.5: [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        expected = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(calc.y_pred, expected)

    def test_custom_threshold(self, sample_predictions):
        """Test classification with custom threshold."""
        y_true, y_pred_proba = sample_predictions

        # Lower threshold should predict more positives
        calc_low = MetricsCalculator(y_true, y_pred_proba, threshold=0.3)
        calc_high = MetricsCalculator(y_true, y_pred_proba, threshold=0.8)

        assert np.sum(calc_low.y_pred) >= np.sum(calc_high.y_pred)

    def test_accuracy(self, sample_predictions):
        """Test accuracy calculation."""
        y_true, y_pred_proba = sample_predictions
        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        acc = calc.accuracy()
        assert isinstance(acc, (float, np.floating))
        assert 0 <= acc <= 1

    def test_precision(self, sample_predictions):
        """Test precision calculation."""
        y_true, y_pred_proba = sample_predictions
        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        prec = calc.precision()
        assert isinstance(prec, (float, np.floating))
        assert 0 <= prec <= 1

    def test_recall(self, sample_predictions):
        """Test recall calculation."""
        y_true, y_pred_proba = sample_predictions
        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        rec = calc.recall()
        assert isinstance(rec, (float, np.floating))
        assert 0 <= rec <= 1

    def test_f1_score(self, sample_predictions):
        """Test F1-score calculation."""
        y_true, y_pred_proba = sample_predictions
        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        f1 = calc.f1_score()
        assert isinstance(f1, (float, np.floating))
        assert 0 <= f1 <= 1

    def test_auc_roc(self, sample_predictions):
        """Test AUC-ROC calculation."""
        y_true, y_pred_proba = sample_predictions
        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        auc_roc = calc.auc_roc()
        assert isinstance(auc_roc, (float, np.floating))
        assert 0 <= auc_roc <= 1

    def test_auc_pr(self, sample_predictions):
        """Test AUC-PR calculation."""
        y_true, y_pred_proba = sample_predictions
        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        auc_pr = calc.auc_pr()
        assert isinstance(auc_pr, (float, np.floating))
        assert 0 <= auc_pr <= 1

    def test_compute_all_metrics(self, sample_predictions):
        """Test computing all metrics at once."""
        y_true, y_pred_proba = sample_predictions
        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        metrics = calc.compute_all_metrics()

        # Check all expected keys are present
        expected_keys = {
            "accuracy", "precision", "recall", "f1_score",
            "auc_roc", "auc_pr", "threshold"
        }
        assert set(metrics.keys()) == expected_keys

        # Check all values are valid
        for key, value in metrics.items():
            if key != "threshold":
                assert 0 <= value <= 1

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.0, 0.1, 0.9, 1.0])

        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        assert calc.accuracy() == 1.0
        assert calc.precision() == 1.0
        assert calc.recall() == 1.0
        assert calc.f1_score() == 1.0
        assert calc.auc_roc() == 1.0

    def test_random_predictions(self):
        """Test with random/poor predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100)

        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        # AUC-ROC should be around 0.5 for random predictions
        auc_roc = calc.auc_roc()
        assert pytest.approx(auc_roc, abs=0.3) == 0.5

    def test_all_zeros(self):
        """Test with all negative samples."""
        y_true = np.array([0, 0, 0, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4])

        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        # With no positive samples, precision/recall should be 0 or nan
        # Accuracy should be 1.0 (all correctly predicted as negative)
        assert calc.accuracy() == 1.0

    def test_all_ones(self):
        """Test with all positive samples."""
        y_true = np.array([1, 1, 1, 1])
        y_pred_proba = np.array([0.6, 0.7, 0.8, 0.9])

        calc = MetricsCalculator(y_true, y_pred_proba, threshold=0.5)

        # With no negative samples, should achieve perfect recall
        assert calc.recall() == 1.0
        assert calc.accuracy() == 1.0

    def test_threshold_effect_on_metrics(self, sample_predictions):
        """Test how threshold affects different metrics."""
        y_true, y_pred_proba = sample_predictions

        calc_low = MetricsCalculator(y_true, y_pred_proba, threshold=0.3)
        calc_high = MetricsCalculator(y_true, y_pred_proba, threshold=0.8)

        # Lower threshold typically increases recall but decreases precision
        # (more positives predicted, so catch more true positives but also more false positives)
        recall_low = calc_low.recall()
        recall_high = calc_high.recall()

        # This might not always be true for small samples, but generally:
        # Lower threshold should find more true positives
        assert recall_low >= recall_high or pytest.approx(recall_low, abs=0.3) == recall_high
