"""
Evaluation metrics module for binary classification.

For binary classification problems, especially imbalanced ones like signal/background
separation, accuracy alone is misleading. We need multiple metrics to understand
model performance across different operating points.

Key metrics explained:

- Accuracy: (TP + TN) / Total. Poor for imbalanced data (high background accuracy
  dominates even if signal detection is terrible).

- Precision: TP / (TP + FP). Of predicted signals, how many are true?
  Important when false alarms are costly.

- Recall (TPR): TP / (TP + FN). Of true signals, how many did we catch?
  Important when missing signals is costly.

- F1-Score: Harmonic mean of precision and recall. Good general metric.

- AUC-ROC: Area under the receiver operating characteristic curve.
  Measures separation at all classification thresholds.
  Robust to class imbalance.

- AUC-PR: Area under precision-recall curve.
  More informative than AUC-ROC for imbalanced data.
"""

import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    auc,
    precision_recall_curve,
    roc_curve,
)


class MetricsCalculator:
    """Computes comprehensive evaluation metrics for binary classification.

    Attributes:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        y_pred: Binary predictions (after thresholding at 0.5).
        threshold: Classification threshold (default 0.5).
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> None:
        """Initialize metrics calculator.

        Args:
            y_true: Ground truth binary labels.
            y_pred_proba: Predicted probabilities (0-1) for positive class.
            threshold: Classification threshold. Probabilities >= threshold
                      are classified as positive (signal).
        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.threshold = threshold
        self.y_pred = (y_pred_proba >= threshold).astype(int)

    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics.

        Returns:
            Dictionary with all metric names and values.
        """
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1_score(),
            "auc_roc": self.auc_roc(),
            "auc_pr": self.auc_pr(),
            "threshold": self.threshold,
        }

    def accuracy(self) -> float:
        """Calculate accuracy: (TP + TN) / Total.

        Note: Can be misleading for imbalanced data.
        With 90% background, a model that predicts all background
        gets 90% accuracy without detecting any signal!
        """
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP).

        Of all samples we predicted as signal, how many were actually signal?
        High precision means few false alarms.
        """
        if np.sum(self.y_pred) == 0:
            return 0.0  # No positive predictions
        return precision_score(self.y_true, self.y_pred, zero_division=0)

    def recall(self) -> float:
        """Calculate recall (sensitivity): TP / (TP + FN).

        Of all true signals, how many did we find?
        High recall means few missed signals.
        """
        return recall_score(self.y_true, self.y_pred, zero_division=0)

    def f1_score(self) -> float:
        """Calculate F1-score: 2 * (precision * recall) / (precision + recall).

        Harmonic mean of precision and recall. Good single metric for imbalanced data.
        """
        return f1_score(self.y_true, self.y_pred, zero_division=0)

    def auc_roc(self) -> float:
        """Calculate AUC-ROC: Area under receiver operating characteristic.

        ROC curve plots TPR vs FPR at all decision thresholds.
        AUC summarizes this: 1.0 = perfect, 0.5 = random, 0.0 = inverted.

        Advantage: Insensitive to class imbalance.
        """
        return roc_auc_score(self.y_true, self.y_pred_proba)

    def auc_pr(self) -> float:
        """Calculate AUC-PR: Area under precision-recall curve.

        PR curve plots precision vs recall at all thresholds.
        More informative than ROC for imbalanced data.

        Why? When one class is rare (signal), random classifiers get
        high AUC-ROC just by predicting the majority class. But they
        get low AUC-PR, making the metric more sensitive.
        """
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        return auc(recall, precision)

    def confusion_matrix_dict(self) -> Dict[str, int]:
        """Calculate confusion matrix components.

        Returns:
            Dictionary with TP, TN, FP, FN counts.
        """
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        return {
            "TP": int(tp),  # True positives (correct signal)
            "TN": int(tn),  # True negatives (correct background)
            "FP": int(fp),  # False positives (background as signal)
            "FN": int(fn),  # False negatives (signal as background)
        }

    def roc_curve_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data for plotting.

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(self.y_true, self.y_pred_proba)

    def pr_curve_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get precision-recall curve data for plotting.

        Returns:
            Tuple of (precision, recall, thresholds)
        """
        return precision_recall_curve(self.y_true, self.y_pred_proba)

    def get_operating_points(self) -> Dict[str, Dict[str, float]]:
        """Get metrics at common operating thresholds.

        Useful for understanding the precision-recall tradeoff.
        As threshold increases:
        - Precision increases (fewer false positives)
        - Recall decreases (fewer true positives caught)

        Returns:
            Dictionary with metrics at different thresholds.
        """
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        points = {}

        for thresh in thresholds:
            calc = MetricsCalculator(self.y_true, self.y_pred_proba, threshold=thresh)
            points[f"threshold_{thresh}"] = {
                "precision": calc.precision(),
                "recall": calc.recall(),
                "f1": calc.f1_score(),
                "accuracy": calc.accuracy(),
            }

        return points


class PerformanceReporter:
    """Formats and reports evaluation results.

    Attributes:
        metrics: MetricsCalculator instance.
        name: Name of the dataset (e.g., "Validation", "Test").
    """

    def __init__(self, metrics: MetricsCalculator, name: str = "Dataset") -> None:
        """Initialize reporter.

        Args:
            metrics: MetricsCalculator instance.
            name: Name of the dataset.
        """
        self.metrics = metrics
        self.name = name

    def print_report(self) -> None:
        """Print formatted evaluation report."""
        all_metrics = self.metrics.compute_all_metrics()
        cm = self.metrics.confusion_matrix_dict()

        print(f"\n{'='*60}")
        print(f"{self.name} Performance Report")
        print(f"{'='*60}")

        print(f"\nClass Distribution:")
        n_signal = np.sum(self.metrics.y_true == 1)
        n_background = np.sum(self.metrics.y_true == 0)
        print(f"  Signal (class 1):     {n_signal:6d} ({100*n_signal/len(self.metrics.y_true):.1f}%)")
        print(f"  Background (class 0): {n_background:6d} ({100*n_background/len(self.metrics.y_true):.1f}%)")

        print(f"\nConfusion Matrix:")
        print(f"  True Positives (TP):  {cm['TP']:6d}  (Signal correctly identified)")
        print(f"  False Negatives (FN): {cm['FN']:6d}  (Signal missed)")
        print(f"  False Positives (FP): {cm['FP']:6d}  (Background false alarm)")
        print(f"  True Negatives (TN):  {cm['TN']:6d}  (Background correctly identified)")

        print(f"\nMetrics at threshold {all_metrics['threshold']:.2f}:")
        print(f"  Accuracy:  {all_metrics['accuracy']:.4f}")
        print(f"  Precision: {all_metrics['precision']:.4f}")
        print(f"  Recall:    {all_metrics['recall']:.4f}")
        print(f"  F1-Score:  {all_metrics['f1_score']:.4f}")

        print(f"\nThreshold-Independent Metrics:")
        print(f"  AUC-ROC:  {all_metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:   {all_metrics['auc_pr']:.4f}")

        print(f"\n{'='*60}\n")

    def get_summary_dict(self) -> Dict[str, Any]:
        """Get all metrics as dictionary.

        Returns:
            Dictionary with comprehensive metrics.
        """
        return {
            "dataset_name": self.name,
            "all_metrics": self.metrics.compute_all_metrics(),
            "confusion_matrix": self.metrics.confusion_matrix_dict(),
            "operating_points": self.metrics.get_operating_points(),
        }


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    dataset_name: str = "Dataset",
    print_report: bool = True,
) -> Dict[str, Any]:
    """Evaluate model performance and optionally print report.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        dataset_name: Name of dataset for reporting.
        print_report: Whether to print formatted report.

    Returns:
        Dictionary with evaluation results.
    """
    calc = MetricsCalculator(y_true, y_pred_proba)
    reporter = PerformanceReporter(calc, dataset_name)

    if print_report:
        reporter.print_report()

    return reporter.get_summary_dict()
