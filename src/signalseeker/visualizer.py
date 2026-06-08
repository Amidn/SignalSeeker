"""
Visualization module for SignalSeeker.

This module creates publication-quality plots for understanding model performance.
Key visualizations include:

1. **Probability Score Distribution**: The core "cut" concept from particle physics.
   Shows how well the model separates signal and background using their probability
   scores. A good model produces two well-separated distributions.

2. **ROC Curve**: Trade-off between true positive rate and false positive rate
   at all decision thresholds.

3. **Precision-Recall Curve**: More informative than ROC for imbalanced data.

4. **Confusion Matrix**: Heatmap of TP/TN/FP/FN.

5. **Feature Importance**: Which features did the model rely on most?

The probability score distribution is analogous to the "BDT score" or "discriminant"
in particle physics experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from .config import VisualizerConfig


class ModelVisualizer:
    """Creates and manages visualizations for model evaluation.

    Attributes:
        config: VisualizerConfig with visualization settings.
    """

    def __init__(self, config: VisualizerConfig) -> None:
        """Initialize visualizer.

        Args:
            config: VisualizerConfig with style and output settings.
        """
        self.config = config
        self._setup_style()

    def _setup_style(self) -> None:
        """Configure matplotlib style."""
        plt.style.use(self.config.style)
        sns.set_palette("husl")

    def _save_if_configured(self, name: str) -> None:
        """Save figure if configured to do so.

        Args:
            name: Base name for the figure file.
        """
        if self.config.save_plots:
            path = self.config.output_dir / f"{name}.png"
            plt.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved: {path}")

    def plot_probability_score_distribution(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        cut_threshold: float = 0.5,
    ) -> None:
        """Plot probability score distribution (the "Cut" concept).

        This is the MOST IMPORTANT plot for signal/background separation.
        It shows the probability scores from the model for signal and background events.

        In particle physics, this is the "BDT score" distribution. A good classifier
        produces:
        - Signal distribution: concentrated near 1.0 (high probability)
        - Background distribution: concentrated near 0.0 (low probability)
        - Good separation: minimal overlap between distributions

        The "cut" refers to the threshold we apply to this score:
        - Events above the cut threshold are classified as signal
        - Events below are classified as background
        - Moving the cut trades off signal acceptance vs background rejection

        Args:
            y_true: Ground truth labels.
            y_pred_proba: Predicted probabilities.
            cut_threshold: Threshold for signal selection (for visualization).
        """
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=100)

        # Separate signal and background scores
        signal_scores = y_pred_proba[y_true == 1]
        background_scores = y_pred_proba[y_true == 0]

        # Plot distributions
        bins = np.linspace(0, 1, 50)
        ax.hist(
            signal_scores,
            bins=bins,
            alpha=0.7,
            label="Signal",
            color="blue",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.hist(
            background_scores,
            bins=bins,
            alpha=0.7,
            label="Background",
            color="red",
            edgecolor="black",
            linewidth=0.5,
        )

        # Mark the cut threshold
        ax.axvline(
            cut_threshold,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Cut at {cut_threshold:.2f}",
        )

        ax.set_xlabel("Probability Score (Higher = More Signal-like)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Events", fontsize=12, fontweight="bold")
        ax.set_title("Probability Score Distribution (Signal/Background Separation)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, loc="upper center")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_if_configured("probability_distribution")
        plt.show()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Plot ROC (Receiver Operating Characteristic) curve.

        The ROC curve shows the trade-off between true positive rate (signal detection)
        and false positive rate (background misidentification) at all thresholds.

        A perfect classifier: curve goes from (0,0) to (0,1) to (1,1), AUC=1.0
        A random classifier: straight diagonal line, AUC=0.5

        Args:
            y_true: Ground truth labels.
            y_pred_proba: Predicted probabilities.

        Returns:
            Tuple of (fpr_array, tpr_array, auc_value).
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=100)

        ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random classifier")

        ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        self._save_if_configured("roc_curve")
        plt.show()

        return fpr, tpr, roc_auc

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Plot Precision-Recall curve.

        More informative than ROC for imbalanced data. Shows the trade-off between
        precision (of predicted signals, how many are true) and recall (of true signals,
        how many we catch).

        Args:
            y_true: Ground truth labels.
            y_pred_proba: Predicted probabilities.

        Returns:
            Tuple of (precision_array, recall_array, auc_pr).
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=100)

        ax.plot(
            recall,
            precision,
            color="purple",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.4f})"
        )

        # Baseline: random classifier with imbalanced data
        baseline = np.sum(y_true == 1) / len(y_true)
        ax.axhline(baseline, color="gray", lw=2, linestyle="--", label=f"Random classifier (P={baseline:.3f})")

        ax.set_xlabel("Recall (True Positive Rate)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
        ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        self._save_if_configured("precision_recall_curve")
        plt.show()

        return precision, recall, pr_auc

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[list] = None,
    ) -> None:
        """Plot confusion matrix heatmap.

        Shows TP, TN, FP, FN in a matrix format.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels (binary, after thresholding).
            labels: Class labels for display.
        """
        if labels is None:
            labels = ["Background", "Signal"]

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 7), dpi=100)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar_kws={"label": "Count"},
            xticklabels=labels,
            yticklabels=labels,
        )

        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

        plt.tight_layout()
        self._save_if_configured("confusion_matrix")
        plt.show()

    def plot_feature_importance(
        self,
        model: xgb.XGBClassifier,
        top_n: int = 15,
    ) -> None:
        """Plot feature importance from trained XGBoost model.

        XGBoost learns the importance of each feature during training.
        Features that contribute more to splits get higher importance scores.

        This helps understand which features the model relied on for classification.

        Args:
            model: Trained XGBoost model.
            top_n: Number of top features to display.
        """
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=100)

        feature_names = [f"Feature {i}" for i in range(len(importances))]
        ax.barh(
            [feature_names[i] for i in indices],
            importances[indices],
            color="steelblue",
            edgecolor="black",
        )

        ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
        ax.invert_yaxis()

        plt.tight_layout()
        self._save_if_configured("feature_importance")
        plt.show()

    def plot_summary(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: np.ndarray,
        model: Optional[xgb.XGBClassifier] = None,
    ) -> None:
        """Create a summary figure with multiple subplots.

        Args:
            y_true: Ground truth labels.
            y_pred_proba: Predicted probabilities.
            y_pred: Binary predictions.
            model: Trained model (optional, for feature importance).
        """
        fig = plt.figure(figsize=(16, 10))

        # Probability distribution
        ax1 = plt.subplot(2, 3, 1)
        signal_scores = y_pred_proba[y_true == 1]
        background_scores = y_pred_proba[y_true == 0]
        ax1.hist(signal_scores, bins=30, alpha=0.6, label="Signal", color="blue")
        ax1.hist(background_scores, bins=30, alpha=0.6, label="Background", color="red")
        ax1.set_xlabel("Probability Score")
        ax1.set_ylabel("Count")
        ax1.set_title("Probability Score Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ROC curve
        ax2 = plt.subplot(2, 3, 2)
        fpr, tpr, roc_auc = roc_curve(y_true, y_pred_proba)[:3], auc(roc_curve(y_true, y_pred_proba)[0], roc_curve(y_true, y_pred_proba)[1])
        fpr_plot, tpr_plot, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr_plot, tpr_plot)
        ax2.plot(fpr_plot, tpr_plot, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax2.plot([0, 1], [0, 1], "--", color="gray")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Precision-Recall curve
        ax3 = plt.subplot(2, 3, 3)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        ax3.plot(recall, precision, lw=2, label=f"AUC = {pr_auc:.3f}")
        ax3.set_xlabel("Recall")
        ax3.set_ylabel("Precision")
        ax3.set_title("Precision-Recall Curve")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Confusion matrix
        ax4 = plt.subplot(2, 3, 4)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4,
                    xticklabels=["Background", "Signal"],
                    yticklabels=["Background", "Signal"])
        ax4.set_ylabel("True Label")
        ax4.set_xlabel("Predicted Label")
        ax4.set_title("Confusion Matrix")

        # Feature importance if model provided
        if model is not None:
            ax5 = plt.subplot(2, 3, 5)
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-10:][::-1]
            ax5.barh(range(len(top_idx)), importances[top_idx], color="steelblue")
            ax5.set_yticks(range(len(top_idx)))
            ax5.set_yticklabels([f"F{i}" for i in top_idx])
            ax5.set_xlabel("Importance")
            ax5.set_title("Top 10 Feature Importances")
            ax5.invert_yaxis()

        # Metrics summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_text = f"""
        Accuracy:  {accuracy:.4f}
        Precision: {precision:.4f}
        Recall:    {recall:.4f}
        F1-Score:  {f1:.4f}
        AUC-ROC:   {roc_auc:.4f}
        AUC-PR:    {pr_auc:.4f}
        """
        ax6.text(0.1, 0.5, metrics_text, fontsize=12, family="monospace",
                verticalalignment="center")
        ax6.set_title("Performance Metrics")

        plt.tight_layout()
        self._save_if_configured("summary")
        plt.show()
