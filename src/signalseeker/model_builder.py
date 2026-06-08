"""
XGBoost model initialization and building module.

This module creates and configures XGBoost Boosted Decision Tree (BDT) models
for binary classification. XGBoost is chosen because:

1. **Boosting mechanism**: Iteratively builds trees to correct previous mistakes,
   leading to high accuracy with relatively few trees.

2. **Gradient-based optimization**: Uses first and second-order derivatives to
   find optimal split points, making it faster than greedy approaches.

3. **Regularization**: Built-in L1/L2 regularization prevents overfitting.

4. **Class weight handling**: Can balance imbalanced datasets automatically.

5. **Feature importance**: Provides interpretability through feature importance scores.

In particle physics, XGBoost has become the de facto standard for signal/background
separation due to its superior performance and interpretability.
"""

import xgboost as xgb
import numpy as np
from typing import Optional
import joblib
from pathlib import Path
from .config import XGBoostConfig


class XGBoostModelBuilder:
    """Builds and manages XGBoost models for binary classification.

    XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting framework
    that builds an ensemble of decision trees sequentially. Each new tree learns to
    correct the residual errors of previous trees, leading to powerful predictions.

    Why Boosted Decision Trees for signal/background separation?
    - Non-parametric: Makes no assumptions about feature distributions
    - Robust: Handles correlated and uninformative features
    - Fast: Efficient training and inference
    - Interpretable: Can extract feature importance
    - Production-ready: Used in industry at scale

    Attributes:
        config: XGBoostConfig with model hyperparameters.
        model: Initialized XGBoost model.
        is_trained: Whether the model has been fitted.
    """

    def __init__(
        self,
        config: XGBoostConfig,
        scale_pos_weight: Optional[float] = None,
    ) -> None:
        """Initialize XGBoost model builder.

        Args:
            config: XGBoostConfig with hyperparameters.
            scale_pos_weight: Class weight for handling imbalance.
                             If None, will be computed from data.
        """
        self.config = config
        self.scale_pos_weight = scale_pos_weight
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained = False
        self._build_model()

    def _build_model(self) -> None:
        """Construct XGBoost classifier with configured hyperparameters.

        Key hyperparameter explanations:

        - max_depth: Maximum tree depth (6-8 typical). Deeper trees fit more
          complex patterns but risk overfitting.

        - learning_rate (eta): Shrinkage factor (0.01-0.3). Lower rates are more
          conservative; higher rates learn faster but may overfit.

        - n_estimators: Number of boosting rounds. More trees → better fit but
          risk overfitting. Use early stopping.

        - subsample: Fraction of samples per tree (0.5-1.0). <1.0 adds randomness,
          reducing overfitting (called "stochastic boosting").

        - colsample_bytree: Fraction of features per tree. <1.0 forces the model
          to learn from different feature subsets, improving generalization.

        - scale_pos_weight: Weight for positive (signal) class. In imbalanced data,
          set to (n_negative / n_positive) to balance loss contributions.

        - objective: 'binary:logistic' outputs probabilities (0-1) via sigmoid.

        - eval_metric: Metric for early stopping. 'logloss' is cross-entropy loss.
        """
        params = {
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "n_estimators": self.config.n_estimators,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "objective": self.config.objective,
            "eval_metric": self.config.eval_metric,
            "random_state": self.config.random_state,
            "verbosity": 0,
            "use_label_encoder": False,
        }

        # Handle class imbalance using scale_pos_weight
        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight
        elif self.config.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.config.scale_pos_weight

        self.model = xgb.XGBClassifier(**params)

    def compute_class_weight(self, y: np.ndarray) -> float:
        """Compute optimal scale_pos_weight for imbalanced data.

        scale_pos_weight = (n_negative / n_positive)

        This weight tells XGBoost to penalize false negatives (missing signal)
        more heavily than false positives (background misclassified as signal).
        For signal/background separation, missing signal is typically worse.

        Args:
            y: Target labels.

        Returns:
            Computed scale_pos_weight value.
        """
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        weight = n_negative / n_positive
        return weight

    def build_with_class_weights(self, y_train: np.ndarray) -> "XGBoostModelBuilder":
        """Rebuild model with automatically computed class weights.

        Args:
            y_train: Training labels.

        Returns:
            Self for method chaining.
        """
        weight = self.compute_class_weight(y_train)
        self.scale_pos_weight = weight
        self._build_model()
        return self

    def get_model(self) -> xgb.XGBClassifier:
        """Get the underlying XGBoost model.

        Returns:
            XGBClassifier instance.
        """
        return self.model

    def get_hyperparameters(self) -> dict:
        """Get current hyperparameters.

        Returns:
            Dictionary of all hyperparameters.
        """
        return self.model.get_params()

    def save(self, path: Path) -> None:
        """Save trained model to disk.

        XGBoost models can be saved in native format (.bst) or using joblib.
        We use joblib for consistency with other scikit-learn estimators.

        Args:
            path: File path to save the model.
        """
        joblib.dump(self.model, path)

    def load(self, path: Path) -> "XGBoostModelBuilder":
        """Load a previously trained model.

        Args:
            path: File path to load from.

        Returns:
            Self for method chaining.
        """
        self.model = joblib.load(path)
        self.is_trained = True
        return self

    def is_fitted(self) -> bool:
        """Check if model has been trained.

        Returns:
            True if model has been fitted.
        """
        return self.is_trained


def create_model_for_data(
    config: XGBoostConfig,
    y_train: np.ndarray,
) -> XGBoostModelBuilder:
    """Convenience function to create a model with automatic class weighting.

    Args:
        config: XGBoostConfig.
        y_train: Training labels for computing class weights.

    Returns:
        XGBoostModelBuilder with computed scale_pos_weight.
    """
    builder = XGBoostModelBuilder(config)
    builder.build_with_class_weights(y_train)
    return builder
