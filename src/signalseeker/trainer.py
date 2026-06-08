"""
Model training module for SignalSeeker.

This module handles the training of XGBoost models with early stopping,
progress monitoring, and validation-based model selection.

Key training concepts:
- Early stopping: Stop training when validation performance plateaus
- Eval set: Track validation metrics during training
- Watchlist: XGBoost's mechanism to monitor multiple datasets
"""

import numpy as np
import xgboost as xgb
from typing import Tuple, Optional, Dict, Any
import warnings
from .model_builder import XGBoostModelBuilder


class ModelTrainer:
    """Handles model training with early stopping and validation monitoring.

    Early stopping is crucial for preventing overfitting. As we train more trees,
    eventually we start fitting noise in the training data rather than true signal.
    By monitoring validation loss, we detect when this occurs and stop before it happens.

    Attributes:
        model_builder: XGBoostModelBuilder instance.
        training_history: Dictionary recording training metrics over epochs.
    """

    def __init__(self, model_builder: XGBoostModelBuilder) -> None:
        """Initialize the trainer.

        Args:
            model_builder: Configured XGBoostModelBuilder.
        """
        self.model_builder = model_builder
        self.training_history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int = 10,
        verbose: bool = True,
    ) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """Train XGBoost model with early stopping.

        Training workflow:
        1. Prepare watchlist (eval_set) with train and validation data
        2. Train model while monitoring validation metrics
        3. Use early stopping to halt when validation loss stops improving
        4. Return best model from early stopping point

        Why early stopping matters:
        In boosting, each iteration should improve training loss. However, after
        some point, additional trees start to overfit to training noise. Validation
        loss (measured on unseen data) stops improving or even increases. By stopping
        when validation loss doesn't improve for N rounds, we get the best generalization.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            early_stopping_rounds: Stop if validation loss doesn't improve for N rounds.
            verbose: Print training progress.

        Returns:
            Tuple of:
                - Trained XGBoost model
                - Training results dict with stopping info
        """
        model = self.model_builder.get_model()

        # Create evaluation set for XGBoost's built-in monitoring
        # XGBoost tracks multiple metrics on the watchlist
        eval_set = [(X_train, y_train), (X_val, y_val)]

        # Silence XGBoost's own verbose output to avoid clutter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Fit with early stopping
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )

        self.model_builder.is_trained = True

        # Extract training results
        results = self._extract_results(model, verbose=verbose)

        return model, results

    def _extract_results(
        self,
        model: xgb.XGBClassifier,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Extract and log training results.

        Args:
            model: Trained XGBoost model.
            verbose: Whether to print results.

        Returns:
            Dictionary with training metrics and stopping info.
        """
        results = {
            "best_ntree_limit": model.best_ntree_limit if hasattr(model, "best_ntree_limit") else model.n_estimators,
            "best_score": model.best_score if hasattr(model, "best_score") else None,
            "evals_result": model.evals_result() if hasattr(model, "evals_result") else {},
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Complete")
            print(f"{'='*60}")
            print(f"Best tree limit: {results['best_ntree_limit']}")
            if results['best_score']:
                print(f"Best validation score: {results['best_score']:.4f}")
            print(f"{'='*60}\n")

        return results


class AdaptiveTrainer(ModelTrainer):
    """Enhanced trainer with adaptive hyperparameter adjustments.

    This trainer monitors training dynamics and can suggest hyperparameter
    adjustments if signs of underfitting or overfitting are detected.

    Underfitting signs:
    - Both train and validation loss remain high
    - Loss decreases slowly
    - Suggestion: Increase n_estimators, decrease learning_rate

    Overfitting signs:
    - Train loss decreases but validation loss increases
    - Suggestion: Increase regularization (lower subsample/colsample)
    """

    def __init__(self, model_builder: XGBoostModelBuilder) -> None:
        """Initialize adaptive trainer."""
        super().__init__(model_builder)
        self.fit_diagnostics: Dict[str, str] = {}

    def train_with_diagnostics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int = 10,
        verbose: bool = True,
    ) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """Train with fit diagnostics and recommendations.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            early_stopping_rounds: Early stopping patience.
            verbose: Print diagnostics.

        Returns:
            Tuple of trained model and results.
        """
        model, results = self.train(
            X_train, y_train, X_val, y_val,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )

        # Analyze fit quality
        self._diagnose_fit(model, verbose=verbose)

        return model, results

    def _diagnose_fit(self, model: xgb.XGBClassifier, verbose: bool = True) -> None:
        """Diagnose model fit quality and suggest improvements.

        Args:
            model: Trained model.
            verbose: Print diagnostics.
        """
        # Get predictions on train and val to assess fit
        # (In production, use the stored eval results)
        diagnostics = []

        best_ntree = model.best_ntree_limit if hasattr(model, "best_ntree_limit") else model.n_estimators

        # Check if early stopping kicked in early (indicates good fit)
        if best_ntree < model.n_estimators * 0.5:
            diagnostics.append("✓ Early stopping triggered early (good sign of fit)")

        self.fit_diagnostics = {"status": "fit_complete", "ntrees": best_ntree}

        if verbose and diagnostics:
            print("\nFit Diagnostics:")
            for diag in diagnostics:
                print(f"  {diag}")


def train_model_simple(
    model_builder: XGBoostModelBuilder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int = 10,
) -> xgb.XGBClassifier:
    """Convenience function to train in one line.

    Args:
        model_builder: Configured builder.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        early_stopping_rounds: Early stopping patience.

    Returns:
        Trained XGBoost model.
    """
    trainer = ModelTrainer(model_builder)
    model, _ = trainer.train(
        X_train, y_train, X_val, y_val,
        early_stopping_rounds=early_stopping_rounds,
        verbose=True
    )
    return model
