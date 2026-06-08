"""
Hyperparameter tuning module for SignalSeeker.

Hyperparameter tuning is the process of finding the best set of hyperparameters
for a given model and dataset. This is typically more important than model selection,
as a well-tuned simple model often beats a poorly-tuned complex one.

This module uses Optuna, a modern Bayesian optimization library, to efficiently
search the hyperparameter space. Optuna is much faster than grid search or
random search for high-dimensional spaces.

Why Bayesian optimization?
- Smarter than grid search: Uses previous trials to guide new searches
- Efficient: Finds good parameters in far fewer trials than grid/random search
- Handles mixed types: Can optimize continuous, categorical, and integer parameters
- Pruning: Stops unpromising trials early, saving computation
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.model_selection import cross_validate
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import xgboost as xgb
from .config import TunerConfig, XGBoostConfig
from .model_builder import XGBoostModelBuilder


class HyperparameterTuner:
    """Performs Bayesian hyperparameter optimization.

    Uses Optuna to find optimal XGBoost hyperparameters by searching the
    parameter space using cross-validation and early stopping.

    Attributes:
        tuner_config: TunerConfig with optimization settings.
        xgb_config: XGBoostConfig baseline configuration.
        best_params: Best parameters found during optimization.
        study: Optuna study object with optimization history.
    """

    def __init__(
        self,
        tuner_config: TunerConfig,
        xgb_config: XGBoostConfig,
    ) -> None:
        """Initialize the tuner.

        Args:
            tuner_config: Configuration for tuning.
            xgb_config: Baseline XGBoost configuration.
        """
        self.tuner_config = tuner_config
        self.xgb_config = xgb_config
        self.best_params: Optional[Dict[str, Any]] = None
        self.study: Optional[optuna.Study] = None

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> float:
        """Objective function for Optuna optimization.

        This function is called for each trial. It:
        1. Suggests hyperparameters
        2. Trains a model with those parameters
        3. Evaluates via cross-validation
        4. Returns the score (higher is better)

        Optuna uses this score to guide the search toward better parameters.

        Args:
            trial: Optuna trial object.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Cross-validation score (AUC).
        """
        # Define hyperparameter search space
        # Each trial suggests different values within these ranges

        # max_depth: typical range 3-10 (deeper = more complex = more overfitting risk)
        max_depth = trial.suggest_int("max_depth", 3, 10)

        # learning_rate: typical range 0.001-0.3 (lower = slower but more stable)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)

        # subsample: fraction of samples per tree (lower = more regularization)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)

        # colsample_bytree: fraction of features per tree
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        # Create model with suggested parameters
        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": self.xgb_config.n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": self.xgb_config.objective,
            "eval_metric": "logloss",
            "random_state": self.xgb_config.random_state,
            "verbosity": 0,
            "use_label_encoder": False,
        }

        # Compute class weight for this trial
        n_positive = np.sum(y_train == 1)
        n_negative = np.sum(y_train == 0)
        params["scale_pos_weight"] = n_negative / n_positive

        model = xgb.XGBClassifier(**params)

        # Evaluate using cross-validation
        # This gives us an estimate of generalization performance
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=self.tuner_config.cv_folds,
            scoring="roc_auc",
            n_jobs=-1,  # Use all CPU cores
        )

        # Return mean CV score
        # Higher AUC is better, so we return the mean AUC
        mean_score = np.mean(scores["test_score"])

        # Optional: Report intermediate value for pruning
        # Optuna can prune (stop) trials that are clearly worse than previous ones
        trial.report(mean_score, step=0)

        return mean_score

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            X_train: Training features.
            y_train: Training labels.
            n_trials: Number of trials (uses config if None).
            verbose: Print progress.

        Returns:
            Dictionary with best parameters and optimization results.
        """
        n_trials = n_trials or self.tuner_config.n_trials

        # Create Optuna study with Bayesian optimization sampler
        sampler = TPESampler(seed=self.xgb_config.random_state)
        pruner = MedianPruner()

        self.study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction="maximize",  # We want to maximize AUC
        )

        # Suppress Optuna's verbose output unless user wants it
        optuna_verbose = 1 if verbose else 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Hyperparameter Optimization")
            print(f"{'='*60}")
            print(f"Number of trials: {n_trials}")
            print(f"Cross-validation folds: {self.tuner_config.cv_folds}")
            print(f"{'='*60}\n")

        # Run optimization
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=verbose,
        )

        # Extract best parameters
        self.best_params = self.study.best_params
        best_value = self.study.best_value

        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimization Complete")
            print(f"{'='*60}")
            print(f"Best CV AUC: {best_value:.4f}")
            print(f"Best parameters found:")
            for param, value in self.best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")
            print(f"{'='*60}\n")

        return {
            "best_params": self.best_params,
            "best_score": best_value,
            "n_trials": len(self.study.trials),
        }

    def create_model_from_best(self) -> XGBoostModelBuilder:
        """Create a model builder using the best found parameters.

        Returns:
            XGBoostModelBuilder with optimal parameters.

        Raises:
            RuntimeError: If optimization hasn't been run.
        """
        if self.best_params is None:
            raise RuntimeError("Must run optimize() before creating model.")

        # Create new config with tuned parameters
        config = XGBoostConfig(
            max_depth=self.best_params["max_depth"],
            learning_rate=self.best_params["learning_rate"],
            n_estimators=self.xgb_config.n_estimators,
            subsample=self.best_params["subsample"],
            colsample_bytree=self.best_params["colsample_bytree"],
            scale_pos_weight=self.xgb_config.scale_pos_weight,
            random_state=self.xgb_config.random_state,
        )

        return XGBoostModelBuilder(config)

    def get_study_summary(self) -> str:
        """Get human-readable summary of optimization results.

        Returns:
            Summary string.
        """
        if self.study is None:
            return "No optimization has been run yet."

        summary = f"\nOptimization Summary:\n"
        summary += f"Total trials: {len(self.study.trials)}\n"
        summary += f"Best value: {self.study.best_value:.4f}\n"
        summary += f"Best trial: {self.study.best_trial.number}\n"
        return summary


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tuner_config: TunerConfig,
    xgb_config: XGBoostConfig,
) -> XGBoostModelBuilder:
    """Convenience function for one-shot hyperparameter tuning.

    Args:
        X_train: Training features.
        y_train: Training labels.
        tuner_config: Tuning configuration.
        xgb_config: Base XGBoost configuration.

    Returns:
        XGBoostModelBuilder with optimized parameters.
    """
    tuner = HyperparameterTuner(tuner_config, xgb_config)
    tuner.optimize(X_train, y_train, verbose=True)
    return tuner.create_model_from_best()
