"""
Main entry point for SignalSeeker pipeline.

This module orchestrates the complete machine learning pipeline:
1. Data generation and loading
2. Data preprocessing
3. Model building
4. Model training
5. Hyperparameter tuning (optional)
6. Model evaluation
7. Visualization
8. Results saving

Run this module to execute the full pipeline:
    python main.py
"""

from pathlib import Path
import argparse
from typing import Tuple, Optional

import numpy as np

from .config import PipelineConfig, DEFAULT_CONFIG
from .data_loader import create_data_pipeline, DataLoader
from .preprocessor import DataPreprocessingPipeline
from .model_builder import create_model_for_data
from .trainer import train_model_simple
from .metrics import MetricsCalculator, PerformanceReporter, evaluate_model
from .visualizer import ModelVisualizer
from .utils import (
    setup_logging,
    save_results,
    create_output_directory,
    ExperimentTracker,
    print_section,
    print_step,
)


class SignalSeekerPipeline:
    """Main pipeline orchestrator for the SignalSeeker package.

    This class manages the complete workflow from data generation to model
    evaluation and visualization.

    Attributes:
        config: PipelineConfig with all settings.
        output_dir: Directory for saving results.
        logger: Logger instance.
        tracker: Experiment tracker.
    """

    def __init__(self, config: PipelineConfig, output_dir: Optional[Path] = None) -> None:
        """Initialize the pipeline.

        Args:
            config: PipelineConfig with all hyperparameters.
            output_dir: Optional output directory. If None, creates timestamped dir.
        """
        self.config = config
        self.output_dir = output_dir or create_output_directory(Path("./results"))
        self.logger = setup_logging(
            log_file=self.output_dir / "pipeline.log",
            verbose=config.verbose
        )

        # Track experiment
        config_dict = {
            "data": config.data.__dict__,
            "preprocess": config.preprocess.__dict__,
            "xgboost": config.xgboost.__dict__,
            "tuner": config.tuner.__dict__,
        }
        self.tracker = ExperimentTracker("SignalSeeker Full Pipeline", config_dict)

        # Create results directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config.verbose:
            print_section("SignalSeeker Pipeline Initialized")
            print(f"Output directory: {self.output_dir}")

    def run(self, use_tuning: bool = False) -> dict:
        """Execute the complete pipeline.

        Args:
            use_tuning: Whether to run hyperparameter tuning (slower but better).

        Returns:
            Dictionary with complete results.
        """
        # Step 1: Generate data
        print_step(1, "Generating synthetic data")
        (X_train, X_val, X_test), (y_train, y_val, y_test) = self._generate_data()

        # Step 2: Preprocess data
        print_step(2, "Preprocessing data")
        X_train, X_val, X_test = self._preprocess_data(X_train, X_val, X_test)

        # Step 3: Build model
        print_step(3, "Building XGBoost model")
        model_builder = self._build_model(y_train)

        # Step 4: Optionally tune hyperparameters
        if use_tuning:
            print_step(4, "Tuning hyperparameters (this may take a while...)")
            model_builder = self._tune_hyperparameters(X_train, y_train, model_builder)
            step_offset = 1
        else:
            print_step(4, "Skipping hyperparameter tuning (use --tune to enable)")
            step_offset = 0

        # Step 5: Train model
        print_step(5 + step_offset, "Training model with early stopping")
        model = self._train_model(model_builder, X_train, y_train, X_val, y_val)

        # Step 6: Evaluate on validation set
        print_step(6 + step_offset, "Evaluating on validation set")
        val_results = self._evaluate(y_val, model.predict_proba(X_val)[:, 1], "Validation")

        # Step 7: Evaluate on test set
        print_step(7 + step_offset, "Evaluating on test set")
        test_results = self._evaluate(y_test, model.predict_proba(X_test)[:, 1], "Test")

        # Step 8: Visualize results
        print_step(8 + step_offset, "Creating visualizations")
        self._visualize_results(y_test, model.predict_proba(X_test)[:, 1], model)

        # Step 9: Save results
        print_step(9 + step_offset, "Saving results")
        self._save_results(model, val_results, test_results)

        # Print summary
        self.tracker.log_metrics({
            "val_auc_roc": val_results["all_metrics"]["auc_roc"],
            "test_auc_roc": test_results["all_metrics"]["auc_roc"],
            "val_auc_pr": val_results["all_metrics"]["auc_pr"],
            "test_auc_pr": test_results["all_metrics"]["auc_pr"],
        })
        self.tracker.print_summary()

        return {
            "model": model,
            "validation_results": val_results,
            "test_results": test_results,
        }

    def _generate_data(self) -> Tuple:
        """Generate synthetic dataset.

        Returns:
            Tuple of (X_arrays, y_arrays) where each is (train, val, test).
        """
        loader = DataLoader(self.config.data)
        dist = loader.get_class_distribution()

        if self.config.verbose:
            print(f"  Generated {self.config.data.n_samples} samples")
            print(f"  Signal: {dist['signal_count']} ({dist['signal_fraction']*100:.1f}%)")
            print(f"  Background: {dist['background_count']} ({dist['background_fraction']*100:.1f}%)")
            print(f"  Imbalance ratio: {dist['imbalance_ratio']:.1f}x")

        return create_data_pipeline(
            self.config.data,
            self.config.preprocess,
            random_state=self.config.random_state,
        )

    def _preprocess_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data.

        Args:
            X_train, X_val, X_test: Feature arrays.

        Returns:
            Preprocessed feature arrays.
        """
        pipeline = DataPreprocessingPipeline(self.config.preprocess)
        X_train, X_val, X_test = pipeline.fit(X_train, X_val, X_test)
        pipeline.save(self.output_dir / "preprocessing")

        if self.config.verbose:
            print(f"  Preprocessing complete")
            print(f"  Scale features: {self.config.preprocess.scale_features}")
            print(f"  Handle missing: {self.config.preprocess.handle_missing}")

        return X_train, X_val, X_test

    def _build_model(self, y_train: np.ndarray):
        """Build XGBoost model with automatic class weighting.

        Args:
            y_train: Training labels.

        Returns:
            XGBoostModelBuilder instance.
        """
        model_builder = create_model_for_data(self.config.xgboost, y_train)
        weight = model_builder.scale_pos_weight

        if self.config.verbose:
            print(f"  Model created with automatic class weighting")
            print(f"  Scale pos weight: {weight:.2f}")
            print(f"  Max depth: {self.config.xgboost.max_depth}")
            print(f"  Learning rate: {self.config.xgboost.learning_rate}")

        return model_builder

    def _tune_hyperparameters(self, X_train, y_train, model_builder):
        """Run hyperparameter tuning.

        Args:
            X_train: Training features.
            y_train: Training labels.
            model_builder: Current model builder.

        Returns:
            Updated model builder with tuned parameters.
        """
        from tuner import HyperparameterTuner

        tuner = HyperparameterTuner(self.config.tuner, self.config.xgboost)
        results = tuner.optimize(X_train, y_train, verbose=self.config.verbose)

        return tuner.create_model_from_best()

    def _train_model(self, model_builder, X_train, y_train, X_val, y_val):
        """Train model with early stopping.

        Args:
            model_builder: XGBoostModelBuilder.
            X_train, y_train: Training data.
            X_val, y_val: Validation data.

        Returns:
            Trained XGBoost model.
        """
        model = train_model_simple(
            model_builder,
            X_train, y_train,
            X_val, y_val,
            early_stopping_rounds=10,
        )

        if self.config.verbose:
            print(f"  Model trained successfully")

        return model

    def _evaluate(self, y_true, y_pred_proba, dataset_name: str) -> dict:
        """Evaluate model on a dataset.

        Args:
            y_true: Ground truth labels.
            y_pred_proba: Predicted probabilities.
            dataset_name: Name of dataset for reporting.

        Returns:
            Evaluation results dictionary.
        """
        return evaluate_model(y_true, y_pred_proba, dataset_name, print_report=True)

    def _visualize_results(self, y_test, y_pred_proba, model):
        """Create visualizations.

        Args:
            y_test: Test labels.
            y_pred_proba: Predicted probabilities on test set.
            model: Trained XGBoost model.
        """
        visualizer = ModelVisualizer(self.config.visualizer)

        # Core visualization: Probability distribution
        visualizer.plot_probability_score_distribution(
            y_test, y_pred_proba, cut_threshold=0.5
        )

        # ROC curve
        visualizer.plot_roc_curve(y_test, y_pred_proba)

        # Precision-recall curve
        visualizer.plot_precision_recall_curve(y_test, y_pred_proba)

        # Confusion matrix
        y_pred = (y_pred_proba >= 0.5).astype(int)
        visualizer.plot_confusion_matrix(y_test, y_pred)

        # Feature importance
        visualizer.plot_feature_importance(model, top_n=15)

        # Summary figure
        visualizer.plot_summary(y_test, y_pred_proba, y_pred, model)

        if self.config.verbose:
            print(f"  Visualizations saved to {self.config.visualizer.output_dir}")

    def _save_results(self, model, val_results, test_results):
        """Save all results and model.

        Args:
            model: Trained model.
            val_results: Validation evaluation results.
            test_results: Test evaluation results.
        """
        # Save model
        model_path = self.output_dir / "model.joblib"
        import joblib
        joblib.dump(model, model_path)

        # Save results
        results = {
            "validation": val_results,
            "test": test_results,
        }
        save_results(results, self.output_dir / "results.json")

        if self.config.verbose:
            print(f"  Model saved to {model_path}")
            print(f"  Results saved to {self.output_dir / 'results.json'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SignalSeeker: XGBoost Binary Classification Pipeline"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning (slower but better results)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--signal-fraction",
        type=float,
        default=0.1,
        help="Fraction of signal samples (0-1)"
    )

    args = parser.parse_args()

    # Configure pipeline
    config = DEFAULT_CONFIG
    if args.n_samples:
        config.data.n_samples = args.n_samples
    if args.signal_fraction:
        config.data.signal_fraction = args.signal_fraction

    # Run pipeline
    pipeline = SignalSeekerPipeline(config, output_dir=args.output_dir)
    results = pipeline.run(use_tuning=args.tune)

    return results


if __name__ == "__main__":
    main()
