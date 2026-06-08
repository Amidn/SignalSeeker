"""
Example usage patterns for SignalSeeker.

This module demonstrates various ways to use the SignalSeeker package,
from simple to advanced applications.

Run examples:
    python example_usage.py
"""

import numpy as np
from pathlib import Path

# Import all major components
from config import (
    PipelineConfig, DataConfig, XGBoostConfig,
    TunerConfig, DEFAULT_CONFIG
)
from data_loader import create_data_pipeline, DataLoader
from preprocessor import DataPreprocessingPipeline
from model_builder import create_model_for_data
from trainer import train_model_simple
from metrics import evaluate_model
from visualizer import ModelVisualizer


def example_1_minimal_pipeline():
    """
    Example 1: Minimal pipeline with default settings.

    This is the simplest way to run the complete pipeline.
    """
    print("\n" + "="*60)
    print("Example 1: Minimal Pipeline (Default Settings)")
    print("="*60 + "\n")

    from main import SignalSeekerPipeline

    # Use default configuration
    config = DEFAULT_CONFIG

    # Create and run pipeline
    pipeline = SignalSeekerPipeline(config)
    results = pipeline.run(use_tuning=False)

    # Access results
    print(f"\nTest AUC-ROC: {results['test_results']['all_metrics']['auc_roc']:.4f}")
    print(f"Test AUC-PR:  {results['test_results']['all_metrics']['auc_pr']:.4f}")
    print(f"Test F1-Score: {results['test_results']['all_metrics']['f1_score']:.4f}")


def example_2_custom_configuration():
    """
    Example 2: Custom configuration for specific use case.

    Shows how to modify hyperparameters before running the pipeline.
    """
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60 + "\n")

    from main import SignalSeekerPipeline

    # Create custom config
    config = PipelineConfig()

    # Modify data generation
    config.data.n_samples = 20000      # More samples
    config.data.signal_fraction = 0.15 # More signal
    config.data.n_features = 30        # More features

    # Modify XGBoost parameters
    config.xgboost.max_depth = 7
    config.xgboost.learning_rate = 0.15
    config.xgboost.n_estimators = 150

    # Run with custom config
    pipeline = SignalSeekerPipeline(config)
    results = pipeline.run(use_tuning=False)

    print(f"\nTest AUC: {results['test_results']['all_metrics']['auc_roc']:.4f}")


def example_3_data_inspection():
    """
    Example 3: Inspect the synthetic data before training.

    Shows how to generate and analyze data separately.
    """
    print("\n" + "="*60)
    print("Example 3: Data Inspection")
    print("="*60 + "\n")

    config = DEFAULT_CONFIG

    # Generate data
    loader = DataLoader(config.data)
    X, y = loader.generate_data()

    # Get distribution
    dist = loader.get_class_distribution()

    print(f"Total samples: {len(y)}")
    print(f"Signal samples: {dist['signal_count']} ({dist['signal_fraction']*100:.1f}%)")
    print(f"Background samples: {dist['background_count']} ({dist['background_fraction']*100:.1f}%)")
    print(f"Imbalance ratio: {dist['imbalance_ratio']:.1f}x")

    print(f"\nFeature statistics:")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature mean: {X.mean(axis=0)[:5]}")  # Show first 5
    print(f"Feature std: {X.std(axis=0)[:5]}")

    # Class-wise feature means (shows separation)
    X_signal = X[y == 1]
    X_background = X[y == 0]
    print(f"\nSignal mean (feature 0): {X_signal[:, 0].mean():.3f}")
    print(f"Background mean (feature 0): {X_background[:, 0].mean():.3f}")


def example_4_training_only():
    """
    Example 4: Focus on training and evaluation.

    Shows how to prepare data, train model, and evaluate.
    """
    print("\n" + "="*60)
    print("Example 4: Training & Evaluation Only")
    print("="*60 + "\n")

    config = DEFAULT_CONFIG

    # Step 1: Generate and split data
    (X_train, X_val, X_test), (y_train, y_val, y_test) = \
        create_data_pipeline(config.data, config.preprocess)

    print(f"Training set size: {len(y_train)}")
    print(f"Validation set size: {len(y_val)}")
    print(f"Test set size: {len(y_test)}")

    # Step 2: Preprocess
    preproc = DataPreprocessingPipeline(config.preprocess)
    X_train, X_val, X_test = preproc.fit(X_train, X_val, X_test)

    print(f"\nPreprocessing complete")

    # Step 3: Build and train
    model_builder = create_model_for_data(config.xgboost, y_train)
    model = train_model_simple(
        model_builder,
        X_train, y_train,
        X_val, y_val,
        early_stopping_rounds=10
    )

    # Step 4: Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    results = evaluate_model(y_test, y_pred_proba, "Test Set")

    print(f"\n✓ Model trained and evaluated successfully")


def example_5_hyperparameter_tuning():
    """
    Example 5: Hyperparameter tuning with Optuna.

    Shows how to use Bayesian optimization to find best parameters.
    """
    print("\n" + "="*60)
    print("Example 5: Hyperparameter Tuning")
    print("="*60 + "\n")

    from tuner import HyperparameterTuner

    config = DEFAULT_CONFIG

    # Generate and prepare data
    (X_train, _, _), (y_train, _, _) = \
        create_data_pipeline(config.data, config.preprocess)

    # Preprocess
    preproc = DataPreprocessingPipeline(config.preprocess)
    X_train, _, _ = preproc.fit(X_train, X_train, X_train)

    # Create tuner with small n_trials for quick demo
    tuner_config = TunerConfig(n_trials=10)  # Quick demo
    tuner = HyperparameterTuner(tuner_config, config.xgboost)

    # Run optimization
    print("Running Bayesian optimization...")
    results = tuner.optimize(X_train, y_train, verbose=True)

    print(f"\n✓ Best CV AUC: {results['best_score']:.4f}")
    print(f"Best parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")


def example_6_visualization_only():
    """
    Example 6: Generate visualizations from pre-computed predictions.

    Shows how to create plots without running the full pipeline.
    """
    print("\n" + "="*60)
    print("Example 6: Visualization Only")
    print("="*60 + "\n")

    config = DEFAULT_CONFIG

    # Quick train-test
    (X_train, X_val, X_test), (y_train, y_val, y_test) = \
        create_data_pipeline(config.data, config.preprocess)

    preproc = DataPreprocessingPipeline(config.preprocess)
    X_train, X_val, X_test = preproc.fit(X_train, X_val, X_test)

    model_builder = create_model_for_data(config.xgboost, y_train)
    model = train_model_simple(
        model_builder, X_train, y_train, X_val, y_val,
        early_stopping_rounds=10
    )

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Create visualizations
    visualizer = ModelVisualizer(config.visualizer)

    print("Creating visualizations...")
    visualizer.plot_probability_score_distribution(y_test, y_pred_proba)
    visualizer.plot_roc_curve(y_test, y_pred_proba)
    visualizer.plot_precision_recall_curve(y_test, y_pred_proba)
    visualizer.plot_confusion_matrix(y_test, y_pred)
    visualizer.plot_feature_importance(model)

    print("\n✓ Visualizations complete")


def example_7_class_weights_analysis():
    """
    Example 7: Understanding class weights for imbalanced data.

    Shows how scale_pos_weight is computed and its effect.
    """
    print("\n" + "="*60)
    print("Example 7: Class Weight Analysis")
    print("="*60 + "\n")

    config = DEFAULT_CONFIG

    # Generate data
    loader = DataLoader(config.data)
    X, y = loader.generate_data()

    # Compute class weight
    n_positive = np.sum(y == 1)
    n_negative = np.sum(y == 0)
    scale_pos_weight = n_negative / n_positive

    print(f"Signal (positive) count: {n_positive}")
    print(f"Background (negative) count: {n_negative}")
    print(f"Imbalance ratio: {n_negative / n_positive:.1f}x")
    print(f"\nscale_pos_weight to balance: {scale_pos_weight:.2f}")

    print("\nExplanation:")
    print(f"- This weight tells XGBoost to treat signal as {scale_pos_weight:.1f}x more important")
    print(f"- Helps prevent the model from simply predicting 'all background'")
    print(f"- Balances the trade-off between precision and recall")


def example_8_batch_experiments():
    """
    Example 8: Run multiple experiments with different configs.

    Shows how to systematically compare different parameter settings.
    """
    print("\n" + "="*60)
    print("Example 8: Batch Experiments")
    print("="*60 + "\n")

    from main import SignalSeekerPipeline

    # Different learning rates to test
    learning_rates = [0.05, 0.1, 0.2]
    results_summary = {}

    for lr in learning_rates:
        print(f"\nTesting learning_rate = {lr}")

        config = DEFAULT_CONFIG
        config.xgboost.learning_rate = lr
        config.data.n_samples = 5000  # Smaller for speed

        pipeline = SignalSeekerPipeline(config)
        results = pipeline.run(use_tuning=False)

        auc = results['test_results']['all_metrics']['auc_roc']
        results_summary[lr] = auc
        print(f"  → Test AUC: {auc:.4f}")

    print(f"\n{'='*60}")
    print("Summary:")
    for lr, auc in sorted(results_summary.items()):
        print(f"  Learning rate {lr}: AUC = {auc:.4f}")
    best_lr = max(results_summary, key=results_summary.get)
    print(f"\nBest learning rate: {best_lr}")


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# SignalSeeker Example Usage")
    print("#"*60)

    examples = [
        (1, "Minimal Pipeline", example_1_minimal_pipeline),
        (2, "Custom Configuration", example_2_custom_configuration),
        (3, "Data Inspection", example_3_data_inspection),
        (4, "Training & Evaluation", example_4_training_only),
        (5, "Hyperparameter Tuning", example_5_hyperparameter_tuning),
        (6, "Visualization Only", example_6_visualization_only),
        (7, "Class Weight Analysis", example_7_class_weights_analysis),
        (8, "Batch Experiments", example_8_batch_experiments),
    ]

    print("\nAvailable examples:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")

    # Run just the quick examples for demo
    print("\n" + "="*60)
    print("Running quick examples (1, 3, 7)...")
    print("="*60)

    try:
        example_1_minimal_pipeline()
    except Exception as e:
        print(f"Example 1 error: {e}")

    try:
        example_3_data_inspection()
    except Exception as e:
        print(f"Example 3 error: {e}")

    try:
        example_7_class_weights_analysis()
    except Exception as e:
        print(f"Example 7 error: {e}")

    print("\n" + "="*60)
    print("Examples complete!")
    print("Run individual examples by calling their functions directly")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
