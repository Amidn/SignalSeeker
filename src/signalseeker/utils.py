"""
Utility functions and helpers for SignalSeeker.

This module contains helper functions for logging, file management,
and other common operations used throughout the pipeline.
"""

import logging
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


def setup_logging(
    log_file: Optional[Path] = None,
    verbose: bool = True,
) -> logging.Logger:
    """Configure logging for the pipeline.

    Args:
        log_file: Optional file path to save logs to.
        verbose: If True, prints logs to console.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("SignalSeeker")
    logger.setLevel(logging.DEBUG)

    # Console handler
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_results(
    results: dict,
    output_path: Path,
) -> None:
    """Save results dictionary to JSON file.

    Args:
        results: Dictionary with results to save.
        output_path: Path to save JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    import numpy as np
    results_serializable = _make_serializable(results)

    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"Results saved to {output_path}")


def _make_serializable(obj):
    """Recursively convert numpy types to native Python types.

    Args:
        obj: Object to convert.

    Returns:
        Serializable object.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def load_results(input_path: Path) -> dict:
    """Load results from JSON file.

    Args:
        input_path: Path to JSON file.

    Returns:
        Results dictionary.
    """
    with open(input_path, "r") as f:
        results = json.load(f)
    return results


def create_output_directory(base_dir: Path) -> Path:
    """Create timestamped output directory for experiment.

    Args:
        base_dir: Base directory for outputs.

    Returns:
        Path to created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class ExperimentTracker:
    """Tracks experiment metadata and results.

    Attributes:
        experiment_name: Name of the experiment.
        config: Configuration dictionary.
        results: Results dictionary.
    """

    def __init__(self, experiment_name: str, config: dict) -> None:
        """Initialize experiment tracker.

        Args:
            experiment_name: Name for this experiment.
            config: Configuration dictionary.
        """
        self.experiment_name = experiment_name
        self.config = config
        self.results = {}
        self.start_time = datetime.now()

    def log_metric(self, name: str, value) -> None:
        """Log a metric result.

        Args:
            name: Metric name.
            value: Metric value.
        """
        self.results[name] = value

    def log_metrics(self, metrics: dict) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metrics.
        """
        self.results.update(metrics)

    def save_experiment(self, output_dir: Path) -> None:
        """Save experiment config and results.

        Args:
            output_dir: Directory to save to.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = output_dir / "config.json"
        save_results(self.config, config_path)

        # Save results
        results_with_metadata = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "metrics": self.results,
        }
        results_path = output_dir / "results.json"
        save_results(results_with_metadata, results_path)

        print(f"Experiment saved to {output_dir}")

    def print_summary(self) -> None:
        """Print experiment summary."""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"Duration: {duration:.1f} seconds")
        print(f"\nResults:")
        for key, value in self.results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*60}\n")


def print_section(title: str) -> None:
    """Print a formatted section header.

    Args:
        title: Section title.
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")


def print_step(step_num: int, description: str) -> None:
    """Print a numbered step.

    Args:
        step_num: Step number.
        description: Step description.
    """
    print(f"Step {step_num}: {description}")
