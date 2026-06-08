"""
Data loader and synthetic dataset generator for SignalSeeker.

In particle physics experiments (the inspiration for this package), "signal" refers
to the desired particle interaction, while "background" refers to all other processes.
Real datasets are heavily imbalanced with much more background than signal.

This module generates synthetic data that mimics these realistic characteristics.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from .config import DataConfig, PreprocessConfig


class DataLoader:
    """Generates and manages synthetic binary classification datasets.

    The class creates imbalanced datasets where the minority class (signal)
    is much rarer than the majority class (background). This is typical in
    particle physics applications.

    Attributes:
        config: DataConfig object containing data generation parameters.
    """

    def __init__(self, config: DataConfig) -> None:
        """Initialize the data loader.

        Args:
            config: DataConfig object with generation parameters.
        """
        self.config = config
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic imbalanced classification dataset.

        Uses sklearn.datasets.make_classification to create a realistic
        binary classification problem with controllable imbalance.

        Why make_classification?
        - Creates linearly separable patterns with controlled difficulty
        - Allows explicit control of feature informativeness
        - Mimics real-world data with noise and redundant features

        Returns:
            Tuple containing:
                - X: Feature matrix of shape (n_samples, n_features)
                - y: Target labels (0=background, 1=signal)
        """
        self._X, self._y = make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_informative=self.config.n_informative,
            n_redundant=self.config.n_features - self.config.n_informative,
            weights=[1 - self.config.signal_fraction, self.config.signal_fraction],
            random_state=self.config.random_state,
            flip_y=0.01,  # 1% label noise to simulate measurement errors
        )
        return self._X, self._y

    @property
    def X(self) -> np.ndarray:
        """Get feature matrix (generates if not already done)."""
        if self._X is None:
            self.generate_data()
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Get target labels (generates if not already done)."""
        if self._y is None:
            self.generate_data()
        return self._y

    def get_class_distribution(self) -> dict:
        """Calculate class distribution statistics.

        Returns:
            Dictionary with class counts and fractions.
        """
        unique, counts = np.unique(self.y, return_counts=True)
        return {
            "background_count": counts[0],
            "signal_count": counts[1],
            "background_fraction": counts[0] / len(self.y),
            "signal_fraction": counts[1] / len(self.y),
            "imbalance_ratio": counts[0] / counts[1],
        }


class DataSplitter:
    """Handles train/validation/test split of the dataset.

    Attributes:
        config: PreprocessConfig object containing split parameters.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        """Initialize the data splitter.

        Args:
            config: PreprocessConfig with test_size and val_size.
        """
        self.config = config

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets.

        Uses stratified splitting to maintain class balance in each split.
        This is critical for imbalanced datasets to avoid test sets that
        don't represent the true distribution.

        Workflow:
        1. First split: X, y → (X_temp, y_temp), (X_test, y_test)
        2. Second split: X_temp, y_temp → (X_train, y_train), (X_val, y_val)

        Args:
            X: Feature matrix.
            y: Target labels.
            random_state: Random seed.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        # First split: separate test set (80% training, 20% testing)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=random_state,
        )

        # Second split: separate validation from training
        # val_size is a fraction of the remaining data (training + val)
        adjusted_val_size = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=adjusted_val_size,
            stratify=y_temp,
            random_state=random_state,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_pipeline(
    data_config: DataConfig,
    preprocess_config: PreprocessConfig,
    random_state: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Convenient function to generate and split data in one step.

    Args:
        data_config: Configuration for data generation.
        preprocess_config: Configuration for data splitting.
        random_state: Random seed.

    Returns:
        Tuple containing:
            - (X_train, X_val, X_test)
            - (y_train, y_val, y_test)
    """
    loader = DataLoader(data_config)
    X, y = loader.generate_data()

    splitter = DataSplitter(preprocess_config)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
        X, y, random_state=random_state
    )

    return (X_train, X_val, X_test), (y_train, y_val, y_test)
