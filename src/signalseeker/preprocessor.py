"""
Data preprocessing module for SignalSeeker.

Preprocessing is crucial for machine learning pipelines. This module handles:
- Feature scaling (standardization)
- Missing value imputation
- Feature normalization

While XGBoost is robust to feature scaling, preprocessing can improve convergence
and help with feature importance interpretation.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
from .config import PreprocessConfig


class FeaturePreprocessor:
    """Handles feature scaling and normalization.

    StandardScaler transforms features to have mean 0 and standard deviation 1.
    This is especially important for distance-based algorithms, though XGBoost
    (which uses tree splits) is relatively insensitive to feature scales.

    The scaler is fit on training data only and applied to all sets, preventing
    data leakage from test/validation sets into the training process.

    Attributes:
        scaler: Fitted StandardScaler object.
        is_fitted: Whether the scaler has been fitted.
    """

    def __init__(self) -> None:
        """Initialize the feature preprocessor."""
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "FeaturePreprocessor":
        """Fit the scaler on training data.

        Args:
            X_train: Training feature matrix.

        Returns:
            Self for method chaining.
        """
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply scaling to features.

        Args:
            X: Feature matrix to scale.

        Returns:
            Scaled features.

        Raises:
            RuntimeError: If scaler hasn't been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def save(self, path: Path) -> None:
        """Save fitted scaler to disk using joblib.

        Args:
            path: File path to save the scaler.
        """
        joblib.dump(self.scaler, path)

    def load(self, path: Path) -> "FeaturePreprocessor":
        """Load a previously fitted scaler.

        Args:
            path: File path to load the scaler from.

        Returns:
            Self for method chaining.
        """
        self.scaler = joblib.load(path)
        self.is_fitted = True
        return self


class MissingValueHandler:
    """Handles missing values in features.

    In real datasets, missing values (NaN, None) are common and must be handled
    carefully. SimpleImputer with strategy='mean' replaces missing values with
    the feature mean calculated from training data.

    This prevents information from test/val sets leaking into training.

    Attributes:
        imputer: Fitted SimpleImputer object.
        is_fitted: Whether the imputer has been fitted.
    """

    def __init__(self, strategy: str = "mean") -> None:
        """Initialize the missing value handler.

        Args:
            strategy: Imputation strategy ('mean', 'median', 'most_frequent').
        """
        self.imputer = SimpleImputer(strategy=strategy)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "MissingValueHandler":
        """Fit the imputer on training data.

        Args:
            X_train: Training feature matrix.

        Returns:
            Self for method chaining.
        """
        self.imputer.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply imputation to features.

        Args:
            X: Feature matrix to impute.

        Returns:
            Features with missing values filled.

        Raises:
            RuntimeError: If imputer hasn't been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Imputer must be fitted before transform.")
        return self.imputer.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class DataPreprocessingPipeline:
    """Complete preprocessing pipeline combining multiple steps.

    This class orchestrates all preprocessing operations in the correct order:
    1. Handle missing values (must be before scaling)
    2. Scale features (standardization)

    The pipeline maintains a consistent transformation from training to test/val.

    Attributes:
        config: PreprocessConfig with preprocessing settings.
        missing_handler: MissingValueHandler instance.
        feature_preprocessor: FeaturePreprocessor instance.
        is_fitted: Whether the entire pipeline is fitted.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        """Initialize the preprocessing pipeline.

        Args:
            config: PreprocessConfig with processing parameters.
        """
        self.config = config
        self.missing_handler = MissingValueHandler()
        self.feature_preprocessor = FeaturePreprocessor()
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit all preprocessing steps on training data.

        Args:
            X_train: Training features.
            X_val: Validation features.
            X_test: Test features.

        Returns:
            Tuple of (X_train, X_val, X_test) after preprocessing.
        """
        # Handle missing values first
        if self.config.handle_missing:
            self.missing_handler.fit(X_train)
            X_train = self.missing_handler.transform(X_train)
            X_val = self.missing_handler.transform(X_val)
            X_test = self.missing_handler.transform(X_test)

        # Scale features
        if self.config.scale_features:
            self.feature_preprocessor.fit(X_train)
            X_train = self.feature_preprocessor.transform(X_train)
            X_val = self.feature_preprocessor.transform(X_val)
            X_test = self.feature_preprocessor.transform(X_test)

        self.is_fitted = True
        return X_train, X_val, X_test

    def transform(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply preprocessing to new data.

        Args:
            X_train: Training features.
            X_val: Validation features.
            X_test: Test features.

        Returns:
            Tuple of (X_train, X_val, X_test) after preprocessing.

        Raises:
            RuntimeError: If pipeline hasn't been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform.")

        if self.config.handle_missing:
            X_train = self.missing_handler.transform(X_train)
            X_val = self.missing_handler.transform(X_val)
            X_test = self.missing_handler.transform(X_test)

        if self.config.scale_features:
            X_train = self.feature_preprocessor.transform(X_train)
            X_val = self.feature_preprocessor.transform(X_val)
            X_test = self.feature_preprocessor.transform(X_test)

        return X_train, X_val, X_test

    def save(self, directory: Path) -> None:
        """Save all preprocessing components.

        Args:
            directory: Directory to save preprocessing artifacts.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self.feature_preprocessor.save(directory / "scaler.joblib")

    def load(self, directory: Path) -> None:
        """Load previously saved preprocessing components.

        Args:
            directory: Directory containing preprocessing artifacts.
        """
        self.feature_preprocessor.load(directory / "scaler.joblib")
        self.is_fitted = True
