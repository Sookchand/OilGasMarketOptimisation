"""
Online Learning Framework for continuous model improvement.
This module provides tools for updating models with new data and evaluating their performance.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import pickle
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.monitoring.drift_detector import AdvancedDriftDetector
from src.utils.data_utils import load_processed_data, load_features_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/online_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Model registry for managing model versions and metadata.
    """

    def __init__(self, registry_dir: str = 'models/registry'):
        """
        Initialize the model registry.

        Parameters
        ----------
        registry_dir : str, optional
            Directory for storing model registry, by default 'models/registry'
        """
        self.registry_dir = registry_dir
        self.registry = {}

        # Create registry directory if it doesn't exist
        os.makedirs(registry_dir, exist_ok=True)

        # Load registry if it exists
        registry_file = os.path.join(registry_dir, 'registry.json')
        if os.path.exists(registry_file):
            try:
                with open(registry_file, 'r') as f:
                    self.registry = json.load(f)
                logger.info(f"Loaded model registry from {registry_file}")
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")

    def register_model(
        self,
        commodity: str,
        model_type: str,
        model: Any,
        metrics: Dict[str, float],
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a model in the registry.

        Parameters
        ----------
        commodity : str
            Name of the commodity
        model_type : str
            Type of the model (e.g., 'arima', 'xgboost', 'lstm')
        model : Any
            Model object
        metrics : Dict[str, float]
            Model evaluation metrics
        version : str, optional
            Model version, by default None (will generate a timestamp-based version)
        metadata : Dict[str, Any], optional
            Additional model metadata, by default None

        Returns
        -------
        str
            Model version
        """
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model ID
        model_id = f"{commodity}_{model_type}_{version}"

        # Create model directory
        model_dir = os.path.join(self.registry_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(model_dir, 'model.pkl')
        try:
            # Try different serialization methods
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception:
                joblib.dump(model, model_path)

            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""

        # Prepare model metadata
        model_metadata = {
            'commodity': commodity,
            'model_type': model_type,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics,
            'model_path': model_path,
            'is_active': True
        }

        # Add additional metadata if provided
        if metadata:
            model_metadata.update(metadata)

        # Add to registry
        if commodity not in self.registry:
            self.registry[commodity] = {}

        if model_type not in self.registry[commodity]:
            self.registry[commodity][model_type] = {}

        self.registry[commodity][model_type][version] = model_metadata

        # Save registry
        self._save_registry()

        logger.info(f"Registered model {model_id}")
        return version

    def get_model(
        self,
        commodity: str,
        model_type: str,
        version: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Get a model from the registry.

        Parameters
        ----------
        commodity : str
            Name of the commodity
        model_type : str
            Type of the model
        version : str, optional
            Model version, by default None (will get the latest version)

        Returns
        -------
        Tuple[Any, Dict[str, Any]]
            Model object and model metadata
        """
        # Check if commodity exists in registry
        if commodity not in self.registry:
            logger.error(f"Commodity {commodity} not found in registry")
            return None, {}

        # Check if model type exists in registry
        if model_type not in self.registry[commodity]:
            logger.error(f"Model type {model_type} not found for commodity {commodity}")
            return None, {}

        # Get version
        if version is None:
            # Get latest version
            versions = list(self.registry[commodity][model_type].keys())
            if not versions:
                logger.error(f"No versions found for {commodity} {model_type}")
                return None, {}

            # Sort versions by creation time
            versions.sort(key=lambda v: self.registry[commodity][model_type][v]['created_at'], reverse=True)
            version = versions[0]

        # Check if version exists
        if version not in self.registry[commodity][model_type]:
            logger.error(f"Version {version} not found for {commodity} {model_type}")
            return None, {}

        # Get model metadata
        model_metadata = self.registry[commodity][model_type][version]

        # Load model
        model_path = model_metadata['model_path']
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None, model_metadata

        try:
            # Try different deserialization methods
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            except Exception:
                model = joblib.load(model_path)

            logger.info(f"Loaded model from {model_path}")
            return model, model_metadata

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, model_metadata

    def get_latest_model(
        self,
        commodity: str,
        model_type: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Get the latest model from the registry.

        Parameters
        ----------
        commodity : str
            Name of the commodity
        model_type : str
            Type of the model

        Returns
        -------
        Tuple[Any, Dict[str, Any]]
            Model object and model metadata
        """
        return self.get_model(commodity, model_type)

    def get_active_model(
        self,
        commodity: str,
        model_type: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Get the active model from the registry.

        Parameters
        ----------
        commodity : str
            Name of the commodity
        model_type : str
            Type of the model

        Returns
        -------
        Tuple[Any, Dict[str, Any]]
            Model object and model metadata
        """
        # Check if commodity exists in registry
        if commodity not in self.registry:
            logger.error(f"Commodity {commodity} not found in registry")
            return None, {}

        # Check if model type exists in registry
        if model_type not in self.registry[commodity]:
            logger.error(f"Model type {model_type} not found for commodity {commodity}")
            return None, {}

        # Find active model
        for version, metadata in self.registry[commodity][model_type].items():
            if metadata.get('is_active', False):
                return self.get_model(commodity, model_type, version)

        # If no active model found, get the latest
        logger.warning(f"No active model found for {commodity} {model_type}, using latest")
        return self.get_latest_model(commodity, model_type)

    def set_active_model(
        self,
        commodity: str,
        model_type: str,
        version: str
    ) -> bool:
        """
        Set a model as active.

        Parameters
        ----------
        commodity : str
            Name of the commodity
        model_type : str
            Type of the model
        version : str
            Model version

        Returns
        -------
        bool
            Whether the operation was successful
        """
        # Check if commodity exists in registry
        if commodity not in self.registry:
            logger.error(f"Commodity {commodity} not found in registry")
            return False

        # Check if model type exists in registry
        if model_type not in self.registry[commodity]:
            logger.error(f"Model type {model_type} not found for commodity {commodity}")
            return False

        # Check if version exists
        if version not in self.registry[commodity][model_type]:
            logger.error(f"Version {version} not found for {commodity} {model_type}")
            return False

        # Set all models as inactive
        for v in self.registry[commodity][model_type]:
            self.registry[commodity][model_type][v]['is_active'] = False

        # Set specified model as active
        self.registry[commodity][model_type][version]['is_active'] = True

        # Save registry
        self._save_registry()

        logger.info(f"Set {commodity} {model_type} {version} as active")
        return True

    def get_all_commodities(self) -> List[str]:
        """
        Get all commodities in the registry.

        Returns
        -------
        List[str]
            List of commodities
        """
        return list(self.registry.keys())

    def get_model_types(self, commodity: str) -> List[str]:
        """
        Get all model types for a commodity.

        Parameters
        ----------
        commodity : str
            Name of the commodity

        Returns
        -------
        List[str]
            List of model types
        """
        if commodity not in self.registry:
            return []

        return list(self.registry[commodity].keys())

    def get_versions(self, commodity: str, model_type: str) -> List[str]:
        """
        Get all versions for a commodity and model type.

        Parameters
        ----------
        commodity : str
            Name of the commodity
        model_type : str
            Type of the model

        Returns
        -------
        List[str]
            List of versions
        """
        if commodity not in self.registry or model_type not in self.registry[commodity]:
            return []

        return list(self.registry[commodity][model_type].keys())

    def _save_registry(self) -> None:
        """Save the registry to disk."""
        registry_file = os.path.join(self.registry_dir, 'registry.json')
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.registry, f, indent=4)
            logger.info(f"Saved model registry to {registry_file}")
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")

class EvaluationMetrics:
    """
    Evaluation metrics for model performance.
    """

    def __init__(self):
        """Initialize the evaluation metrics."""
        pass

    def evaluate(
        self,
        model: Any,
        data: pd.DataFrame,
        target_column: str = 'price',
        features: Optional[List[str]] = None,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Parameters
        ----------
        model : Any
            Model object
        data : pd.DataFrame
            Data for evaluation
        target_column : str, optional
            Name of the target column, by default 'price'
        features : List[str], optional
            List of feature columns, by default None (will use all columns except target)
        test_size : float, optional
            Fraction of data to use for testing, by default 0.2

        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        try:
            # Check if model has predict method
            if not hasattr(model, 'predict'):
                logger.error("Model does not have predict method")
                return {}

            # Split data into train and test
            split_idx = int(len(data) * (1 - test_size))
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]

            # Get features
            if features is None:
                features = [col for col in data.columns if col != target_column]

            # Get target
            y_test = test_data[target_column]

            # Make predictions
            try:
                # Try different prediction methods
                if hasattr(model, 'predict_in_sample'):
                    # For models that need the full dataset
                    predictions = model.predict_in_sample(data)
                    predictions = predictions.iloc[split_idx:]
                else:
                    # For models that can predict on new data
                    X_test = test_data[features]
                    predictions = model.predict(X_test)
            except Exception as e:
                logger.error(f"Error making predictions: {e}")
                return {}

            # Calculate metrics
            metrics = {}

            # Mean Absolute Error
            metrics['mae'] = mean_absolute_error(y_test, predictions)

            # Mean Squared Error
            metrics['mse'] = mean_squared_error(y_test, predictions)

            # Root Mean Squared Error
            metrics['rmse'] = np.sqrt(metrics['mse'])

            # R-squared
            metrics['r2'] = r2_score(y_test, predictions)

            # Mean Absolute Percentage Error
            metrics['mape'] = np.mean(np.abs((y_test - predictions) / y_test)) * 100

            # Direction Accuracy
            if len(y_test) > 1:
                actual_direction = np.sign(y_test.diff().dropna())
                pred_direction = np.sign(pd.Series(predictions).diff().dropna())

                # Align the arrays
                actual_direction = actual_direction.iloc[1:]
                pred_direction = pred_direction.iloc[:len(actual_direction)]

                metrics['direction_accuracy'] = np.mean(actual_direction == pred_direction) * 100

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}

    def compare_models(
        self,
        model1: Any,
        model2: Any,
        data: pd.DataFrame,
        target_column: str = 'price',
        features: Optional[List[str]] = None,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Compare two models.

        Parameters
        ----------
        model1 : Any
            First model object
        model2 : Any
            Second model object
        data : pd.DataFrame
            Data for evaluation
        target_column : str, optional
            Name of the target column, by default 'price'
        features : List[str], optional
            List of feature columns, by default None (will use all columns except target)
        test_size : float, optional
            Fraction of data to use for testing, by default 0.2

        Returns
        -------
        Dict[str, Any]
            Comparison results
        """
        # Evaluate both models
        metrics1 = self.evaluate(model1, data, target_column, features, test_size)
        metrics2 = self.evaluate(model2, data, target_column, features, test_size)

        if not metrics1 or not metrics2:
            return {}

        # Calculate improvement
        improvement = {}
        for metric in metrics1:
            if metric in metrics2:
                # For metrics where lower is better (MAE, MSE, RMSE, MAPE)
                if metric in ['mae', 'mse', 'rmse', 'mape']:
                    improvement[metric] = (metrics1[metric] - metrics2[metric]) / metrics1[metric] * 100
                # For metrics where higher is better (R2, Direction Accuracy)
                else:
                    improvement[metric] = (metrics2[metric] - metrics1[metric]) / abs(metrics1[metric]) * 100 if metrics1[metric] != 0 else float('inf')

        # Determine if model2 is better overall
        better_count = sum(1 for metric, value in improvement.items() if
                          (metric in ['mae', 'mse', 'rmse', 'mape'] and value > 0) or
                          (metric not in ['mae', 'mse', 'rmse', 'mape'] and value > 0))

        is_better = better_count > len(improvement) / 2

        return {
            'model1_metrics': metrics1,
            'model2_metrics': metrics2,
            'improvement': improvement,
            'is_better': is_better
        }

class DriftDetector:
    """
    Drift detector for detecting data and model drift.
    """

    def __init__(self):
        """Initialize the drift detector."""
        self.detector = AdvancedDriftDetector()

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        threshold: float = 0.05
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect drift between reference and current data.

        Parameters
        ----------
        reference_data : pd.DataFrame
            Reference data
        current_data : pd.DataFrame
            Current data
        feature_names : List[str], optional
            List of feature names to check for drift, by default None (all columns)
        categorical_features : List[str], optional
            List of categorical features, by default None
        threshold : float, optional
            Threshold for drift detection, by default 0.05

        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            A tuple containing:
            - Boolean indicating if significant drift was detected
            - Dictionary with detailed drift results
        """
        # Set reference data
        self.detector.reference_data = reference_data

        # Detect drift
        return self.detector.detect_drift(
            current_data,
            feature_names=feature_names,
            categorical_features=categorical_features
        )

class OnlineLearningManager:
    """
    Online Learning Manager for continuous model improvement.
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        evaluation_metrics: Optional[EvaluationMetrics] = None,
        drift_detector: Optional[DriftDetector] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the online learning manager.

        Parameters
        ----------
        model_registry : ModelRegistry, optional
            Model registry, by default None (will create a new one)
        evaluation_metrics : EvaluationMetrics, optional
            Evaluation metrics, by default None (will create a new one)
        drift_detector : DriftDetector, optional
            Drift detector, by default None (will create a new one)
        config_path : str, optional
            Path to configuration file, by default None
        """
        self.model_registry = model_registry or ModelRegistry()
        self.evaluation_metrics = evaluation_metrics or EvaluationMetrics()
        self.drift_detector = drift_detector or DriftDetector()

        # Default configuration
        self.config = {
            'update_frequency': 'daily',  # 'hourly', 'daily', 'weekly'
            'drift_threshold': 0.05,
            'improvement_threshold': 5.0,  # Percentage improvement required to update model
            'auto_update': True,
            'commodities': ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel'],
            'model_types': ['arima', 'xgboost', 'lstm', 'price_drivers'],
            'categorical_features': []
        }

        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.config.update(config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")

        # Create directories
        os.makedirs('logs', exist_ok=True)

        # Last update timestamp
        self.last_update = None

    def update_models(
        self,
        commodities: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        force_update: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Update models with new data if significant drift is detected.

        Parameters
        ----------
        commodities : List[str], optional
            List of commodities to update, by default None (will use all from config)
        model_types : List[str], optional
            List of model types to update, by default None (will use all from config)
        force_update : bool, optional
            Whether to force update regardless of drift, by default False

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with update results for each commodity and model type
        """
        # Check if it's time to update based on frequency
        if not force_update and self.last_update is not None:
            frequency = self.config.get('update_frequency', 'daily')

            if frequency == 'hourly':
                if datetime.now() - self.last_update < timedelta(hours=1):
                    logger.info("Skipping update (less than 1 hour since last update)")
                    return {"status": "skipped", "reason": "Too soon since last update"}

            elif frequency == 'daily':
                if datetime.now() - self.last_update < timedelta(days=1):
                    logger.info("Skipping update (less than 1 day since last update)")
                    return {"status": "skipped", "reason": "Too soon since last update"}

            elif frequency == 'weekly':
                if datetime.now() - self.last_update < timedelta(weeks=1):
                    logger.info("Skipping update (less than 1 week since last update)")
                    return {"status": "skipped", "reason": "Too soon since last update"}

        # Use commodities from config if not specified
        if commodities is None:
            commodities = self.config.get('commodities', [])

        # Use model types from config if not specified
        if model_types is None:
            model_types = self.config.get('model_types', [])

        # Update results
        results = {}

        for commodity in commodities:
            results[commodity] = {}

            # Load new data
            new_data = load_processed_data(commodity)

            if new_data.empty:
                logger.warning(f"No data found for commodity {commodity}")
                results[commodity]['status'] = 'error'
                results[commodity]['reason'] = 'No data found'
                continue

            # Load features data
            features_data = load_features_data(commodity)

            for model_type in model_types:
                logger.info(f"Checking {model_type} model for {commodity}")

                # Get current model
                current_model, model_metadata = self.model_registry.get_active_model(commodity, model_type)

                if current_model is None:
                    logger.warning(f"No active {model_type} model found for {commodity}")
                    results[commodity][model_type] = {
                        'status': 'error',
                        'reason': 'No active model found'
                    }
                    continue

                # Check for drift if not forcing update
                if not force_update:
                    # Get training data distribution from metadata
                    training_data = None
                    if 'training_data_path' in model_metadata:
                        training_data_path = model_metadata['training_data_path']
                        if os.path.exists(training_data_path):
                            try:
                                training_data = pd.read_parquet(training_data_path)
                            except Exception as e:
                                logger.error(f"Error loading training data: {e}")

                    # If training data not available, skip drift detection
                    if training_data is None:
                        logger.warning(f"No training data found for {commodity} {model_type}, skipping drift detection")
                    else:
                        # Detect drift
                        drift_detected, drift_results = self.drift_detector.detect_drift(
                            training_data,
                            new_data,
                            categorical_features=self.config.get('categorical_features', []),
                            threshold=self.config.get('drift_threshold', 0.05)
                        )

                        if not drift_detected:
                            logger.info(f"No significant drift detected for {commodity} {model_type}")
                            results[commodity][model_type] = {
                                'status': 'skipped',
                                'reason': 'No significant drift detected',
                                'drift_results': drift_results
                            }
                            continue

                        logger.info(f"Drift detected for {commodity} {model_type}, updating model")

                # Update model
                try:
                    # Get model update function
                    update_func = self._get_update_function(model_type)

                    if update_func is None:
                        logger.error(f"No update function found for model type {model_type}")
                        results[commodity][model_type] = {
                            'status': 'error',
                            'reason': 'No update function found'
                        }
                        continue

                    # Update model
                    updated_model = update_func(current_model, new_data, features_data)

                    if updated_model is None:
                        logger.error(f"Failed to update {model_type} model for {commodity}")
                        results[commodity][model_type] = {
                            'status': 'error',
                            'reason': 'Failed to update model'
                        }
                        continue

                    # Evaluate updated model
                    comparison = self.evaluation_metrics.compare_models(
                        current_model,
                        updated_model,
                        new_data
                    )

                    if not comparison:
                        logger.error(f"Failed to evaluate models for {commodity} {model_type}")
                        results[commodity][model_type] = {
                            'status': 'error',
                            'reason': 'Failed to evaluate models'
                        }
                        continue

                    # Check if updated model is better
                    is_better = comparison.get('is_better', False)
                    improvement = comparison.get('improvement', {})

                    # Calculate average improvement
                    avg_improvement = 0
                    if improvement:
                        avg_improvement = sum(improvement.values()) / len(improvement)

                    # Register updated model if it's better
                    if is_better and avg_improvement > self.config.get('improvement_threshold', 5.0):
                        # Save training data
                        training_data_dir = os.path.join(self.model_registry.registry_dir, 'training_data')
                        os.makedirs(training_data_dir, exist_ok=True)

                        training_data_path = os.path.join(
                            training_data_dir,
                            f"{commodity}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                        )

                        new_data.to_parquet(training_data_path)

                        # Register updated model
                        version = self.model_registry.register_model(
                            commodity=commodity,
                            model_type=model_type,
                            model=updated_model,
                            metrics=comparison['model2_metrics'],
                            metadata={
                                'training_data_path': training_data_path,
                                'improvement': improvement,
                                'avg_improvement': avg_improvement,
                                'updated_at': datetime.now().isoformat(),
                                'updated_from': model_metadata.get('version', 'unknown')
                            }
                        )

                        if version:
                            # Set as active if auto-update is enabled
                            if self.config.get('auto_update', True):
                                self.model_registry.set_active_model(commodity, model_type, version)

                            logger.info(f"Updated {model_type} model for {commodity} (version: {version})")
                            results[commodity][model_type] = {
                                'status': 'updated',
                                'version': version,
                                'improvement': improvement,
                                'avg_improvement': avg_improvement
                            }
                        else:
                            logger.error(f"Failed to register updated model for {commodity} {model_type}")
                            results[commodity][model_type] = {
                                'status': 'error',
                                'reason': 'Failed to register updated model'
                            }
                    else:
                        logger.info(f"Updated model for {commodity} {model_type} is not significantly better")
                        results[commodity][model_type] = {
                            'status': 'skipped',
                            'reason': 'Updated model is not significantly better',
                            'improvement': improvement,
                            'avg_improvement': avg_improvement
                        }

                except Exception as e:
                    logger.error(f"Error updating {model_type} model for {commodity}: {e}")
                    results[commodity][model_type] = {
                        'status': 'error',
                        'reason': str(e)
                    }

        # Update last update timestamp
        self.last_update = datetime.now()

        return results

    def _get_update_function(self, model_type: str):
        """
        Get the update function for a model type.

        Parameters
        ----------
        model_type : str
            Type of the model

        Returns
        -------
        function
            Update function for the model type
        """
        if model_type == 'arima':
            return self._update_arima_model
        elif model_type == 'xgboost':
            return self._update_xgboost_model
        elif model_type == 'lstm':
            return self._update_lstm_model
        elif model_type == 'price_drivers':
            return self._update_price_drivers_model
        else:
            return None

    def _update_arima_model(self, model, new_data, features_data):
        """Update ARIMA model with new data."""
        try:
            # Check if model has update method
            if hasattr(model, 'update'):
                return model.update(new_data)

            # Otherwise, retrain model
            from src.models.forecasting.arima_forecaster import ARIMAForecaster

            if isinstance(model, ARIMAForecaster):
                # Create a new model with the same parameters
                new_model = ARIMAForecaster(
                    order=model.order,
                    seasonal_order=model.seasonal_order,
                    trend=model.trend
                )

                # Fit the new model with new data
                new_model.fit(new_data)

                return new_model

            return None

        except Exception as e:
            logger.error(f"Error updating ARIMA model: {e}")
            return None

    def _update_xgboost_model(self, model, new_data, features_data):
        """Update XGBoost model with new data."""
        try:
            # Check if model has update method
            if hasattr(model, 'update'):
                return model.update(new_data, features_data)

            # Otherwise, retrain model
            from src.models.forecasting.xgboost_forecaster import XGBoostForecaster

            if isinstance(model, XGBoostForecaster):
                # Create a new model with the same parameters
                new_model = XGBoostForecaster(
                    params=model.params,
                    target_column=model.target_column,
                    feature_columns=model.feature_columns
                )

                # Fit the new model with new data
                new_model.fit(new_data, features_data)

                return new_model

            return None

        except Exception as e:
            logger.error(f"Error updating XGBoost model: {e}")
            return None

    def _update_lstm_model(self, model, new_data, features_data):
        """Update LSTM model with new data."""
        try:
            # Check if model has update method
            if hasattr(model, 'update'):
                return model.update(new_data)

            # Otherwise, retrain model
            from src.models.forecasting.lstm_forecaster import LSTMForecaster

            if isinstance(model, LSTMForecaster):
                # Create a new model with the same parameters
                new_model = LSTMForecaster(
                    sequence_length=model.sequence_length,
                    n_features=model.n_features,
                    n_units=model.n_units,
                    dropout=model.dropout
                )

                # Fit the new model with new data
                new_model.fit(new_data)

                return new_model

            return None

        except Exception as e:
            logger.error(f"Error updating LSTM model: {e}")
            return None

    def _update_price_drivers_model(self, model, new_data, features_data):
        """Update Price Drivers model with new data."""
        try:
            # Check if model has update method
            if hasattr(model, 'update'):
                return model.update(new_data, features_data)

            # Otherwise, retrain model
            from src.models.forecasting.price_drivers_forecaster import PriceDriversForecaster

            if isinstance(model, PriceDriversForecaster):
                # Create a new model with the same parameters
                new_model = PriceDriversForecaster(
                    params=model.params,
                    target_column=model.target_column,
                    feature_columns=model.feature_columns
                )

                # Fit the new model with new data
                new_model.fit(new_data, features_data)

                return new_model

            return None

        except Exception as e:
            logger.error(f"Error updating Price Drivers model: {e}")
            return None
