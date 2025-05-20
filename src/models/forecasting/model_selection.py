"""
Model selection module for forecasting models.
This module compares different forecasting models and selects the best one.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.models.forecasting.arima_forecaster import ARIMAForecaster, train_arima_model
from src.models.forecasting.xgboost_forecaster import XGBoostForecaster, train_xgboost_model
from src.models.forecasting.lstm_forecaster import LSTMForecaster, train_lstm_model
from src.models.forecasting.price_drivers_forecaster import PriceDriversForecaster, train_price_drivers_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_selection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
FEATURES_DATA_DIR = 'data/features'
MODELS_DIR = 'models/forecasting'
RESULTS_DIR = 'results/model_selection'

class ModelEvaluator:
    """
    Class for evaluating and comparing forecasting models.
    """

    def __init__(self, commodity: str, target_column: Optional[str] = None):
        """
        Initialize the model evaluator.

        Parameters
        ----------
        commodity : str
            Name of the commodity (e.g., 'crude_oil')
        target_column : str, optional
            Name of the target column, by default None (will read from target.txt)
        """
        self.commodity = commodity
        self.target_column = target_column
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

        # Get target column if not provided
        if self.target_column is None:
            target_file_path = os.path.join(FEATURES_DATA_DIR, f"{commodity}_target.txt")
            try:
                with open(target_file_path, 'r') as f:
                    self.target_column = f.read().strip()
                logger.info(f"Using target column: {self.target_column}")
            except Exception as e:
                logger.error(f"Error reading target column: {e}")
                raise

        # Load data
        self.data = self._load_data()

        # Split data
        self.train_data, self.test_data = self._split_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Load features data for the commodity.

        Returns
        -------
        pd.DataFrame
            Features data
        """
        features_file_path = os.path.join(FEATURES_DATA_DIR, f"{self.commodity}_features.parquet")

        try:
            if os.path.exists(features_file_path):
                df = pd.read_parquet(features_file_path)
                logger.info(f"Loaded {len(df)} rows for {self.commodity} from {features_file_path}")
                return df
            else:
                logger.error(f"Features file not found: {features_file_path}")
                raise FileNotFoundError(f"Features file not found: {features_file_path}")
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            raise

    def _split_data(self, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.

        Parameters
        ----------
        train_size : float, optional
            Fraction of data to use for training, by default 0.8

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Train and test DataFrames
        """
        train_idx = int(len(self.data) * train_size)
        train_df = self.data.iloc[:train_idx]
        test_df = self.data.iloc[train_idx:]

        logger.info(f"Split data into {len(train_df)} training samples and {len(test_df)} test samples")

        return train_df, test_df

    def train_models(
        self,
        models_to_train: List[str] = ['arima', 'xgboost', 'lstm', 'price_drivers'],
        optimize_params: bool = True
    ) -> Dict[str, Any]:
        """
        Train multiple forecasting models.

        Parameters
        ----------
        models_to_train : List[str], optional
            List of models to train, by default ['arima', 'xgboost', 'lstm', 'price_drivers']
        optimize_params : bool, optional
            Whether to optimize model parameters, by default True

        Returns
        -------
        Dict[str, Any]
            Dictionary of trained models
        """
        # Create directories if they don't exist
        os.makedirs(MODELS_DIR, exist_ok=True)

        for model_name in models_to_train:
            logger.info(f"Training {model_name.upper()} model for {self.commodity}")

            try:
                if model_name.lower() == 'arima':
                    model = train_arima_model(
                        self.commodity,
                        target_column=self.target_column,
                        optimize_order=optimize_params
                    )
                    self.models['arima'] = model

                elif model_name.lower() == 'xgboost':
                    model = train_xgboost_model(
                        self.commodity,
                        target_column=self.target_column,
                        optimize_params=optimize_params
                    )
                    self.models['xgboost'] = model

                elif model_name.lower() == 'lstm':
                    model = train_lstm_model(
                        self.commodity,
                        target_column=self.target_column
                    )
                    self.models['lstm'] = model

                elif model_name.lower() == 'price_drivers':
                    model = train_price_drivers_model(
                        self.commodity,
                        target_column=self.target_column,
                        optimize_params=optimize_params
                    )
                    self.models['price_drivers'] = model

                else:
                    logger.warning(f"Unknown model type: {model_name}")

            except Exception as e:
                logger.error(f"Error training {model_name} model: {e}")

        logger.info(f"Trained {len(self.models)} models")
        return self.models

    def load_models(
        self,
        models_to_load: List[str] = ['arima', 'xgboost', 'lstm', 'price_drivers']
    ) -> Dict[str, Any]:
        """
        Load trained models from files.

        Parameters
        ----------
        models_to_load : List[str], optional
            List of models to load, by default ['arima', 'xgboost', 'lstm', 'price_drivers']

        Returns
        -------
        Dict[str, Any]
            Dictionary of loaded models
        """
        for model_name in models_to_load:
            model_file_path = os.path.join(MODELS_DIR, f"{self.commodity}_{model_name}.pkl")

            try:
                if os.path.exists(model_file_path):
                    if model_name.lower() == 'arima':
                        model = ARIMAForecaster.load(model_file_path)
                        self.models['arima'] = model

                    elif model_name.lower() == 'xgboost':
                        model = XGBoostForecaster.load(model_file_path)
                        self.models['xgboost'] = model

                    elif model_name.lower() == 'lstm':
                        model = LSTMForecaster.load(model_file_path)
                        self.models['lstm'] = model

                    elif model_name.lower() == 'price_drivers':
                        model = PriceDriversForecaster.load(model_file_path)
                        self.models['price_drivers'] = model

                    else:
                        logger.warning(f"Unknown model type: {model_name}")

                    logger.info(f"Loaded {model_name} model from {model_file_path}")
                else:
                    logger.warning(f"Model file not found: {model_file_path}")

            except Exception as e:
                logger.error(f"Error loading {model_name} model: {e}")

        logger.info(f"Loaded {len(self.models)} models")
        return self.models

    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all loaded models on the test data.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of evaluation metrics for each model
        """
        if not self.models:
            logger.warning("No models to evaluate")
            return {}

        for model_name, model in self.models.items():
            try:
                metrics = model.evaluate(self.test_data, self.target_column)
                self.results[model_name] = metrics
                logger.info(f"{model_name.upper()} model evaluation: {metrics}")

            except Exception as e:
                logger.error(f"Error evaluating {model_name} model: {e}")

        # Determine best model based on RMSE
        if self.results:
            self.best_model_name = min(
                self.results.keys(),
                key=lambda k: self.results[k].get('rmse', float('inf'))
            )
            self.best_model = self.models[self.best_model_name]

            logger.info(f"Best model: {self.best_model_name.upper()} with RMSE: {self.results[self.best_model_name]['rmse']}")

        return self.results

    def compare_forecasts(
        self,
        steps: int = 30,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Compare forecasts from different models.

        Parameters
        ----------
        steps : int, optional
            Number of steps to forecast, by default 30
        figsize : Tuple[int, int], optional
            Figure size, by default (12, 8)

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if not self.models:
            logger.warning("No models to compare")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot actual values
        actual = self.test_data[self.target_column]
        ax.plot(actual.index, actual, label='Actual', color='black')

        # Plot forecasts for each model
        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                # Generate in-sample predictions
                predictions = model.predict(self.test_data)

                # Generate forecast
                last_date = self.test_data.index[-1]
                forecast_index = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=steps,
                    freq='D'
                )

                if hasattr(model, 'forecast'):
                    forecast = model.forecast(self.test_data, steps=steps)
                else:
                    # For models without a forecast method, use model_fit.forecast
                    forecast = model.model_fit.forecast(steps=steps)
                    forecast = pd.Series(forecast, index=forecast_index)

                # Plot predictions and forecast
                color = colors[i % len(colors)]
                ax.plot(predictions.index, predictions, label=f'{model_name.upper()} Fitted', color=color, alpha=0.7)
                ax.plot(forecast.index, forecast, label=f'{model_name.upper()} Forecast', color=color, linestyle='--')

                # Highlight best model
                if model_name == self.best_model_name:
                    ax.plot(predictions.index, predictions, color=color, linewidth=3, alpha=0.5)
                    ax.plot(forecast.index, forecast, color=color, linewidth=3, linestyle='--', alpha=0.5)

            except Exception as e:
                logger.error(f"Error generating forecast for {model_name} model: {e}")

        # Add labels and legend
        ax.set_title(f'Model Comparison for {self.commodity.replace("_", " ").title()}')
        ax.set_xlabel('Date')
        ax.set_ylabel(self.target_column)
        ax.legend()

        # Format x-axis
        fig.autofmt_xdate()

        return fig

    def save_results(self) -> None:
        """
        Save evaluation results and comparison plot.
        """
        # Create directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Save evaluation results
        results_file_path = os.path.join(RESULTS_DIR, f"{self.commodity}_model_comparison.json")

        try:
            # Convert results to JSON-serializable format
            json_results = {}
            for model_name, metrics in self.results.items():
                json_results[model_name] = {k: float(v) for k, v in metrics.items()}

            with open(results_file_path, 'w') as f:
                json.dump(json_results, f, indent=4)

            logger.info(f"Saved evaluation results to {results_file_path}")

        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")

        # Save comparison plot
        plot_file_path = os.path.join(RESULTS_DIR, f"{self.commodity}_model_comparison.png")

        try:
            fig = self.compare_forecasts()
            if fig:
                fig.savefig(plot_file_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved comparison plot to {plot_file_path}")

        except Exception as e:
            logger.error(f"Error saving comparison plot: {e}")

        # Save best model info
        if self.best_model_name:
            best_model_file_path = os.path.join(RESULTS_DIR, f"{self.commodity}_best_model.json")

            try:
                with open(best_model_file_path, 'w') as f:
                    json.dump({
                        'commodity': self.commodity,
                        'best_model': self.best_model_name,
                        'metrics': {k: float(v) for k, v in self.results[self.best_model_name].items()}
                    }, f, indent=4)

                logger.info(f"Saved best model info to {best_model_file_path}")

            except Exception as e:
                logger.error(f"Error saving best model info: {e}")

def select_best_model(
    commodity: str,
    target_column: Optional[str] = None,
    models_to_train: List[str] = ['arima', 'xgboost', 'lstm', 'price_drivers'],
    optimize_params: bool = True,
    train_new_models: bool = True
) -> Tuple[str, Any]:
    """
    Train and evaluate multiple models, then select the best one.

    Parameters
    ----------
    commodity : str
        Name of the commodity (e.g., 'crude_oil')
    target_column : str, optional
        Name of the target column, by default None (will read from target.txt)
    models_to_train : List[str], optional
        List of models to train, by default ['arima', 'xgboost', 'lstm', 'price_drivers']
    optimize_params : bool, optional
        Whether to optimize model parameters, by default True
    train_new_models : bool, optional
        Whether to train new models or load existing ones, by default True

    Returns
    -------
    Tuple[str, Any]
        Name of the best model and the model object
    """
    # Create evaluator
    evaluator = ModelEvaluator(commodity, target_column)

    # Train or load models
    if train_new_models:
        evaluator.train_models(models_to_train, optimize_params)
    else:
        evaluator.load_models(models_to_train)

    # Evaluate models
    evaluator.evaluate_models()

    # Save results
    evaluator.save_results()

    return evaluator.best_model_name, evaluator.best_model

if __name__ == "__main__":
    # Example usage
    best_model_name, best_model = select_best_model('crude_oil')
