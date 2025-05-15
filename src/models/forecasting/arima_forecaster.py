"""
ARIMA forecasting model for oil and gas commodities.
This module implements ARIMA/SARIMA models for time series forecasting.
"""

import os
import logging
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/arima_forecaster.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
FEATURES_DATA_DIR = 'data/features'
MODELS_DIR = 'models/forecasting'

class ARIMAForecaster:
    """
    ARIMA forecasting model for time series data.
    """
    
    def __init__(
        self, 
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        exog_columns: Optional[List[str]] = None
    ):
        """
        Initialize the ARIMA forecaster.
        
        Parameters
        ----------
        order : Tuple[int, int, int], optional
            ARIMA order (p, d, q), by default (1, 1, 1)
        seasonal_order : Tuple[int, int, int, int], optional
            Seasonal order (P, D, Q, s), by default None (no seasonal component)
        exog_columns : List[str], optional
            List of exogenous variables to include, by default None
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_columns = exog_columns
        self.model = None
        self.model_fit = None
        self.target_column = None
        self.scaler = None
    
    def fit(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        exog_columns: Optional[List[str]] = None
    ) -> 'ARIMAForecaster':
        """
        Fit the ARIMA model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        target_column : str
            Name of the target column
        exog_columns : List[str], optional
            List of exogenous variables to include, by default None
        
        Returns
        -------
        ARIMAForecaster
            Fitted model
        """
        self.target_column = target_column
        
        # Update exog_columns if provided
        if exog_columns is not None:
            self.exog_columns = exog_columns
        
        # Extract target and exogenous variables
        y = df[target_column]
        X = None
        if self.exog_columns:
            # Filter to only include columns that exist in the DataFrame
            valid_exog = [col for col in self.exog_columns if col in df.columns]
            if valid_exog:
                X = df[valid_exog]
            else:
                logger.warning("None of the specified exogenous variables exist in the DataFrame")
        
        try:
            # Create and fit the model
            if self.seasonal_order:
                logger.info(f"Fitting SARIMA model with order={self.order}, seasonal_order={self.seasonal_order}")
                self.model = SARIMAX(
                    y, 
                    exog=X,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                logger.info(f"Fitting ARIMA model with order={self.order}")
                self.model = ARIMA(
                    y,
                    exog=X,
                    order=self.order
                )
            
            self.model_fit = self.model.fit()
            logger.info("Model fitting completed")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
    
    def predict(
        self, 
        df: pd.DataFrame, 
        steps: int = 1,
        dynamic: bool = False
    ) -> pd.Series:
        """
        Generate predictions from the fitted model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to predict on
        steps : int, optional
            Number of steps to forecast, by default 1
        dynamic : bool, optional
            Whether to use dynamic forecasting, by default False
        
        Returns
        -------
        pd.Series
            Predicted values
        """
        if self.model_fit is None:
            raise ValueError("Model has not been fit yet")
        
        try:
            # Extract exogenous variables if needed
            X = None
            if self.exog_columns:
                # Filter to only include columns that exist in the DataFrame
                valid_exog = [col for col in self.exog_columns if col in df.columns]
                if valid_exog:
                    X = df[valid_exog]
                else:
                    logger.warning("None of the specified exogenous variables exist in the DataFrame")
            
            # Generate predictions
            if steps == 1:
                # In-sample prediction
                predictions = self.model_fit.predict(exog=X, dynamic=dynamic)
            else:
                # Out-of-sample forecast
                predictions = self.model_fit.forecast(steps=steps, exog=X)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise
    
    def evaluate(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Test data
        target_column : str, optional
            Name of the target column, by default None (use the one from fitting)
        
        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        if self.model_fit is None:
            raise ValueError("Model has not been fit yet")
        
        if target_column is None:
            target_column = self.target_column
        
        try:
            # Generate predictions
            y_true = df[target_column]
            y_pred = self.predict(df)
            
            # Align predictions with actual values
            y_pred = y_pred.reindex(y_true.index)
            
            # Calculate metrics
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            
            # Calculate MAPE if no zeros in y_true
            if not np.any(y_true == 0):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                metrics['mape'] = mape
            
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def plot_forecast(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None,
        steps: int = 30,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot the forecast against actual values.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to predict on
        target_column : str, optional
            Name of the target column, by default None (use the one from fitting)
        steps : int, optional
            Number of steps to forecast, by default 30
        figsize : Tuple[int, int], optional
            Figure size, by default (12, 6)
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if self.model_fit is None:
            raise ValueError("Model has not been fit yet")
        
        if target_column is None:
            target_column = self.target_column
        
        try:
            # Generate predictions
            y_true = df[target_column]
            y_pred = self.predict(df)
            
            # Create forecast for future steps
            forecast = self.model_fit.forecast(steps=steps)
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot actual values
            ax.plot(y_true.index, y_true, label='Actual')
            
            # Plot fitted values
            ax.plot(y_pred.index, y_pred, label='Fitted', alpha=0.7)
            
            # Plot forecast
            forecast_index = pd.date_range(
                start=y_true.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
            ax.plot(forecast_index, forecast, label='Forecast', color='red')
            
            # Add confidence intervals for forecast
            if hasattr(self.model_fit, 'get_forecast'):
                forecast_obj = self.model_fit.get_forecast(steps=steps)
                conf_int = forecast_obj.conf_int()
                ax.fill_between(
                    forecast_index,
                    conf_int.iloc[:, 0],
                    conf_int.iloc[:, 1],
                    color='red',
                    alpha=0.2
                )
            
            # Add labels and legend
            ax.set_title(f'ARIMA Forecast for {target_column}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            
            # Format x-axis
            fig.autofmt_xdate()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
            raise
    
    def save(self, file_path: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        file_path : str
            Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save model
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'order': self.order,
                    'seasonal_order': self.seasonal_order,
                    'exog_columns': self.exog_columns,
                    'target_column': self.target_column,
                    'model_fit': self.model_fit
                }, f)
            
            logger.info(f"Model saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load(cls, file_path: str) -> 'ARIMAForecaster':
        """
        Load a model from a file.
        
        Parameters
        ----------
        file_path : str
            Path to load the model from
        
        Returns
        -------
        ARIMAForecaster
            Loaded model
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create a new instance
            forecaster = cls(
                order=model_data['order'],
                seasonal_order=model_data['seasonal_order'],
                exog_columns=model_data['exog_columns']
            )
            
            # Set attributes
            forecaster.target_column = model_data['target_column']
            forecaster.model_fit = model_data['model_fit']
            
            logger.info(f"Model loaded from {file_path}")
            return forecaster
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def find_optimal_arima_order(
    df: pd.DataFrame,
    target_column: str,
    p_values: List[int] = [0, 1, 2],
    d_values: List[int] = [0, 1],
    q_values: List[int] = [0, 1, 2],
    exog_columns: Optional[List[str]] = None
) -> Tuple[Tuple[int, int, int], Dict[str, float]]:
    """
    Find the optimal ARIMA order using grid search.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data
    target_column : str
        Name of the target column
    p_values : List[int], optional
        List of p values to try, by default [0, 1, 2]
    d_values : List[int], optional
        List of d values to try, by default [0, 1]
    q_values : List[int], optional
        List of q values to try, by default [0, 1, 2]
    exog_columns : List[str], optional
        List of exogenous variables to include, by default None
    
    Returns
    -------
    Tuple[Tuple[int, int, int], Dict[str, float]]
        Optimal order and corresponding metrics
    """
    best_order = None
    best_aic = float('inf')
    best_metrics = None
    
    # Split data into train and validation sets
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    logger.info(f"Grid search for optimal ARIMA order: p={p_values}, d={d_values}, q={q_values}")
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    # Create and fit model
                    model = ARIMAForecaster(order=order, exog_columns=exog_columns)
                    model.fit(train_df, target_column, exog_columns)
                    
                    # Get AIC
                    aic = model.model_fit.aic
                    
                    # Evaluate on validation set
                    metrics = model.evaluate(val_df, target_column)
                    
                    logger.info(f"Order {order}: AIC={aic}, RMSE={metrics['rmse']}")
                    
                    # Update best model if this one is better
                    if aic < best_aic:
                        best_order = order
                        best_aic = aic
                        best_metrics = metrics
                        
                except Exception as e:
                    logger.warning(f"Error fitting ARIMA with order {order}: {e}")
                    continue
    
    logger.info(f"Best ARIMA order: {best_order} with AIC={best_aic}")
    return best_order, best_metrics

def train_arima_model(
    commodity: str,
    target_column: Optional[str] = None,
    exog_columns: Optional[List[str]] = None,
    optimize_order: bool = True,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
) -> ARIMAForecaster:
    """
    Train an ARIMA model for a specific commodity.
    
    Parameters
    ----------
    commodity : str
        Name of the commodity (e.g., 'crude_oil')
    target_column : str, optional
        Name of the target column, by default None (will read from target.txt)
    exog_columns : List[str], optional
        List of exogenous variables to include, by default None
    optimize_order : bool, optional
        Whether to optimize the ARIMA order, by default True
    order : Tuple[int, int, int], optional
        ARIMA order if not optimizing, by default (1, 1, 1)
    seasonal_order : Tuple[int, int, int, int], optional
        Seasonal order if using SARIMA, by default None
    
    Returns
    -------
    ARIMAForecaster
        Trained model
    """
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load features data
    features_file_path = os.path.join(FEATURES_DATA_DIR, f"{commodity}_features.parquet")
    
    try:
        if os.path.exists(features_file_path):
            df = pd.read_parquet(features_file_path)
            logger.info(f"Loaded {len(df)} rows for {commodity} from {features_file_path}")
        else:
            logger.error(f"Features file not found: {features_file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return None
    
    # Get target column if not provided
    if target_column is None:
        target_file_path = os.path.join(FEATURES_DATA_DIR, f"{commodity}_target.txt")
        try:
            with open(target_file_path, 'r') as f:
                target_column = f.read().strip()
            logger.info(f"Using target column: {target_column}")
        except Exception as e:
            logger.error(f"Error reading target column: {e}")
            return None
    
    # Split data into train and test sets
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    logger.info(f"Split data into {len(train_df)} training samples and {len(test_df)} test samples")
    
    # Optimize ARIMA order if requested
    if optimize_order:
        logger.info("Optimizing ARIMA order")
        order, _ = find_optimal_arima_order(
            train_df,
            target_column,
            exog_columns=exog_columns
        )
    
    # Create and train model
    model = ARIMAForecaster(order=order, seasonal_order=seasonal_order, exog_columns=exog_columns)
    model.fit(train_df, target_column, exog_columns)
    
    # Evaluate on test set
    metrics = model.evaluate(test_df, target_column)
    logger.info(f"Test set evaluation: {metrics}")
    
    # Save model
    model_file_path = os.path.join(MODELS_DIR, f"{commodity}_arima.pkl")
    model.save(model_file_path)
    
    return model

if __name__ == "__main__":
    # Example usage
    train_arima_model('crude_oil', optimize_order=True)
