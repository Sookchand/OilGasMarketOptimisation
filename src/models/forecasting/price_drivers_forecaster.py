"""
Price Drivers Forecaster model.
This module implements a forecasting model that leverages EIA price drivers data.
"""

import os
import pickle
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from src.utils.data_utils import load_features_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/price_drivers_forecaster.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
MODELS_DIR = 'models'

class PriceDriversForecaster:
    """
    Forecasting model that leverages EIA price drivers data.
    """
    
    def __init__(
        self, 
        params: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the Price Drivers Forecaster.
        
        Parameters
        ----------
        params : Dict[str, Any], optional
            Model parameters, by default None (use default parameters)
        feature_columns : List[str], optional
            List of feature columns to use, by default None (use all except target)
        """
        self.params = params or {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'loss': 'squared_error',
            'random_state': 42
        }
        self.feature_columns = feature_columns
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.price_driver_columns = []
    
    def fit(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> 'PriceDriversForecaster':
        """
        Fit the model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        target_column : str
            Name of the target column
        feature_columns : List[str], optional
            List of feature columns to use, by default None (use all except target)
        
        Returns
        -------
        PriceDriversForecaster
            Fitted model
        """
        self.target_column = target_column
        
        # Update feature_columns if provided
        if feature_columns is not None:
            self.feature_columns = feature_columns
        
        # If feature_columns is still None, use all columns except target
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != target_column]
        
        # Identify price driver columns
        self.price_driver_columns = [
            col for col in self.feature_columns if any(
                driver in col for driver in [
                    'production', 'consumption', 'inventories', 'balance', 
                    'supply', 'demand', 'opec', 'non_opec', 'days_supply'
                ]
            )
        ]
        
        # Ensure all feature columns exist in the DataFrame
        valid_features = [col for col in self.feature_columns if col in df.columns]
        
        if len(valid_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(valid_features)
            logger.warning(f"Some feature columns are missing from the DataFrame: {missing}")
        
        logger.info(f"Using {len(valid_features)} features for training")
        logger.info(f"Found {len(self.price_driver_columns)} price driver features")
        
        # Extract features and target
        X = df[valid_features]
        y = df[target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        try:
            # Create and fit the model
            self.model = GradientBoostingRegressor(**self.params)
            self.model.fit(X_scaled, y)
            
            # Get feature importance
            self.feature_importance = pd.DataFrame({
                'feature': valid_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Calculate price driver importance
            if self.price_driver_columns:
                price_driver_importance = self.feature_importance[
                    self.feature_importance['feature'].isin(self.price_driver_columns)
                ]
                total_importance = price_driver_importance['importance'].sum()
                logger.info(f"Price driver features contribute {total_importance:.2%} of total importance")
                
                # Log top price driver features
                top_drivers = price_driver_importance.head(5)
                logger.info(f"Top 5 price driver features: {top_drivers['feature'].tolist()}")
            
            logger.info("Model fitting completed")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting Price Drivers model: {e}")
            raise
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate predictions from the fitted model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to predict on
        
        Returns
        -------
        pd.Series
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        
        # Ensure all feature columns exist in the DataFrame
        valid_features = [col for col in self.feature_columns if col in df.columns]
        
        if len(valid_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(valid_features)
            logger.warning(f"Some feature columns are missing from the DataFrame: {missing}")
        
        # Extract features
        X = df[valid_features]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        predictions = self.model.predict(X_scaled)
        
        # Create Series with the same index as the input DataFrame
        return pd.Series(predictions, index=df.index, name=f"predicted_{self.target_column}")
    
    def evaluate(self, df: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Test data
        target_column : str
            Name of the target column
        
        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        
        # Generate predictions
        predictions = self.predict(df)
        
        # Calculate metrics
        mse = mean_squared_error(df[target_column], predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(df[target_column], predictions)
        r2 = r2_score(df[target_column], predictions)
        
        # Calculate MAPE
        mape = np.mean(np.abs((df[target_column] - predictions) / df[target_column])) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def forecast(self, df: pd.DataFrame, steps: int = 30) -> pd.Series:
        """
        Generate a forecast for future time steps.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical data
        steps : int, optional
            Number of steps to forecast, by default 30
        
        Returns
        -------
        pd.Series
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        
        # Get the last date in the DataFrame
        last_date = df.index[-1]
        
        # Create a date range for the forecast
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        # Initialize the forecast DataFrame
        forecast_df = pd.DataFrame(index=forecast_index)
        
        # Copy the last row of the input DataFrame
        last_row = df.iloc[[-1]].copy()
        
        # Generate the forecast iteratively
        for i in range(steps):
            # Predict the next value
            next_value = self.predict(last_row).iloc[0]
            
            # Update the last row with the predicted value
            last_row[self.target_column] = next_value
            
            # Shift the time series features
            for col in last_row.columns:
                if col.startswith('lag_'):
                    lag_col = col.split('_')[-1]
                    if lag_col.isdigit() and int(lag_col) > 1:
                        lag_n = int(lag_col)
                        lag_n_minus_1 = lag_n - 1
                        lag_col_minus_1 = f"lag_{self.target_column}_{lag_n_minus_1}"
                        if lag_col_minus_1 in last_row.columns:
                            last_row[col] = last_row[lag_col_minus_1]
                
                # Update the lag_1 column with the current prediction
                lag_1_col = f"lag_{self.target_column}_1"
                if lag_1_col in last_row.columns:
                    last_row[lag_1_col] = next_value
            
            # Store the predicted value
            forecast_df.loc[forecast_index[i], self.target_column] = next_value
        
        return forecast_df[self.target_column]
    
    def save(self, file_path: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        file_path : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'params': self.params,
                'feature_importance': self.feature_importance,
                'price_driver_columns': self.price_driver_columns
            }, f)
        
        logger.info(f"Model saved to {file_path}")
    
    def load(self, file_path: str) -> 'PriceDriversForecaster':
        """
        Load the model from a file.
        
        Parameters
        ----------
        file_path : str
            Path to load the model from
        
        Returns
        -------
        PriceDriversForecaster
            Loaded model
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            self.target_column = data['target_column']
            self.params = data['params']
            self.feature_importance = data['feature_importance']
            self.price_driver_columns = data.get('price_driver_columns', [])
            
            logger.info(f"Model loaded from {file_path}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def optimize_price_drivers_model_params(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    cv: int = 5,
    n_iter: int = 20
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Optimize hyperparameters for the Price Drivers model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data
    target_column : str
        Name of the target column
    feature_columns : List[str], optional
        List of feature columns to use, by default None (use all except target)
    cv : int, optional
        Number of cross-validation folds, by default 5
    n_iter : int, optional
        Number of iterations for Bayesian optimization, by default 20
    
    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, float]]
        Best parameters and evaluation metrics
    """
    logger.info("Optimizing Price Drivers model hyperparameters")
    
    # If feature_columns is None, use all columns except target
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # Ensure all feature columns exist in the DataFrame
    valid_features = [col for col in feature_columns if col in df.columns]
    
    # Extract features and target
    X = df[valid_features]
    y = df[target_column]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define parameter space
    param_space = {
        'n_estimators': Integer(50, 500),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'max_depth': Integer(3, 10),
        'subsample': Real(0.5, 1.0),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10)
    }
    
    # Create time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Create base model
    base_model = GradientBoostingRegressor(
        loss='squared_error',
        random_state=42
    )
    
    # Create BayesSearchCV
    optimizer = BayesSearchCV(
        base_model,
        param_space,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit optimizer
    optimizer.fit(X_scaled, y)
    
    # Get best parameters and score
    best_params = optimizer.best_params_
    best_score = optimizer.best_score_
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best RMSE: {np.sqrt(-best_score)}")
    
    # Train model with best parameters
    best_model = PriceDriversForecaster(params=best_params, feature_columns=valid_features)
    best_model.fit(df, target_column)
    
    # Evaluate on the entire dataset
    metrics = best_model.evaluate(df, target_column)
    
    return best_params, metrics

def train_price_drivers_model(
    commodity: str,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    optimize_params: bool = True,
    params: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    n_iter: int = 20
) -> PriceDriversForecaster:
    """
    Train a Price Drivers model for a specific commodity.
    
    Parameters
    ----------
    commodity : str
        Name of the commodity (e.g., 'crude_oil')
    target_column : str, optional
        Name of the target column, by default None (will read from target.txt)
    feature_columns : List[str], optional
        List of feature columns to use, by default None (use all except target)
    optimize_params : bool, optional
        Whether to optimize hyperparameters, by default True
    params : Dict[str, Any], optional
        Model parameters, by default None (use default or optimized parameters)
    cv : int, optional
        Number of cross-validation folds, by default 5
    n_iter : int, optional
        Number of iterations for Bayesian optimization, by default 20
    
    Returns
    -------
    PriceDriversForecaster
        Trained model
    """
    logger.info(f"Training Price Drivers model for {commodity}")
    
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load features data with price drivers
    df = load_features_data(commodity, with_price_drivers=True)
    
    if df.empty:
        raise ValueError(f"No data found for {commodity}")
    
    # If target_column not specified, read from target.txt
    if target_column is None:
        target_file = os.path.join('data/features', f"{commodity}_target.txt")
        if os.path.exists(target_file):
            with open(target_file, 'r') as f:
                target_column = f.read().strip()
            logger.info(f"Using target column from file: {target_column}")
        else:
            raise ValueError(f"Target column not specified and {target_file} not found")
    
    # Split data into train and test sets
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    logger.info(f"Split data into {len(train_df)} training samples and {len(test_df)} test samples")
    
    # Optimize hyperparameters if requested
    if optimize_params:
        logger.info("Optimizing Price Drivers model hyperparameters")
        best_params, _ = optimize_price_drivers_model_params(
            train_df,
            target_column,
            feature_columns=feature_columns,
            cv=cv,
            n_iter=n_iter
        )
        params = best_params
    
    # Use default params if not optimizing and not provided
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'loss': 'squared_error',
            'random_state': 42
        }
    
    # Create and train model
    model = PriceDriversForecaster(params=params, feature_columns=feature_columns)
    model.fit(train_df, target_column, feature_columns)
    
    # Evaluate on test set
    metrics = model.evaluate(test_df, target_column)
    logger.info(f"Test set evaluation: {metrics}")
    
    # Save model
    model_file_path = os.path.join(MODELS_DIR, f"{commodity}_price_drivers.pkl")
    model.save(model_file_path)
    
    return model

if __name__ == "__main__":
    # Example usage
    train_price_drivers_model('crude_oil', optimize_params=True)
