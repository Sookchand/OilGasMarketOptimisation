"""
XGBoost forecasting model for oil and gas commodities.
This module implements XGBoost models for time series forecasting.
"""

import os
import logging
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/xgboost_forecaster.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
FEATURES_DATA_DIR = 'data/features'
MODELS_DIR = 'models/forecasting'

class XGBoostForecaster:
    """
    XGBoost forecasting model for time series data.
    """
    
    def __init__(
        self, 
        params: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the XGBoost forecaster.
        
        Parameters
        ----------
        params : Dict[str, Any], optional
            XGBoost parameters, by default None (use default parameters)
        feature_columns : List[str], optional
            List of feature columns to use, by default None (use all except target)
        """
        self.params = params or {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        self.feature_columns = feature_columns
        self.model = None
        self.target_column = None
        self.scaler = StandardScaler()
        self.feature_importance = None
    
    def fit(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> 'XGBoostForecaster':
        """
        Fit the XGBoost model.
        
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
        XGBoostForecaster
            Fitted model
        """
        self.target_column = target_column
        
        # Update feature_columns if provided
        if feature_columns is not None:
            self.feature_columns = feature_columns
        
        # If feature_columns not specified, use all numeric columns except target
        if self.feature_columns is None:
            self.feature_columns = [
                col for col in df.select_dtypes(include=np.number).columns 
                if col != target_column
            ]
        
        # Filter to only include columns that exist in the DataFrame
        valid_features = [col for col in self.feature_columns if col in df.columns]
        
        if not valid_features:
            raise ValueError("No valid feature columns found in the DataFrame")
        
        logger.info(f"Using {len(valid_features)} features for training")
        
        # Extract features and target
        X = df[valid_features]
        y = df[target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        try:
            # Create and fit the model
            self.model = XGBRegressor(**self.params)
            self.model.fit(X_scaled, y)
            
            # Get feature importance
            self.feature_importance = pd.DataFrame({
                'feature': valid_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Model fitting completed")
            logger.info(f"Top 5 important features: {self.feature_importance.head(5)['feature'].tolist()}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting XGBoost model: {e}")
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
        
        try:
            # Filter to only include columns that exist in the DataFrame
            valid_features = [col for col in self.feature_columns if col in df.columns]
            
            if not valid_features:
                raise ValueError("No valid feature columns found in the DataFrame")
            
            # Extract features
            X = df[valid_features]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Generate predictions
            predictions = self.model.predict(X_scaled)
            
            # Convert to Series with the same index as the input
            return pd.Series(predictions, index=df.index)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise
    
    def forecast(
        self, 
        df: pd.DataFrame, 
        steps: int = 1,
        dynamic: bool = True
    ) -> pd.Series:
        """
        Generate multi-step forecasts.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical data to start forecasting from
        steps : int, optional
            Number of steps to forecast, by default 1
        dynamic : bool, optional
            Whether to use dynamic forecasting, by default True
        
        Returns
        -------
        pd.Series
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        
        try:
            # Make a copy of the data
            forecast_df = df.copy()
            
            # Get the last date
            last_date = forecast_df.index[-1]
            
            # Create forecast index
            forecast_index = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
            
            # Initialize forecast array
            forecasts = np.zeros(steps)
            
            # For each step
            for i in range(steps):
                # Generate prediction for the next step
                next_pred = self.predict(forecast_df.iloc[-1:]).iloc[0]
                forecasts[i] = next_pred
                
                if i < steps - 1 and dynamic:
                    # Create a new row for the next day
                    next_date = forecast_index[i]
                    new_row = pd.DataFrame(index=[next_date])
                    
                    # Add the prediction as the target
                    new_row[self.target_column] = next_pred
                    
                    # Add calendar features
                    new_row['day_of_week'] = next_date.dayofweek
                    new_row['day_of_month'] = next_date.day
                    new_row['month'] = next_date.month
                    new_row['quarter'] = next_date.quarter
                    new_row['year'] = next_date.year
                    new_row['is_month_start'] = next_date.is_month_start
                    new_row['is_month_end'] = next_date.is_month_end
                    
                    # Add lagged features from the forecast_df
                    for lag in [1, 3, 5, 10, 20]:
                        lag_col = f'lag_{self.target_column}_{lag}'
                        if lag_col in self.feature_columns:
                            if lag == 1:
                                new_row[lag_col] = next_pred
                            else:
                                idx = min(lag - 1, len(forecast_df))
                                new_row[lag_col] = forecast_df[self.target_column].iloc[-idx]
                    
                    # Add rolling features
                    for window in [5, 10, 20, 50]:
                        mean_col = f'{self.target_column}_mean_{window}'
                        std_col = f'{self.target_column}_std_{window}'
                        
                        if mean_col in self.feature_columns:
                            values = list(forecast_df[self.target_column].iloc[-window+1:]) + [next_pred]
                            new_row[mean_col] = np.mean(values)
                        
                        if std_col in self.feature_columns:
                            values = list(forecast_df[self.target_column].iloc[-window+1:]) + [next_pred]
                            new_row[std_col] = np.std(values)
                    
                    # Append to forecast_df
                    forecast_df = pd.concat([forecast_df, new_row])
            
            # Return as Series
            return pd.Series(forecasts, index=forecast_index)
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
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
        if self.model is None:
            raise ValueError("Model has not been fit yet")
        
        if target_column is None:
            target_column = self.target_column
        
        try:
            # Generate predictions
            y_true = df[target_column]
            y_pred = self.predict(df)
            
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
    
    def plot_feature_importance(
        self, 
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Parameters
        ----------
        top_n : int, optional
            Number of top features to show, by default 20
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 8)
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been fit yet or has no feature importance")
        
        # Get top N features
        top_features = self.feature_importance.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        ax.barh(
            top_features['feature'],
            top_features['importance'],
            color='skyblue'
        )
        
        # Add labels and title
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(f'Top {top_n} Feature Importance')
        
        # Invert y-axis to show most important at the top
        ax.invert_yaxis()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
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
                    'params': self.params,
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_importance': self.feature_importance
                }, f)
            
            logger.info(f"Model saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load(cls, file_path: str) -> 'XGBoostForecaster':
        """
        Load a model from a file.
        
        Parameters
        ----------
        file_path : str
            Path to load the model from
        
        Returns
        -------
        XGBoostForecaster
            Loaded model
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create a new instance
            forecaster = cls(
                params=model_data['params'],
                feature_columns=model_data['feature_columns']
            )
            
            # Set attributes
            forecaster.target_column = model_data['target_column']
            forecaster.model = model_data['model']
            forecaster.scaler = model_data['scaler']
            forecaster.feature_importance = model_data['feature_importance']
            
            logger.info(f"Model loaded from {file_path}")
            return forecaster
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def optimize_xgboost_params(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    cv: int = 5,
    n_iter: int = 20
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Optimize XGBoost hyperparameters using Bayesian optimization.
    
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
        Number of iterations for optimization, by default 20
    
    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, float]]
        Best parameters and corresponding metrics
    """
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer
    except ImportError:
        logger.error("scikit-optimize not installed. Install with 'pip install scikit-optimize'")
        raise
    
    # If feature_columns not specified, use all numeric columns except target
    if feature_columns is None:
        feature_columns = [
            col for col in df.select_dtypes(include=np.number).columns 
            if col != target_column
        ]
    
    # Filter to only include columns that exist in the DataFrame
    valid_features = [col for col in feature_columns if col in df.columns]
    
    if not valid_features:
        raise ValueError("No valid feature columns found in the DataFrame")
    
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
        'colsample_bytree': Real(0.5, 1.0),
        'min_child_weight': Integer(1, 10)
    }
    
    # Create time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Create base model
    base_model = XGBRegressor(
        objective='reg:squarederror',
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
    best_model = XGBoostForecaster(params=best_params, feature_columns=valid_features)
    best_model.fit(df, target_column)
    
    # Evaluate on the entire dataset
    metrics = best_model.evaluate(df, target_column)
    
    return best_params, metrics

def train_xgboost_model(
    commodity: str,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    optimize_params: bool = True,
    params: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    n_iter: int = 20
) -> XGBoostForecaster:
    """
    Train an XGBoost model for a specific commodity.
    
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
        XGBoost parameters if not optimizing, by default None
    cv : int, optional
        Number of cross-validation folds, by default 5
    n_iter : int, optional
        Number of iterations for optimization, by default 20
    
    Returns
    -------
    XGBoostForecaster
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
    
    # Optimize hyperparameters if requested
    if optimize_params:
        logger.info("Optimizing XGBoost hyperparameters")
        best_params, _ = optimize_xgboost_params(
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
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
    
    # Create and train model
    model = XGBoostForecaster(params=params, feature_columns=feature_columns)
    model.fit(train_df, target_column, feature_columns)
    
    # Evaluate on test set
    metrics = model.evaluate(test_df, target_column)
    logger.info(f"Test set evaluation: {metrics}")
    
    # Save model
    model_file_path = os.path.join(MODELS_DIR, f"{commodity}_xgboost.pkl")
    model.save(model_file_path)
    
    return model

if __name__ == "__main__":
    # Example usage
    train_xgboost_model('crude_oil', optimize_params=True)
