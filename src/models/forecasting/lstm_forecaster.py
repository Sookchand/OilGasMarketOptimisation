"""
LSTM forecasting model for oil and gas commodities.
This module implements LSTM models for time series forecasting.
"""

import os
import logging
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
except ImportError:
    logging.error("TensorFlow not installed. Install with 'pip install tensorflow'")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lstm_forecaster.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
FEATURES_DATA_DIR = 'data/features'
MODELS_DIR = 'models/forecasting'

class LSTMForecaster:
    """
    LSTM forecasting model for time series data.
    """
    
    def __init__(
        self, 
        params: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None,
        sequence_length: int = 10
    ):
        """
        Initialize the LSTM forecaster.
        
        Parameters
        ----------
        params : Dict[str, Any], optional
            LSTM parameters, by default None (use default parameters)
        feature_columns : List[str], optional
            List of feature columns to use, by default None (use all except target)
        sequence_length : int, optional
            Length of input sequences, by default 10
        """
        self.params = params or {
            'units': 50,
            'layers': 1,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10
        }
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.model = None
        self.target_column = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None
    
    def _create_sequences(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        target_column : str
            Name of the target column
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X and y arrays for LSTM
        """
        # Extract features and target
        X = df[self.feature_columns].values
        y = df[target_column].values
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq = []
        y_seq = []
        
        for i in range(len(df) - self.sequence_length):
            X_seq.append(X_scaled[i:i+self.sequence_length])
            y_seq.append(y_scaled[i+self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build the LSTM model.
        
        Parameters
        ----------
        input_shape : Tuple[int, int]
            Shape of input sequences (sequence_length, n_features)
        
        Returns
        -------
        tf.keras.Model
            LSTM model
        """
        model = Sequential()
        
        # Add LSTM layers
        for i in range(self.params['layers']):
            if i == 0:
                # First layer
                model.add(LSTM(
                    units=self.params['units'],
                    return_sequences=i < self.params['layers'] - 1,
                    input_shape=input_shape
                ))
            else:
                # Subsequent layers
                model.add(LSTM(
                    units=self.params['units'],
                    return_sequences=i < self.params['layers'] - 1
                ))
            
            # Add dropout after each LSTM layer
            model.add(Dropout(self.params['dropout']))
        
        # Add output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.params['learning_rate']),
            loss='mse'
        )
        
        return model
    
    def fit(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> 'LSTMForecaster':
        """
        Fit the LSTM model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        target_column : str
            Name of the target column
        feature_columns : List[str], optional
            List of feature columns to use, by default None (use all except target)
        validation_split : float, optional
            Fraction of data to use for validation, by default 0.2
        verbose : int, optional
            Verbosity level, by default 1
        
        Returns
        -------
        LSTMForecaster
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
        
        self.feature_columns = valid_features
        logger.info(f"Using {len(valid_features)} features for training")
        
        try:
            # Create sequences
            X, y = self._create_sequences(df, target_column)
            
            if len(X) == 0:
                raise ValueError("No sequences created. Check sequence_length and data size.")
            
            logger.info(f"Created {len(X)} sequences of length {self.sequence_length}")
            
            # Build model
            self.model = self._build_model((self.sequence_length, len(self.feature_columns)))
            
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.params['patience'],
                    restore_best_weights=True
                )
            ]
            
            # Create temporary model checkpoint
            checkpoint_path = os.path.join(MODELS_DIR, 'temp_lstm_checkpoint.h5')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True
                )
            )
            
            # Train model
            self.history = self.model.fit(
                X, y,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            
            # Load best model if checkpoint exists
            if os.path.exists(checkpoint_path):
                self.model = load_model(checkpoint_path)
                os.remove(checkpoint_path)
            
            logger.info("Model fitting completed")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {e}")
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
            X = df[valid_features].values
            
            # Scale features
            X_scaled = self.scaler_X.transform(X)
            
            # Create sequences
            X_seq = []
            
            for i in range(len(df) - self.sequence_length):
                X_seq.append(X_scaled[i:i+self.sequence_length])
            
            if not X_seq:
                raise ValueError("Not enough data to create sequences")
            
            # Generate predictions
            y_pred_scaled = self.model.predict(np.array(X_seq))
            
            # Inverse transform predictions
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
            
            # Create index for predictions
            pred_index = df.index[self.sequence_length:]
            
            # Return as Series
            return pd.Series(y_pred, index=pred_index)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise
    
    def forecast(
        self, 
        df: pd.DataFrame, 
        steps: int = 1
    ) -> pd.Series:
        """
        Generate multi-step forecasts.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical data to start forecasting from
        steps : int, optional
            Number of steps to forecast, by default 1
        
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
            
            # Extract features
            X = forecast_df[self.feature_columns].values
            
            # Scale features
            X_scaled = self.scaler_X.transform(X)
            
            # Get the last sequence
            last_sequence = X_scaled[-self.sequence_length:]
            
            # For each step
            for i in range(steps):
                # Reshape sequence for prediction
                sequence = last_sequence.reshape(1, self.sequence_length, len(self.feature_columns))
                
                # Generate prediction
                pred_scaled = self.model.predict(sequence)
                
                # Inverse transform prediction
                pred = self.scaler_y.inverse_transform(pred_scaled)[0, 0]
                
                # Store forecast
                forecasts[i] = pred
                
                if i < steps - 1:
                    # Update sequence for next prediction
                    # This is a simplified approach - in a real application, you would
                    # need to update all features, not just shift the sequence
                    last_sequence = np.roll(last_sequence, -1, axis=0)
                    
                    # Create a new feature vector for the predicted point
                    # This is a placeholder - in reality, you would need to generate
                    # appropriate feature values for the new time point
                    new_features = np.zeros(len(self.feature_columns))
                    new_features[0] = pred  # Assuming first feature is the target
                    
                    # Scale the new features
                    new_features_scaled = self.scaler_X.transform(new_features.reshape(1, -1))
                    
                    # Update the last position in the sequence
                    last_sequence[-1] = new_features_scaled
            
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
            y_pred = self.predict(df)
            
            # Get actual values (aligned with predictions)
            y_true = df[target_column].loc[y_pred.index]
            
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
    
    def plot_training_history(
        self, 
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot training history.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 6)
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training and validation loss
        ax.plot(self.history.history['loss'], label='Training Loss')
        ax.plot(self.history.history['val_loss'], label='Validation Loss')
        
        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        
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
            
            # Save Keras model separately
            keras_model_path = file_path.replace('.pkl', '_keras.h5')
            self.model.save(keras_model_path)
            
            # Save other attributes
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'params': self.params,
                    'feature_columns': self.feature_columns,
                    'sequence_length': self.sequence_length,
                    'target_column': self.target_column,
                    'scaler_X': self.scaler_X,
                    'scaler_y': self.scaler_y,
                    'keras_model_path': keras_model_path
                }, f)
            
            logger.info(f"Model saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load(cls, file_path: str) -> 'LSTMForecaster':
        """
        Load a model from a file.
        
        Parameters
        ----------
        file_path : str
            Path to load the model from
        
        Returns
        -------
        LSTMForecaster
            Loaded model
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create a new instance
            forecaster = cls(
                params=model_data['params'],
                feature_columns=model_data['feature_columns'],
                sequence_length=model_data['sequence_length']
            )
            
            # Set attributes
            forecaster.target_column = model_data['target_column']
            forecaster.scaler_X = model_data['scaler_X']
            forecaster.scaler_y = model_data['scaler_y']
            
            # Load Keras model
            keras_model_path = model_data['keras_model_path']
            if os.path.exists(keras_model_path):
                forecaster.model = load_model(keras_model_path)
            else:
                logger.warning(f"Keras model file not found: {keras_model_path}")
            
            logger.info(f"Model loaded from {file_path}")
            return forecaster
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def train_lstm_model(
    commodity: str,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    sequence_length: int = 10,
    params: Optional[Dict[str, Any]] = None,
    validation_split: float = 0.2
) -> LSTMForecaster:
    """
    Train an LSTM model for a specific commodity.
    
    Parameters
    ----------
    commodity : str
        Name of the commodity (e.g., 'crude_oil')
    target_column : str, optional
        Name of the target column, by default None (will read from target.txt)
    feature_columns : List[str], optional
        List of feature columns to use, by default None (use all except target)
    sequence_length : int, optional
        Length of input sequences, by default 10
    params : Dict[str, Any], optional
        LSTM parameters, by default None (use default parameters)
    validation_split : float, optional
        Fraction of data to use for validation, by default 0.2
    
    Returns
    -------
    LSTMForecaster
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
    
    # Use default params if not provided
    if params is None:
        params = {
            'units': 50,
            'layers': 1,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10
        }
    
    # Create and train model
    model = LSTMForecaster(
        params=params,
        feature_columns=feature_columns,
        sequence_length=sequence_length
    )
    
    model.fit(
        train_df,
        target_column,
        feature_columns=feature_columns,
        validation_split=validation_split
    )
    
    # Evaluate on test set
    metrics = model.evaluate(test_df, target_column)
    logger.info(f"Test set evaluation: {metrics}")
    
    # Save model
    model_file_path = os.path.join(MODELS_DIR, f"{commodity}_lstm.pkl")
    model.save(model_file_path)
    
    return model

if __name__ == "__main__":
    # Example usage
    train_lstm_model('crude_oil')
