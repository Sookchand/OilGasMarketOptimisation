"""
Trend following trading strategies.
This module implements trend following strategies like moving average crossover.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np

from src.trading.strategies.base_strategy import BaseStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trend_following.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MovingAverageCrossover(BaseStrategy):
    """
    Moving average crossover strategy.
    
    This strategy generates buy signals when the fast moving average crosses above
    the slow moving average, and sell signals when it crosses below.
    """
    
    def __init__(
        self, 
        fast_window: int = 10,
        slow_window: int = 50,
        price_col: str = 'close',
        signal_threshold: float = 0.0
    ):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        fast_window : int, optional
            Fast moving average window, by default 10
        slow_window : int, optional
            Slow moving average window, by default 50
        price_col : str, optional
            Price column name, by default 'close'
        signal_threshold : float, optional
            Signal threshold to filter out small crossovers, by default 0.0
        """
        parameters = {
            'fast_window': fast_window,
            'slow_window': slow_window,
            'price_col': price_col,
            'signal_threshold': signal_threshold
        }
        
        super().__init__(name="Moving Average Crossover", parameters=parameters)
        
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.price_col = price_col
        self.signal_threshold = signal_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossover.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price and feature data
        
        Returns
        -------
        pd.DataFrame
            DataFrame with signals
        """
        # Make a copy of the data
        df = data.copy()
        
        # Check if price column exists
        if self.price_col not in df.columns:
            logger.warning(f"Price column '{self.price_col}' not found in data")
            return pd.DataFrame(index=df.index)
        
        # Calculate moving averages
        df[f'ma_fast'] = df[self.price_col].rolling(window=self.fast_window).mean()
        df[f'ma_slow'] = df[self.price_col].rolling(window=self.slow_window).mean()
        
        # Calculate crossover
        df['crossover'] = df['ma_fast'] - df['ma_slow']
        
        # Generate signals
        df['signal'] = 0.0
        
        # Buy signal: fast MA crosses above slow MA
        df.loc[df['crossover'] > self.signal_threshold, 'signal'] = 1.0
        
        # Sell signal: fast MA crosses below slow MA
        df.loc[df['crossover'] < -self.signal_threshold, 'signal'] = -1.0
        
        # Drop NaN values
        df = df.dropna()
        
        logger.info(f"Generated {len(df)} signals")
        
        return df[['signal', 'ma_fast', 'ma_slow', 'crossover']]
    
    def get_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Convert signals to positions.
        
        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame with signals
        
        Returns
        -------
        pd.DataFrame
            DataFrame with positions
        """
        # Make a copy of the signals
        positions = signals.copy()
        
        # Initialize position column
        positions['position'] = 0.0
        
        # Set position based on signal
        positions.loc[positions['signal'] > 0, 'position'] = 1.0
        positions.loc[positions['signal'] < 0, 'position'] = -1.0
        
        return positions

class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) strategy.
    
    This strategy generates buy signals when the MACD line crosses above
    the signal line, and sell signals when it crosses below.
    """
    
    def __init__(
        self, 
        fast_window: int = 12,
        slow_window: int = 26,
        signal_window: int = 9,
        price_col: str = 'close',
        signal_threshold: float = 0.0
    ):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        fast_window : int, optional
            Fast EMA window, by default 12
        slow_window : int, optional
            Slow EMA window, by default 26
        signal_window : int, optional
            Signal line window, by default 9
        price_col : str, optional
            Price column name, by default 'close'
        signal_threshold : float, optional
            Signal threshold to filter out small crossovers, by default 0.0
        """
        parameters = {
            'fast_window': fast_window,
            'slow_window': slow_window,
            'signal_window': signal_window,
            'price_col': price_col,
            'signal_threshold': signal_threshold
        }
        
        super().__init__(name="MACD", parameters=parameters)
        
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window
        self.price_col = price_col
        self.signal_threshold = signal_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MACD crossover.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price and feature data
        
        Returns
        -------
        pd.DataFrame
            DataFrame with signals
        """
        # Make a copy of the data
        df = data.copy()
        
        # Check if price column exists
        if self.price_col not in df.columns:
            logger.warning(f"Price column '{self.price_col}' not found in data")
            return pd.DataFrame(index=df.index)
        
        # Calculate MACD
        df['ema_fast'] = df[self.price_col].ewm(span=self.fast_window, adjust=False).mean()
        df['ema_slow'] = df[self.price_col].ewm(span=self.slow_window, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['signal_line'] = df['macd'].ewm(span=self.signal_window, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal_line']
        
        # Generate signals
        df['signal'] = 0.0
        
        # Buy signal: MACD crosses above signal line
        df.loc[df['histogram'] > self.signal_threshold, 'signal'] = 1.0
        
        # Sell signal: MACD crosses below signal line
        df.loc[df['histogram'] < -self.signal_threshold, 'signal'] = -1.0
        
        # Drop NaN values
        df = df.dropna()
        
        logger.info(f"Generated {len(df)} signals")
        
        return df[['signal', 'macd', 'signal_line', 'histogram']]
    
    def get_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Convert signals to positions.
        
        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame with signals
        
        Returns
        -------
        pd.DataFrame
            DataFrame with positions
        """
        # Make a copy of the signals
        positions = signals.copy()
        
        # Initialize position column
        positions['position'] = 0.0
        
        # Set position based on signal
        positions.loc[positions['signal'] > 0, 'position'] = 1.0
        positions.loc[positions['signal'] < 0, 'position'] = -1.0
        
        return positions
