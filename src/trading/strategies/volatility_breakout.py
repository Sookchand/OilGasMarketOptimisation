"""
Volatility breakout trading strategies.
This module implements volatility breakout strategies like Donchian Channel and ATR.
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
        logging.FileHandler('logs/volatility_breakout.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DonchianChannelStrategy(BaseStrategy):
    """
    Donchian Channel strategy.
    
    This strategy generates buy signals when price breaks above the upper channel,
    and sell signals when it breaks below the lower channel.
    """
    
    def __init__(
        self, 
        window: int = 20,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low'
    ):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        window : int, optional
            Lookback window, by default 20
        price_col : str, optional
            Price column name, by default 'close'
        high_col : str, optional
            High price column name, by default 'high'
        low_col : str, optional
            Low price column name, by default 'low'
        """
        parameters = {
            'window': window,
            'price_col': price_col,
            'high_col': high_col,
            'low_col': low_col
        }
        
        super().__init__(name="Donchian Channel", parameters=parameters)
        
        self.window = window
        self.price_col = price_col
        self.high_col = high_col
        self.low_col = low_col
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Donchian Channel breakouts.
        
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
        
        # Check if required columns exist
        required_cols = [self.price_col, self.high_col, self.low_col]
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Required column '{col}' not found in data")
                return pd.DataFrame(index=df.index)
        
        # Calculate Donchian Channels
        df['upper_channel'] = df[self.high_col].rolling(window=self.window).max()
        df['lower_channel'] = df[self.low_col].rolling(window=self.window).min()
        df['middle_channel'] = (df['upper_channel'] + df['lower_channel']) / 2
        
        # Calculate previous channels
        df['prev_upper'] = df['upper_channel'].shift(1)
        df['prev_lower'] = df['lower_channel'].shift(1)
        
        # Generate signals
        df['signal'] = 0.0
        
        # Buy signal: price breaks above upper channel
        df.loc[df[self.price_col] > df['prev_upper'], 'signal'] = 1.0
        
        # Sell signal: price breaks below lower channel
        df.loc[df[self.price_col] < df['prev_lower'], 'signal'] = -1.0
        
        # Drop NaN values
        df = df.dropna()
        
        logger.info(f"Generated {len(df)} signals")
        
        return df[['signal', 'upper_channel', 'middle_channel', 'lower_channel']]
    
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

class ATRChannelStrategy(BaseStrategy):
    """
    Average True Range (ATR) Channel strategy.
    
    This strategy generates buy signals when price moves up by a multiple of ATR,
    and sell signals when it moves down by a multiple of ATR.
    """
    
    def __init__(
        self, 
        window: int = 14,
        multiplier: float = 2.0,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low'
    ):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        window : int, optional
            ATR calculation window, by default 14
        multiplier : float, optional
            ATR multiplier for channel width, by default 2.0
        price_col : str, optional
            Price column name, by default 'close'
        high_col : str, optional
            High price column name, by default 'high'
        low_col : str, optional
            Low price column name, by default 'low'
        """
        parameters = {
            'window': window,
            'multiplier': multiplier,
            'price_col': price_col,
            'high_col': high_col,
            'low_col': low_col
        }
        
        super().__init__(name="ATR Channel", parameters=parameters)
        
        self.window = window
        self.multiplier = multiplier
        self.price_col = price_col
        self.high_col = high_col
        self.low_col = low_col
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on ATR Channel breakouts.
        
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
        
        # Check if required columns exist
        required_cols = [self.price_col, self.high_col, self.low_col]
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Required column '{col}' not found in data")
                return pd.DataFrame(index=df.index)
        
        # Calculate True Range
        df['prev_close'] = df[self.price_col].shift(1)
        df['tr1'] = df[self.high_col] - df[self.low_col]
        df['tr2'] = abs(df[self.high_col] - df['prev_close'])
        df['tr3'] = abs(df[self.low_col] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['true_range'].rolling(window=self.window).mean()
        
        # Calculate ATR Channels
        df['upper_channel'] = df[self.price_col] + (df['atr'] * self.multiplier)
        df['lower_channel'] = df[self.price_col] - (df['atr'] * self.multiplier)
        
        # Calculate price changes
        df['price_change'] = df[self.price_col].diff()
        
        # Generate signals
        df['signal'] = 0.0
        
        # Buy signal: price moves up by ATR multiplier
        df.loc[df['price_change'] > df['atr'] * self.multiplier, 'signal'] = 1.0
        
        # Sell signal: price moves down by ATR multiplier
        df.loc[df['price_change'] < -df['atr'] * self.multiplier, 'signal'] = -1.0
        
        # Drop NaN values
        df = df.dropna()
        
        logger.info(f"Generated {len(df)} signals")
        
        return df[['signal', 'atr', 'upper_channel', 'lower_channel']]
    
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
