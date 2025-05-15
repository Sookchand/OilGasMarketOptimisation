"""
Mean reversion trading strategies.
This module implements mean reversion strategies like RSI and Bollinger Bands.
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
        logging.FileHandler('logs/mean_reversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RSIStrategy(BaseStrategy):
    """
    Relative Strength Index (RSI) strategy.
    
    This strategy generates buy signals when RSI falls below the oversold level,
    and sell signals when it rises above the overbought level.
    """
    
    def __init__(
        self, 
        window: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        price_col: str = 'close'
    ):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        window : int, optional
            RSI calculation window, by default 14
        oversold : float, optional
            Oversold threshold, by default 30.0
        overbought : float, optional
            Overbought threshold, by default 70.0
        price_col : str, optional
            Price column name, by default 'close'
        """
        parameters = {
            'window': window,
            'oversold': oversold,
            'overbought': overbought,
            'price_col': price_col
        }
        
        super().__init__(name="RSI", parameters=parameters)
        
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        self.price_col = price_col
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI.
        
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
        
        # Calculate RSI
        delta = df[self.price_col].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.window).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df['signal'] = 0.0
        
        # Buy signal: RSI below oversold level
        df.loc[df['rsi'] < self.oversold, 'signal'] = 1.0
        
        # Sell signal: RSI above overbought level
        df.loc[df['rsi'] > self.overbought, 'signal'] = -1.0
        
        # Drop NaN values
        df = df.dropna()
        
        logger.info(f"Generated {len(df)} signals")
        
        return df[['signal', 'rsi']]
    
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

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands strategy.
    
    This strategy generates buy signals when price touches the lower band,
    and sell signals when it touches the upper band.
    """
    
    def __init__(
        self, 
        window: int = 20,
        num_std: float = 2.0,
        price_col: str = 'close'
    ):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        window : int, optional
            Moving average window, by default 20
        num_std : float, optional
            Number of standard deviations for bands, by default 2.0
        price_col : str, optional
            Price column name, by default 'close'
        """
        parameters = {
            'window': window,
            'num_std': num_std,
            'price_col': price_col
        }
        
        super().__init__(name="Bollinger Bands", parameters=parameters)
        
        self.window = window
        self.num_std = num_std
        self.price_col = price_col
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.
        
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
        
        # Calculate Bollinger Bands
        df['middle_band'] = df[self.price_col].rolling(window=self.window).mean()
        df['std'] = df[self.price_col].rolling(window=self.window).std()
        df['upper_band'] = df['middle_band'] + (df['std'] * self.num_std)
        df['lower_band'] = df['middle_band'] - (df['std'] * self.num_std)
        
        # Calculate band width
        df['band_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
        
        # Calculate percent B
        df['percent_b'] = (df[self.price_col] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        
        # Generate signals
        df['signal'] = 0.0
        
        # Buy signal: price touches lower band
        df.loc[df['percent_b'] <= 0.0, 'signal'] = 1.0
        
        # Sell signal: price touches upper band
        df.loc[df['percent_b'] >= 1.0, 'signal'] = -1.0
        
        # Drop NaN values
        df = df.dropna()
        
        logger.info(f"Generated {len(df)} signals")
        
        return df[['signal', 'middle_band', 'upper_band', 'lower_band', 'percent_b', 'band_width']]
    
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
