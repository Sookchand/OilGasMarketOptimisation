"""
Feature engineering module for the Oil & Gas Market Optimization system.
This module provides functions for adding technical indicators and features to the data.
"""

import pandas as pd
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

def add_features(df):
    """
    Add technical indicators and features to the data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to add features to
        
    Returns:
    --------
    pandas.DataFrame
        The dataframe with added features
    """
    logger.info("Adding technical indicators and features...")
    
    # Make a copy of the data
    df_features = df.copy()
    
    # Ensure we have a price column
    price_col = 'Price'
    if price_col not in df_features.columns:
        price_candidates = ['Close', 'close', 'Adj Close', 'adj_close']
        for col in price_candidates:
            if col in df_features.columns:
                price_col = col
                break
        else:
            # If no price column found, use the first numeric column
            numeric_cols = df_features.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
                logger.warning(f"No standard price column found. Using '{price_col}' as price column.")
            else:
                logger.error("No numeric columns found in dataframe. Cannot add features.")
                return df_features
    
    # Calculate returns
    df_features['Returns'] = df_features[price_col].pct_change()
    logger.info("Added returns")
    
    # Calculate log returns
    df_features['Log_Returns'] = np.log(df_features[price_col] / df_features[price_col].shift(1))
    logger.info("Added log returns")
    
    # Calculate moving averages
    for window in [5, 10, 20, 50, 100, 200]:
        df_features[f'MA_{window}'] = df_features[price_col].rolling(window=window).mean()
    logger.info("Added moving averages")
    
    # Calculate exponential moving averages
    for window in [5, 10, 20, 50, 100, 200]:
        df_features[f'EMA_{window}'] = df_features[price_col].ewm(span=window, adjust=False).mean()
    logger.info("Added exponential moving averages")
    
    # Calculate volatility
    for window in [5, 10, 20, 50]:
        df_features[f'Volatility_{window}'] = df_features['Returns'].rolling(window=window).std() * np.sqrt(252)
    logger.info("Added volatility indicators")
    
    # Calculate RSI
    for window in [7, 14, 21]:
        delta = df_features[price_col].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        df_features[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    logger.info("Added RSI indicators")
    
    # Calculate Bollinger Bands
    for window in [20]:
        for std in [2.0]:
            middle = df_features[price_col].rolling(window=window).mean()
            stdev = df_features[price_col].rolling(window=window).std()
            
            df_features[f'BB_Middle_{window}'] = middle
            df_features[f'BB_Upper_{window}_{std}'] = middle + std * stdev
            df_features[f'BB_Lower_{window}_{std}'] = middle - std * stdev
            df_features[f'BB_Width_{window}_{std}'] = (df_features[f'BB_Upper_{window}_{std}'] - df_features[f'BB_Lower_{window}_{std}']) / middle
    logger.info("Added Bollinger Bands")
    
    # Calculate MACD
    fast_window = 12
    slow_window = 26
    signal_window = 9
    
    df_features['EMA_12'] = df_features[price_col].ewm(span=fast_window, adjust=False).mean()
    df_features['EMA_26'] = df_features[price_col].ewm(span=slow_window, adjust=False).mean()
    df_features['MACD'] = df_features['EMA_12'] - df_features['EMA_26']
    df_features['MACD_Signal'] = df_features['MACD'].ewm(span=signal_window, adjust=False).mean()
    df_features['MACD_Histogram'] = df_features['MACD'] - df_features['MACD_Signal']
    logger.info("Added MACD indicators")
    
    # Calculate Stochastic Oscillator
    for k_window in [14]:
        for d_window in [3]:
            low_min = df_features[price_col].rolling(window=k_window).min()
            high_max = df_features[price_col].rolling(window=k_window).max()
            
            df_features[f'Stoch_%K_{k_window}'] = 100 * ((df_features[price_col] - low_min) / (high_max - low_min))
            df_features[f'Stoch_%D_{k_window}_{d_window}'] = df_features[f'Stoch_%K_{k_window}'].rolling(window=d_window).mean()
    logger.info("Added Stochastic Oscillator")
    
    # Calculate Average True Range (ATR)
    for window in [14]:
        high = df_features['High'] if 'High' in df_features.columns else df_features[price_col]
        low = df_features['Low'] if 'Low' in df_features.columns else df_features[price_col]
        close = df_features[price_col]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_features[f'ATR_{window}'] = tr.rolling(window=window).mean()
    logger.info("Added ATR")
    
    # Calculate Rate of Change (ROC)
    for window in [5, 10, 20, 50]:
        df_features[f'ROC_{window}'] = (df_features[price_col] - df_features[price_col].shift(window)) / df_features[price_col].shift(window) * 100
    logger.info("Added Rate of Change indicators")
    
    # Calculate Commodity Channel Index (CCI)
    for window in [20]:
        typical_price = df_features[price_col]
        if 'High' in df_features.columns and 'Low' in df_features.columns:
            typical_price = (df_features['High'] + df_features['Low'] + df_features[price_col]) / 3
        
        moving_avg = typical_price.rolling(window=window).mean()
        mean_deviation = abs(typical_price - moving_avg).rolling(window=window).mean()
        
        df_features[f'CCI_{window}'] = (typical_price - moving_avg) / (0.015 * mean_deviation)
    logger.info("Added CCI")
    
    # Calculate On-Balance Volume (OBV)
    if 'Volume' in df_features.columns:
        obv = pd.Series(index=df_features.index)
        obv.iloc[0] = 0
        
        for i in range(1, len(df_features)):
            if df_features[price_col].iloc[i] > df_features[price_col].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df_features['Volume'].iloc[i]
            elif df_features[price_col].iloc[i] < df_features[price_col].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df_features['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        df_features['OBV'] = obv
        logger.info("Added On-Balance Volume")
    
    # Calculate Ichimoku Cloud
    tenkan_window = 9
    kijun_window = 26
    senkou_span_b_window = 52
    
    # Tenkan-sen (Conversion Line)
    high_tenkan = df_features[price_col].rolling(window=tenkan_window).max()
    low_tenkan = df_features[price_col].rolling(window=tenkan_window).min()
    df_features['Ichimoku_Tenkan'] = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line)
    high_kijun = df_features[price_col].rolling(window=kijun_window).max()
    low_kijun = df_features[price_col].rolling(window=kijun_window).min()
    df_features['Ichimoku_Kijun'] = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A)
    df_features['Ichimoku_Senkou_Span_A'] = ((df_features['Ichimoku_Tenkan'] + df_features['Ichimoku_Kijun']) / 2).shift(kijun_window)
    
    # Senkou Span B (Leading Span B)
    high_senkou = df_features[price_col].rolling(window=senkou_span_b_window).max()
    low_senkou = df_features[price_col].rolling(window=senkou_span_b_window).min()
    df_features['Ichimoku_Senkou_Span_B'] = ((high_senkou + low_senkou) / 2).shift(kijun_window)
    
    # Chikou Span (Lagging Span)
    df_features['Ichimoku_Chikou'] = df_features[price_col].shift(-kijun_window)
    logger.info("Added Ichimoku Cloud indicators")
    
    # Calculate price momentum
    for window in [5, 10, 20, 50]:
        df_features[f'Momentum_{window}'] = df_features[price_col] / df_features[price_col].shift(window)
    logger.info("Added momentum indicators")
    
    # Calculate Z-Score
    for window in [20]:
        rolling_mean = df_features[price_col].rolling(window=window).mean()
        rolling_std = df_features[price_col].rolling(window=window).std()
        df_features[f'Z_Score_{window}'] = (df_features[price_col] - rolling_mean) / rolling_std
    logger.info("Added Z-Score")
    
    logger.info(f"Added {len(df_features.columns) - len(df.columns)} new features")
    
    return df_features

def create_lagged_features(df, columns=None, lags=[1, 5, 10]):
    """
    Create lagged features for specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : list, optional
        List of columns to create lags for. If None, uses all numeric columns.
    lags : list, optional
        List of lag periods to create
        
    Returns:
    --------
    pandas.DataFrame
        The dataframe with added lagged features
    """
    logger.info(f"Creating lagged features with lags {lags}")
    
    # Make a copy of the data
    df_lagged = df.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    # Create lagged features
    for col in columns:
        for lag in lags:
            df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
    
    logger.info(f"Added {len(columns) * len(lags)} lagged features")
    
    return df_lagged

def create_rolling_features(df, columns=None, windows=[5, 10, 20], functions=['mean', 'std', 'min', 'max']):
    """
    Create rolling window features for specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : list, optional
        List of columns to create rolling features for. If None, uses all numeric columns.
    windows : list, optional
        List of window sizes to use
    functions : list, optional
        List of functions to apply to rolling windows
        
    Returns:
    --------
    pandas.DataFrame
        The dataframe with added rolling features
    """
    logger.info(f"Creating rolling features with windows {windows} and functions {functions}")
    
    # Make a copy of the data
    df_rolling = df.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    # Create rolling features
    for col in columns:
        for window in windows:
            for func in functions:
                if func == 'mean':
                    df_rolling[f'{col}_rolling_{window}_mean'] = df_rolling[col].rolling(window=window).mean()
                elif func == 'std':
                    df_rolling[f'{col}_rolling_{window}_std'] = df_rolling[col].rolling(window=window).std()
                elif func == 'min':
                    df_rolling[f'{col}_rolling_{window}_min'] = df_rolling[col].rolling(window=window).min()
                elif func == 'max':
                    df_rolling[f'{col}_rolling_{window}_max'] = df_rolling[col].rolling(window=window).max()
    
    logger.info(f"Added {len(columns) * len(windows) * len(functions)} rolling features")
    
    return df_rolling

def create_return_features(df, price_col='Price', periods=[1, 5, 10, 20]):
    """
    Create return features for different periods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    price_col : str, optional
        The price column to use
    periods : list, optional
        List of periods to calculate returns for
        
    Returns:
    --------
    pandas.DataFrame
        The dataframe with added return features
    """
    logger.info(f"Creating return features for periods {periods}")
    
    # Make a copy of the data
    df_returns = df.copy()
    
    # Ensure price column exists
    if price_col not in df_returns.columns:
        price_candidates = ['Close', 'close', 'Adj Close', 'adj_close']
        for col in price_candidates:
            if col in df_returns.columns:
                price_col = col
                break
        else:
            # If no price column found, use the first numeric column
            numeric_cols = df_returns.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
                logger.warning(f"No standard price column found. Using '{price_col}' as price column.")
            else:
                logger.error("No numeric columns found in dataframe. Cannot create return features.")
                return df_returns
    
    # Create return features
    for period in periods:
        df_returns[f'Return_{period}d'] = df_returns[price_col].pct_change(period)
        df_returns[f'Log_Return_{period}d'] = np.log(df_returns[price_col] / df_returns[price_col].shift(period))
    
    logger.info(f"Added {len(periods) * 2} return features")
    
    return df_returns
