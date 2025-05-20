"""
Feature engineering for oil and gas commodities.
This module creates features for time series forecasting and market analysis.
"""

import os
import logging
import glob
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.data_utils import save_to_parquet
from src.pipeline.data.price_drivers_features import merge_price_drivers_with_commodity_data, load_price_drivers_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
PROCESSED_DATA_DIR = 'data/processed'
FEATURES_DATA_DIR = 'data/features'

def load_processed_data(commodity: str) -> pd.DataFrame:
    """
    Load processed data for a specific commodity.

    Parameters
    ----------
    commodity : str
        Name of the commodity (e.g., 'crude_oil')

    Returns
    -------
    pd.DataFrame
        Processed data for the commodity
    """
    file_path = os.path.join(PROCESSED_DATA_DIR, f"{commodity}.parquet")

    try:
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} rows for {commodity} from {file_path}")
            return df
        else:
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index

    Returns
    -------
    pd.DataFrame
        DataFrame with calendar features added
    """
    df_features = df.copy()

    # Ensure we have a datetime index
    if not isinstance(df_features.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex. Attempting to convert.")
        df_features.index = pd.to_datetime(df_features.index)

    # Extract calendar features
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_month'] = df_features.index.day
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    df_features['year'] = df_features.index.year
    df_features['is_month_start'] = df_features.index.is_month_start.astype(int)
    df_features['is_month_end'] = df_features.index.is_month_end.astype(int)
    df_features['is_quarter_start'] = df_features.index.is_quarter_start.astype(int)
    df_features['is_quarter_end'] = df_features.index.is_quarter_end.astype(int)
    df_features['is_year_start'] = df_features.index.is_year_start.astype(int)
    df_features['is_year_end'] = df_features.index.is_year_end.astype(int)

    # Add US holiday flag (simplified approach)
    # For a more comprehensive approach, consider using the holidays package
    df_features['is_holiday'] = 0

    logger.info("Added calendar features")
    return df_features

def add_lagged_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lags: List[int] = [1, 3, 5, 10, 20]
) -> pd.DataFrame:
    """
    Add lagged values as features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : List[str], optional
        Columns to create lags for, by default None (all numeric columns)
    lags : List[int], optional
        List of lag periods, by default [1, 3, 5, 10, 20]

    Returns
    -------
    pd.DataFrame
        DataFrame with lagged features added
    """
    df_features = df.copy()

    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df_features.select_dtypes(include=np.number).columns.tolist()
        # Exclude any existing lag columns
        columns = [col for col in columns if not col.startswith('lag_')]

    # Add lagged features
    for col in columns:
        for lag in lags:
            lag_col_name = f'lag_{col}_{lag}'
            df_features[lag_col_name] = df_features[col].shift(lag)

    logger.info(f"Added lagged features for {len(columns)} columns with lags {lags}")
    return df_features

def add_rolling_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    windows: List[int] = [5, 10, 20, 50],
    functions: Dict[str, callable] = {'mean': np.mean, 'std': np.std}
) -> pd.DataFrame:
    """
    Add rolling window features (e.g., moving averages).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : List[str], optional
        Columns to create rolling features for, by default None (all numeric columns)
    windows : List[int], optional
        List of window sizes, by default [5, 10, 20, 50]
    functions : Dict[str, callable], optional
        Dictionary of functions to apply to rolling windows, by default {'mean': np.mean, 'std': np.std}

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features added
    """
    df_features = df.copy()

    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df_features.select_dtypes(include=np.number).columns.tolist()
        # Exclude any existing rolling columns
        columns = [col for col in columns if not any(f"_{func}_" in col for func in functions.keys())]

    # Add rolling features
    for col in columns:
        for window in windows:
            for func_name, func in functions.items():
                feature_name = f'{col}_{func_name}_{window}'
                df_features[feature_name] = df_features[col].rolling(window=window, min_periods=1).apply(func)

    logger.info(f"Added rolling features for {len(columns)} columns with windows {windows}")
    return df_features

def add_technical_indicators(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    Add technical indicators commonly used in trading.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    price_col : str
        Column name containing the price data

    Returns
    -------
    pd.DataFrame
        DataFrame with technical indicators added
    """
    df_features = df.copy()

    # Ensure the price column exists
    if price_col not in df_features.columns:
        logger.error(f"Price column '{price_col}' not found in DataFrame")
        return df_features

    # Calculate returns
    df_features['return_1d'] = df_features[price_col].pct_change(1)

    # Relative Strength Index (RSI)
    # Simplified implementation - for production, consider using a library like ta-lib
    delta = df_features[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    df_features['rsi_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = df_features[price_col].rolling(window=20).mean()
    rolling_std = df_features[price_col].rolling(window=20).std()

    df_features['bollinger_upper'] = rolling_mean + (rolling_std * 2)
    df_features['bollinger_middle'] = rolling_mean
    df_features['bollinger_lower'] = rolling_mean - (rolling_std * 2)

    # MACD (Moving Average Convergence Divergence)
    ema_12 = df_features[price_col].ewm(span=12, adjust=False).mean()
    ema_26 = df_features[price_col].ewm(span=26, adjust=False).mean()
    df_features['macd'] = ema_12 - ema_26
    df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
    df_features['macd_histogram'] = df_features['macd'] - df_features['macd_signal']

    # Momentum
    df_features['momentum_14'] = df_features[price_col].diff(14)

    logger.info(f"Added technical indicators based on {price_col}")
    return df_features

def add_volatility_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    Add volatility-related features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    price_col : str
        Column name containing the price data

    Returns
    -------
    pd.DataFrame
        DataFrame with volatility features added
    """
    df_features = df.copy()

    # Ensure the price column exists
    if price_col not in df_features.columns:
        logger.error(f"Price column '{price_col}' not found in DataFrame")
        return df_features

    # Calculate returns if not already present
    if 'return_1d' not in df_features.columns:
        df_features['return_1d'] = df_features[price_col].pct_change(1)

    # Historical volatility (standard deviation of returns)
    for window in [5, 10, 20, 30]:
        df_features[f'volatility_{window}d'] = df_features['return_1d'].rolling(window=window).std() * np.sqrt(252)  # Annualized

    # High-Low Range
    if 'high' in df_features.columns and 'low' in df_features.columns:
        df_features['daily_range'] = df_features['high'] - df_features['low']
        df_features['daily_range_pct'] = df_features['daily_range'] / df_features[price_col]

        # Average True Range (ATR) - simplified version
        df_features['tr'] = np.maximum(
            df_features['high'] - df_features['low'],
            np.maximum(
                abs(df_features['high'] - df_features[price_col].shift(1)),
                abs(df_features['low'] - df_features[price_col].shift(1))
            )
        )
        df_features['atr_14'] = df_features['tr'].rolling(window=14).mean()

    logger.info(f"Added volatility features based on {price_col}")
    return df_features

def create_target_variable(
    df: pd.DataFrame,
    price_col: str,
    horizon: int = 1,
    target_type: str = 'return'
) -> Tuple[pd.DataFrame, str]:
    """
    Create target variable for forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    price_col : str
        Column name containing the price data
    horizon : int, optional
        Forecast horizon in days, by default 1
    target_type : str, optional
        Type of target ('return', 'price', 'direction'), by default 'return'

    Returns
    -------
    Tuple[pd.DataFrame, str]
        DataFrame with target variable and name of the target column
    """
    df_with_target = df.copy()

    if target_type == 'return':
        # Future return
        target_col = f'future_return_{horizon}d'
        df_with_target[target_col] = df_with_target[price_col].pct_change(horizon).shift(-horizon)

    elif target_type == 'price':
        # Future price
        target_col = f'future_price_{horizon}d'
        df_with_target[target_col] = df_with_target[price_col].shift(-horizon)

    elif target_type == 'direction':
        # Future direction (1 if price goes up, 0 if down)
        future_price = df_with_target[price_col].shift(-horizon)
        target_col = f'future_direction_{horizon}d'
        df_with_target[target_col] = (future_price > df_with_target[price_col]).astype(int)

    else:
        raise ValueError(f"Unknown target type: {target_type}")

    logger.info(f"Created target variable '{target_col}' with horizon {horizon}")
    return df_with_target, target_col

def engineer_features_for_commodity(
    commodity: str,
    price_col: Optional[str] = None,
    add_calendar: bool = True,
    add_lags: bool = True,
    add_rolling: bool = True,
    add_technical: bool = True,
    add_volatility: bool = True,
    add_price_drivers: bool = True,
    forecast_horizon: int = 1,
    target_type: str = 'return'
) -> Tuple[pd.DataFrame, str]:
    """
    Engineer features for a specific commodity.

    Parameters
    ----------
    commodity : str
        Name of the commodity (e.g., 'crude_oil')
    price_col : str, optional
        Column name containing the price data, by default None (will use first column)
    add_calendar : bool, optional
        Whether to add calendar features, by default True
    add_lags : bool, optional
        Whether to add lagged features, by default True
    add_rolling : bool, optional
        Whether to add rolling features, by default True
    add_technical : bool, optional
        Whether to add technical indicators, by default True
    add_volatility : bool, optional
        Whether to add volatility features, by default True
    add_price_drivers : bool, optional
        Whether to add EIA price drivers features, by default True
    forecast_horizon : int, optional
        Forecast horizon in days, by default 1
    target_type : str, optional
        Type of target ('return', 'price', 'direction'), by default 'return'

    Returns
    -------
    Tuple[pd.DataFrame, str]
        DataFrame with engineered features and name of the target column
    """
    # Load processed data
    df = load_processed_data(commodity)

    if df.empty:
        logger.warning(f"No data found for {commodity}")
        return df, ""

    # If price_col not specified, use the first column
    if price_col is None:
        price_col = df.columns[0]
        logger.info(f"Using {price_col} as the price column")

    # Add features
    if add_calendar:
        df = add_calendar_features(df)

    if add_lags:
        df = add_lagged_features(df, columns=[price_col])

    if add_rolling:
        df = add_rolling_features(df, columns=[price_col])

    if add_technical:
        df = add_technical_indicators(df, price_col)

    if add_volatility:
        df = add_volatility_features(df, price_col)

    # Add EIA price drivers features if requested
    if add_price_drivers:
        try:
            # Load price drivers data
            price_drivers_df = load_price_drivers_data()

            if not price_drivers_df.empty:
                # Merge price drivers with commodity data
                df = merge_price_drivers_with_commodity_data(price_drivers_df, df)
                logger.info(f"Added price drivers features to {commodity} data")

                # Add lagged features for price drivers
                price_driver_cols = [col for col in df.columns if col in price_drivers_df.columns]
                if price_driver_cols and add_lags:
                    df = add_lagged_features(df, columns=price_driver_cols, lags=[1, 3, 6, 12])
                    logger.info(f"Added lagged features for {len(price_driver_cols)} price driver columns")
            else:
                logger.warning("No price drivers data available to add")
        except Exception as e:
            logger.error(f"Error adding price drivers features: {e}")

    # Create target variable
    df, target_col = create_target_variable(df, price_col, forecast_horizon, target_type)

    # Drop rows with NaN values (due to lagging/rolling operations)
    df_clean = df.dropna()
    rows_dropped = len(df) - len(df_clean)
    if rows_dropped > 0:
        logger.info(f"Dropped {rows_dropped} rows with NaN values")

    return df_clean, target_col

def run_feature_engineering(
    commodities: Optional[List[str]] = None,
    forecast_horizon: int = 1,
    target_type: str = 'return',
    add_price_drivers: bool = True
) -> None:
    """
    Run the feature engineering pipeline for all commodities.

    Parameters
    ----------
    commodities : List[str], optional
        List of commodities to process, by default None (all available)
    forecast_horizon : int, optional
        Forecast horizon in days, by default 1
    target_type : str, optional
        Type of target ('return', 'price', 'direction'), by default 'return'
    add_price_drivers : bool, optional
        Whether to add EIA price drivers features, by default True
    """
    # Create directories if they don't exist
    os.makedirs(FEATURES_DATA_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # If no commodities specified, process all available in processed data directory
    if commodities is None:
        parquet_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*.parquet"))
        commodities = [os.path.splitext(os.path.basename(f))[0] for f in parquet_files]

    logger.info(f"Engineering features for commodities: {commodities}")

    for commodity in commodities:
        try:
            # Engineer features
            df_features, target_col = engineer_features_for_commodity(
                commodity,
                add_calendar=True,
                add_lags=True,
                add_rolling=True,
                add_technical=True,
                add_volatility=True,
                add_price_drivers=add_price_drivers,
                forecast_horizon=forecast_horizon,
                target_type=target_type
            )

            if not df_features.empty:
                # Save to features directory
                features_file_path = os.path.join(FEATURES_DATA_DIR, f"{commodity}_features.parquet")
                save_to_parquet(df_features, features_file_path)
                logger.info(f"Saved features for {commodity} to {features_file_path}")

                # Save target column name for reference
                with open(os.path.join(FEATURES_DATA_DIR, f"{commodity}_target.txt"), 'w') as f:
                    f.write(target_col)
            else:
                logger.warning(f"No features to save for {commodity}")

        except Exception as e:
            logger.error(f"Error engineering features for {commodity}: {e}")

    logger.info("Feature engineering completed")

if __name__ == "__main__":
    # Run the feature engineering pipeline
    run_feature_engineering()
