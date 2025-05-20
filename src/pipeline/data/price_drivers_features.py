"""
Feature engineering for EIA price drivers.
This module creates features from EIA price drivers data for use in forecasting models.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np

from src.utils.data_utils import load_processed_data, save_to_parquet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/price_drivers_features.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
PRICE_DRIVERS_DIR = 'data/processed/price_drivers'
FEATURES_DATA_DIR = 'data/features'

def load_price_drivers_data() -> pd.DataFrame:
    """
    Load the merged price drivers data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all price drivers
    """
    file_path = os.path.join(PRICE_DRIVERS_DIR, "all_price_drivers.parquet")
    
    if not os.path.exists(file_path):
        logger.warning(f"Price drivers data file not found: {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded price drivers data: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading price drivers data: {e}")
        return pd.DataFrame()

def resample_price_drivers_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample monthly price drivers data to daily frequency using forward fill.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with monthly price drivers data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with daily price drivers data
    """
    if df.empty:
        return df
    
    # Make sure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame index is not a DatetimeIndex")
        return df
    
    # Resample to daily frequency
    df_daily = df.resample('D').ffill()
    
    logger.info(f"Resampled price drivers data from {len(df)} rows to {len(df_daily)} rows")
    return df_daily

def calculate_supply_demand_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate supply-demand balance features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price drivers data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added supply-demand balance features
    """
    df_features = df.copy()
    
    # Check if required columns exist
    supply_cols = ['global_production', 'opec_production', 'non_opec_production']
    demand_cols = ['global_consumption', 'oecd_consumption', 'non_oecd_consumption']
    
    supply_available = all(col in df_features.columns for col in supply_cols)
    demand_available = all(col in df_features.columns for col in demand_cols)
    
    if supply_available and demand_available:
        # Global supply-demand balance
        df_features['global_balance'] = df_features['global_production'] - df_features['global_consumption']
        
        # OPEC supply-demand balance
        df_features['opec_balance'] = df_features['opec_production'] - df_features['global_consumption']
        
        # Non-OPEC supply-demand balance
        df_features['non_opec_balance'] = df_features['non_opec_production'] - df_features['global_consumption']
        
        # Supply-demand ratio
        df_features['supply_demand_ratio'] = df_features['global_production'] / df_features['global_consumption']
        
        logger.info("Added supply-demand balance features")
    else:
        logger.warning("Could not calculate supply-demand balance features due to missing columns")
    
    return df_features

def calculate_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate growth rates for price drivers.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price drivers data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added growth rate features
    """
    df_features = df.copy()
    
    # Calculate month-over-month growth rates
    for col in df_features.columns:
        # Skip columns that are already growth rates
        if 'growth' in col or 'ratio' in col or 'balance' in col:
            continue
        
        # Calculate month-over-month growth rate
        df_features[f'{col}_mom_growth'] = df_features[col].pct_change()
        
        # Calculate year-over-year growth rate (12 months)
        df_features[f'{col}_yoy_growth'] = df_features[col].pct_change(12)
    
    logger.info(f"Added growth rate features for {len(df.columns)} columns")
    return df_features

def calculate_relative_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate relative levels compared to historical averages.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price drivers data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added relative level features
    """
    df_features = df.copy()
    
    # Calculate z-scores relative to 3-year and 5-year moving averages
    for col in df_features.columns:
        # Skip derived columns
        if 'growth' in col or 'ratio' in col or 'balance' in col or 'relative' in col:
            continue
        
        # 3-year moving average (36 months)
        ma_36 = df_features[col].rolling(window=36).mean()
        std_36 = df_features[col].rolling(window=36).std()
        df_features[f'{col}_relative_3yr'] = (df_features[col] - ma_36) / std_36
        
        # 5-year moving average (60 months)
        ma_60 = df_features[col].rolling(window=60).mean()
        std_60 = df_features[col].rolling(window=60).std()
        df_features[f'{col}_relative_5yr'] = (df_features[col] - ma_60) / std_60
    
    logger.info(f"Added relative level features for {len(df.columns)} columns")
    return df_features

def calculate_inventory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate inventory-specific metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price drivers data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added inventory metric features
    """
    df_features = df.copy()
    
    # Check if inventory columns exist
    inventory_cols = ['oecd_inventories', 'global_inventories']
    consumption_cols = ['oecd_consumption', 'global_consumption']
    
    inventory_available = any(col in df_features.columns for col in inventory_cols)
    consumption_available = any(col in df_features.columns for col in consumption_cols)
    
    if inventory_available and consumption_available:
        # Calculate days of supply (inventory / daily consumption)
        if 'oecd_inventories' in df_features.columns and 'oecd_consumption' in df_features.columns:
            # Convert monthly consumption to daily
            daily_consumption = df_features['oecd_consumption'] / 30
            df_features['oecd_days_supply'] = df_features['oecd_inventories'] / daily_consumption
        
        if 'global_inventories' in df_features.columns and 'global_consumption' in df_features.columns:
            # Convert monthly consumption to daily
            daily_consumption = df_features['global_consumption'] / 30
            df_features['global_days_supply'] = df_features['global_inventories'] / daily_consumption
        
        logger.info("Added inventory metric features")
    else:
        logger.warning("Could not calculate inventory metrics due to missing columns")
    
    return df_features

def engineer_price_driver_features() -> pd.DataFrame:
    """
    Engineer features from price drivers data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features
    """
    # Load price drivers data
    df = load_price_drivers_data()
    
    if df.empty:
        logger.warning("No price drivers data available")
        return df
    
    # Resample to daily frequency
    df_daily = resample_price_drivers_to_daily(df)
    
    # Calculate features
    df_features = df_daily.copy()
    
    # Add supply-demand balance features
    df_features = calculate_supply_demand_balance(df_features)
    
    # Add growth rate features
    df_features = calculate_growth_rates(df_features)
    
    # Add relative level features
    df_features = calculate_relative_levels(df_features)
    
    # Add inventory metric features
    df_features = calculate_inventory_metrics(df_features)
    
    # Drop rows with NaN values
    df_clean = df_features.dropna()
    rows_dropped = len(df_features) - len(df_clean)
    if rows_dropped > 0:
        logger.info(f"Dropped {rows_dropped} rows with NaN values")
    
    return df_clean

def merge_price_drivers_with_commodity_data(
    price_drivers_df: pd.DataFrame,
    commodity: str
) -> pd.DataFrame:
    """
    Merge price drivers features with commodity data.
    
    Parameters
    ----------
    price_drivers_df : pd.DataFrame
        DataFrame with price drivers features
    commodity : str
        Name of the commodity (e.g., 'crude_oil')
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame
    """
    # Load commodity data
    commodity_df = load_processed_data(commodity)
    
    if commodity_df.empty:
        logger.warning(f"No data found for {commodity}")
        return pd.DataFrame()
    
    if price_drivers_df.empty:
        logger.warning("No price drivers features available")
        return commodity_df
    
    # Merge on date index
    merged_df = commodity_df.join(price_drivers_df, how='left')
    
    # Forward fill missing values (since price drivers are monthly)
    merged_df = merged_df.ffill()
    
    logger.info(f"Merged price drivers features with {commodity} data")
    return merged_df

def run_price_drivers_feature_engineering(commodities: List[str]) -> None:
    """
    Run the price drivers feature engineering pipeline.
    
    Parameters
    ----------
    commodities : List[str]
        List of commodities to process
    """
    logger.info("Running price drivers feature engineering")
    
    # Create directories if they don't exist
    os.makedirs(FEATURES_DATA_DIR, exist_ok=True)
    
    # Engineer price driver features
    price_drivers_features = engineer_price_driver_features()
    
    if price_drivers_features.empty:
        logger.warning("No price drivers features to save")
        return
    
    # Save price drivers features
    price_drivers_file_path = os.path.join(FEATURES_DATA_DIR, "price_drivers_features.parquet")
    save_to_parquet(price_drivers_features, price_drivers_file_path)
    logger.info(f"Saved price drivers features to {price_drivers_file_path}")
    
    # Merge with commodity data
    for commodity in commodities:
        try:
            # Merge price drivers with commodity data
            merged_df = merge_price_drivers_with_commodity_data(price_drivers_features, commodity)
            
            if not merged_df.empty:
                # Save to features directory
                merged_file_path = os.path.join(FEATURES_DATA_DIR, f"{commodity}_with_drivers.parquet")
                save_to_parquet(merged_df, merged_file_path)
                logger.info(f"Saved merged features for {commodity} to {merged_file_path}")
            else:
                logger.warning(f"No merged features to save for {commodity}")
                
        except Exception as e:
            logger.error(f"Error processing {commodity}: {e}")
    
    logger.info("Price drivers feature engineering completed")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the price drivers feature engineering pipeline
    run_price_drivers_feature_engineering(['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel'])
