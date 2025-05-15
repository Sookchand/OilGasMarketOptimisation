"""
Data cleaning and preprocessing pipeline for oil and gas commodities.
This module processes raw data and prepares it for feature engineering and modeling.
"""

import os
import logging
import glob
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np

from src.utils.data_utils import (
    clean_time_series_data,
    detect_outliers,
    save_to_parquet
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
RAW_DATA_DIR = 'data/raw'
INTERIM_DATA_DIR = 'data/interim'
PROCESSED_DATA_DIR = 'data/processed'

def load_raw_data(commodity: str) -> pd.DataFrame:
    """
    Load raw data for a specific commodity.
    
    Parameters
    ----------
    commodity : str
        Name of the commodity (e.g., 'crude_oil')
    
    Returns
    -------
    pd.DataFrame
        Raw data for the commodity
    """
    file_path = os.path.join(RAW_DATA_DIR, f"{commodity}.parquet")
    
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

def handle_outliers(
    df: pd.DataFrame, 
    method: str = 'iqr',
    threshold: float = 1.5,
    treatment: str = 'winsorize'
) -> pd.DataFrame:
    """
    Detect and handle outliers in the data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    method : str, optional
        Method to detect outliers ('iqr', 'zscore'), by default 'iqr'
    threshold : float, optional
        Threshold for outlier detection, by default 1.5
    treatment : str, optional
        How to handle outliers ('winsorize', 'remove', 'none'), by default 'winsorize'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with outliers handled
    """
    df_clean = df.copy()
    
    for column in df_clean.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df_clean[column]):
            continue
            
        # Detect outliers
        outliers = detect_outliers(df_clean[column], method, threshold)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            logger.info(f"Detected {outlier_count} outliers in {column} using {method} method")
            
            if treatment == 'winsorize':
                # Cap outliers at threshold values
                if method == 'iqr':
                    q1 = df_clean[column].quantile(0.25)
                    q3 = df_clean[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                elif method == 'zscore':
                    mean = df_clean[column].mean()
                    std = df_clean[column].std()
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                
                # Apply winsorization
                df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
                df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
                logger.info(f"Winsorized outliers in {column}")
                
            elif treatment == 'remove':
                # Remove rows with outliers
                df_clean = df_clean.loc[~outliers]
                logger.info(f"Removed {outlier_count} rows with outliers in {column}")
                
            elif treatment == 'none':
                logger.info(f"No treatment applied to outliers in {column}")
    
    return df_clean

def normalize_data(
    df: pd.DataFrame, 
    method: str = 'minmax',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize numeric columns in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    method : str, optional
        Normalization method ('minmax', 'zscore'), by default 'minmax'
    columns : List[str], optional
        Columns to normalize, by default None (all numeric columns)
    
    Returns
    -------
    pd.DataFrame
        Normalized DataFrame
    """
    df_norm = df.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df_norm.select_dtypes(include=np.number).columns.tolist()
    
    # Filter to ensure we only normalize existing numeric columns
    columns = [col for col in columns if col in df_norm.columns and pd.api.types.is_numeric_dtype(df_norm[col])]
    
    for column in columns:
        if method == 'minmax':
            min_val = df_norm[column].min()
            max_val = df_norm[column].max()
            
            # Avoid division by zero
            if max_val > min_val:
                df_norm[column] = (df_norm[column] - min_val) / (max_val - min_val)
                logger.info(f"Applied min-max normalization to {column}")
            else:
                logger.warning(f"Skipped normalization for {column}: min equals max")
                
        elif method == 'zscore':
            mean = df_norm[column].mean()
            std = df_norm[column].std()
            
            # Avoid division by zero
            if std > 0:
                df_norm[column] = (df_norm[column] - mean) / std
                logger.info(f"Applied z-score normalization to {column}")
            else:
                logger.warning(f"Skipped normalization for {column}: standard deviation is zero")
    
    return df_norm

def process_commodity_data(
    commodity: str,
    handle_missing: bool = True,
    handle_outliers_method: str = 'iqr',
    normalize: bool = False
) -> pd.DataFrame:
    """
    Process raw data for a specific commodity.
    
    Parameters
    ----------
    commodity : str
        Name of the commodity (e.g., 'crude_oil')
    handle_missing : bool, optional
        Whether to handle missing values, by default True
    handle_outliers_method : str, optional
        Method to handle outliers ('iqr', 'zscore', 'none'), by default 'iqr'
    normalize : bool, optional
        Whether to normalize the data, by default False
    
    Returns
    -------
    pd.DataFrame
        Processed data for the commodity
    """
    # Load raw data
    df = load_raw_data(commodity)
    
    if df.empty:
        logger.warning(f"No data found for {commodity}")
        return df
    
    # Handle missing values
    if handle_missing:
        df = clean_time_series_data(df, fill_method='ffill')
        logger.info(f"Handled missing values for {commodity}")
    
    # Handle outliers
    if handle_outliers_method != 'none':
        df = handle_outliers(df, method=handle_outliers_method, treatment='winsorize')
        logger.info(f"Handled outliers for {commodity} using {handle_outliers_method} method")
    
    # Normalize data if requested
    if normalize:
        df = normalize_data(df, method='minmax')
        logger.info(f"Normalized data for {commodity}")
    
    return df

def run_data_cleaning(
    commodities: Optional[List[str]] = None,
    normalize: bool = False
) -> None:
    """
    Run the data cleaning pipeline for all commodities.
    
    Parameters
    ----------
    commodities : List[str], optional
        List of commodities to process, by default None (all available)
    normalize : bool, optional
        Whether to normalize the data, by default False
    """
    # Create directories if they don't exist
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # If no commodities specified, process all available in raw data directory
    if commodities is None:
        parquet_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.parquet"))
        commodities = [os.path.splitext(os.path.basename(f))[0] for f in parquet_files]
    
    logger.info(f"Processing data for commodities: {commodities}")
    
    for commodity in commodities:
        try:
            # Process the data
            df_processed = process_commodity_data(
                commodity,
                handle_missing=True,
                handle_outliers_method='iqr',
                normalize=normalize
            )
            
            if not df_processed.empty:
                # Save to processed directory
                processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"{commodity}.parquet")
                save_to_parquet(df_processed, processed_file_path)
                logger.info(f"Saved processed data for {commodity} to {processed_file_path}")
            else:
                logger.warning(f"No processed data to save for {commodity}")
                
        except Exception as e:
            logger.error(f"Error processing {commodity}: {e}")
    
    logger.info("Data cleaning completed")

if __name__ == "__main__":
    # Run the data cleaning pipeline
    run_data_cleaning()
