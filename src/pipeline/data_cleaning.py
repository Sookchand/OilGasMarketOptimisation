"""
Data cleaning module for the Oil & Gas Market Optimization system.
This module provides functions for cleaning and preprocessing data.
"""

import pandas as pd
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

def clean_data(df):
    """
    Clean data by handling missing values and outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to clean
        
    Returns:
    --------
    pandas.DataFrame
        The cleaned dataframe
    """
    logger.info("Cleaning data...")
    
    # Make a copy of the data
    df_cleaned = df.copy()
    
    # Handle missing values
    df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
    logger.info(f"Handled {df.isna().sum().sum()} missing values")
    
    # Handle outliers using IQR method
    for col in df_cleaned.select_dtypes(include=['number']).columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers_count = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
        
        # Replace outliers with bounds
        df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
        df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
        
        logger.info(f"Handled {outliers_count} outliers in column {col}")
    
    return df_cleaned

def normalize_data(df, method='minmax'):
    """
    Normalize numerical data in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to normalize
    method : str, optional
        The normalization method ('minmax' or 'zscore')
        
    Returns:
    --------
    pandas.DataFrame
        The normalized dataframe
    """
    logger.info(f"Normalizing data using {method} method...")
    
    # Make a copy of the data
    df_normalized = df.copy()
    
    # Select numerical columns
    num_cols = df_normalized.select_dtypes(include=['number']).columns
    
    if method == 'minmax':
        # Min-Max normalization
        for col in num_cols:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        # Z-score normalization
        for col in num_cols:
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    else:
        logger.warning(f"Unknown normalization method: {method}. Data not normalized.")
    
    return df_normalized

def resample_data(df, freq='D'):
    """
    Resample time series data to a specified frequency.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to resample
    freq : str, optional
        The frequency to resample to ('D' for daily, 'W' for weekly, etc.)
        
    Returns:
    --------
    pandas.DataFrame
        The resampled dataframe
    """
    logger.info(f"Resampling data to {freq} frequency...")
    
    # Ensure the index is a datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Index is not a DatetimeIndex. Attempting to convert...")
        try:
            df.index = pd.to_datetime(df.index)
        except:
            logger.error("Failed to convert index to DatetimeIndex. Cannot resample.")
            return df
    
    # Resample the data
    resampled = df.resample(freq)
    
    # For price data, use the following aggregations
    agg_dict = {
        'Price': 'last',
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # Filter the aggregation dictionary to only include columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    # If no matching columns, use default aggregation
    if not agg_dict:
        logger.warning("No standard columns found. Using default aggregation.")
        return resampled.last()
    
    # Apply the aggregation
    result = resampled.agg(agg_dict)
    
    logger.info(f"Resampled data from {len(df)} rows to {len(result)} rows")
    
    return result

def handle_duplicates(df):
    """
    Handle duplicate rows in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to process
        
    Returns:
    --------
    pandas.DataFrame
        The dataframe with duplicates handled
    """
    logger.info("Checking for duplicate rows...")
    
    # Count duplicates
    dup_count = df.duplicated().sum()
    
    if dup_count > 0:
        logger.info(f"Found {dup_count} duplicate rows. Removing...")
        df = df.drop_duplicates()
    else:
        logger.info("No duplicate rows found.")
    
    return df

def preprocess_data(df, clean=True, normalize=False, resample_freq=None, handle_dups=True):
    """
    Preprocess data with multiple steps.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to preprocess
    clean : bool, optional
        Whether to clean the data
    normalize : bool or str, optional
        Whether to normalize the data and which method to use
    resample_freq : str, optional
        The frequency to resample to, if any
    handle_dups : bool, optional
        Whether to handle duplicate rows
        
    Returns:
    --------
    pandas.DataFrame
        The preprocessed dataframe
    """
    logger.info("Starting data preprocessing...")
    
    # Make a copy of the data
    result = df.copy()
    
    # Handle duplicates
    if handle_dups:
        result = handle_duplicates(result)
    
    # Clean data
    if clean:
        result = clean_data(result)
    
    # Resample data
    if resample_freq:
        result = resample_data(result, freq=resample_freq)
    
    # Normalize data
    if normalize:
        method = 'minmax' if normalize is True else normalize
        result = normalize_data(result, method=method)
    
    logger.info("Data preprocessing completed.")
    
    return result
