"""
Utility functions for data acquisition, processing, and management.
This module provides functions to fetch data from various sources including EIA API,
Yahoo Finance, and other data providers.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Configure logging
logger = logging.getLogger(__name__)

def fetch_eia_data(
    series_id: str,
    start_date: str,
    end_date: str,
    frequency: str = 'daily',
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch data from EIA API with error handling and rate limiting.

    Parameters
    ----------
    series_id : str
        The EIA series ID to fetch
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    frequency : str, optional
        Data frequency ('daily', 'weekly', 'monthly'), by default 'daily'
    api_key : str, optional
        EIA API key, by default None (will use environment variable)

    Returns
    -------
    Dict[str, Any]
        JSON response from the EIA API

    Raises
    ------
    Exception
        If the API request fails after multiple retries
    """
    if api_key is None:
        api_key = os.getenv('EIA_API_KEY')
        if not api_key:
            raise ValueError("EIA API key not found. Set the EIA_API_KEY environment variable.")

    url = f"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key={api_key}&frequency={frequency}"
    url += f"&data[0]=value&facets[series][]={series_id}&start={start_date}&end={end_date}"

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = requests.get(url)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:  # Rate limited
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry.")
                time.sleep(wait_time)
                retry_count += 1
                continue

            # Handle other error codes
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from EIA API: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed to fetch data after {max_retries} retries") from e
            time.sleep(2 ** retry_count)  # Exponential backoff

    raise Exception(f"Failed to fetch data after {max_retries} retries")

def eia_json_to_dataframe(json_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert EIA API JSON response to a pandas DataFrame.

    Parameters
    ----------
    json_data : Dict[str, Any]
        JSON response from the EIA API

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and price values
    """
    try:
        # Extract the data from the response
        data = json_data.get('response', {}).get('data', [])

        if not data:
            logger.warning("No data found in EIA API response")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Convert period to datetime
        df['date'] = pd.to_datetime(df['period'])

        # Set date as index and select only value column
        df = df.set_index('date')[['value']]

        # Rename column based on series description
        series_name = json_data.get('response', {}).get('series', [])[0].get('name', 'price')
        df = df.rename(columns={'value': series_name})

        return df

    except Exception as e:
        logger.error(f"Error converting EIA JSON to DataFrame: {e}")
        raise

def fetch_yahoo_finance_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance API.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g., 'CL=F' for WTI Crude)
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    interval : str, optional
        Data interval ('1d', '1wk', '1mo'), by default '1d'

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    try:
        # Convert string dates to datetime if they're not already
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        if data.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()

        # Add ticker as column name prefix
        data.columns = [f"{ticker}_{col}" for col in data.columns]

        return data

    except Exception as e:
        logger.error(f"Error fetching data from Yahoo Finance: {e}")
        raise

def clean_time_series_data(df: pd.DataFrame, fill_method: str = 'ffill') -> pd.DataFrame:
    """
    Clean time series data by handling missing values and outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data
    fill_method : str, optional
        Method to fill missing values ('ffill', 'bfill', 'linear'), by default 'ffill'

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()

    # Ensure the index is datetime
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex. Attempting to convert.")
        df_clean.index = pd.to_datetime(df_clean.index)

    # Handle missing values
    if df_clean.isnull().any().any():
        logger.info(f"Filling missing values using method: {fill_method}")
        if fill_method == 'linear':
            df_clean = df_clean.interpolate(method='linear')
        else:
            df_clean = df_clean.fillna(method=fill_method)

    # Handle remaining missing values (e.g., at the beginning if using ffill)
    if df_clean.isnull().any().any():
        logger.warning("Some missing values remain after initial filling")
        df_clean = df_clean.fillna(method='bfill')

    return df_clean

def detect_outliers(
    series: pd.Series,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.Series:
    """
    Detect outliers in a time series.

    Parameters
    ----------
    series : pd.Series
        Input time series
    method : str, optional
        Method to detect outliers ('iqr', 'zscore'), by default 'iqr'
    threshold : float, optional
        Threshold for outlier detection, by default 1.5 for IQR, 3.0 for zscore

    Returns
    -------
    pd.Series
        Boolean series where True indicates an outlier
    """
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (series < lower_bound) | (series > upper_bound)

    elif method == 'zscore':
        z_scores = (series - series.mean()) / series.std()
        return abs(z_scores) > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

def save_to_parquet(df: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to parquet file with proper directory creation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str
        Path to save the file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save to parquet
        df.to_parquet(file_path)
        logger.info(f"Successfully saved data to {file_path}")

    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def load_processed_data(commodity: str, data_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Load processed data for a specific commodity.

    Parameters
    ----------
    commodity : str
        Name of the commodity (e.g., 'crude_oil', 'regular_gasoline')
    data_dir : str, optional
        Directory containing processed data files, by default 'data/processed'

    Returns
    -------
    pd.DataFrame
        Processed data for the commodity
    """
    try:
        # Construct file path
        file_path = os.path.join(data_dir, f"{commodity}.parquet")

        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"Processed data file not found: {file_path}")

            # Try CSV as fallback
            csv_path = os.path.join(data_dir, f"{commodity}.csv")
            if os.path.exists(csv_path):
                logger.info(f"Loading CSV file instead: {csv_path}")
                return pd.read_csv(csv_path, index_col=0, parse_dates=True)

            return pd.DataFrame()

        # Load parquet file
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        return df

    except Exception as e:
        logger.error(f"Error loading processed data for {commodity}: {e}")
        return pd.DataFrame()
