"""
Data acquisition pipeline for oil and gas commodities.
This module fetches data from various sources and saves it to the data/raw directory.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np

from src.utils.data_utils import (
    fetch_eia_data, 
    eia_json_to_dataframe,
    fetch_yahoo_finance_data,
    clean_time_series_data,
    save_to_parquet
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_acquisition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
RAW_DATA_DIR = 'data/raw'
INTERIM_DATA_DIR = 'data/interim'
PROCESSED_DATA_DIR = 'data/processed'

# EIA Series IDs
EIA_SERIES = {
    'crude_oil': 'RWTC',  # WTI Crude Oil
    'regular_gasoline': 'EER_EPMRR_PF4_Y35NY_DPG',  # Regular Gasoline NY Harbor
    'conventional_gasoline': 'EER_EPMRU_PF4_Y35NY_DPG',  # Conventional Gasoline NY Harbor
    'diesel': 'EER_EPD2F_PF4_Y35NY_DPG'  # Ultra-Low Sulfur Diesel NY Harbor
}

# Yahoo Finance Tickers
YAHOO_TICKERS = {
    'crude_oil': 'CL=F',  # WTI Crude Oil
    'regular_gasoline': 'RB=F',  # RBOB Gasoline
    'heating_oil': 'HO=F'  # Heating Oil (proxy for diesel)
}

def fetch_all_eia_data(
    start_date: str,
    end_date: str,
    frequency: str = 'daily',
    api_key: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for all commodities from EIA API.
    
    Parameters
    ----------
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
    Dict[str, pd.DataFrame]
        Dictionary of DataFrames with commodity data
    """
    results = {}
    
    for commodity, series_id in EIA_SERIES.items():
        try:
            logger.info(f"Fetching {commodity} data from EIA API")
            json_data = fetch_eia_data(series_id, start_date, end_date, frequency, api_key)
            df = eia_json_to_dataframe(json_data)
            
            if not df.empty:
                results[commodity] = df
                logger.info(f"Successfully fetched {commodity} data: {len(df)} rows")
            else:
                logger.warning(f"No data returned for {commodity}")
                
        except Exception as e:
            logger.error(f"Error fetching {commodity} data: {e}")
    
    return results

def fetch_all_yahoo_data(
    start_date: str,
    end_date: str,
    interval: str = '1d'
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for all commodities from Yahoo Finance as a fallback.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    interval : str, optional
        Data interval ('1d', '1wk', '1mo'), by default '1d'
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of DataFrames with commodity data
    """
    results = {}
    
    for commodity, ticker in YAHOO_TICKERS.items():
        try:
            logger.info(f"Fetching {commodity} data from Yahoo Finance")
            df = fetch_yahoo_finance_data(ticker, start_date, end_date, interval)
            
            if not df.empty:
                # Extract just the closing price and rename
                close_col = f"{ticker}_Close"
                df = df[[close_col]].rename(columns={close_col: commodity})
                
                results[commodity] = df
                logger.info(f"Successfully fetched {commodity} data: {len(df)} rows")
            else:
                logger.warning(f"No data returned for {commodity}")
                
        except Exception as e:
            logger.error(f"Error fetching {commodity} data: {e}")
    
    return results

def save_commodity_data(data_dict: Dict[str, pd.DataFrame], directory: str) -> None:
    """
    Save commodity data to parquet files.
    
    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with commodity data
    directory : str
        Directory to save the files
    """
    for commodity, df in data_dict.items():
        try:
            file_path = os.path.join(directory, f"{commodity}.parquet")
            save_to_parquet(df, file_path)
            logger.info(f"Saved {commodity} data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving {commodity} data: {e}")

def run_data_acquisition(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_eia: bool = True,
    use_yahoo: bool = True
) -> None:
    """
    Run the data acquisition pipeline.
    
    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format, by default 5 years ago
    end_date : str, optional
        End date in YYYY-MM-DD format, by default today
    use_eia : bool, optional
        Whether to use EIA API, by default True
    use_yahoo : bool, optional
        Whether to use Yahoo Finance as fallback, by default True
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        # Default to 5 years of data
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    logger.info(f"Running data acquisition from {start_date} to {end_date}")
    
    # Create directories if they don't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Fetch data from EIA API
    eia_data = {}
    if use_eia:
        try:
            eia_data = fetch_all_eia_data(start_date, end_date)
            if eia_data:
                save_commodity_data(eia_data, RAW_DATA_DIR)
        except Exception as e:
            logger.error(f"Error in EIA data acquisition: {e}")
    
    # Fetch data from Yahoo Finance as fallback
    if use_yahoo and (not use_eia or not all(commodity in eia_data for commodity in YAHOO_TICKERS)):
        try:
            yahoo_data = fetch_all_yahoo_data(start_date, end_date)
            
            # For commodities that weren't fetched from EIA
            missing_commodities = {
                commodity: df for commodity, df in yahoo_data.items() 
                if commodity not in eia_data or eia_data[commodity].empty
            }
            
            if missing_commodities:
                save_commodity_data(missing_commodities, RAW_DATA_DIR)
        except Exception as e:
            logger.error(f"Error in Yahoo Finance data acquisition: {e}")
    
    logger.info("Data acquisition completed")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the data acquisition pipeline
    run_data_acquisition()
