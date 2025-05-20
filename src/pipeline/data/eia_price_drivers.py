"""
EIA price drivers data acquisition module.
This module fetches data on crude oil price drivers from the EIA API.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import requests

from src.utils.data_utils import save_to_parquet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/eia_price_drivers.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
RAW_DATA_DIR = 'data/raw/price_drivers'
PROCESSED_DATA_DIR = 'data/processed/price_drivers'

# EIA Series IDs for price drivers
EIA_PRICE_DRIVERS = {
    # Supply factors
    'non_opec_production': 'STEO.COPR_NONOPEC.M',  # Non-OPEC Production
    'us_production': 'STEO.COPR_US.M',  # US Production
    'global_production': 'STEO.COPR_WORLD.M',  # Global Production
    
    # OPEC factors
    'opec_production': 'STEO.COPR_OPEC.M',  # OPEC Production
    'opec_spare_capacity': 'STEO.OPEC_SPARE_CAPACITY_WORLD.M',  # OPEC Spare Capacity
    
    # Balance factors
    'oecd_inventories': 'STEO.OECD_STOCKS.M',  # OECD Inventories
    'global_inventories': 'STEO.STOCKS_WORLD.M',  # Global Inventories
    
    # Demand factors
    'oecd_consumption': 'STEO.COPC_OECD.M',  # OECD Consumption
    'non_oecd_consumption': 'STEO.COPC_NONOECD.M',  # Non-OECD Consumption
    'global_consumption': 'STEO.COPC_WORLD.M',  # Global Consumption
    'us_consumption': 'STEO.COPC_US.M',  # US Consumption
    
    # Financial factors
    'wti_price': 'STEO.WTIPUUS.M',  # WTI Price
    'brent_price': 'STEO.BREPUUS.M'  # Brent Price
}

def fetch_eia_price_driver(
    series_id: str,
    start_date: str,
    end_date: str,
    frequency: str = 'monthly',
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch price driver data from EIA API with error handling and rate limiting.

    Parameters
    ----------
    series_id : str
        The EIA series ID to fetch
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    frequency : str, optional
        Data frequency ('monthly', 'quarterly', 'annual'), by default 'monthly'
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

    # Extract the category from the series ID (e.g., 'STEO' from 'STEO.COPR_NONOPEC.M')
    category = series_id.split('.')[0].lower()
    
    # Construct the URL based on the category
    url = f"https://api.eia.gov/v2/{category}/data/?api_key={api_key}&frequency={frequency}"
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

def eia_json_to_dataframe(json_data: Dict[str, Any], driver_name: str) -> pd.DataFrame:
    """
    Convert EIA API JSON response to a pandas DataFrame.

    Parameters
    ----------
    json_data : Dict[str, Any]
        JSON response from the EIA API
    driver_name : str
        Name of the price driver

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and price driver values
    """
    try:
        # Extract the data from the response
        data = json_data.get('response', {}).get('data', [])

        if not data:
            logger.warning(f"No data found in EIA API response for {driver_name}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Convert period to datetime
        df['date'] = pd.to_datetime(df['period'])

        # Set date as index and select only value column
        df = df.set_index('date')[['value']]

        # Rename column to the driver name
        df = df.rename(columns={'value': driver_name})

        return df

    except Exception as e:
        logger.error(f"Error converting EIA JSON to DataFrame for {driver_name}: {e}")
        raise

def fetch_all_price_drivers(
    start_date: str,
    end_date: str,
    frequency: str = 'monthly',
    api_key: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch all price drivers from EIA API.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    frequency : str, optional
        Data frequency ('monthly', 'quarterly', 'annual'), by default 'monthly'
    api_key : str, optional
        EIA API key, by default None (will use environment variable)
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of DataFrames with price driver data
    """
    results = {}
    
    for driver_name, series_id in EIA_PRICE_DRIVERS.items():
        try:
            logger.info(f"Fetching {driver_name} data from EIA API")
            json_data = fetch_eia_price_driver(series_id, start_date, end_date, frequency, api_key)
            df = eia_json_to_dataframe(json_data, driver_name)
            
            if not df.empty:
                results[driver_name] = df
                logger.info(f"Successfully fetched {driver_name} data: {len(df)} rows")
            else:
                logger.warning(f"No data returned for {driver_name}")
                
        except Exception as e:
            logger.error(f"Error fetching {driver_name} data: {e}")
    
    return results

def save_price_driver_data(data_dict: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Save price driver data to parquet files.
    
    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with price driver data
    output_dir : str
        Directory to save the data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for driver_name, df in data_dict.items():
        if not df.empty:
            file_path = os.path.join(output_dir, f"{driver_name}.parquet")
            save_to_parquet(df, file_path)
            logger.info(f"Saved {driver_name} data to {file_path}")

def merge_price_drivers(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all price drivers into a single DataFrame.
    
    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with price driver data
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all price drivers
    """
    if not data_dict:
        logger.warning("No price driver data to merge")
        return pd.DataFrame()
    
    # Start with the first DataFrame
    merged_df = next(iter(data_dict.values())).copy()
    
    # Merge the rest
    for driver_name, df in data_dict.items():
        if df is not merged_df:  # Skip the first one
            merged_df = merged_df.join(df, how='outer')
    
    logger.info(f"Merged {len(data_dict)} price drivers into a single DataFrame")
    return merged_df

def run_price_drivers_acquisition(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    frequency: str = 'monthly'
) -> None:
    """
    Run the price drivers acquisition pipeline.
    
    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format, by default 10 years ago
    end_date : str, optional
        End date in YYYY-MM-DD format, by default today
    frequency : str, optional
        Data frequency ('monthly', 'quarterly', 'annual'), by default 'monthly'
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        # Default to 10 years of data
        start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    
    logger.info(f"Running price drivers acquisition from {start_date} to {end_date}")
    
    # Create directories if they don't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Fetch price drivers data
    try:
        price_drivers_data = fetch_all_price_drivers(start_date, end_date, frequency)
        
        if price_drivers_data:
            # Save individual price drivers
            save_price_driver_data(price_drivers_data, RAW_DATA_DIR)
            
            # Merge and save combined dataset
            merged_df = merge_price_drivers(price_drivers_data)
            if not merged_df.empty:
                merged_file_path = os.path.join(PROCESSED_DATA_DIR, "all_price_drivers.parquet")
                save_to_parquet(merged_df, merged_file_path)
                logger.info(f"Saved merged price drivers data to {merged_file_path}")
        
    except Exception as e:
        logger.error(f"Error in price drivers acquisition: {e}")
    
    logger.info("Price drivers acquisition completed")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the price drivers acquisition pipeline
    run_price_drivers_acquisition()
