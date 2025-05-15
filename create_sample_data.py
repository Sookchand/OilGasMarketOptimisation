#!/usr/bin/env python
"""
Create sample data for the Oil & Gas Market Optimization project.

This script generates synthetic price data for crude oil, regular gasoline,
conventional gasoline, and diesel with realistic trends, seasonality, and
correlations between the commodities.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/raw',
        'data/processed',
        'data/features',
        'data/insights',
        'data/chroma',
        'logs',
        'results/forecasting',
        'results/backtests',
        'results/model_selection',
        'results/monte_carlo',
        'results/trading'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def generate_correlated_random_walks(n_series, n_steps, correlation_matrix, volatilities, drift=0.0):
    """
    Generate correlated random walks for multiple time series.
    
    Parameters
    ----------
    n_series : int
        Number of time series to generate
    n_steps : int
        Number of time steps
    correlation_matrix : numpy.ndarray
        Correlation matrix (n_series x n_series)
    volatilities : list
        List of volatilities for each series
    drift : float or list, optional
        Drift term(s) for the random walks
    
    Returns
    -------
    numpy.ndarray
        Array of shape (n_steps, n_series) containing the correlated random walks
    """
    # Convert drift to array if it's a scalar
    if isinstance(drift, (int, float)):
        drift = np.ones(n_series) * drift
    
    # Generate correlated normal random variables
    cholesky_matrix = np.linalg.cholesky(correlation_matrix)
    uncorrelated_rvs = np.random.normal(0, 1, size=(n_steps, n_series))
    correlated_rvs = np.dot(uncorrelated_rvs, cholesky_matrix.T)
    
    # Scale by volatilities
    for i in range(n_series):
        correlated_rvs[:, i] *= volatilities[i]
    
    # Add drift
    for i in range(n_series):
        correlated_rvs[:, i] += drift[i]
    
    # Cumulative sum to get random walks
    random_walks = np.cumsum(correlated_rvs, axis=0)
    
    return random_walks

def add_seasonality(data, periods, amplitudes):
    """
    Add seasonality components to the data.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data of shape (n_steps, n_series)
    periods : list
        List of periods for each seasonality component
    amplitudes : list
        List of amplitudes for each seasonality component
    
    Returns
    -------
    numpy.ndarray
        Data with added seasonality
    """
    n_steps, n_series = data.shape
    result = data.copy()
    
    # Create time index
    t = np.arange(n_steps)
    
    # Add seasonality components
    for period, amplitude in zip(periods, amplitudes):
        seasonal_component = amplitude * np.sin(2 * np.pi * t / period)
        for i in range(n_series):
            # Vary the amplitude slightly for each series
            series_amplitude = amplitude * (0.8 + 0.4 * np.random.random())
            result[:, i] += series_amplitude * np.sin(2 * np.pi * t / period + np.random.random() * np.pi)
    
    return result

def add_price_shocks(data, n_shocks, max_shock_size, shock_decay):
    """
    Add random price shocks to the data.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data of shape (n_steps, n_series)
    n_shocks : int
        Number of shocks to add
    max_shock_size : float
        Maximum size of the shock as a percentage of the price
    shock_decay : float
        Rate at which the shock decays (between 0 and 1)
    
    Returns
    -------
    numpy.ndarray
        Data with added price shocks
    """
    n_steps, n_series = data.shape
    result = data.copy()
    
    for _ in range(n_shocks):
        # Random shock time
        shock_time = np.random.randint(0, n_steps // 2)  # Place shocks in first half for more interesting backtests
        
        # Random shock sizes for each series (correlated)
        base_shock = np.random.choice([-1, 1]) * max_shock_size * np.random.random()
        shock_sizes = []
        
        for i in range(n_series):
            # Add some randomness to the shock size for each series
            series_shock = base_shock * (0.8 + 0.4 * np.random.random())
            shock_sizes.append(series_shock)
        
        # Apply the shock and its decay
        for t in range(shock_time, n_steps):
            decay_factor = shock_decay ** (t - shock_time)
            for i in range(n_series):
                shock_effect = shock_sizes[i] * decay_factor * result[shock_time, i]
                result[t, i] += shock_effect
    
    return result

def create_sample_data():
    """Create sample data for all commodities."""
    logger.info("Creating sample data for oil and gas commodities...")
    
    # Parameters
    commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']
    n_series = len(commodities)
    
    # Date range (5 years of daily data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_steps = len(dates)
    
    # Starting prices
    base_prices = {
        'crude_oil': 60.0,  # USD per barrel
        'regular_gasoline': 2.5,  # USD per gallon
        'conventional_gasoline': 2.4,  # USD per gallon
        'diesel': 3.0  # USD per gallon
    }
    
    # Correlation matrix (crude oil price affects gasoline and diesel prices)
    correlation_matrix = np.array([
        [1.0, 0.8, 0.7, 0.6],  # Crude oil
        [0.8, 1.0, 0.9, 0.5],  # Regular gasoline
        [0.7, 0.9, 1.0, 0.5],  # Conventional gasoline
        [0.6, 0.5, 0.5, 1.0]   # Diesel
    ])
    
    # Volatilities (annualized)
    volatilities = [0.3, 0.25, 0.25, 0.2]  # Crude oil is most volatile
    
    # Convert to daily volatilities
    daily_volatilities = [vol / np.sqrt(252) for vol in volatilities]
    
    # Drift terms (annualized)
    drift = [0.05, 0.04, 0.04, 0.03]  # Slight upward trend
    
    # Convert to daily drift
    daily_drift = [d / 252 for d in drift]
    
    # Generate correlated random walks
    logger.info("Generating correlated price movements...")
    price_changes = generate_correlated_random_walks(
        n_series=n_series,
        n_steps=n_steps,
        correlation_matrix=correlation_matrix,
        volatilities=daily_volatilities,
        drift=daily_drift
    )
    
    # Add seasonality (annual and quarterly patterns)
    logger.info("Adding seasonality components...")
    price_changes = add_seasonality(
        data=price_changes,
        periods=[365, 90],  # Annual and quarterly seasonality
        amplitudes=[0.1, 0.05]  # 10% annual, 5% quarterly
    )
    
    # Add price shocks
    logger.info("Adding random price shocks...")
    price_changes = add_price_shocks(
        data=price_changes,
        n_shocks=10,
        max_shock_size=0.15,  # 15% maximum shock
        shock_decay=0.95  # 5% decay per day
    )
    
    # Convert to actual prices
    prices = {}
    for i, commodity in enumerate(commodities):
        # Start from base price and apply changes
        base_price = base_prices[commodity]
        commodity_prices = base_price * np.exp(price_changes[:, i])
        
        # Ensure no negative prices
        commodity_prices = np.maximum(commodity_prices, 0.1 * base_price)
        
        prices[commodity] = commodity_prices
    
    # Create and save DataFrames
    for i, commodity in enumerate(commodities):
        logger.info(f"Creating DataFrame for {commodity}...")
        
        # Create DataFrame with date index
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices[commodity]
        })
        df.set_index('Date', inplace=True)
        
        # Add some additional columns for more realistic data
        df['Open'] = df['Price'].shift(1).fillna(df['Price'])
        df['High'] = df['Price'] * (1 + 0.01 * np.random.random(size=len(df)))
        df['Low'] = df['Price'] * (1 - 0.01 * np.random.random(size=len(df)))
        df['Volume'] = np.random.randint(1000, 10000, size=len(df))
        
        # Add some missing values to test data cleaning
        mask = np.random.random(size=len(df)) < 0.01  # 1% missing data
        df.loc[mask, 'Price'] = np.nan
        
        # Save to CSV and parquet
        csv_path = f'data/raw/{commodity}.csv'
        parquet_path = f'data/raw/{commodity}.parquet'
        
        df.to_csv(csv_path)
        df.to_parquet(parquet_path)
        
        logger.info(f"Saved {commodity} data to {csv_path} and {parquet_path}")
    
    logger.info("Sample data creation complete!")

if __name__ == "__main__":
    create_directories()
    create_sample_data()
