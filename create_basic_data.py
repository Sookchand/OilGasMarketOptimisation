#!/usr/bin/env python
"""
Create basic sample data for the Oil & Gas Market Optimization project.
This is a simplified version that just creates CSV files with random data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def main():
    """Create basic sample data."""
    print("Creating basic sample data...")
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results/forecasting', exist_ok=True)
    
    print("Created directories")
    
    # Generate dates (1 year of daily data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Commodities
    commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']
    
    # Generate data for each commodity
    for commodity in commodities:
        print(f"Creating data for {commodity}...")
        
        # Generate random price data
        np.random.seed(42 + commodities.index(commodity))  # Different seed for each commodity
        prices = 50 + 10 * np.random.randn(len(dates))  # Random prices around $50
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices
        })
        df.set_index('Date', inplace=True)
        
        # Save to CSV
        csv_path = f'data/raw/{commodity}.csv'
        df.to_csv(csv_path)
        print(f"  Saved to {csv_path}")
        
        # Process the data (simple moving averages)
        df['MA_10'] = df['Price'].rolling(window=10).mean()
        df['MA_30'] = df['Price'].rolling(window=30).mean()
        
        # Save processed data
        processed_path = f'data/processed/{commodity}.csv'
        df.to_csv(processed_path)
        print(f"  Saved processed data to {processed_path}")
    
    print("Basic sample data creation complete!")

if __name__ == "__main__":
    main()
