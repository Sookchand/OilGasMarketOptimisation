#!/usr/bin/env python
"""
Simplified data processing pipeline for the Oil & Gas Market Optimization project.
This script focuses only on the data processing part.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    """Run the data processing pipeline."""
    print("Starting data processing pipeline...")
    
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results/forecasting', exist_ok=True)
    
    # List of commodities
    commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']
    
    # Process each commodity
    for commodity in commodities:
        print(f"Processing {commodity}...")
        
        # Load raw data
        raw_path = f'data/raw/{commodity}.parquet'
        if os.path.exists(raw_path):
            try:
                # Load data
                df = pd.read_parquet(raw_path)
                print(f"  Loaded {len(df)} rows from {raw_path}")
                
                # Basic data cleaning
                # Fill missing values
                if 'Price' in df.columns:
                    df['Price'] = df['Price'].interpolate(method='linear')
                
                # Calculate returns
                if 'Price' in df.columns:
                    df['Returns'] = df['Price'].pct_change()
                
                # Calculate moving averages
                if 'Price' in df.columns:
                    df['MA_10'] = df['Price'].rolling(window=10).mean()
                    df['MA_30'] = df['Price'].rolling(window=30).mean()
                
                # Calculate volatility
                if 'Returns' in df.columns:
                    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
                
                # Drop NaN values
                df = df.dropna()
                
                # Save processed data
                processed_path = f'data/processed/{commodity}.parquet'
                df.to_parquet(processed_path)
                print(f"  Saved processed data to {processed_path}")
                
                # Create a simple plot
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df['Price'], label='Price')
                if 'MA_10' in df.columns and 'MA_30' in df.columns:
                    plt.plot(df.index, df['MA_10'], label='10-day MA')
                    plt.plot(df.index, df['MA_30'], label='30-day MA')
                plt.title(f"{commodity.replace('_', ' ').title()} Price")
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True)
                
                # Save the plot
                plot_path = f'results/forecasting/{commodity}_price.png'
                plt.savefig(plot_path)
                plt.close()
                print(f"  Saved price plot to {plot_path}")
                
            except Exception as e:
                print(f"  Error processing {commodity}: {e}")
        else:
            print(f"  Raw data file not found: {raw_path}")
    
    print("Data processing pipeline completed.")

if __name__ == "__main__":
    main()
