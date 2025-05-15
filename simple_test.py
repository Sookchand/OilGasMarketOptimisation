#!/usr/bin/env python
"""
Simple test script for the Oil & Gas Market Optimization system.
"""

import os
import pandas as pd

def main():
    """Run a simple test."""
    print("Starting simple test...")
    
    # Check if data files exist
    data_dir = 'data/raw'
    commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']
    
    for commodity in commodities:
        csv_path = os.path.join(data_dir, f"{commodity}.csv")
        parquet_path = os.path.join(data_dir, f"{commodity}.parquet")
        
        if os.path.exists(csv_path):
            print(f"Found CSV file: {csv_path}")
        else:
            print(f"CSV file not found: {csv_path}")
        
        if os.path.exists(parquet_path):
            print(f"Found Parquet file: {parquet_path}")
            
            # Try to load the parquet file
            try:
                df = pd.read_parquet(parquet_path)
                print(f"  Successfully loaded {commodity} data: {len(df)} rows, columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"  Error loading {parquet_path}: {e}")
        else:
            print(f"Parquet file not found: {parquet_path}")
    
    # Check if insight files exist
    insights_dir = 'data/insights'
    if os.path.exists(insights_dir):
        insight_files = [f for f in os.listdir(insights_dir) if f.endswith('.md')]
        print(f"Found {len(insight_files)} insight files: {insight_files}")
    else:
        print(f"Insights directory not found: {insights_dir}")
    
    print("Simple test completed.")

if __name__ == "__main__":
    main()
