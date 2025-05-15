#!/usr/bin/env python
"""
Check the status of the Oil & Gas Market Optimization system.
"""

import os
import sys
import importlib
import traceback

def check_directory(directory):
    """Check if a directory exists and list its contents."""
    print(f"Checking directory: {directory}")
    if os.path.exists(directory):
        print(f"  Directory exists")
        try:
            contents = os.listdir(directory)
            print(f"  Contents: {contents}")
        except Exception as e:
            print(f"  Error listing contents: {e}")
    else:
        print(f"  Directory does not exist")

def check_file(file_path):
    """Check if a file exists and print its size."""
    print(f"Checking file: {file_path}")
    if os.path.exists(file_path):
        print(f"  File exists")
        try:
            size = os.path.getsize(file_path)
            print(f"  Size: {size} bytes")
        except Exception as e:
            print(f"  Error getting file size: {e}")
    else:
        print(f"  File does not exist")

def check_module(module_name):
    """Check if a module can be imported."""
    print(f"Checking module: {module_name}")
    try:
        module = importlib.import_module(module_name)
        print(f"  Module imported successfully")
        return module
    except Exception as e:
        print(f"  Error importing module: {e}")
        traceback.print_exc()
        return None

def main():
    """Run system checks."""
    print("Starting system checks...")
    
    # Check directories
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'data/insights',
        'logs',
        'results',
        'results/forecasting',
        'src',
        'src/trading',
        'src/risk',
        'src/rag'
    ]
    
    for directory in directories:
        check_directory(directory)
    
    # Check files
    files = [
        'data/raw/crude_oil.csv',
        'data/raw/crude_oil.parquet',
        'data/insights/crude_oil_insights.md',
        'data/insights/regular_gasoline_insights.md'
    ]
    
    for file_path in files:
        check_file(file_path)
    
    # Check modules
    modules = [
        'pandas',
        'numpy',
        'matplotlib',
        'streamlit',
        'plotly',
        'sentence_transformers',
        'chromadb'
    ]
    
    for module_name in modules:
        check_module(module_name)
    
    # Check Python path
    print(f"Python path: {sys.path}")
    
    print("System checks completed.")

if __name__ == "__main__":
    main()
