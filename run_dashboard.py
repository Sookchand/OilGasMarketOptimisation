"""
Script to run the Streamlit dashboard for the Oil & Gas Market Optimization project.
"""

import os
import subprocess
import sys

def main():
    """Run the Streamlit dashboard."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Path to the dashboard app
    dashboard_path = os.path.join('src', 'dashboard', 'app.py')
    
    # Check if the dashboard app exists
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard app not found at {dashboard_path}")
        sys.exit(1)
    
    # Run the Streamlit app
    print(f"Starting Streamlit dashboard from {dashboard_path}")
    subprocess.run(['streamlit', 'run', dashboard_path])

if __name__ == "__main__":
    main()
