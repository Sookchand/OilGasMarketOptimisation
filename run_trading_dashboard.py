"""
Script to run the trading dashboard for the Oil & Gas Market Optimization project.
"""

import os
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/run_trading_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories."""
    directories = [
        'logs',
        'results/trading',
        'results/backtests',
        'results/monte_carlo'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_data():
    """Check if processed data exists."""
    data_dir = 'data/processed'

    if not os.path.exists(data_dir):
        logger.error(f"Processed data directory not found: {data_dir}")
        logger.info("Please run the data pipeline first.")
        return False

    # Check for commodity data files
    commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']
    found = False

    for commodity in commodities:
        parquet_path = os.path.join(data_dir, f"{commodity}.parquet")
        csv_path = os.path.join(data_dir, f"{commodity}.csv")

        if os.path.exists(parquet_path) or os.path.exists(csv_path):
            found = True
            logger.info(f"Found data for {commodity}")

    if not found:
        logger.error("No commodity data found.")
        logger.info("Please run the data pipeline first.")
        return False

    return True

def main():
    """Run the trading dashboard."""
    try:
        # Create directories
        create_directories()

        # Check data
        if not check_data():
            return

        # Try to use the full dashboard first
        dashboard_path = os.path.join('src', 'dashboard', 'trading_dashboard.py')

        # If the full dashboard doesn't exist, use the simplified one
        if not os.path.exists(dashboard_path):
            logger.warning(f"Full dashboard not found at {dashboard_path}")
            logger.info("Using simplified dashboard instead.")
            dashboard_path = "simple_trading_dashboard.py"

            if not os.path.exists(dashboard_path):
                logger.error(f"Simplified dashboard not found at {dashboard_path}")
                return

        # Run the Streamlit app
        logger.info(f"Starting trading dashboard from {dashboard_path}")
        subprocess.run(['streamlit', 'run', dashboard_path])

    except Exception as e:
        logger.error(f"Error running trading dashboard: {e}")

if __name__ == "__main__":
    main()
