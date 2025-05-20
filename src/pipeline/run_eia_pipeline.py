"""
Run the EIA price drivers data acquisition and feature engineering pipeline.
This script fetches EIA price drivers data, engineers features, and integrates them with commodity data.
"""

import os
import logging
import argparse
from datetime import datetime, timedelta

from src.pipeline.data.eia_price_drivers import run_price_drivers_acquisition
from src.pipeline.data.price_drivers_features import run_price_drivers_feature_engineering
from src.pipeline.data.feature_engineering import run_feature_engineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/eia_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the EIA price drivers pipeline')
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for EIA data (YYYY-MM-DD)',
        default=None
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for EIA data (YYYY-MM-DD)',
        default=None
    )
    
    parser.add_argument(
        '--frequency',
        type=str,
        choices=['monthly', 'quarterly', 'annual'],
        default='monthly',
        help='Data frequency'
    )
    
    parser.add_argument(
        '--commodities',
        type=str,
        nargs='+',
        default=['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel'],
        help='Commodities to process'
    )
    
    parser.add_argument(
        '--forecast-horizon',
        type=int,
        default=30,
        help='Forecast horizon in days'
    )
    
    parser.add_argument(
        '--target-type',
        type=str,
        choices=['return', 'price', 'direction'],
        default='return',
        help='Type of target variable'
    )
    
    parser.add_argument(
        '--skip-eia-acquisition',
        action='store_true',
        help='Skip EIA data acquisition step'
    )
    
    parser.add_argument(
        '--skip-feature-engineering',
        action='store_true',
        help='Skip feature engineering step'
    )
    
    return parser.parse_args()

def run_pipeline(args):
    """
    Run the complete EIA price drivers pipeline.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/raw/price_drivers', exist_ok=True)
    os.makedirs('data/processed/price_drivers', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    
    logger.info("Starting EIA price drivers pipeline")
    
    # Step 1: Acquire EIA price drivers data
    if not args.skip_eia_acquisition:
        logger.info("Running EIA price drivers data acquisition")
        run_price_drivers_acquisition(
            start_date=args.start_date,
            end_date=args.end_date,
            frequency=args.frequency
        )
    else:
        logger.info("Skipping EIA data acquisition step")
    
    # Step 2: Engineer price drivers features
    logger.info("Running price drivers feature engineering")
    run_price_drivers_feature_engineering(args.commodities)
    
    # Step 3: Run feature engineering with price drivers
    if not args.skip_feature_engineering:
        logger.info("Running feature engineering with price drivers")
        run_feature_engineering(
            commodities=args.commodities,
            forecast_horizon=args.forecast_horizon,
            target_type=args.target_type,
            add_price_drivers=True
        )
    else:
        logger.info("Skipping feature engineering step")
    
    logger.info("EIA price drivers pipeline completed")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Run the pipeline
    run_pipeline(args)
