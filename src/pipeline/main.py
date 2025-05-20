"""
Main pipeline script for the Oil & Gas Market Optimization project.
This script orchestrates the entire data processing and modeling pipeline.
"""

import os
import logging
import argparse
from datetime import datetime, timedelta

from src.pipeline.data.data_acquisition import run_data_acquisition
from src.pipeline.data.data_cleaning import run_data_cleaning
from src.pipeline.data.feature_engineering import run_feature_engineering
from src.pipeline.data.eia_price_drivers import run_price_drivers_acquisition
from src.pipeline.data.price_drivers_features import run_price_drivers_feature_engineering
from src.pipeline.run_eia_pipeline import run_pipeline as run_eia_pipeline
from src.models.forecasting.arima_forecaster import train_arima_model
from src.models.forecasting.xgboost_forecaster import train_xgboost_model
from src.models.forecasting.lstm_forecaster import train_lstm_model
from src.models.forecasting.price_drivers_forecaster import train_price_drivers_model
from src.models.forecasting.model_selection import select_best_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline(
    commodities=None,
    start_date=None,
    end_date=None,
    steps=None,
    forecast_horizon=1,
    target_type='return',
    model_type='all',
    optimize_params=True,
    use_price_drivers=True
):
    """
    Run the complete data processing and modeling pipeline.

    Parameters
    ----------
    commodities : list, optional
        List of commodities to process, by default None (all available)
    start_date : str, optional
        Start date for data acquisition, by default None (5 years ago)
    end_date : str, optional
        End date for data acquisition, by default None (today)
    steps : list, optional
        List of pipeline steps to run, by default None (all steps)
    forecast_horizon : int, optional
        Forecast horizon in days, by default 1
    target_type : str, optional
        Type of target variable ('return', 'price', 'direction'), by default 'return'
    model_type : str, optional
        Type of model to train ('all', 'arima', 'xgboost', 'lstm', 'price_drivers'), by default 'all'
    optimize_params : bool, optional
        Whether to optimize model parameters, by default True
    use_price_drivers : bool, optional
        Whether to use EIA price drivers data, by default True
    """
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/raw/price_drivers', exist_ok=True)
    os.makedirs('data/interim', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/processed/price_drivers', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    os.makedirs('models/forecasting', exist_ok=True)

    # Default commodities if not specified
    if commodities is None:
        commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']

    # Default steps if not specified
    if steps is None:
        steps = ['acquisition', 'cleaning', 'feature_engineering', 'modeling']

    logger.info(f"Starting pipeline for commodities: {commodities}")
    logger.info(f"Pipeline steps: {steps}")

    # Step 1: Data Acquisition
    if 'acquisition' in steps:
        logger.info("Step 1: Data Acquisition")
        run_data_acquisition(start_date=start_date, end_date=end_date)

        # Acquire EIA price drivers data if requested
        if use_price_drivers:
            logger.info("Step 1b: EIA Price Drivers Acquisition")
            run_eia_pipeline(
                skip_eia_acquisition=False,
                skip_feature_engineering=True,
                start_date=start_date,
                end_date=end_date,
                commodities=commodities
            )

    # Step 2: Data Cleaning
    if 'cleaning' in steps:
        logger.info("Step 2: Data Cleaning")
        run_data_cleaning(commodities=commodities)

    # Step 3: Feature Engineering
    if 'feature_engineering' in steps:
        logger.info("Step 3: Feature Engineering")

        # Process EIA price drivers data if requested
        if use_price_drivers:
            logger.info("Step 3b: EIA Price Drivers Feature Engineering")
            run_eia_pipeline(
                skip_eia_acquisition=True,
                skip_feature_engineering=False,
                commodities=commodities
            )

        # Run standard feature engineering
        run_feature_engineering(
            commodities=commodities,
            forecast_horizon=forecast_horizon,
            target_type=target_type,
            add_price_drivers=use_price_drivers
        )

    # Step 4: Modeling
    if 'modeling' in steps:
        logger.info("Step 4: Modeling")

        # Create results directory if it doesn't exist
        os.makedirs('results/model_selection', exist_ok=True)

        for commodity in commodities:
            if model_type == 'all':
                logger.info(f"Training and selecting best model for {commodity}")

                # Include price_drivers model if requested
                if use_price_drivers:
                    models_to_train = ['arima', 'xgboost', 'lstm', 'price_drivers']
                else:
                    models_to_train = ['arima', 'xgboost', 'lstm']

                # Train and select best model
                best_model_name, _ = select_best_model(
                    commodity=commodity,
                    models_to_train=models_to_train,
                    optimize_params=optimize_params,
                    train_new_models=True
                )

                logger.info(f"Best model for {commodity}: {best_model_name}")

            elif model_type == 'arima':
                logger.info(f"Training ARIMA model for {commodity}")
                train_arima_model(
                    commodity=commodity,
                    optimize_order=optimize_params
                )

            elif model_type == 'xgboost':
                logger.info(f"Training XGBoost model for {commodity}")
                train_xgboost_model(
                    commodity=commodity,
                    optimize_params=optimize_params
                )

            elif model_type == 'lstm':
                logger.info(f"Training LSTM model for {commodity}")
                train_lstm_model(
                    commodity=commodity
                )

            elif model_type == 'price_drivers':
                logger.info(f"Training Price Drivers model for {commodity}")
                train_price_drivers_model(
                    commodity=commodity,
                    optimize_params=optimize_params
                )

    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Oil & Gas Market Optimization pipeline')

    parser.add_argument('--commodities', nargs='+', help='List of commodities to process')
    parser.add_argument('--start-date', help='Start date for data acquisition (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for data acquisition (YYYY-MM-DD)')
    parser.add_argument('--steps', nargs='+', choices=['acquisition', 'cleaning', 'feature_engineering', 'modeling'],
                        help='Pipeline steps to run')
    parser.add_argument('--forecast-horizon', type=int, default=1, help='Forecast horizon in days')
    parser.add_argument('--target-type', choices=['return', 'price', 'direction'], default='return',
                        help='Type of target variable')
    parser.add_argument('--model-type', choices=['all', 'arima', 'xgboost', 'lstm', 'price_drivers'], default='all',
                        help='Type of model to train')
    parser.add_argument('--no-optimize-params', action='store_false', dest='optimize_params',
                        help='Disable model parameter optimization')
    parser.add_argument('--no-price-drivers', action='store_false', dest='use_price_drivers',
                        help='Disable EIA price drivers data')

    args = parser.parse_args()

    run_pipeline(
        commodities=args.commodities,
        start_date=args.start_date,
        end_date=args.end_date,
        steps=args.steps,
        forecast_horizon=args.forecast_horizon,
        target_type=args.target_type,
        model_type=args.model_type,
        optimize_params=args.optimize_params,
        use_price_drivers=args.use_price_drivers
    )
