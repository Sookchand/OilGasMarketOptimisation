#!/usr/bin/env python
"""
Full pipeline script for the Oil & Gas Market Optimization project.
This script runs all the pipelines in sequence for optimal results.
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/full_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a shell command and log the output."""
    logger.info(f"Running {description}...")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line.strip())
            logger.info(line.strip())
        
        # Wait for process to complete
        process.wait()
        
        # Check return code
        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"Error running {description}: {stderr}")
            return False
        
        logger.info(f"Successfully completed {description}")
        return True
    
    except Exception as e:
        logger.error(f"Exception running {description}: {e}")
        return False

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
        'results/trading',
        'docs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_data_pipeline(args):
    """Run the data processing pipeline."""
    command = f"python -m src.pipeline.main --commodities {' '.join(args.commodities)}"
    
    if args.start_date:
        command += f" --start-date {args.start_date}"
    
    if args.end_date:
        command += f" --end-date {args.end_date}"
    
    if args.steps:
        command += f" --steps {' '.join(args.steps)}"
    
    if args.forecast_horizon:
        command += f" --forecast-horizon {args.forecast_horizon}"
    
    if args.target_type:
        command += f" --target-type {args.target_type}"
    
    command += f" --model-type {args.model_type}"
    
    if not args.optimize_params:
        command += " --no-optimize-params"
    
    return run_command(command, "data processing pipeline")

def run_rag_pipeline(args):
    """Run the RAG system pipeline."""
    command = "python -m src.pipeline.rag_pipeline"
    
    if args.insights_dir:
        command += f" --insights-dir {args.insights_dir}"
    
    if args.processed_dir:
        command += f" --processed-dir {args.processed_dir}"
    
    if args.chroma_dir:
        command += f" --chroma-dir {args.chroma_dir}"
    
    if args.rag_steps:
        command += f" --steps {' '.join(args.rag_steps)}"
    
    return run_command(command, "RAG system pipeline")

def run_trading_pipeline(args):
    """Run the trading pipeline."""
    command = f"python -m src.pipeline.trading_pipeline --commodities {' '.join(args.commodities)}"
    
    if args.strategy != 'all':
        command += f" --strategy {args.strategy}"
    
    if args.initial_capital:
        command += f" --initial-capital {args.initial_capital}"
    
    if args.commission:
        command += f" --commission {args.commission}"
    
    if args.slippage:
        command += f" --slippage {args.slippage}"
    
    if not args.risk_analysis:
        command += " --no-risk-analysis"
    
    if not args.market_analysis:
        command += " --no-market-analysis"
    
    if not args.save_results:
        command += " --no-save-results"
    
    if args.output_dir:
        command += f" --output-dir {args.output_dir}"
    
    return run_command(command, "trading pipeline")

def run_dashboards(args):
    """Run the dashboards."""
    if args.run_main_dashboard:
        run_command("python run_dashboard.py", "main dashboard")
    
    if args.run_trading_dashboard:
        run_command("python run_trading_dashboard.py", "trading dashboard")

def main():
    """Main function to run all pipelines."""
    parser = argparse.ArgumentParser(description='Run the full pipeline for the Oil & Gas Market Optimization project')
    
    # General arguments
    parser.add_argument('--commodities', nargs='+', default=['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel'],
                        help='List of commodities to process')
    parser.add_argument('--skip-data-pipeline', action='store_true', help='Skip the data processing pipeline')
    parser.add_argument('--skip-rag-pipeline', action='store_true', help='Skip the RAG system pipeline')
    parser.add_argument('--skip-trading-pipeline', action='store_true', help='Skip the trading pipeline')
    
    # Data pipeline arguments
    parser.add_argument('--start-date', help='Start date for data acquisition')
    parser.add_argument('--end-date', help='End date for data acquisition')
    parser.add_argument('--steps', nargs='+', choices=['acquisition', 'cleaning', 'feature_engineering', 'modeling'],
                        help='Pipeline steps to run')
    parser.add_argument('--forecast-horizon', type=int, default=1, help='Forecast horizon in days')
    parser.add_argument('--target-type', choices=['return', 'price', 'direction'], default='return',
                        help='Type of target variable')
    parser.add_argument('--model-type', choices=['all', 'arima', 'xgboost', 'lstm'], default='all',
                        help='Type of model to train')
    parser.add_argument('--no-optimize-params', action='store_false', dest='optimize_params',
                        help='Disable model parameter optimization')
    
    # RAG pipeline arguments
    parser.add_argument('--insights-dir', default='data/insights', help='Directory containing insight files')
    parser.add_argument('--processed-dir', default='data/processed', help='Directory for processed data')
    parser.add_argument('--chroma-dir', default='data/chroma', help='Directory for ChromaDB')
    parser.add_argument('--rag-steps', nargs='+', choices=['load', 'index', 'qa'],
                        help='RAG pipeline steps to run')
    
    # Trading pipeline arguments
    parser.add_argument('--strategy', choices=['ma_crossover', 'macd', 'rsi', 'bollinger', 'donchian', 'atr', 'all'],
                        default='all', help='Trading strategy to use')
    parser.add_argument('--initial-capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--slippage', type=float, default=0.001, help='Slippage rate')
    parser.add_argument('--no-risk-analysis', action='store_false', dest='risk_analysis',
                        help='Disable risk analysis')
    parser.add_argument('--no-market-analysis', action='store_false', dest='market_analysis',
                        help='Disable market analysis')
    parser.add_argument('--no-save-results', action='store_false', dest='save_results',
                        help='Disable saving results')
    parser.add_argument('--output-dir', default='results/trading', help='Output directory')
    
    # Dashboard arguments
    parser.add_argument('--run-main-dashboard', action='store_true', help='Run the main dashboard after pipelines')
    parser.add_argument('--run-trading-dashboard', action='store_true', help='Run the trading dashboard after pipelines')
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Run pipelines
    success = True
    
    if not args.skip_data_pipeline:
        success = run_data_pipeline(args)
        if not success:
            logger.error("Data pipeline failed. Stopping execution.")
            return
    
    if not args.skip_rag_pipeline:
        success = run_rag_pipeline(args)
        if not success:
            logger.error("RAG pipeline failed. Stopping execution.")
            return
    
    if not args.skip_trading_pipeline:
        success = run_trading_pipeline(args)
        if not success:
            logger.error("Trading pipeline failed. Stopping execution.")
            return
    
    # Run dashboards if requested
    if success and (args.run_main_dashboard or args.run_trading_dashboard):
        run_dashboards(args)
    
    logger.info("Full pipeline completed successfully!")

if __name__ == "__main__":
    main()
