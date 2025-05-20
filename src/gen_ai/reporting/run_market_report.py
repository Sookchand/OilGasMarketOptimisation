"""
Script to demonstrate the usage of the Market Report Generator.
This script generates a comprehensive market report for specified commodities.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.gen_ai.reporting.market_report_generator import MarketReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_report.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate market report')
    
    parser.add_argument('--commodities', nargs='+', default=['crude_oil', 'natural_gas', 'gasoline', 'diesel'],
                        help='List of commodities to include in the report')
    
    parser.add_argument('--date', type=str, default=None,
                        help='Date for the report (YYYY-MM-DD), defaults to today')
    
    parser.add_argument('--lookback-days', type=int, default=30,
                        help='Number of days to look back for historical data')
    
    parser.add_argument('--forecast-days', type=int, default=14,
                        help='Number of days to forecast')
    
    parser.add_argument('--template', type=str, default='daily_report.html',
                        help='Name of the template to use')
    
    parser.add_argument('--output-format', choices=['html', 'pdf'], default='html',
                        help='Output format (html or pdf)')
    
    parser.add_argument('--output-dir', type=str, default='reports/market',
                        help='Directory to save the report')
    
    parser.add_argument('--template-dir', type=str, default='templates/reports',
                        help='Directory containing report templates')
    
    return parser.parse_args()

def main():
    """Generate market report."""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.template_dir, exist_ok=True)
    
    # Parse date
    if args.date:
        try:
            date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Using today's date.")
            date = datetime.now()
    else:
        date = datetime.now()
    
    # Initialize report generator
    report_generator = MarketReportGenerator(
        template_dir=args.template_dir,
        output_dir=args.output_dir
    )
    
    # Generate report
    report_path = report_generator.generate_daily_report(
        commodities=args.commodities,
        date=date,
        lookback_days=args.lookback_days,
        forecast_days=args.forecast_days,
        template_name=args.template,
        output_format=args.output_format
    )
    
    if report_path:
        print(f"\nMarket report generated successfully: {report_path}")
        print(f"Open this file in a web browser to view the report.")
    else:
        print("\nFailed to generate market report. Check logs for details.")

if __name__ == "__main__":
    main()
