"""
This module provides logging configuration for the Oil and Gas Market Optimization project.

It sets up a logging system that:
- Creates timestamped log files in a 'logs' directory
- Uses a standardized logging format with timestamp, line number, logger name, level, and message
- Sets the default logging level to INFO

The log files are named using the format: MM_DD_YYYY_HH_MM_SS.log
"""

import os
import logging
from datetime import datetime

def setup_logging():
    """Set up logging configuration and return a configured logger instance."""
    # Get the project root directory (parent of src)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Create a log file with the current date and time
    # The log file will be named in the format "MM_DD_YYYY_HH_MM_SS.log" 
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

    # Create logs directory using absolute path
    logs_path = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_path, exist_ok=True)

    # Create the full path for the log file
    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

    # Configure the logging system with standardized format
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    # Create and return a logger instance
    logger = logging.getLogger("OilGasMarketOptimization")
    return logger

# Create a logger instance that can be imported by other modules
logger = setup_logging()