import sys
import os

# Add the project root directory to sys.path before any local imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from src.logging.logger import logger

"""
Custom exception class for Oil and Gas Market Optimization project.
This class provides detailed error information including the file name,
line number, and error message where the exception occurred.
"""

class OilGasException(Exception):
    def __init__(self, error_message, error_detail: sys, filename="N/A", lineno="N/A"):
        """
        Initialize the custom exception with detailed error information.
        
        Args:
            error_message: The error message to be displayed
            error_detail: The sys module containing exception info
            filename: Optional default filename if traceback is not available
            lineno: Optional default line number if traceback is not available
        """
        super().__init__(error_message)
        _, _, exc_tb = error_detail.exc_info()
        
        # Extracting the filename and line number from the traceback
        if exc_tb is None:
            logger.warning("Exception traceback is None. Using default filename and line number.")
        self.error_message = error_message
        self.lineno = exc_tb.tb_lineno if exc_tb else lineno  # Use provided lineno if exc_tb is None
        self.filename = exc_tb.tb_frame.f_code.co_filename if exc_tb else filename  # Use provided filename if exc_tb is None

    def __str__(self):
        return f"Error occurred in python script name [{self.filename}] line number [{self.lineno}] error message [{self.error_message}]"


# if __name__=='__main__':
#     try:
#         logger.info("Enter the try block")
#         a = 1/0
#         print("This will not be printed", a)
#     except Exception as e:
#         raise OilGasException(e, sys)