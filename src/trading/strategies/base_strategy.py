"""
Base trading strategy class.
This module defines the base class for all trading strategies.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_strategies.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    """
    
    def __init__(
        self, 
        name: str,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the strategy.
        
        Parameters
        ----------
        name : str
            Strategy name
        parameters : Dict[str, Any], optional
            Strategy parameters, by default None
        """
        self.name = name
        self.parameters = parameters or {}
        self.signals = None
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price and feature data
        
        Returns
        -------
        pd.DataFrame
            DataFrame with signals
        """
        pass
    
    def get_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Convert signals to positions.
        
        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame with signals
        
        Returns
        -------
        pd.DataFrame
            DataFrame with positions
        """
        # Default implementation: positions are the same as signals
        return signals.copy()
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the strategy on the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price and feature data
        
        Returns
        -------
        pd.DataFrame
            DataFrame with positions
        """
        logger.info(f"Running strategy: {self.name}")
        
        # Generate signals
        self.signals = self.generate_signals(data)
        
        # Convert signals to positions
        positions = self.get_positions(self.signals)
        
        logger.info(f"Generated {len(positions)} positions")
        
        return positions
    
    def __str__(self) -> str:
        """
        String representation of the strategy.
        
        Returns
        -------
        str
            String representation
        """
        return f"{self.name} Strategy"
    
    def __repr__(self) -> str:
        """
        Representation of the strategy.
        
        Returns
        -------
        str
            Representation
        """
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"
