"""
Value at Risk (VaR) calculator.
This module implements various methods for calculating Value at Risk.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/var_calculator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VaRCalculator:
    """
    Value at Risk (VaR) calculator.
    """

    def __init__(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ):
        """
        Initialize the VaR calculator.

        Parameters
        ----------
        returns : pd.Series
            Series of returns
        confidence_level : float, optional
            Confidence level, by default 0.95
        time_horizon : int, optional
            Time horizon in days, by default 1
        """
        self.returns = returns
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon

    def historical_var(self, confidence_level: Optional[float] = None) -> float:
        """
        Calculate historical VaR.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level, by default None (use the instance's confidence_level)

        Returns
        -------
        float
            Value at Risk
        """
        # Use provided confidence level or instance's confidence level
        cl = confidence_level if confidence_level is not None else self.confidence_level

        # Sort returns
        sorted_returns = self.returns.sort_values()

        # Calculate percentile
        var_percentile = 1 - cl

        # Calculate VaR
        var = -sorted_returns.quantile(var_percentile)

        # Scale for time horizon
        var = var * np.sqrt(self.time_horizon)

        logger.info(f"Historical VaR ({cl:.0%}, {self.time_horizon}-day): {var:.4f}")

        return var

    def parametric_var(self, confidence_level: Optional[float] = None) -> float:
        """
        Calculate parametric VaR assuming normal distribution.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level, by default None (use the instance's confidence_level)

        Returns
        -------
        float
            Value at Risk
        """
        # Use provided confidence level or instance's confidence level
        cl = confidence_level if confidence_level is not None else self.confidence_level

        # Calculate mean and standard deviation
        mu = self.returns.mean()
        sigma = self.returns.std()

        # Calculate z-score
        z = stats.norm.ppf(1 - cl)

        # Calculate VaR
        var = -(mu * self.time_horizon + z * sigma * np.sqrt(self.time_horizon))

        logger.info(f"Parametric VaR ({cl:.0%}, {self.time_horizon}-day): {var:.4f}")

        return var

    def monte_carlo_var(
        self,
        num_simulations: int = 10000,
        seed: Optional[int] = None,
        confidence_level: Optional[float] = None
    ) -> float:
        """
        Calculate Monte Carlo VaR.

        Parameters
        ----------
        num_simulations : int, optional
            Number of simulations, by default 10000
        seed : int, optional
            Random seed, by default None
        confidence_level : float, optional
            Confidence level, by default None (use the instance's confidence_level)

        Returns
        -------
        float
            Value at Risk
        """
        # Use provided confidence level or instance's confidence level
        cl = confidence_level if confidence_level is not None else self.confidence_level

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Calculate mean and standard deviation
        mu = self.returns.mean()
        sigma = self.returns.std()

        # Generate random returns
        random_returns = np.random.normal(
            mu * self.time_horizon,
            sigma * np.sqrt(self.time_horizon),
            num_simulations
        )

        # Sort returns
        sorted_returns = np.sort(random_returns)

        # Calculate VaR
        var_index = int(num_simulations * (1 - cl))
        var = -sorted_returns[var_index]

        logger.info(f"Monte Carlo VaR ({cl:.0%}, {self.time_horizon}-day): {var:.4f}")

        return var

    def calculate_all(self, confidence_level: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate VaR using all methods.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level, by default None (use the instance's confidence_level)

        Returns
        -------
        Dict[str, float]
            Dictionary of VaR values
        """
        historical = self.historical_var(confidence_level=confidence_level)
        parametric = self.parametric_var(confidence_level=confidence_level)
        monte_carlo = self.monte_carlo_var(confidence_level=confidence_level)

        return {
            'historical': historical,
            'parametric': parametric,
            'monte_carlo': monte_carlo
        }

    def calculate_expected_shortfall(
        self,
        method: str = 'historical',
        confidence_level: Optional[float] = None
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Parameters
        ----------
        method : str, optional
            Method to use ('historical', 'parametric', 'monte_carlo'), by default 'historical'
        confidence_level : float, optional
            Confidence level, by default None (use the instance's confidence_level)

        Returns
        -------
        float
            Expected Shortfall
        """
        # Use provided confidence level or instance's confidence level
        cl = confidence_level if confidence_level is not None else self.confidence_level

        if method == 'historical':
            # Sort returns
            sorted_returns = self.returns.sort_values()

            # Calculate percentile
            var_percentile = 1 - cl

            # Get returns beyond VaR
            var_threshold = sorted_returns.quantile(var_percentile)
            beyond_var = sorted_returns[sorted_returns <= var_threshold]

            # Calculate Expected Shortfall
            es = -beyond_var.mean()

            # Scale for time horizon
            es = es * np.sqrt(self.time_horizon)

        elif method == 'parametric':
            # Calculate mean and standard deviation
            mu = self.returns.mean()
            sigma = self.returns.std()

            # Calculate z-score
            z = stats.norm.ppf(1 - cl)

            # Calculate Expected Shortfall
            es = -(mu * self.time_horizon + sigma * np.sqrt(self.time_horizon) * stats.norm.pdf(z) / (1 - cl))

        elif method == 'monte_carlo':
            # Set random seed for reproducibility
            np.random.seed(42)

            # Calculate mean and standard deviation
            mu = self.returns.mean()
            sigma = self.returns.std()

            # Generate random returns
            num_simulations = 10000
            random_returns = np.random.normal(
                mu * self.time_horizon,
                sigma * np.sqrt(self.time_horizon),
                num_simulations
            )

            # Sort returns
            sorted_returns = np.sort(random_returns)

            # Calculate VaR index
            var_index = int(num_simulations * (1 - cl))

            # Calculate Expected Shortfall
            es = -sorted_returns[:var_index].mean()

        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Expected Shortfall ({method}, {cl:.0%}, {self.time_horizon}-day): {es:.4f}")

        return es

def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    time_horizon: int = 1,
    method: str = 'all'
) -> Union[float, Dict[str, float]]:
    """
    Calculate Value at Risk.

    Parameters
    ----------
    returns : pd.Series
        Series of returns
    confidence_level : float, optional
        Confidence level, by default 0.95
    time_horizon : int, optional
        Time horizon in days, by default 1
    method : str, optional
        Method to use ('historical', 'parametric', 'monte_carlo', 'all'), by default 'all'

    Returns
    -------
    Union[float, Dict[str, float]]
        Value at Risk or dictionary of VaR values
    """
    # Create VaR calculator
    calculator = VaRCalculator(returns, confidence_level, time_horizon)

    # Calculate VaR
    if method == 'historical':
        return calculator.historical_var(confidence_level=confidence_level)
    elif method == 'parametric':
        return calculator.parametric_var(confidence_level=confidence_level)
    elif method == 'monte_carlo':
        return calculator.monte_carlo_var(confidence_level=confidence_level)
    elif method == 'all':
        return calculator.calculate_all(confidence_level=confidence_level)
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_expected_shortfall(
    returns: pd.Series,
    confidence_level: float = 0.95,
    time_horizon: int = 1,
    method: str = 'historical'
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).

    Parameters
    ----------
    returns : pd.Series
        Series of returns
    confidence_level : float, optional
        Confidence level, by default 0.95
    time_horizon : int, optional
        Time horizon in days, by default 1
    method : str, optional
        Method to use ('historical', 'parametric', 'monte_carlo'), by default 'historical'

    Returns
    -------
    float
        Expected Shortfall
    """
    # Create VaR calculator
    calculator = VaRCalculator(returns, confidence_level, time_horizon)

    # Calculate Expected Shortfall
    return calculator.calculate_expected_shortfall(method, confidence_level=confidence_level)
