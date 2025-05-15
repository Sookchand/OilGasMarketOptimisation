"""
Risk limits framework for trading.
This module implements risk limits and monitoring.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np

from src.risk.var_calculator import calculate_var, calculate_expected_shortfall

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/risk_limits.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RiskLimits:
    """
    Risk limits framework for trading.
    """
    
    def __init__(
        self, 
        portfolio_value: float,
        limits: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the risk limits framework.
        
        Parameters
        ----------
        portfolio_value : float
            Current portfolio value
        limits : Dict[str, float], optional
            Dictionary of risk limits, by default None
        """
        self.portfolio_value = portfolio_value
        
        # Default limits
        default_limits = {
            'max_position_size': 0.1,  # Maximum position size as fraction of portfolio
            'max_sector_exposure': 0.3,  # Maximum sector exposure as fraction of portfolio
            'max_drawdown': 0.2,  # Maximum drawdown as fraction of portfolio
            'max_var_95': 0.05,  # Maximum 95% VaR as fraction of portfolio
            'max_leverage': 1.5,  # Maximum leverage ratio
            'max_concentration': 0.2  # Maximum concentration in a single asset
        }
        
        # Use provided limits or defaults
        self.limits = limits or default_limits
        
        # Initialize current exposures
        self.exposures = {
            'position_sizes': {},
            'sector_exposures': {},
            'drawdown': 0.0,
            'var_95': 0.0,
            'leverage': 1.0,
            'concentration': 0.0
        }
        
        # Initialize violations
        self.violations = {}
    
    def set_limit(self, limit_name: str, value: float) -> None:
        """
        Set a risk limit.
        
        Parameters
        ----------
        limit_name : str
            Name of the limit
        value : float
            Limit value
        """
        self.limits[limit_name] = value
        logger.info(f"Set {limit_name} limit to {value}")
    
    def update_portfolio_value(self, value: float) -> None:
        """
        Update portfolio value.
        
        Parameters
        ----------
        value : float
            New portfolio value
        """
        self.portfolio_value = value
        logger.info(f"Updated portfolio value to {value}")
    
    def update_position_size(
        self, 
        asset: str, 
        value: float
    ) -> bool:
        """
        Update position size and check limit.
        
        Parameters
        ----------
        asset : str
            Asset name
        value : float
            Position value
        
        Returns
        -------
        bool
            True if within limit, False if violated
        """
        # Calculate position size as fraction of portfolio
        position_size = value / self.portfolio_value
        
        # Update exposure
        self.exposures['position_sizes'][asset] = position_size
        
        # Check limit
        limit = self.limits.get('max_position_size', float('inf'))
        within_limit = position_size <= limit
        
        if not within_limit:
            self.violations[f'position_size_{asset}'] = {
                'limit': limit,
                'actual': position_size,
                'excess': position_size - limit
            }
            logger.warning(f"Position size limit violated for {asset}: {position_size:.2%} > {limit:.2%}")
        
        return within_limit
    
    def update_sector_exposure(
        self, 
        sector: str, 
        value: float
    ) -> bool:
        """
        Update sector exposure and check limit.
        
        Parameters
        ----------
        sector : str
            Sector name
        value : float
            Sector exposure value
        
        Returns
        -------
        bool
            True if within limit, False if violated
        """
        # Calculate sector exposure as fraction of portfolio
        sector_exposure = value / self.portfolio_value
        
        # Update exposure
        self.exposures['sector_exposures'][sector] = sector_exposure
        
        # Check limit
        limit = self.limits.get('max_sector_exposure', float('inf'))
        within_limit = sector_exposure <= limit
        
        if not within_limit:
            self.violations[f'sector_exposure_{sector}'] = {
                'limit': limit,
                'actual': sector_exposure,
                'excess': sector_exposure - limit
            }
            logger.warning(f"Sector exposure limit violated for {sector}: {sector_exposure:.2%} > {limit:.2%}")
        
        return within_limit
    
    def update_drawdown(self, peak_value: float) -> bool:
        """
        Update drawdown and check limit.
        
        Parameters
        ----------
        peak_value : float
            Peak portfolio value
        
        Returns
        -------
        bool
            True if within limit, False if violated
        """
        # Calculate drawdown
        drawdown = (peak_value - self.portfolio_value) / peak_value
        
        # Update exposure
        self.exposures['drawdown'] = drawdown
        
        # Check limit
        limit = self.limits.get('max_drawdown', float('inf'))
        within_limit = drawdown <= limit
        
        if not within_limit:
            self.violations['drawdown'] = {
                'limit': limit,
                'actual': drawdown,
                'excess': drawdown - limit
            }
            logger.warning(f"Drawdown limit violated: {drawdown:.2%} > {limit:.2%}")
        
        return within_limit
    
    def update_var(
        self, 
        returns: pd.Series,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        method: str = 'historical'
    ) -> bool:
        """
        Update Value at Risk and check limit.
        
        Parameters
        ----------
        returns : pd.Series
            Series of historical returns
        confidence_level : float, optional
            Confidence level, by default 0.95
        time_horizon : int, optional
            Time horizon in days, by default 1
        method : str, optional
            VaR calculation method, by default 'historical'
        
        Returns
        -------
        bool
            True if within limit, False if violated
        """
        # Calculate VaR
        var = calculate_var(returns, confidence_level, time_horizon, method)
        
        # Convert to fraction of portfolio
        var_pct = var
        
        # Update exposure
        self.exposures['var_95'] = var_pct
        
        # Check limit
        limit = self.limits.get('max_var_95', float('inf'))
        within_limit = var_pct <= limit
        
        if not within_limit:
            self.violations['var_95'] = {
                'limit': limit,
                'actual': var_pct,
                'excess': var_pct - limit
            }
            logger.warning(f"VaR limit violated: {var_pct:.2%} > {limit:.2%}")
        
        return within_limit
    
    def update_leverage(self, total_exposure: float) -> bool:
        """
        Update leverage and check limit.
        
        Parameters
        ----------
        total_exposure : float
            Total exposure value
        
        Returns
        -------
        bool
            True if within limit, False if violated
        """
        # Calculate leverage ratio
        leverage = total_exposure / self.portfolio_value
        
        # Update exposure
        self.exposures['leverage'] = leverage
        
        # Check limit
        limit = self.limits.get('max_leverage', float('inf'))
        within_limit = leverage <= limit
        
        if not within_limit:
            self.violations['leverage'] = {
                'limit': limit,
                'actual': leverage,
                'excess': leverage - limit
            }
            logger.warning(f"Leverage limit violated: {leverage:.2f} > {limit:.2f}")
        
        return within_limit
    
    def update_concentration(self, max_position_value: float) -> bool:
        """
        Update concentration and check limit.
        
        Parameters
        ----------
        max_position_value : float
            Value of the largest position
        
        Returns
        -------
        bool
            True if within limit, False if violated
        """
        # Calculate concentration
        concentration = max_position_value / self.portfolio_value
        
        # Update exposure
        self.exposures['concentration'] = concentration
        
        # Check limit
        limit = self.limits.get('max_concentration', float('inf'))
        within_limit = concentration <= limit
        
        if not within_limit:
            self.violations['concentration'] = {
                'limit': limit,
                'actual': concentration,
                'excess': concentration - limit
            }
            logger.warning(f"Concentration limit violated: {concentration:.2%} > {limit:.2%}")
        
        return within_limit
    
    def check_all_limits(self) -> Dict[str, bool]:
        """
        Check all risk limits.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary of limit checks
        """
        results = {}
        
        # Check position size limits
        for asset, size in self.exposures['position_sizes'].items():
            limit = self.limits.get('max_position_size', float('inf'))
            results[f'position_size_{asset}'] = size <= limit
        
        # Check sector exposure limits
        for sector, exposure in self.exposures['sector_exposures'].items():
            limit = self.limits.get('max_sector_exposure', float('inf'))
            results[f'sector_exposure_{sector}'] = exposure <= limit
        
        # Check other limits
        results['drawdown'] = self.exposures['drawdown'] <= self.limits.get('max_drawdown', float('inf'))
        results['var_95'] = self.exposures['var_95'] <= self.limits.get('max_var_95', float('inf'))
        results['leverage'] = self.exposures['leverage'] <= self.limits.get('max_leverage', float('inf'))
        results['concentration'] = self.exposures['concentration'] <= self.limits.get('max_concentration', float('inf'))
        
        return results
    
    def get_violations(self) -> Dict[str, Dict[str, float]]:
        """
        Get all limit violations.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of violations
        """
        return self.violations
    
    def get_exposure_report(self) -> Dict[str, Any]:
        """
        Get exposure report.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of exposures and limits
        """
        report = {
            'portfolio_value': self.portfolio_value,
            'limits': self.limits,
            'exposures': self.exposures,
            'violations': self.violations
        }
        
        return report

def create_risk_limits(
    portfolio_value: float,
    max_position_size: float = 0.1,
    max_sector_exposure: float = 0.3,
    max_drawdown: float = 0.2,
    max_var_95: float = 0.05,
    max_leverage: float = 1.5,
    max_concentration: float = 0.2
) -> RiskLimits:
    """
    Create risk limits framework.
    
    Parameters
    ----------
    portfolio_value : float
        Current portfolio value
    max_position_size : float, optional
        Maximum position size as fraction of portfolio, by default 0.1
    max_sector_exposure : float, optional
        Maximum sector exposure as fraction of portfolio, by default 0.3
    max_drawdown : float, optional
        Maximum drawdown as fraction of portfolio, by default 0.2
    max_var_95 : float, optional
        Maximum 95% VaR as fraction of portfolio, by default 0.05
    max_leverage : float, optional
        Maximum leverage ratio, by default 1.5
    max_concentration : float, optional
        Maximum concentration in a single asset, by default 0.2
    
    Returns
    -------
    RiskLimits
        Risk limits framework
    """
    limits = {
        'max_position_size': max_position_size,
        'max_sector_exposure': max_sector_exposure,
        'max_drawdown': max_drawdown,
        'max_var_95': max_var_95,
        'max_leverage': max_leverage,
        'max_concentration': max_concentration
    }
    
    return RiskLimits(portfolio_value, limits)
