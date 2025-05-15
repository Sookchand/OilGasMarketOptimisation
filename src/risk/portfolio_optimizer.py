"""
Portfolio optimizer for risk management.
This module implements portfolio optimization techniques.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/portfolio_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Portfolio optimizer for risk management.
    """
    
    def __init__(
        self, 
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize the portfolio optimizer.
        
        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of asset returns
        risk_free_rate : float, optional
            Risk-free rate, by default 0.0
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.num_assets = returns.shape[1]
        self.assets = returns.columns
        
        # Calculate mean returns and covariance matrix
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        
        # Initialize results
        self.efficient_frontier = None
        self.optimal_portfolio = None
    
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio return.
        
        Parameters
        ----------
        weights : np.ndarray
            Array of asset weights
        
        Returns
        -------
        float
            Portfolio return
        """
        return np.sum(self.mean_returns * weights)
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility.
        
        Parameters
        ----------
        weights : np.ndarray
            Array of asset weights
        
        Returns
        -------
        float
            Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def _portfolio_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio Sharpe ratio.
        
        Parameters
        ----------
        weights : np.ndarray
            Array of asset weights
        
        Returns
        -------
        float
            Portfolio Sharpe ratio
        """
        return (self._portfolio_return(weights) - self.risk_free_rate) / self._portfolio_volatility(weights)
    
    def _negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate negative Sharpe ratio for minimization.
        
        Parameters
        ----------
        weights : np.ndarray
            Array of asset weights
        
        Returns
        -------
        float
            Negative Sharpe ratio
        """
        return -self._portfolio_sharpe_ratio(weights)
    
    def _minimize_volatility(self, target_return: float) -> Dict[str, Any]:
        """
        Minimize portfolio volatility for a target return.
        
        Parameters
        ----------
        target_return : float
            Target portfolio return
        
        Returns
        -------
        Dict[str, Any]
            Optimization results
        """
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: self._portfolio_return(x) - target_return}  # Target return
        ]
        
        # Define bounds
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Initial guess
        initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)
        
        # Minimize volatility
        result = minimize(
            self._portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result
    
    def optimize_sharpe_ratio(self) -> Dict[str, Any]:
        """
        Optimize portfolio for maximum Sharpe ratio.
        
        Returns
        -------
        Dict[str, Any]
            Optimization results
        """
        # Define constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Define bounds
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Initial guess
        initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)
        
        # Maximize Sharpe ratio
        result = minimize(
            self._negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Store optimal portfolio
        self.optimal_portfolio = {
            'weights': result['x'],
            'return': self._portfolio_return(result['x']),
            'volatility': self._portfolio_volatility(result['x']),
            'sharpe_ratio': self._portfolio_sharpe_ratio(result['x'])
        }
        
        logger.info(f"Optimized portfolio for maximum Sharpe ratio: {self.optimal_portfolio['sharpe_ratio']:.4f}")
        
        return self.optimal_portfolio
    
    def calculate_efficient_frontier(
        self, 
        num_portfolios: int = 100
    ) -> pd.DataFrame:
        """
        Calculate the efficient frontier.
        
        Parameters
        ----------
        num_portfolios : int, optional
            Number of portfolios to calculate, by default 100
        
        Returns
        -------
        pd.DataFrame
            DataFrame of efficient frontier portfolios
        """
        # Calculate minimum and maximum returns
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        # Initialize results
        efficient_portfolios = []
        
        # Calculate efficient portfolios
        for target_return in target_returns:
            try:
                result = self._minimize_volatility(target_return)
                
                if result['success']:
                    weights = result['x']
                    portfolio = {
                        'return': self._portfolio_return(weights),
                        'volatility': self._portfolio_volatility(weights),
                        'sharpe_ratio': self._portfolio_sharpe_ratio(weights)
                    }
                    
                    # Add weights
                    for i, asset in enumerate(self.assets):
                        portfolio[f'weight_{asset}'] = weights[i]
                    
                    efficient_portfolios.append(portfolio)
            except Exception as e:
                logger.warning(f"Error calculating efficient portfolio for return {target_return}: {e}")
        
        # Convert to DataFrame
        self.efficient_frontier = pd.DataFrame(efficient_portfolios)
        
        logger.info(f"Calculated {len(self.efficient_frontier)} efficient portfolios")
        
        return self.efficient_frontier
    
    def plot_efficient_frontier(
        self, 
        figsize: Tuple[int, int] = (12, 8),
        show_assets: bool = True,
        show_optimal: bool = True
    ) -> plt.Figure:
        """
        Plot the efficient frontier.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size, by default (12, 8)
        show_assets : bool, optional
            Whether to show individual assets, by default True
        show_optimal : bool, optional
            Whether to show the optimal portfolio, by default True
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if self.efficient_frontier is None:
            logger.warning("Efficient frontier not calculated")
            self.calculate_efficient_frontier()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot efficient frontier
        ax.plot(
            self.efficient_frontier['volatility'],
            self.efficient_frontier['return'],
            'b-',
            label='Efficient Frontier'
        )
        
        # Plot individual assets
        if show_assets:
            for i, asset in enumerate(self.assets):
                ax.scatter(
                    np.sqrt(self.cov_matrix.iloc[i, i]),
                    self.mean_returns[i],
                    marker='o',
                    label=asset
                )
        
        # Plot optimal portfolio
        if show_optimal and self.optimal_portfolio is not None:
            ax.scatter(
                self.optimal_portfolio['volatility'],
                self.optimal_portfolio['return'],
                marker='*',
                color='r',
                s=100,
                label=f"Optimal Portfolio (Sharpe: {self.optimal_portfolio['sharpe_ratio']:.4f})"
            )
        
        # Add labels and title
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def get_optimal_weights(self) -> pd.Series:
        """
        Get optimal portfolio weights.
        
        Returns
        -------
        pd.Series
            Series of optimal weights
        """
        if self.optimal_portfolio is None:
            logger.warning("Optimal portfolio not calculated")
            self.optimize_sharpe_ratio()
        
        return pd.Series(self.optimal_portfolio['weights'], index=self.assets)
    
    def get_portfolio_stats(
        self, 
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Get portfolio statistics.
        
        Parameters
        ----------
        weights : np.ndarray, optional
            Array of asset weights, by default None (use optimal weights)
        
        Returns
        -------
        Dict[str, float]
            Dictionary of portfolio statistics
        """
        if weights is None:
            if self.optimal_portfolio is None:
                logger.warning("Optimal portfolio not calculated")
                self.optimize_sharpe_ratio()
            weights = self.optimal_portfolio['weights']
        
        return {
            'return': self._portfolio_return(weights),
            'volatility': self._portfolio_volatility(weights),
            'sharpe_ratio': self._portfolio_sharpe_ratio(weights)
        }

def optimize_portfolio(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    plot_results: bool = True,
    save_plot: bool = False,
    output_dir: str = 'results/portfolio'
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Optimize portfolio weights.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns
    risk_free_rate : float, optional
        Risk-free rate, by default 0.0
    plot_results : bool, optional
        Whether to plot results, by default True
    save_plot : bool, optional
        Whether to save plot, by default False
    output_dir : str, optional
        Output directory, by default 'results/portfolio'
    
    Returns
    -------
    Tuple[pd.Series, Dict[str, float]]
        Optimal weights and portfolio statistics
    """
    # Create optimizer
    optimizer = PortfolioOptimizer(returns, risk_free_rate)
    
    # Optimize portfolio
    optimizer.optimize_sharpe_ratio()
    
    # Calculate efficient frontier
    optimizer.calculate_efficient_frontier()
    
    # Plot results
    if plot_results:
        fig = optimizer.plot_efficient_frontier()
        
        # Save plot
        if save_plot:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save plot
            fig.savefig(os.path.join(output_dir, 'efficient_frontier.png'), dpi=300, bbox_inches='tight')
            
            # Save weights
            weights = optimizer.get_optimal_weights()
            weights.to_csv(os.path.join(output_dir, 'optimal_weights.csv'))
            
            # Save efficient frontier
            optimizer.efficient_frontier.to_csv(os.path.join(output_dir, 'efficient_frontier.csv'))
            
            logger.info(f"Saved results to {output_dir}")
    
    return optimizer.get_optimal_weights(), optimizer.get_portfolio_stats()
