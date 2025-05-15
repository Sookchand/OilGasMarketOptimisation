"""
Monte Carlo simulation engine for risk analysis.
This module implements Monte Carlo simulations for scenario analysis.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monte_carlo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for risk analysis.
    """
    
    def __init__(
        self, 
        returns: pd.Series,
        initial_value: float = 10000.0,
        num_simulations: int = 1000,
        time_horizon: int = 252,
        seed: Optional[int] = None
    ):
        """
        Initialize the Monte Carlo simulator.
        
        Parameters
        ----------
        returns : pd.Series
            Series of historical returns
        initial_value : float, optional
            Initial portfolio value, by default 10000.0
        num_simulations : int, optional
            Number of simulations, by default 1000
        time_horizon : int, optional
            Time horizon in days, by default 252 (1 year)
        seed : int, optional
            Random seed, by default None
        """
        self.returns = returns
        self.initial_value = initial_value
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon
        self.seed = seed
        self.simulations = None
    
    def run_simulation(
        self, 
        method: str = 'bootstrap',
        drift: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulation.
        
        Parameters
        ----------
        method : str, optional
            Simulation method ('bootstrap', 'normal', 'gbm'), by default 'bootstrap'
        drift : float, optional
            Drift parameter for GBM, by default None (use historical mean)
        volatility : float, optional
            Volatility parameter for GBM, by default None (use historical std)
        
        Returns
        -------
        pd.DataFrame
            DataFrame of simulated paths
        """
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Initialize simulations DataFrame
        self.simulations = pd.DataFrame(
            index=range(self.time_horizon),
            columns=range(self.num_simulations)
        )
        
        # Set initial value
        self.simulations.loc[0] = self.initial_value
        
        if method == 'bootstrap':
            # Bootstrap from historical returns
            for i in range(1, self.time_horizon):
                # Sample returns with replacement
                sampled_returns = np.random.choice(self.returns, size=self.num_simulations)
                
                # Update portfolio value
                self.simulations.loc[i] = self.simulations.loc[i-1] * (1 + sampled_returns)
        
        elif method == 'normal':
            # Calculate mean and standard deviation
            mu = self.returns.mean()
            sigma = self.returns.std()
            
            # Generate random returns from normal distribution
            for i in range(1, self.time_horizon):
                random_returns = np.random.normal(mu, sigma, size=self.num_simulations)
                self.simulations.loc[i] = self.simulations.loc[i-1] * (1 + random_returns)
        
        elif method == 'gbm':
            # Geometric Brownian Motion
            # Use provided parameters or calculate from historical data
            mu = drift if drift is not None else self.returns.mean()
            sigma = volatility if volatility is not None else self.returns.std()
            
            # Generate random returns from GBM
            for i in range(1, self.time_horizon):
                random_returns = np.random.normal(
                    (mu - 0.5 * sigma**2),  # Drift term
                    sigma,                   # Volatility term
                    size=self.num_simulations
                )
                self.simulations.loc[i] = self.simulations.loc[i-1] * np.exp(random_returns)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Completed {self.num_simulations} Monte Carlo simulations using {method} method")
        
        return self.simulations
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics from simulations.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of statistics
        """
        if self.simulations is None:
            logger.error("No simulations to calculate statistics")
            return {}
        
        # Get final values
        final_values = self.simulations.iloc[-1]
        
        # Calculate returns
        total_returns = (final_values / self.initial_value) - 1
        
        # Calculate statistics
        stats = {
            'mean': final_values.mean(),
            'median': final_values.median(),
            'std': final_values.std(),
            'min': final_values.min(),
            'max': final_values.max(),
            'mean_return': total_returns.mean(),
            'median_return': total_returns.median(),
            'var_95': -np.percentile(total_returns, 5),
            'var_99': -np.percentile(total_returns, 1),
            'expected_shortfall_95': -total_returns[total_returns <= np.percentile(total_returns, 5)].mean(),
            'expected_shortfall_99': -total_returns[total_returns <= np.percentile(total_returns, 1)].mean(),
            'probability_of_loss': (total_returns < 0).mean()
        }
        
        logger.info(f"Calculated statistics from {self.num_simulations} simulations")
        
        return stats
    
    def plot_simulations(
        self, 
        num_paths: int = 100,
        figsize: Tuple[int, int] = (12, 8),
        alpha: float = 0.1
    ) -> plt.Figure:
        """
        Plot Monte Carlo simulations.
        
        Parameters
        ----------
        num_paths : int, optional
            Number of paths to plot, by default 100
        figsize : Tuple[int, int], optional
            Figure size, by default (12, 8)
        alpha : float, optional
            Transparency of paths, by default 0.1
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if self.simulations is None:
            logger.error("No simulations to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot a subset of simulations
        num_paths = min(num_paths, self.num_simulations)
        paths_to_plot = np.random.choice(self.num_simulations, size=num_paths, replace=False)
        
        for path in paths_to_plot:
            ax.plot(self.simulations[path], alpha=alpha, color='blue')
        
        # Plot mean path
        mean_path = self.simulations.mean(axis=1)
        ax.plot(mean_path, color='red', linewidth=2, label='Mean Path')
        
        # Plot confidence intervals
        percentile_5 = self.simulations.quantile(0.05, axis=1)
        percentile_95 = self.simulations.quantile(0.95, axis=1)
        
        ax.fill_between(
            range(self.time_horizon),
            percentile_5,
            percentile_95,
            color='red',
            alpha=0.2,
            label='90% Confidence Interval'
        )
        
        # Add labels and title
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Monte Carlo Simulation')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_histogram(
        self, 
        figsize: Tuple[int, int] = (12, 8),
        bins: int = 50
    ) -> plt.Figure:
        """
        Plot histogram of final values.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size, by default (12, 8)
        bins : int, optional
            Number of bins, by default 50
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if self.simulations is None:
            logger.error("No simulations to plot")
            return None
        
        # Get final values
        final_values = self.simulations.iloc[-1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        ax.hist(final_values, bins=bins, alpha=0.7, color='blue')
        
        # Add vertical lines for statistics
        ax.axvline(final_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${final_values.mean():.2f}')
        ax.axvline(final_values.median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${final_values.median():.2f}')
        ax.axvline(self.initial_value, color='black', linestyle='-', linewidth=2, label=f'Initial: ${self.initial_value:.2f}')
        
        # Add VaR lines
        var_95 = np.percentile(final_values, 5)
        ax.axvline(var_95, color='orange', linestyle='--', linewidth=2, label=f'5% VaR: ${var_95:.2f}')
        
        # Add labels and title
        ax.set_xlabel('Portfolio Value ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Final Portfolio Values')
        ax.legend()
        ax.grid(True)
        
        return fig

def run_monte_carlo_analysis(
    returns: pd.Series,
    initial_value: float = 10000.0,
    num_simulations: int = 1000,
    time_horizon: int = 252,
    method: str = 'bootstrap',
    plot_results: bool = True,
    save_plots: bool = False,
    output_dir: str = 'results/monte_carlo'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run Monte Carlo analysis.
    
    Parameters
    ----------
    returns : pd.Series
        Series of historical returns
    initial_value : float, optional
        Initial portfolio value, by default 10000.0
    num_simulations : int, optional
        Number of simulations, by default 1000
    time_horizon : int, optional
        Time horizon in days, by default 252 (1 year)
    method : str, optional
        Simulation method ('bootstrap', 'normal', 'gbm'), by default 'bootstrap'
    plot_results : bool, optional
        Whether to plot results, by default True
    save_plots : bool, optional
        Whether to save plots, by default False
    output_dir : str, optional
        Output directory, by default 'results/monte_carlo'
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Simulations and statistics
    """
    # Create simulator
    simulator = MonteCarloSimulator(
        returns=returns,
        initial_value=initial_value,
        num_simulations=num_simulations,
        time_horizon=time_horizon
    )
    
    # Run simulation
    simulations = simulator.run_simulation(method=method)
    
    # Calculate statistics
    statistics = simulator.calculate_statistics()
    
    # Plot results
    if plot_results:
        # Plot simulations
        fig_sim = simulator.plot_simulations()
        
        # Plot histogram
        fig_hist = simulator.plot_histogram()
        
        # Save plots
        if save_plots:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save plots
            fig_sim.savefig(os.path.join(output_dir, 'monte_carlo_simulations.png'), dpi=300, bbox_inches='tight')
            fig_hist.savefig(os.path.join(output_dir, 'monte_carlo_histogram.png'), dpi=300, bbox_inches='tight')
            
            # Save statistics
            pd.Series(statistics).to_csv(os.path.join(output_dir, 'monte_carlo_statistics.csv'))
            
            logger.info(f"Saved plots and statistics to {output_dir}")
    
    return simulations, statistics
