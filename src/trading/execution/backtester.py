"""
Backtesting framework for trading strategies.
This module implements a backtester to evaluate trading strategies.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.trading.strategies.base_strategy import BaseStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtester.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtester for evaluating trading strategies.
    """
    
    def __init__(
        self, 
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.001
    ):
        """
        Initialize the backtester.
        
        Parameters
        ----------
        strategy : BaseStrategy
            Trading strategy to backtest
        initial_capital : float, optional
            Initial capital, by default 10000.0
        commission : float, optional
            Commission rate per trade, by default 0.001 (0.1%)
        slippage : float, optional
            Slippage rate per trade, by default 0.001 (0.1%)
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = None
        self.positions = None
        self.portfolio = None
    
    def run(
        self, 
        data: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Run the backtest.
        
        Parameters
        ----------
        data : pd.DataFrame
            Price and feature data
        price_col : str, optional
            Price column name, by default 'close'
        
        Returns
        -------
        pd.DataFrame
            Backtest results
        """
        logger.info(f"Running backtest for {self.strategy.name} strategy")
        
        # Check if price column exists
        if price_col not in data.columns:
            logger.error(f"Price column '{price_col}' not found in data")
            return pd.DataFrame()
        
        # Generate positions
        self.positions = self.strategy.run(data)
        
        # Check if positions were generated
        if self.positions is None or self.positions.empty:
            logger.error("No positions generated")
            return pd.DataFrame()
        
        # Merge positions with price data
        self.results = pd.merge(
            self.positions,
            data[[price_col]],
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Calculate returns
        self.results['returns'] = self.results[price_col].pct_change()
        
        # Calculate strategy returns
        self.results['strategy_returns'] = self.results['position'].shift(1) * self.results['returns']
        
        # Calculate transaction costs
        self.results['position_change'] = self.results['position'].diff().abs()
        self.results['transaction_costs'] = self.results['position_change'] * (self.commission + self.slippage)
        
        # Calculate net returns
        self.results['net_returns'] = self.results['strategy_returns'] - self.results['transaction_costs']
        
        # Calculate cumulative returns
        self.results['cumulative_returns'] = (1 + self.results['returns']).cumprod() - 1
        self.results['cumulative_strategy_returns'] = (1 + self.results['strategy_returns']).cumprod() - 1
        self.results['cumulative_net_returns'] = (1 + self.results['net_returns']).cumprod() - 1
        
        # Calculate portfolio value
        self.results['portfolio_value'] = self.initial_capital * (1 + self.results['cumulative_net_returns'])
        
        # Calculate drawdown
        self.results['peak'] = self.results['portfolio_value'].cummax()
        self.results['drawdown'] = (self.results['portfolio_value'] - self.results['peak']) / self.results['peak']
        
        logger.info(f"Backtest completed with {len(self.results)} data points")
        
        return self.results
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of performance metrics
        """
        if self.results is None or self.results.empty:
            logger.error("No backtest results to calculate metrics")
            return {}
        
        # Calculate annualized return
        total_days = (self.results.index[-1] - self.results.index[0]).days
        if total_days <= 0:
            total_years = 1
        else:
            total_years = total_days / 365.25
        
        final_return = self.results['cumulative_net_returns'].iloc[-1]
        annualized_return = (1 + final_return) ** (1 / total_years) - 1
        
        # Calculate volatility
        daily_std = self.results['net_returns'].std()
        annualized_std = daily_std * np.sqrt(252)
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.0  # Simplified
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std if annualized_std > 0 else 0
        
        # Calculate Sortino ratio
        downside_returns = self.results['net_returns'][self.results['net_returns'] < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Calculate maximum drawdown
        max_drawdown = self.results['drawdown'].min()
        
        # Calculate win rate
        trades = self.results['position_change'] > 0
        if trades.sum() > 0:
            winning_trades = (self.results['net_returns'][trades] > 0).sum()
            win_rate = winning_trades / trades.sum()
        else:
            win_rate = 0
        
        # Calculate profit factor
        gross_profits = self.results['net_returns'][self.results['net_returns'] > 0].sum()
        gross_losses = abs(self.results['net_returns'][self.results['net_returns'] < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Calculate average trade
        avg_trade = self.results['net_returns'][trades].mean()
        
        metrics = {
            'total_return': final_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_std,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'num_trades': trades.sum()
        }
        
        logger.info(f"Calculated performance metrics: {metrics}")
        
        return metrics
    
    def plot_results(
        self, 
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot backtest results.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size, by default (12, 8)
        
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if self.results is None or self.results.empty:
            logger.error("No backtest results to plot")
            return None
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot portfolio value
        axes[0].plot(self.results.index, self.results['portfolio_value'], label='Portfolio Value')
        axes[0].set_title(f"{self.strategy.name} Strategy - Portfolio Value")
        axes[0].set_ylabel('Value ($)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot cumulative returns
        axes[1].plot(self.results.index, self.results['cumulative_returns'], label='Buy & Hold')
        axes[1].plot(self.results.index, self.results['cumulative_net_returns'], label='Strategy')
        axes[1].set_title('Cumulative Returns')
        axes[1].set_ylabel('Returns (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot drawdown
        axes[2].fill_between(self.results.index, self.results['drawdown'], 0, color='red', alpha=0.3)
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        
        # Format x-axis
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def save_results(
        self, 
        output_dir: str = 'results/backtests',
        include_plot: bool = True
    ) -> None:
        """
        Save backtest results.
        
        Parameters
        ----------
        output_dir : str, optional
            Output directory, by default 'results/backtests'
        include_plot : bool, optional
            Whether to save plot, by default True
        """
        if self.results is None or self.results.empty:
            logger.error("No backtest results to save")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save results to CSV
        results_file = os.path.join(output_dir, f"{self.strategy.name.replace(' ', '_')}_{timestamp}.csv")
        self.results.to_csv(results_file)
        logger.info(f"Saved results to {results_file}")
        
        # Calculate and save metrics
        metrics = self.calculate_metrics()
        metrics_file = os.path.join(output_dir, f"{self.strategy.name.replace(' ', '_')}_{timestamp}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Save plot
        if include_plot:
            plot_file = os.path.join(output_dir, f"{self.strategy.name.replace(' ', '_')}_{timestamp}.png")
            fig = self.plot_results()
            if fig:
                fig.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved plot to {plot_file}")

def run_backtest(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    price_col: str = 'close',
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.001,
    save_results: bool = True,
    output_dir: str = 'results/backtests'
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run a backtest for a strategy.
    
    Parameters
    ----------
    strategy : BaseStrategy
        Trading strategy to backtest
    data : pd.DataFrame
        Price and feature data
    price_col : str, optional
        Price column name, by default 'close'
    initial_capital : float, optional
        Initial capital, by default 10000.0
    commission : float, optional
        Commission rate per trade, by default 0.001 (0.1%)
    slippage : float, optional
        Slippage rate per trade, by default 0.001 (0.1%)
    save_results : bool, optional
        Whether to save results, by default True
    output_dir : str, optional
        Output directory, by default 'results/backtests'
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        Backtest results and performance metrics
    """
    # Create backtester
    backtester = Backtester(
        strategy=strategy,
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage
    )
    
    # Run backtest
    results = backtester.run(data, price_col)
    
    # Calculate metrics
    metrics = backtester.calculate_metrics()
    
    # Save results
    if save_results:
        backtester.save_results(output_dir)
    
    return results, metrics
