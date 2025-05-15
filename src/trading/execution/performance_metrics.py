"""
Performance metrics for trading strategies.
This module calculates various performance metrics for trading strategies.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/performance_metrics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_returns_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate return-based performance metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    risk_free_rate : float, optional
        Risk-free rate, by default 0.0
    
    Returns
    -------
    Dict[str, float]
        Dictionary of performance metrics
    """
    # Calculate total return
    total_return = (1 + returns).prod() - 1
    
    # Calculate annualized return
    n_periods = len(returns)
    n_years = n_periods / 252  # Assuming daily returns
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Calculate volatility
    volatility = returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Calculate Sortino ratio
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod() - 1
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / (1 + peak)
    max_drawdown = drawdown.min()
    
    # Calculate Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
    
    # Calculate win rate
    win_rate = (returns > 0).mean()
    
    # Calculate average win and loss
    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()
    
    # Calculate profit factor
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    # Calculate recovery factor
    recovery_factor = total_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'recovery_factor': recovery_factor
    }
    
    return metrics

def calculate_trade_metrics(
    trades: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate trade-based performance metrics.
    
    Parameters
    ----------
    trades : pd.DataFrame
        DataFrame of trades with columns: 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'position', 'pnl'
    
    Returns
    -------
    Dict[str, float]
        Dictionary of performance metrics
    """
    if trades.empty:
        logger.warning("No trades to calculate metrics")
        return {}
    
    # Calculate number of trades
    num_trades = len(trades)
    
    # Calculate win rate
    num_wins = (trades['pnl'] > 0).sum()
    win_rate = num_wins / num_trades
    
    # Calculate average trade
    avg_trade = trades['pnl'].mean()
    
    # Calculate average win and loss
    avg_win = trades.loc[trades['pnl'] > 0, 'pnl'].mean()
    avg_loss = trades.loc[trades['pnl'] < 0, 'pnl'].mean()
    
    # Calculate profit factor
    gross_profits = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
    gross_losses = abs(trades.loc[trades['pnl'] < 0, 'pnl'].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    # Calculate average holding period
    trades['holding_period'] = (trades['exit_date'] - trades['entry_date']).dt.days
    avg_holding_period = trades['holding_period'].mean()
    
    # Calculate maximum consecutive wins and losses
    trades['win'] = trades['pnl'] > 0
    
    # Calculate consecutive wins
    consecutive_wins = []
    current_streak = 0
    
    for win in trades['win']:
        if win:
            current_streak += 1
        else:
            if current_streak > 0:
                consecutive_wins.append(current_streak)
                current_streak = 0
    
    if current_streak > 0:
        consecutive_wins.append(current_streak)
    
    max_consecutive_wins = max(consecutive_wins) if consecutive_wins else 0
    
    # Calculate consecutive losses
    consecutive_losses = []
    current_streak = 0
    
    for win in trades['win']:
        if not win:
            current_streak += 1
        else:
            if current_streak > 0:
                consecutive_losses.append(current_streak)
                current_streak = 0
    
    if current_streak > 0:
        consecutive_losses.append(current_streak)
    
    max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
    
    metrics = {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_holding_period': avg_holding_period,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses
    }
    
    return metrics

def extract_trades_from_positions(
    positions: pd.DataFrame,
    prices: pd.Series,
    commission: float = 0.001,
    slippage: float = 0.001
) -> pd.DataFrame:
    """
    Extract trades from positions.
    
    Parameters
    ----------
    positions : pd.DataFrame
        DataFrame with position column
    prices : pd.Series
        Series of prices
    commission : float, optional
        Commission rate per trade, by default 0.001 (0.1%)
    slippage : float, optional
        Slippage rate per trade, by default 0.001 (0.1%)
    
    Returns
    -------
    pd.DataFrame
        DataFrame of trades
    """
    if 'position' not in positions.columns:
        logger.error("No position column in positions DataFrame")
        return pd.DataFrame()
    
    # Merge positions with prices
    data = pd.merge(
        positions[['position']],
        prices.to_frame('price'),
        left_index=True,
        right_index=True,
        how='left'
    )
    
    # Calculate position changes
    data['position_change'] = data['position'].diff()
    
    # Extract trade entries
    entries = data[data['position_change'] != 0].copy()
    entries.rename(columns={
        'price': 'entry_price',
        'position': 'position',
        'position_change': 'size'
    }, inplace=True)
    
    # Initialize trades list
    trades = []
    
    # Process each entry
    for i, entry in entries.iterrows():
        # Skip if size is zero
        if entry['size'] == 0:
            continue
        
        # Find exit
        exit_mask = (data.index > i) & (data['position_change'] == -entry['size'])
        
        if exit_mask.any():
            # Get exit data
            exit_idx = data[exit_mask].index[0]
            exit_price = data.loc[exit_idx, 'price']
            
            # Calculate P&L
            entry_price_with_costs = entry['entry_price'] * (1 + np.sign(entry['size']) * (commission + slippage))
            exit_price_with_costs = exit_price * (1 - np.sign(entry['size']) * (commission + slippage))
            
            pnl = entry['size'] * (exit_price_with_costs - entry_price_with_costs)
            
            # Add trade to list
            trades.append({
                'entry_date': i,
                'exit_date': exit_idx,
                'entry_price': entry['entry_price'],
                'exit_price': exit_price,
                'position': np.sign(entry['size']),
                'size': abs(entry['size']),
                'pnl': pnl
            })
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)
    
    return trades_df

def plot_performance(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot performance metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    benchmark_returns : pd.Series, optional
        Series of benchmark returns, by default None
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 8)
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # Calculate drawdown
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / (1 + peak)
    
    # Calculate rolling metrics
    rolling_return = returns.rolling(252).mean() * 252
    rolling_vol = returns.rolling(252).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Plot cumulative returns
    axes[0].plot(cumulative_returns.index, cumulative_returns, label='Strategy')
    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
        axes[0].plot(benchmark_cumulative.index, benchmark_cumulative, label='Benchmark')
    axes[0].set_title('Cumulative Returns')
    axes[0].set_ylabel('Returns (%)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot drawdown
    axes[1].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].grid(True)
    
    # Plot rolling annualized return
    axes[2].plot(rolling_return.index, rolling_return)
    axes[2].set_title('Rolling 1-Year Annualized Return')
    axes[2].set_ylabel('Return (%)')
    axes[2].grid(True)
    
    # Plot rolling Sharpe ratio
    axes[3].plot(rolling_sharpe.index, rolling_sharpe)
    axes[3].set_title('Rolling 1-Year Sharpe Ratio')
    axes[3].set_ylabel('Sharpe Ratio')
    axes[3].set_xlabel('Date')
    axes[3].grid(True)
    
    # Format x-axis
    fig.autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
