"""
Trading execution package.
"""

from src.trading.execution.backtester import Backtester, run_backtest
from src.trading.execution.performance_metrics import calculate_returns_metrics, plot_performance

__all__ = [
    'Backtester',
    'run_backtest',
    'calculate_returns_metrics',
    'plot_performance'
]
