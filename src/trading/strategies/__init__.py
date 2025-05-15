"""
Trading strategies package.
"""

from src.trading.strategies.base_strategy import BaseStrategy
from src.trading.strategies.trend_following import MovingAverageCrossover, MACDStrategy
from src.trading.strategies.mean_reversion import RSIStrategy, BollingerBandsStrategy
from src.trading.strategies.volatility_breakout import DonchianChannelStrategy, ATRChannelStrategy

__all__ = [
    'BaseStrategy',
    'MovingAverageCrossover',
    'MACDStrategy',
    'RSIStrategy',
    'BollingerBandsStrategy',
    'DonchianChannelStrategy',
    'ATRChannelStrategy'
]
