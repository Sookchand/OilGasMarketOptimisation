#!/usr/bin/env python
"""
Test script for the Oil & Gas Market Optimization system.
This script tests the basic functionality of each component.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test loading the sample data."""
    logger.info("Testing data loading...")
    
    commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']
    data_dir = 'data/raw'
    
    for commodity in commodities:
        file_path = os.path.join(data_dir, f"{commodity}.parquet")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                logger.info(f"Successfully loaded {commodity} data: {len(df)} rows, columns: {df.columns.tolist()}")
                
                # Plot the data
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df['Price'])
                plt.title(f"{commodity.replace('_', ' ').title()} Price")
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.grid(True)
                
                # Save the plot
                os.makedirs('results/test', exist_ok=True)
                plt.savefig(f"results/test/{commodity}_price.png")
                plt.close()
                
                logger.info(f"Saved price plot to results/test/{commodity}_price.png")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    logger.info("Data loading test completed.")

def test_trading_strategies():
    """Test basic trading strategies."""
    logger.info("Testing trading strategies...")
    
    try:
        # Import trading strategy classes
        from src.trading.strategies.trend_following import MovingAverageCrossover
        from src.trading.strategies.mean_reversion import RSIStrategy
        
        # Load sample data
        df = pd.read_parquet('data/raw/crude_oil.parquet')
        
        # Prepare data for trading
        data = df.copy()
        data['close'] = data['Price']
        data['open'] = data['Open'] if 'Open' in data.columns else data['Price'].shift(1)
        data['high'] = data['High'] if 'High' in data.columns else data['Price'] * 1.01
        data['low'] = data['Low'] if 'Low' in data.columns else data['Price'] * 0.99
        data['volume'] = data['Volume'] if 'Volume' in data.columns else np.random.randint(1000, 10000, size=len(data))
        data['returns'] = data['close'].pct_change()
        data = data.dropna()
        
        # Test MA Crossover strategy
        ma_strategy = MovingAverageCrossover(fast_window=10, slow_window=30)
        signals = ma_strategy.generate_signals(data)
        
        # Test RSI strategy
        rsi_strategy = RSIStrategy(window=14, oversold=30, overbought=70)
        signals = rsi_strategy.generate_signals(data)
        
        logger.info("Successfully tested trading strategies.")
        
    except Exception as e:
        logger.error(f"Error testing trading strategies: {e}")
    
    logger.info("Trading strategies test completed.")

def test_risk_management():
    """Test risk management components."""
    logger.info("Testing risk management components...")
    
    try:
        # Import risk management classes
        from src.risk.var_calculator import VaRCalculator
        
        # Load sample data
        df = pd.read_parquet('data/raw/crude_oil.parquet')
        
        # Calculate returns
        returns = df['Price'].pct_change().dropna()
        
        # Calculate VaR
        calculator = VaRCalculator(returns)
        var_95 = calculator.historical_var()
        var_99 = calculator.historical_var(confidence_level=0.99)
        
        logger.info(f"95% VaR: {var_95:.2%}")
        logger.info(f"99% VaR: {var_99:.2%}")
        
        logger.info("Successfully tested risk management components.")
        
    except Exception as e:
        logger.error(f"Error testing risk management components: {e}")
    
    logger.info("Risk management test completed.")

def test_rag_system():
    """Test RAG system components."""
    logger.info("Testing RAG system components...")
    
    try:
        # Check if insight files exist
        insights_dir = 'data/insights'
        insight_files = [f for f in os.listdir(insights_dir) if f.endswith('.md')]
        
        if insight_files:
            logger.info(f"Found {len(insight_files)} insight files: {insight_files}")
            
            # Read the first insight file
            with open(os.path.join(insights_dir, insight_files[0]), 'r') as f:
                content = f.read()
            
            logger.info(f"Successfully read insight file: {insight_files[0]}")
            logger.info(f"Content length: {len(content)} characters")
        else:
            logger.warning("No insight files found in data/insights")
        
    except Exception as e:
        logger.error(f"Error testing RAG system components: {e}")
    
    logger.info("RAG system test completed.")

def main():
    """Run all tests."""
    logger.info("Starting system tests...")
    
    # Create necessary directories
    os.makedirs('results/test', exist_ok=True)
    
    # Run tests
    test_data_loading()
    test_trading_strategies()
    test_risk_management()
    test_rag_system()
    
    logger.info("All tests completed.")

if __name__ == "__main__":
    main()
