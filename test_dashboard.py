#!/usr/bin/env python
"""
Test script for the trading dashboard.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_data(commodity, data_dir='data/processed'):
    """Load processed data for a commodity."""
    file_path = os.path.join(data_dir, f"{commodity}.parquet")
    
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded {len(df)} rows from {file_path}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        return df
    
    # Try CSV as fallback
    csv_path = os.path.join(data_dir, f"{commodity}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"Successfully loaded {len(df)} rows from {csv_path}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        return df
    
    print(f"No data found for {commodity}")
    return pd.DataFrame()

def calculate_moving_average_signals(df, fast_window=10, slow_window=30):
    """Calculate moving average crossover signals."""
    # Make a copy of the data
    data = df.copy()
    
    # Calculate moving averages
    data['fast_ma'] = data['Price'].rolling(window=fast_window).mean()
    data['slow_ma'] = data['Price'].rolling(window=slow_window).mean()
    
    # Calculate signals
    data['signal'] = 0
    data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1
    data.loc[data['fast_ma'] < data['slow_ma'], 'signal'] = -1
    
    # Calculate position changes
    data['position_change'] = data['signal'].diff()
    
    # Calculate returns
    data['returns'] = data['Price'].pct_change()
    
    # Calculate strategy returns
    data['strategy_returns'] = data['signal'].shift(1) * data['returns']
    
    # Calculate cumulative returns
    data['cumulative_returns'] = (1 + data['returns']).cumprod() - 1
    data['strategy_cumulative_returns'] = (1 + data['strategy_returns']).cumprod() - 1
    
    return data

def calculate_performance_metrics(returns):
    """Calculate performance metrics."""
    # Calculate total return
    total_return = (1 + returns.dropna()).prod() - 1
    
    # Calculate annualized return
    n_periods = len(returns.dropna())
    n_years = n_periods / 252  # Assuming daily returns
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Calculate volatility
    volatility = returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + returns.dropna()).cumprod() - 1
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / (1 + peak)
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    win_rate = (returns > 0).mean()
    
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Win Rate': f"{win_rate:.2%}"
    }
    
    return metrics

def main():
    """Main function."""
    print("Testing dashboard functionality...")
    
    # Available commodities
    commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']
    
    # Load data for all commodities
    data_dict = {}
    for commodity in commodities:
        df = load_data(commodity)
        if not df.empty:
            data_dict[commodity] = df
    
    if not data_dict:
        print("No commodity data found. Please run the data pipeline first.")
        return
    
    # Test strategy on first commodity
    commodity = list(data_dict.keys())[0]
    print(f"\nTesting strategy on {commodity}...")
    
    df = data_dict[commodity]
    
    # Run moving average crossover strategy
    results = calculate_moving_average_signals(df, fast_window=10, slow_window=30)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(results['strategy_returns'])
    
    # Print metrics
    print("\nStrategy Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Plot results
    print("\nCreating performance chart...")
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot price and signals
    ax[0].plot(results.index, results['Price'], label='Price')
    ax[0].plot(results.index, results['fast_ma'], label='10-day MA')
    ax[0].plot(results.index, results['slow_ma'], label='30-day MA')
    
    # Plot buy/sell signals
    buy_signals = results[results['position_change'] > 0]
    sell_signals = results[results['position_change'] < 0]
    
    ax[0].scatter(buy_signals.index, buy_signals['Price'], marker='^', color='green', label='Buy')
    ax[0].scatter(sell_signals.index, sell_signals['Price'], marker='v', color='red', label='Sell')
    
    ax[0].set_ylabel('Price')
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot cumulative returns
    ax[1].plot(results.index, results['cumulative_returns'], label='Buy & Hold')
    ax[1].plot(results.index, results['strategy_cumulative_returns'], label='Strategy')
    ax[1].set_ylabel('Cumulative Returns')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('results/test', exist_ok=True)
    plt.savefig('results/test/strategy_performance.png')
    print("Saved performance chart to results/test/strategy_performance.png")
    
    # Test VaR calculation
    print("\nTesting VaR calculation...")
    
    # Calculate returns
    returns = df['Price'].pct_change().dropna()
    
    # Calculate VaR
    confidence_levels = [0.95, 0.99]
    
    for cl in confidence_levels:
        # Historical VaR
        var_percentile = 1 - cl
        historical_var = -np.percentile(returns, var_percentile * 100)
        
        # Parametric VaR
        z_score = stats.norm.ppf(cl)
        parametric_var = -(returns.mean() + z_score * returns.std())
        
        print(f"{cl:.0%} Historical VaR: {historical_var:.4%}")
        print(f"{cl:.0%} Parametric VaR: {parametric_var:.4%}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
