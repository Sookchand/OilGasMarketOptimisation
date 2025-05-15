#!/usr/bin/env python
"""
Simple trading dashboard for the Oil & Gas Market Optimization project.
This is a simplified version that works with the processed data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats as scipy_stats

def load_data(commodity, data_dir='data/processed'):
    """Load processed data for a commodity."""
    file_path = os.path.join(data_dir, f"{commodity}.parquet")

    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        return df

    # Try CSV as fallback
    csv_path = os.path.join(data_dir, f"{commodity}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return df

    return pd.DataFrame()

def calculate_moving_average_signals(df, fast_window=10, slow_window=30):
    """Calculate moving average crossover signals."""
    # Make a copy of the data
    data = df.copy()

    # Determine price column
    price_col = 'Price' if 'Price' in data.columns else 'close'
    if price_col not in data.columns:
        # Try to find a suitable price column
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
        else:
            raise ValueError("No suitable price column found in data")

    # Calculate moving averages
    data['fast_ma'] = data[price_col].rolling(window=fast_window).mean()
    data['slow_ma'] = data[price_col].rolling(window=slow_window).mean()

    # Calculate signals
    data['signal'] = 0
    data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1
    data.loc[data['fast_ma'] < data['slow_ma'], 'signal'] = -1

    # Calculate position changes
    data['position_change'] = data['signal'].diff()

    # Calculate returns
    data['returns'] = data[price_col].pct_change()

    # Calculate strategy returns
    data['strategy_returns'] = data['signal'].shift(1) * data['returns']

    # Calculate cumulative returns
    data['cumulative_returns'] = (1 + data['returns']).cumprod() - 1
    data['strategy_cumulative_returns'] = (1 + data['strategy_returns']).cumprod() - 1

    return data

def calculate_rsi_signals(df, window=14, oversold=30, overbought=70):
    """Calculate RSI signals."""
    # Make a copy of the data
    data = df.copy()

    # Determine price column
    price_col = 'Price' if 'Price' in data.columns else 'close'
    if price_col not in data.columns:
        # Try to find a suitable price column
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
        else:
            raise ValueError("No suitable price column found in data")

    # Calculate price changes
    delta = data[price_col].diff()

    # Calculate gains and losses
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)

    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Calculate signals
    data['signal'] = 0
    data.loc[data['rsi'] < oversold, 'signal'] = 1  # Buy when oversold
    data.loc[data['rsi'] > overbought, 'signal'] = -1  # Sell when overbought

    # Calculate position changes
    data['position_change'] = data['signal'].diff()

    # Calculate returns
    data['returns'] = data[price_col].pct_change()

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
    """Main function for the simple trading dashboard."""
    # Set page config
    st.set_page_config(
        page_title="Oil & Gas Trading Dashboard",
        page_icon="📈",
        layout="wide"
    )

    # Title and description
    st.title("📈 Oil & Gas Trading Dashboard")
    st.markdown(
        """
        This dashboard provides tools for backtesting trading strategies and analyzing performance
        for oil and gas commodities.
        """
    )

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Strategy Backtesting", "Risk Analysis"]
    )

    # Available commodities
    commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']
    available_commodities = []

    # Load data for all commodities
    data_dict = {}
    for commodity in commodities:
        df = load_data(commodity)
        if not df.empty:
            data_dict[commodity] = df
            available_commodities.append(commodity)

    if not available_commodities:
        st.error("No commodity data found. Please run the data pipeline first.")
        return

    # Strategy Backtesting page
    if page == "Strategy Backtesting":
        st.header("Strategy Backtesting")

        # Select commodity
        selected_commodity = st.selectbox(
            "Select a commodity",
            available_commodities,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Select strategy
        strategy_types = {
            'ma_crossover': 'Moving Average Crossover',
            'rsi': 'RSI'
        }

        selected_strategy = st.selectbox(
            "Select a strategy",
            list(strategy_types.keys()),
            format_func=lambda x: strategy_types[x]
        )

        # Strategy parameters
        st.subheader("Strategy Parameters")

        strategy_params = {}

        if selected_strategy == 'ma_crossover':
            col1, col2 = st.columns(2)
            with col1:
                fast_window = st.slider("Fast Window", 5, 50, 10)
            with col2:
                slow_window = st.slider("Slow Window", 20, 200, 50)

            strategy_params = {
                'fast_window': fast_window,
                'slow_window': slow_window
            }

        elif selected_strategy == 'rsi':
            col1, col2, col3 = st.columns(3)
            with col1:
                window = st.slider("RSI Window", 5, 30, 14)
            with col2:
                oversold = st.slider("Oversold Level", 10, 40, 30)
            with col3:
                overbought = st.slider("Overbought Level", 60, 90, 70)

            strategy_params = {
                'window': window,
                'oversold': oversold,
                'overbought': overbought
            }

        # Run backtest button
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    # Load data
                    df = data_dict[selected_commodity]

                    # Run strategy
                    if selected_strategy == 'ma_crossover':
                        results = calculate_moving_average_signals(
                            df,
                            fast_window=strategy_params['fast_window'],
                            slow_window=strategy_params['slow_window']
                        )
                    elif selected_strategy == 'rsi':
                        results = calculate_rsi_signals(
                            df,
                            window=strategy_params['window'],
                            oversold=strategy_params['oversold'],
                            overbought=strategy_params['overbought']
                        )

                    # Calculate metrics
                    metrics = calculate_performance_metrics(results['strategy_returns'])

                    # Display results
                    st.subheader("Backtest Results")

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Return", metrics['Total Return'])
                    with col2:
                        st.metric("Annualized Return", metrics['Annualized Return'])
                    with col3:
                        st.metric("Sharpe Ratio", metrics['Sharpe Ratio'])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Volatility", metrics['Volatility'])
                    with col2:
                        st.metric("Max Drawdown", metrics['Max Drawdown'])
                    with col3:
                        st.metric("Win Rate", metrics['Win Rate'])

                    # Plot results
                    st.subheader("Performance Chart")

                    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

                    # Plot price and signals
                    ax[0].plot(results.index, results['Price'], label='Price')

                    if selected_strategy == 'ma_crossover':
                        ax[0].plot(results.index, results['fast_ma'], label=f'{strategy_params["fast_window"]}-day MA')
                        ax[0].plot(results.index, results['slow_ma'], label=f'{strategy_params["slow_window"]}-day MA')
                    elif selected_strategy == 'rsi':
                        ax2 = ax[0].twinx()
                        ax2.plot(results.index, results['rsi'], label='RSI', color='purple', alpha=0.5)
                        ax2.axhline(y=strategy_params['oversold'], color='green', linestyle='--')
                        ax2.axhline(y=strategy_params['overbought'], color='red', linestyle='--')
                        ax2.set_ylabel('RSI')
                        ax2.legend(loc='upper right')

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
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error running backtest: {e}")

    # Risk Analysis page
    elif page == "Risk Analysis":
        st.header("Risk Analysis")

        # Select commodity
        selected_commodity = st.selectbox(
            "Select a commodity",
            available_commodities,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Load data
        df = data_dict[selected_commodity]

        # Calculate returns
        returns = df['Price'].pct_change().dropna()

        # Display basic statistics
        st.subheader("Return Statistics")

        stats = {
            'Mean Daily Return': f"{returns.mean():.4%}",
            'Daily Volatility': f"{returns.std():.4%}",
            'Annualized Volatility': f"{returns.std() * np.sqrt(252):.4%}",
            'Minimum Return': f"{returns.min():.4%}",
            'Maximum Return': f"{returns.max():.4%}"
        }

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Daily Return", stats['Mean Daily Return'])
        with col2:
            st.metric("Daily Volatility", stats['Daily Volatility'])
        with col3:
            st.metric("Annualized Volatility", stats['Annualized Volatility'])

        # Plot return distribution
        st.subheader("Return Distribution")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(returns, bins=50, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--')
        ax.axvline(x=returns.mean(), color='red', linestyle='-', label=f'Mean: {returns.mean():.4%}')
        ax.axvline(x=returns.mean() - 2*returns.std(), color='orange', linestyle='--', label=f'2σ Down: {returns.mean() - 2*returns.std():.4%}')
        ax.axvline(x=returns.mean() + 2*returns.std(), color='green', linestyle='--', label=f'2σ Up: {returns.mean() + 2*returns.std():.4%}')

        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{selected_commodity.replace("_", " ").title()} Daily Return Distribution')
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # Calculate and display Value at Risk
        st.subheader("Value at Risk (VaR)")

        confidence_levels = [0.95, 0.99]
        var_values = {}

        for cl in confidence_levels:
            # Historical VaR
            var_percentile = 1 - cl
            historical_var = -np.percentile(returns, var_percentile * 100)

            # Parametric VaR
            z_score = scipy_stats.norm.ppf(cl)
            parametric_var = -(returns.mean() + z_score * returns.std())

            var_values[f'{cl:.0%} Historical VaR'] = f"{historical_var:.4%}"
            var_values[f'{cl:.0%} Parametric VaR'] = f"{parametric_var:.4%}"

        col1, col2 = st.columns(2)
        with col1:
            st.metric("95% Historical VaR", var_values['95% Historical VaR'])
            st.metric("99% Historical VaR", var_values['99% Historical VaR'])
        with col2:
            st.metric("95% Parametric VaR", var_values['95% Parametric VaR'])
            st.metric("99% Parametric VaR", var_values['99% Parametric VaR'])

if __name__ == "__main__":
    main()
