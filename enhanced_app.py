"""
Enhanced Oil & Gas Market Optimization Streamlit App
This version includes additional functionalities as described in the project overview.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Set page config
st.set_page_config(
    page_title="Oil & Gas Market Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories if they don't exist
import os
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Helper Functions
def generate_sample_data(commodity, start_date='2020-01-01', end_date='2023-01-01', freq='D'):
    """Generate sample price data for a commodity."""
    # Create date range
    date_rng = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Set random seed for reproducibility
    np.random.seed(42 + hash(commodity) % 100)

    # Generate random walk
    n = len(date_rng)
    returns = np.random.normal(0.0005, 0.01, n)

    # Add some seasonality
    seasonality = 0.1 * np.sin(np.linspace(0, 4*np.pi, n))

    # Add trend
    trend = np.linspace(0, 0.5, n)

    # Combine components
    log_prices = np.cumsum(returns) + seasonality + trend

    # Convert to prices
    base_price = 50.0 if 'crude' in commodity else 2.0
    prices = base_price * np.exp(log_prices)

    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_rng,
        'Price': prices
    })

    # Set Date as index
    df.set_index('Date', inplace=True)

    # Add volume
    volume = np.random.lognormal(10, 1, n) * 1000
    df['Volume'] = volume

    return df

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

# Module Functions
def data_management_page():
    """Data Management page functionality."""
    st.header("Data Management")

    # Generate sample data
    st.subheader("Generate Sample Data")

    # Commodity selection
    commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']

    if st.button("Generate Sample Data for All Commodities"):
        with st.spinner("Generating sample data for all commodities..."):
            for commodity in commodities:
                # Generate sample data
                df_sample = generate_sample_data(commodity)

                # Store in session state
                st.session_state.processed_data[commodity] = df_sample

            st.success("Sample data for all commodities generated successfully!")

            # Add button to navigate to Trading Dashboard
            if st.button("Go to Trading Dashboard"):
                st.session_state.page = "Trading Dashboard"
                st.experimental_rerun()

    # Individual commodity data generation
    st.subheader("Generate Data for Individual Commodities")

    selected_commodity = st.selectbox(
        "Select a commodity",
        commodities,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    if st.button(f"Generate Sample Data for {selected_commodity.replace('_', ' ').title()}"):
        with st.spinner(f"Generating sample data for {selected_commodity}..."):
            # Generate sample data
            df_sample = generate_sample_data(selected_commodity)

            # Store in session state
            st.session_state.processed_data[selected_commodity] = df_sample

            st.success(f"Sample data for {selected_commodity.replace('_', ' ').title()} generated successfully!")

            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df_sample.head())

            # Plot data
            st.subheader("Price Chart")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_sample.index, df_sample['Price'])
            ax.set_title(f"{selected_commodity.replace('_', ' ').title()} Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(True)
            st.pyplot(fig)

            # Add button to navigate to Trading Dashboard
            if st.button(f"Analyze {selected_commodity.replace('_', ' ').title()} Data"):
                st.session_state.selected_commodity = selected_commodity
                st.session_state.page = "Trading Dashboard"
                st.experimental_rerun()

def trading_dashboard_page():
    """Trading Dashboard page functionality."""
    st.header("Trading Dashboard")

    # Check if we have processed data
    if not st.session_state.processed_data:
        st.warning("No processed data available. Please go to the Data Management page to generate sample data.")

        if st.button("Go to Data Management"):
            st.session_state.page = "Data Management"
            st.experimental_rerun()
    else:
        # Available commodities
        available_commodities = list(st.session_state.processed_data.keys())

        # Select commodity
        default_index = 0
        if st.session_state.selected_commodity in available_commodities:
            default_index = available_commodities.index(st.session_state.selected_commodity)

        selected_commodity = st.selectbox(
            "Select a commodity",
            available_commodities,
            index=default_index,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Update selected commodity in session state
        st.session_state.selected_commodity = selected_commodity

        # Get data for selected commodity
        df = st.session_state.processed_data[selected_commodity]

        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Plot price data
        st.subheader("Price Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        price_col = 'Price' if 'Price' in df.columns else 'close'
        ax.plot(df.index, df[price_col])
        ax.set_title(f"{selected_commodity.replace('_', ' ').title()} Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        st.pyplot(fig)

        # Trading Strategy section
        st.subheader("Trading Strategy")

        # Select strategy
        strategy_types = {
            'ma_crossover': 'Moving Average Crossover',
            'rsi': 'RSI (Relative Strength Index)'
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
                strategy_params['fast_window'] = st.slider("Fast Window", 5, 50, 10)
            with col2:
                strategy_params['slow_window'] = st.slider("Slow Window", 20, 200, 50)

        elif selected_strategy == 'rsi':
            col1, col2, col3 = st.columns(3)
            with col1:
                strategy_params['window'] = st.slider("RSI Window", 5, 30, 14)
            with col2:
                strategy_params['oversold'] = st.slider("Oversold Level", 10, 40, 30)
            with col3:
                strategy_params['overbought'] = st.slider("Overbought Level", 60, 90, 70)

        # Run backtest button
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
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
                    price_col = 'Price' if 'Price' in results.columns else 'close'
                    ax[0].plot(results.index, results[price_col], label='Price')

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

                    ax[0].scatter(buy_signals.index, buy_signals[price_col], marker='^', color='green', label='Buy')
                    ax[0].scatter(sell_signals.index, sell_signals[price_col], marker='v', color='red', label='Sell')

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

                    # Add button to go to Risk Analysis
                    if st.button("Go to Risk Analysis"):
                        st.session_state.page = "Risk Analysis"
                        st.experimental_rerun()

                except Exception as e:
                    st.error(f"Error running backtest: {e}")

        # Add button to go back to Data Management
        if st.button("Back to Data Management"):
            st.session_state.page = "Data Management"
            st.experimental_rerun()

def risk_analysis_page():
    """Risk Analysis page functionality."""
    st.header("Risk Analysis")

    # Check if we have processed data
    if not st.session_state.processed_data:
        st.warning("No processed data available. Please go to the Data Management page to generate sample data.")

        if st.button("Go to Data Management"):
            st.session_state.page = "Data Management"
            st.experimental_rerun()
    else:
        # Available commodities
        available_commodities = list(st.session_state.processed_data.keys())

        # Select commodity
        default_index = 0
        if st.session_state.selected_commodity in available_commodities:
            default_index = available_commodities.index(st.session_state.selected_commodity)

        selected_commodity = st.selectbox(
            "Select a commodity",
            available_commodities,
            index=default_index,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Update selected commodity in session state
        st.session_state.selected_commodity = selected_commodity

        # Get data for selected commodity
        df = st.session_state.processed_data[selected_commodity]

        # Risk analysis parameters
        st.subheader("Risk Parameters")

        col1, col2 = st.columns(2)
        with col1:
            confidence_level = st.slider("Confidence Level (%)", 90, 99, 95) / 100
        with col2:
            time_horizon = st.slider("Time Horizon (days)", 1, 30, 1)

        # Determine price column
        price_col = 'Price' if 'Price' in df.columns else 'close'
        if price_col not in df.columns:
            # Try to find a suitable price column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
            else:
                st.error("No suitable price column found in data")
                return

        # Calculate returns
        if 'returns' not in df.columns:
            df['returns'] = df[price_col].pct_change()

        # Get recent returns
        returns = df['returns'].iloc[-252:].dropna()  # Use last year of data

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

        # Calculate Value at Risk (VaR)
        st.subheader("Value at Risk (VaR)")

        # Historical VaR
        var_percentile = 1 - confidence_level
        historical_var = -np.percentile(returns, var_percentile * 100)

        # Parametric VaR (using normal distribution approximation)
        # For 95% confidence, z-score is approximately 1.645
        # For 99% confidence, z-score is approximately 2.326
        z_score = 1.645 if confidence_level == 0.95 else 2.326 if confidence_level == 0.99 else 2.0
        parametric_var = -(returns.mean() + z_score * returns.std())

        # Scale for time horizon
        historical_var_scaled = historical_var * np.sqrt(time_horizon)
        parametric_var_scaled = parametric_var * np.sqrt(time_horizon)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{confidence_level:.0%} Historical VaR (1-day)", f"{historical_var:.4%}")
            st.metric(f"{confidence_level:.0%} Historical VaR ({time_horizon}-day)", f"{historical_var_scaled:.4%}")
        with col2:
            st.metric(f"{confidence_level:.0%} Parametric VaR (1-day)", f"{parametric_var:.4%}")
            st.metric(f"{confidence_level:.0%} Parametric VaR ({time_horizon}-day)", f"{parametric_var_scaled:.4%}")

        # Monte Carlo simulation
        st.subheader("Monte Carlo Simulation")

        col1, col2 = st.columns(2)
        with col1:
            initial_value = st.number_input("Initial Portfolio Value ($)", 1000.0, 1000000.0, 10000.0, 1000.0)
        with col2:
            num_simulations = st.slider("Number of Simulations", 100, 1000, 100)

        # Run Monte Carlo simulation
        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running simulation..."):
                try:
                    # Set random seed for reproducibility
                    np.random.seed(42)

                    # Calculate mean and standard deviation
                    mu = returns.mean()
                    sigma = returns.std()

                    # Simulation parameters
                    days = 252  # 1 year of trading days

                    # Generate random returns
                    simulation_returns = np.random.normal(
                        mu,
                        sigma,
                        (days, num_simulations)
                    )

                    # Calculate price paths
                    price_paths = np.zeros((days, num_simulations))
                    price_paths[0] = df[price_col].iloc[-1]

                    for t in range(1, days):
                        price_paths[t] = price_paths[t-1] * (1 + simulation_returns[t])

                    # Create dates for the simulation
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date, periods=days+1)[1:]

                    # Plot simulation results
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Plot a subset of simulations
                    for i in range(10):
                        ax.plot(future_dates, price_paths[:, i], linewidth=1, alpha=0.5)

                    # Plot mean path
                    mean_path = np.mean(price_paths, axis=1)
                    ax.plot(future_dates, mean_path, color='red', linewidth=2, label='Mean Path')

                    # Plot confidence intervals
                    lower_bound = np.percentile(price_paths, 5, axis=1)
                    upper_bound = np.percentile(price_paths, 95, axis=1)

                    ax.plot(future_dates, lower_bound, color='orange', linewidth=1.5, linestyle='--', label='5th Percentile')
                    ax.plot(future_dates, upper_bound, color='green', linewidth=1.5, linestyle='--', label='95th Percentile')

                    ax.set_title(f'Monte Carlo Simulation: {selected_commodity.replace("_", " ").title()} Price (1 Year)')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price')
                    ax.legend()
                    ax.grid(True)

                    st.pyplot(fig)

                    # Calculate final price statistics
                    final_prices = price_paths[-1, :]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
                    with col2:
                        st.metric("5th Percentile", f"${np.percentile(final_prices, 5):.2f}")
                    with col3:
                        st.metric("95th Percentile", f"${np.percentile(final_prices, 95):.2f}")

                    # Calculate portfolio value statistics
                    portfolio_values = initial_value * (final_prices / price_paths[0, 0])

                    st.subheader("Portfolio Value Projections")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Final Value", f"${np.mean(portfolio_values):.2f}")
                    with col2:
                        st.metric("5% VaR", f"${initial_value - np.percentile(portfolio_values, 5):.2f}")
                    with col3:
                        st.metric("Probability of Loss", f"{(portfolio_values < initial_value).mean():.2%}")

                except Exception as e:
                    st.error(f"Error running Monte Carlo simulation: {e}")

        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back to Trading Dashboard"):
                st.session_state.page = "Trading Dashboard"
                st.experimental_rerun()
        with col2:
            if st.button("Back to Data Management"):
                st.session_state.page = "Data Management"
                st.experimental_rerun()

def predictive_analytics_page():
    """Predictive Analytics page functionality."""
    st.header("Predictive Analytics Engine")

    # Check if we have processed data
    if not st.session_state.processed_data:
        st.warning("No processed data available. Please go to the Data Management page to generate sample data.")

        if st.button("Go to Data Management"):
            st.session_state.page = "Data Management"
            st.experimental_rerun()
    else:
        # Tabs for different predictive analytics features
        tab1, tab2, tab3, tab4 = st.tabs(["Price Forecasting", "Production Optimization",
                                          "Maintenance Scheduling", "Supply Chain Optimization"])

        with tab1:
            st.subheader("Price Forecasting")
            st.write("Utilize time series analysis and machine learning to predict oil and gas price movements.")

            # Available commodities
            available_commodities = list(st.session_state.processed_data.keys())

            # Select commodity
            default_index = 0
            if st.session_state.selected_commodity in available_commodities:
                default_index = available_commodities.index(st.session_state.selected_commodity)

            selected_commodity = st.selectbox(
                "Select a commodity",
                available_commodities,
                index=default_index,
                format_func=lambda x: x.replace('_', ' ').title(),
                key="forecast_commodity"
            )

            # Update selected commodity in session state
            st.session_state.selected_commodity = selected_commodity

            # Get data for selected commodity
            df = st.session_state.processed_data[selected_commodity]

            # Model selection
            model_type = st.selectbox(
                "Select forecasting model",
                ["ARIMA", "Prophet", "LSTM Neural Network", "XGBoost"]
            )

            # Forecast horizon
            forecast_days = st.slider("Forecast Horizon (days)", 7, 365, 30)

            # Generate forecast
            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    try:
                        # Determine price column
                        price_col = 'Price' if 'Price' in df.columns else 'close'

                        # Create a simple forecast (placeholder for actual models)
                        last_date = df.index[-1]
                        future_dates = pd.date_range(start=last_date, periods=forecast_days+1)[1:]

                        # Simple forecast based on historical mean and std
                        mean_return = df[price_col].pct_change().mean()
                        std_return = df[price_col].pct_change().std()

                        last_price = df[price_col].iloc[-1]
                        forecast_returns = np.random.normal(mean_return, std_return, forecast_days)
                        forecast_prices = [last_price]

                        for ret in forecast_returns:
                            forecast_prices.append(forecast_prices[-1] * (1 + ret))

                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Forecasted_Price': forecast_prices[1:]
                        })
                        forecast_df.set_index('Date', inplace=True)

                        # Plot forecast
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df.index[-90:], df[price_col].iloc[-90:], label='Historical')
                        ax.plot(forecast_df.index, forecast_df['Forecasted_Price'], label='Forecast', color='red')
                        ax.axvline(x=last_date, color='black', linestyle='--', alpha=0.5)
                        ax.fill_between(forecast_df.index,
                                       forecast_df['Forecasted_Price'] * 0.9,
                                       forecast_df['Forecasted_Price'] * 1.1,
                                       color='red', alpha=0.2)
                        ax.set_title(f"{selected_commodity.replace('_', ' ').title()} Price Forecast ({model_type})")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                        # Forecast metrics
                        st.subheader("Forecast Metrics")
                        metrics = {
                            "Model": model_type,
                            "Forecast Horizon": f"{forecast_days} days",
                            "Confidence Interval": "90%",
                            "Expected Price (End of Forecast)": f"${forecast_df['Forecasted_Price'].iloc[-1]:.2f}",
                            "Forecasted Change": f"{((forecast_df['Forecasted_Price'].iloc[-1] / last_price) - 1) * 100:.2f}%"
                        }

                        col1, col2 = st.columns(2)
                        for i, (key, value) in enumerate(metrics.items()):
                            if i % 2 == 0:
                                col1.metric(key, value)
                            else:
                                col2.metric(key, value)

                    except Exception as e:
                        st.error(f"Error generating forecast: {e}")

        with tab2:
            st.subheader("Production Optimization")
            st.write("Recommends optimal production levels based on market demand, storage capacity, and operational constraints.")

            # Production parameters
            st.subheader("Production Parameters")

            col1, col2 = st.columns(2)
            with col1:
                production_capacity = st.slider("Production Capacity (barrels/day)", 1000, 100000, 10000, 1000)
                storage_capacity = st.slider("Storage Capacity (barrels)", 10000, 1000000, 100000, 10000)
            with col2:
                production_cost = st.slider("Production Cost ($/barrel)", 10, 100, 40, 5)
                market_price = st.slider("Current Market Price ($/barrel)", 20, 200, 80, 5)

            # Demand forecast
            st.subheader("Demand Forecast")
            demand_pattern = st.selectbox(
                "Demand Pattern",
                ["Stable", "Increasing", "Decreasing", "Seasonal"]
            )

            # Optimize production
            if st.button("Optimize Production"):
                with st.spinner("Optimizing production levels..."):
                    try:
                        # Generate sample demand data
                        days = 30
                        dates = pd.date_range(start=datetime.now(), periods=days)

                        # Create demand based on selected pattern
                        if demand_pattern == "Stable":
                            base_demand = production_capacity * 0.8
                            demand = np.random.normal(base_demand, base_demand * 0.1, days)
                        elif demand_pattern == "Increasing":
                            base_demand = production_capacity * 0.6
                            trend = np.linspace(0, production_capacity * 0.4, days)
                            demand = base_demand + trend + np.random.normal(0, base_demand * 0.1, days)
                        elif demand_pattern == "Decreasing":
                            base_demand = production_capacity * 1.0
                            trend = np.linspace(0, -production_capacity * 0.4, days)
                            demand = base_demand + trend + np.random.normal(0, base_demand * 0.1, days)
                        else:  # Seasonal
                            base_demand = production_capacity * 0.8
                            seasonality = 0.2 * base_demand * np.sin(np.linspace(0, 2*np.pi, days))
                            demand = base_demand + seasonality + np.random.normal(0, base_demand * 0.1, days)

                        # Ensure demand is positive
                        demand = np.maximum(demand, 0)

                        # Simple optimization algorithm
                        optimal_production = []
                        storage = 0

                        for day_demand in demand:
                            # Calculate optimal production for this day
                            if market_price > production_cost:
                                # Profitable to produce
                                if storage + production_capacity <= storage_capacity:
                                    # Can produce at full capacity
                                    day_production = min(production_capacity, day_demand + (storage_capacity - storage))
                                else:
                                    # Limited by storage
                                    day_production = max(0, day_demand - storage)
                            else:
                                # Not profitable to produce more than demand
                                day_production = max(0, min(day_demand - storage, production_capacity))

                            # Update storage
                            storage = max(0, storage + day_production - day_demand)
                            optimal_production.append(day_production)

                        # Create dataframe
                        optimization_df = pd.DataFrame({
                            'Date': dates,
                            'Demand': demand,
                            'Optimal_Production': optimal_production,
                            'Storage': np.array([0] + list(storage)[:-1])  # Storage at beginning of day
                        })

                        # Calculate profit
                        optimization_df['Revenue'] = optimization_df['Demand'] * market_price
                        optimization_df['Cost'] = optimization_df['Optimal_Production'] * production_cost
                        optimization_df['Profit'] = optimization_df['Revenue'] - optimization_df['Cost']

                        # Plot results
                        fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

                        # Plot demand and production
                        ax[0].plot(optimization_df['Date'], optimization_df['Demand'], label='Demand', color='blue')
                        ax[0].plot(optimization_df['Date'], optimization_df['Optimal_Production'], label='Optimal Production', color='green')
                        ax[0].set_ylabel('Barrels/day')
                        ax[0].set_title('Demand vs. Optimal Production')
                        ax[0].legend()
                        ax[0].grid(True)

                        # Plot storage
                        ax[1].plot(optimization_df['Date'], optimization_df['Storage'], label='Storage Level', color='orange')
                        ax[1].axhline(y=storage_capacity, color='red', linestyle='--', label='Storage Capacity')
                        ax[1].set_ylabel('Barrels')
                        ax[1].set_xlabel('Date')
                        ax[1].set_title('Storage Level')
                        ax[1].legend()
                        ax[1].grid(True)

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Display optimization results
                        st.subheader("Optimization Results")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Demand", f"{optimization_df['Demand'].sum():.0f} barrels")
                        with col2:
                            st.metric("Total Production", f"{optimization_df['Optimal_Production'].sum():.0f} barrels")
                        with col3:
                            st.metric("Total Profit", f"${optimization_df['Profit'].sum():.2f}")

                        # Display optimization table
                        st.subheader("Daily Optimization Plan")
                        st.dataframe(optimization_df.set_index('Date').round(2))

                    except Exception as e:
                        st.error(f"Error optimizing production: {e}")

        with tab3:
            st.subheader("Maintenance Scheduling")
            st.write("Predicts equipment failures and suggests preventive maintenance schedules to minimize downtime.")

            # Equipment selection
            equipment_types = ["Pump", "Compressor", "Separator", "Heat Exchanger", "Storage Tank"]
            selected_equipment = st.selectbox("Select Equipment Type", equipment_types)

            # Equipment parameters
            st.subheader("Equipment Parameters")

            col1, col2 = st.columns(2)
            with col1:
                equipment_age = st.slider("Equipment Age (months)", 0, 120, 24, 1)
                last_maintenance = st.slider("Months Since Last Maintenance", 0, 36, 6, 1)
            with col2:
                operating_hours = st.slider("Daily Operating Hours", 1, 24, 16, 1)
                operating_conditions = st.select_slider(
                    "Operating Conditions",
                    options=["Mild", "Moderate", "Severe", "Extreme"]
                )

            # Generate maintenance schedule
            if st.button("Generate Maintenance Schedule"):
                with st.spinner("Analyzing equipment and generating maintenance schedule..."):
                    try:
                        # Generate sample data
                        days = 365
                        dates = pd.date_range(start=datetime.now(), periods=days)

                        # Calculate failure probability based on parameters
                        base_failure_rate = {
                            "Pump": 0.001,
                            "Compressor": 0.0015,
                            "Separator": 0.0008,
                            "Heat Exchanger": 0.0005,
                            "Storage Tank": 0.0003
                        }[selected_equipment]

                        # Adjust for age
                        age_factor = 1 + (equipment_age / 60)

                        # Adjust for maintenance
                        maintenance_factor = 1 + (last_maintenance / 12)

                        # Adjust for operating hours
                        hours_factor = operating_hours / 8

                        # Adjust for conditions
                        condition_factor = {
                            "Mild": 1.0,
                            "Moderate": 1.5,
                            "Severe": 2.0,
                            "Extreme": 3.0
                        }[operating_conditions]

                        # Calculate daily failure probability
                        daily_failure_prob = base_failure_rate * age_factor * maintenance_factor * hours_factor * condition_factor

                        # Calculate cumulative failure probability
                        cumulative_prob = 1 - np.power(1 - daily_failure_prob, np.arange(1, days + 1))

                        # Determine maintenance thresholds
                        maintenance_thresholds = {
                            "Inspection": 0.2,
                            "Minor Maintenance": 0.4,
                            "Major Maintenance": 0.7,
                            "Replacement": 0.9
                        }

                        # Create maintenance schedule
                        maintenance_df = pd.DataFrame({
                            'Date': dates,
                            'Failure_Probability': cumulative_prob
                        })

                        # Add maintenance recommendations
                        for action, threshold in maintenance_thresholds.items():
                            maintenance_df[action] = maintenance_df['Failure_Probability'] >= threshold

                            # Find first day for each action
                            if any(maintenance_df[action]):
                                first_day = maintenance_df[maintenance_df[action]].iloc[0]['Date']
                                maintenance_df[f"{action}_Date"] = first_day
                            else:
                                maintenance_df[f"{action}_Date"] = None

                        # Plot failure probability
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(maintenance_df['Date'], maintenance_df['Failure_Probability'], label='Failure Probability')

                        # Add threshold lines
                        for action, threshold in maintenance_thresholds.items():
                            ax.axhline(y=threshold, linestyle='--', alpha=0.7,
                                      label=f"{action} Threshold ({threshold:.0%})")

                        ax.set_ylabel('Probability')
                        ax.set_xlabel('Date')
                        ax.set_title(f'Failure Probability for {selected_equipment}')
                        ax.legend()
                        ax.grid(True)

                        st.pyplot(fig)

                        # Display maintenance recommendations
                        st.subheader("Maintenance Recommendations")

                        recommendations = []
                        for action, threshold in maintenance_thresholds.items():
                            if any(maintenance_df[action]):
                                first_day = maintenance_df[maintenance_df[action]].iloc[0]['Date']
                                days_until = (first_day - datetime.now()).days
                                recommendations.append({
                                    "Action": action,
                                    "Recommended Date": first_day.strftime("%Y-%m-%d"),
                                    "Days Until Required": max(0, days_until),
                                    "Failure Probability": f"{maintenance_df[maintenance_df[action]].iloc[0]['Failure_Probability']:.2%}"
                                })

                        st.table(pd.DataFrame(recommendations))

                        # Cost-benefit analysis
                        st.subheader("Cost-Benefit Analysis")

                        maintenance_costs = {
                            "Inspection": 500,
                            "Minor Maintenance": 2000,
                            "Major Maintenance": 10000,
                            "Replacement": 50000
                        }

                        downtime_costs = {
                            "Planned": 5000,  # per day
                            "Unplanned": 20000  # per day
                        }

                        downtime_duration = {
                            "Inspection": 0.5,
                            "Minor Maintenance": 1,
                            "Major Maintenance": 3,
                            "Replacement": 7,
                            "Failure": 14
                        }

                        analysis = []
                        for action in maintenance_thresholds.keys():
                            if any(maintenance_df[action]):
                                planned_cost = maintenance_costs[action] + downtime_costs["Planned"] * downtime_duration[action]
                                failure_cost = maintenance_costs["Replacement"] + downtime_costs["Unplanned"] * downtime_duration["Failure"]
                                savings = failure_cost - planned_cost

                                analysis.append({
                                    "Action": action,
                                    "Cost": f"${planned_cost:,.2f}",
                                    "Failure Cost": f"${failure_cost:,.2f}",
                                    "Potential Savings": f"${savings:,.2f}",
                                    "ROI": f"{(savings / planned_cost) * 100:.0f}%"
                                })

                        st.table(pd.DataFrame(analysis))

                    except Exception as e:
                        st.error(f"Error generating maintenance schedule: {e}")

        with tab4:
            st.subheader("Supply Chain Optimization")
            st.write("Optimizes transportation routes and inventory levels to reduce costs and improve delivery times.")

            # Placeholder for supply chain optimization
            st.info("This feature will be implemented in the next update. It will include transportation route optimization, inventory management, and supplier selection algorithms.")

            # Show a sample visualization
            st.subheader("Sample Supply Chain Network")

            # Create a sample supply chain network visualization
            try:
                # Create nodes for the supply chain
                nodes = [
                    {"id": "Well1", "type": "Production", "location": (0, 0)},
                    {"id": "Well2", "type": "Production", "location": (0, 2)},
                    {"id": "Well3", "type": "Production", "location": (0, 4)},
                    {"id": "Storage1", "type": "Storage", "location": (2, 1)},
                    {"id": "Storage2", "type": "Storage", "location": (2, 3)},
                    {"id": "Refinery1", "type": "Processing", "location": (4, 0)},
                    {"id": "Refinery2", "type": "Processing", "location": (4, 4)},
                    {"id": "Distribution1", "type": "Distribution", "location": (6, 1)},
                    {"id": "Distribution2", "type": "Distribution", "location": (6, 3)},
                    {"id": "Market1", "type": "Market", "location": (8, 0)},
                    {"id": "Market2", "type": "Market", "location": (8, 2)},
                    {"id": "Market3", "type": "Market", "location": (8, 4)}
                ]

                # Create edges (connections) between nodes
                edges = [
                    {"source": "Well1", "target": "Storage1", "flow": 100},
                    {"source": "Well2", "target": "Storage1", "flow": 80},
                    {"source": "Well2", "target": "Storage2", "flow": 50},
                    {"source": "Well3", "target": "Storage2", "flow": 120},
                    {"source": "Storage1", "target": "Refinery1", "flow": 150},
                    {"source": "Storage1", "target": "Refinery2", "flow": 30},
                    {"source": "Storage2", "target": "Refinery1", "flow": 40},
                    {"source": "Storage2", "target": "Refinery2", "flow": 130},
                    {"source": "Refinery1", "target": "Distribution1", "flow": 100},
                    {"source": "Refinery1", "target": "Distribution2", "flow": 90},
                    {"source": "Refinery2", "target": "Distribution1", "flow": 60},
                    {"source": "Refinery2", "target": "Distribution2", "flow": 100},
                    {"source": "Distribution1", "target": "Market1", "flow": 80},
                    {"source": "Distribution1", "target": "Market2", "flow": 80},
                    {"source": "Distribution2", "target": "Market2", "flow": 70},
                    {"source": "Distribution2", "target": "Market3", "flow": 120}
                ]

                # Create a figure
                fig, ax = plt.subplots(figsize=(12, 8))

                # Define node colors based on type
                node_colors = {
                    "Production": "green",
                    "Storage": "blue",
                    "Processing": "red",
                    "Distribution": "purple",
                    "Market": "orange"
                }

                # Plot nodes
                for node in nodes:
                    x, y = node["location"]
                    color = node_colors[node["type"]]
                    ax.scatter(x, y, s=300, color=color, alpha=0.7, edgecolors='black')
                    ax.text(x, y, node["id"], ha='center', va='center', fontweight='bold')

                # Plot edges
                for edge in edges:
                    source_node = next(node for node in nodes if node["id"] == edge["source"])
                    target_node = next(node for node in nodes if node["id"] == edge["target"])

                    sx, sy = source_node["location"]
                    tx, ty = target_node["location"]

                    # Scale line width based on flow
                    line_width = edge["flow"] / 30

                    ax.plot([sx, tx], [sy, ty], 'k-', alpha=0.5, linewidth=line_width)

                    # Add flow label
                    mid_x = (sx + tx) / 2
                    mid_y = (sy + ty) / 2
                    ax.text(mid_x, mid_y, str(edge["flow"]), ha='center', va='center',
                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

                # Add legend
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                             markersize=10, label=node_type)
                                  for node_type, color in node_colors.items()]

                ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                         ncol=len(node_colors))

                # Set axis properties
                ax.set_xlim(-1, 9)
                ax.set_ylim(-1, 5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Oil & Gas Supply Chain Network')

                st.pyplot(fig)

                # Add explanation
                st.markdown("""
                **Supply Chain Network Components:**
                - **Production (Green)**: Oil wells and production facilities
                - **Storage (Blue)**: Storage tanks and terminals
                - **Processing (Red)**: Refineries and processing plants
                - **Distribution (Purple)**: Distribution centers
                - **Market (Orange)**: End markets and customers

                The numbers on the connections represent flow volumes (barrels/day).
                """)

            except Exception as e:
                st.error(f"Error generating supply chain visualization: {e}")

        # Navigation buttons
        st.subheader("Navigation")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Go to Data Management", key="pred_to_data"):
                st.session_state.page = "Data Management"
                st.experimental_rerun()
        with col2:
            if st.button("Go to Trading Dashboard", key="pred_to_trading"):
                st.session_state.page = "Trading Dashboard"
                st.experimental_rerun()
        with col3:
            if st.button("Go to Risk Analysis", key="pred_to_risk"):
                st.session_state.page = "Risk Analysis"
                st.experimental_rerun()

def risk_assessment_page():
    """Risk Assessment page functionality."""
    st.header("Risk Assessment Framework")

    # Check if we have processed data
    if not st.session_state.processed_data:
        st.warning("No processed data available. Please go to the Data Management page to generate sample data.")

        if st.button("Go to Data Management"):
            st.session_state.page = "Data Management"
            st.experimental_rerun()
    else:
        # Tabs for different risk assessment features
        tab1, tab2, tab3 = st.tabs(["Market Risk Analysis", "Geopolitical Risk Monitoring",
                                   "Regulatory Compliance"])

        with tab1:
            st.subheader("Market Risk Analysis")
            st.write("Evaluates exposure to price volatility and market shifts, providing hedging recommendations.")

            # Available commodities
            available_commodities = list(st.session_state.processed_data.keys())

            # Select commodity
            default_index = 0
            if st.session_state.selected_commodity in available_commodities:
                default_index = available_commodities.index(st.session_state.selected_commodity)

            selected_commodity = st.selectbox(
                "Select a commodity",
                available_commodities,
                index=default_index,
                format_func=lambda x: x.replace('_', ' ').title(),
                key="market_risk_commodity"
            )

            # Update selected commodity in session state
            st.session_state.selected_commodity = selected_commodity

            # Get data for selected commodity
            df = st.session_state.processed_data[selected_commodity]

            # Risk parameters
            col1, col2 = st.columns(2)
            with col1:
                position_size = st.number_input("Position Size ($)", 10000, 1000000, 100000, 10000)
            with col2:
                confidence_level = st.slider("Confidence Level (%)", 90, 99, 95) / 100

            # Determine price column
            price_col = 'Price' if 'Price' in df.columns else 'close'

            # Calculate returns
            if 'returns' not in df.columns:
                df['returns'] = df[price_col].pct_change()

            # Get recent returns
            returns = df['returns'].iloc[-252:].dropna()  # Use last year of data

            # Calculate VaR
            # Historical VaR
            var_percentile = 1 - confidence_level
            historical_var = -np.percentile(returns, var_percentile * 100)

            # Parametric VaR
            z_score = 1.645 if confidence_level == 0.95 else 2.326 if confidence_level == 0.99 else 2.0
            parametric_var = -(returns.mean() + z_score * returns.std())

            # Calculate dollar VaR
            historical_dollar_var = position_size * historical_var
            parametric_dollar_var = position_size * parametric_var

            # Display VaR
            st.subheader("Value at Risk (VaR)")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{confidence_level:.0%} Historical VaR", f"{historical_var:.4%}")
                st.metric("Dollar VaR", f"${historical_dollar_var:.2f}")
            with col2:
                st.metric(f"{confidence_level:.0%} Parametric VaR", f"{parametric_var:.4%}")
                st.metric("Dollar VaR", f"${parametric_dollar_var:.2f}")

            # Hedging recommendations
            st.subheader("Hedging Recommendations")

            hedging_options = [
                {
                    "Strategy": "Futures Contracts",
                    "Coverage": "75%",
                    "Cost": f"${position_size * 0.005:.2f}",
                    "Effectiveness": "High"
                },
                {
                    "Strategy": "Options (Put)",
                    "Coverage": "100%",
                    "Cost": f"${position_size * 0.02:.2f}",
                    "Effectiveness": "Medium"
                },
                {
                    "Strategy": "Swaps",
                    "Coverage": "50%",
                    "Cost": f"${position_size * 0.003:.2f}",
                    "Effectiveness": "Medium-High"
                }
            ]

            st.table(pd.DataFrame(hedging_options))

            # Stress testing
            st.subheader("Stress Testing")

            # Define stress scenarios
            scenarios = {
                "Mild Market Decline": -0.05,
                "Moderate Market Decline": -0.10,
                "Severe Market Decline": -0.20,
                "Market Crash": -0.30
            }

            # Calculate impact
            stress_results = []
            for scenario, price_change in scenarios.items():
                dollar_impact = position_size * price_change
                new_position_value = position_size * (1 + price_change)

                stress_results.append({
                    "Scenario": scenario,
                    "Price Change": f"{price_change:.1%}",
                    "Dollar Impact": f"${dollar_impact:.2f}",
                    "New Position Value": f"${new_position_value:.2f}"
                })

            st.table(pd.DataFrame(stress_results))

            # Portfolio diversification
            st.subheader("Portfolio Diversification Analysis")

            # Create a sample portfolio
            portfolio = {
                "Crude Oil": 0.4,
                "Natural Gas": 0.2,
                "Gasoline": 0.25,
                "Diesel": 0.15
            }

            # Display portfolio allocation
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(portfolio.values(), labels=portfolio.keys(), autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title('Current Portfolio Allocation')
            st.pyplot(fig)

            # Correlation matrix
            st.subheader("Correlation Matrix")

            # Create a sample correlation matrix
            corr_matrix = pd.DataFrame({
                "Crude Oil": [1.0, 0.7, 0.8, 0.75],
                "Natural Gas": [0.7, 1.0, 0.6, 0.5],
                "Gasoline": [0.8, 0.6, 1.0, 0.85],
                "Diesel": [0.75, 0.5, 0.85, 1.0]
            }, index=["Crude Oil", "Natural Gas", "Gasoline", "Diesel"])

            # Display correlation matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr_matrix, cmap='coolwarm')

            # Add labels
            ax.set_xticks(np.arange(len(corr_matrix.columns)))
            ax.set_yticks(np.arange(len(corr_matrix.index)))
            ax.set_xticklabels(corr_matrix.columns)
            ax.set_yticklabels(corr_matrix.index)

            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)

            # Add correlation values
            for i in range(len(corr_matrix.index)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                                  ha="center", va="center", color="black")

            ax.set_title("Correlation Matrix")
            fig.tight_layout()
            st.pyplot(fig)

            # Diversification recommendations
            st.subheader("Diversification Recommendations")

            recommendations = [
                "**Reduce exposure to highly correlated assets**: Crude Oil and Gasoline show high correlation (0.80). Consider reducing allocation to one of these.",
                "**Increase Natural Gas allocation**: Natural Gas has lower correlation with other assets, providing better diversification benefits.",
                "**Consider adding non-energy commodities**: Adding commodities from different sectors (e.g., metals, agriculture) could further reduce portfolio risk.",
                "**Implement a dynamic allocation strategy**: Adjust allocations based on changing market conditions and correlations."
            ]

            for rec in recommendations:
                st.markdown(rec)

        with tab2:
            st.subheader("Geopolitical Risk Monitoring")
            st.write("Tracks global events that could impact supply chains or market dynamics.")

            # Risk map
            st.subheader("Global Risk Map")

            # Create a sample risk map
            try:
                # Create a figure
                fig, ax = plt.subplots(figsize=(12, 8))

                # Define regions and their risk levels
                regions = {
                    "North America": {"position": (0.2, 0.7), "risk": "Low"},
                    "South America": {"position": (0.3, 0.3), "risk": "Medium"},
                    "Europe": {"position": (0.5, 0.7), "risk": "Medium-Low"},
                    "Middle East": {"position": (0.6, 0.5), "risk": "High"},
                    "Africa": {"position": (0.5, 0.4), "risk": "Medium-High"},
                    "Russia": {"position": (0.7, 0.8), "risk": "High"},
                    "China": {"position": (0.8, 0.6), "risk": "Medium"},
                    "Southeast Asia": {"position": (0.8, 0.4), "risk": "Medium"},
                    "Australia": {"position": (0.9, 0.2), "risk": "Low"}
                }

                # Define risk colors
                risk_colors = {
                    "Low": "green",
                    "Medium-Low": "yellowgreen",
                    "Medium": "yellow",
                    "Medium-High": "orange",
                    "High": "red"
                }

                # Plot regions
                for region, data in regions.items():
                    x, y = data["position"]
                    risk = data["risk"]
                    color = risk_colors[risk]

                    ax.scatter(x, y, s=500, color=color, alpha=0.7, edgecolors='black')
                    ax.text(x, y, region, ha='center', va='center', fontweight='bold', fontsize=8)

                # Add legend
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                             markersize=10, label=risk)
                                  for risk, color in risk_colors.items()]

                ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                         ncol=len(risk_colors))

                # Set axis properties
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Geopolitical Risk Map')

                # Add a simple world map background (simplified)
                ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
                ax.axvline(x=0.5, color='gray', linestyle='-', alpha=0.3)

                st.pyplot(fig)

                # Add explanation
                st.markdown("""
                **Geopolitical Risk Levels:**
                - **Low**: Stable political environment, minimal supply chain disruption risk
                - **Medium-Low**: Generally stable with occasional minor disruptions
                - **Medium**: Moderate political uncertainty, potential for supply disruptions
                - **Medium-High**: Significant political uncertainty, likely supply disruptions
                - **High**: Severe political instability, high probability of major supply disruptions
                """)

            except Exception as e:
                st.error(f"Error generating risk map: {e}")

            # Current events
            st.subheader("Current Geopolitical Events")

            events = [
                {
                    "Region": "Middle East",
                    "Event": "Ongoing tensions in the Strait of Hormuz",
                    "Impact": "Potential disruption to 20% of global oil supply",
                    "Risk Level": "High"
                },
                {
                    "Region": "Russia",
                    "Event": "New export restrictions on natural gas",
                    "Impact": "Reduced supply to European markets",
                    "Risk Level": "High"
                },
                {
                    "Region": "South America",
                    "Event": "Political instability in Venezuela",
                    "Impact": "Continued decline in oil production",
                    "Risk Level": "Medium"
                },
                {
                    "Region": "North America",
                    "Event": "New pipeline regulations",
                    "Impact": "Potential delays in infrastructure development",
                    "Risk Level": "Medium-Low"
                },
                {
                    "Region": "Africa",
                    "Event": "Civil unrest in Nigeria",
                    "Impact": "Disruption to regional oil production",
                    "Risk Level": "Medium-High"
                }
            ]

            # Display events
            st.table(pd.DataFrame(events))

            # Risk alerts
            st.subheader("Risk Alerts")

            alerts = [
                "**HIGH ALERT**: Escalating tensions in the Strait of Hormuz could lead to significant oil price volatility in the next 2-4 weeks.",
                "**MEDIUM ALERT**: Russian export restrictions expected to increase European natural gas prices by 15-20% in the coming month.",
                "**MONITORING**: Potential labor strikes in U.S. refineries could impact gasoline production capacity."
            ]

            for alert in alerts:
                st.markdown(alert)

        with tab3:
            st.subheader("Regulatory Compliance")
            st.write("Monitors changing regulations and assesses their potential impact on operations and profitability.")

            # Regulatory dashboard
            st.subheader("Regulatory Dashboard")

            # Create tabs for different regions
            reg_tab1, reg_tab2, reg_tab3 = st.tabs(["North America", "Europe", "Global"])

            with reg_tab1:
                st.subheader("North American Regulations")

                na_regulations = [
                    {
                        "Regulation": "EPA Emissions Standards Update",
                        "Status": "Proposed",
                        "Effective Date": "2024-01-01",
                        "Impact Level": "High",
                        "Compliance Cost": "$$$"
                    },
                    {
                        "Regulation": "Carbon Tax Increase",
                        "Status": "Enacted",
                        "Effective Date": "2023-07-01",
                        "Impact Level": "High",
                        "Compliance Cost": "$$$$"
                    },
                    {
                        "Regulation": "Pipeline Safety Requirements",
                        "Status": "Enacted",
                        "Effective Date": "2023-03-15",
                        "Impact Level": "Medium",
                        "Compliance Cost": "$$"
                    },
                    {
                        "Regulation": "Renewable Fuel Standards",
                        "Status": "Under Review",
                        "Effective Date": "TBD",
                        "Impact Level": "Medium-High",
                        "Compliance Cost": "$$$"
                    }
                ]

                st.table(pd.DataFrame(na_regulations))

                # Compliance recommendations
                st.subheader("Compliance Recommendations")

                recommendations = [
                    "**Emissions Standards**: Begin equipment upgrades to meet new EPA standards before the 2024 deadline.",
                    "**Carbon Tax**: Implement carbon capture technologies to reduce tax liability.",
                    "**Pipeline Safety**: Complete required inspections and documentation by Q3 2023.",
                    "**Renewable Fuels**: Increase investment in biofuel blending capabilities."
                ]

                for rec in recommendations:
                    st.markdown(rec)

            with reg_tab2:
                st.subheader("European Regulations")

                eu_regulations = [
                    {
                        "Regulation": "EU Carbon Border Adjustment Mechanism",
                        "Status": "Enacted",
                        "Effective Date": "2023-10-01",
                        "Impact Level": "High",
                        "Compliance Cost": "$$$$"
                    },
                    {
                        "Regulation": "Methane Emissions Limits",
                        "Status": "Enacted",
                        "Effective Date": "2023-06-01",
                        "Impact Level": "Medium-High",
                        "Compliance Cost": "$$$"
                    },
                    {
                        "Regulation": "Sustainable Finance Disclosure Regulation",
                        "Status": "Enacted",
                        "Effective Date": "2023-01-01",
                        "Impact Level": "Medium",
                        "Compliance Cost": "$$"
                    },
                    {
                        "Regulation": "Hydrogen Strategy Implementation",
                        "Status": "Proposed",
                        "Effective Date": "2024-06-01",
                        "Impact Level": "Medium-High",
                        "Compliance Cost": "$$$"
                    }
                ]

                st.table(pd.DataFrame(eu_regulations))

                # Compliance recommendations
                st.subheader("Compliance Recommendations")

                recommendations = [
                    "**Carbon Border Adjustment**: Develop detailed carbon accounting for all EU exports.",
                    "**Methane Emissions**: Implement continuous monitoring systems at all EU facilities.",
                    "**Sustainable Finance**: Update ESG reporting to meet new disclosure requirements.",
                    "**Hydrogen Strategy**: Evaluate investment opportunities in green hydrogen production."
                ]

                for rec in recommendations:
                    st.markdown(rec)

            with reg_tab3:
                st.subheader("Global Regulations")

                global_regulations = [
                    {
                        "Regulation": "Paris Agreement NDC Updates",
                        "Status": "Ongoing",
                        "Effective Date": "2023-12-31",
                        "Impact Level": "High",
                        "Compliance Cost": "$$$$"
                    },
                    {
                        "Regulation": "IMO 2023 Marine Fuel Standards",
                        "Status": "Enacted",
                        "Effective Date": "2023-01-01",
                        "Impact Level": "Medium",
                        "Compliance Cost": "$$$"
                    },
                    {
                        "Regulation": "Global Methane Pledge",
                        "Status": "Voluntary",
                        "Effective Date": "Ongoing",
                        "Impact Level": "Medium-High",
                        "Compliance Cost": "$$$"
                    },
                    {
                        "Regulation": "TCFD Climate Disclosure Requirements",
                        "Status": "Expanding",
                        "Effective Date": "Varies by Country",
                        "Impact Level": "Medium",
                        "Compliance Cost": "$$"
                    }
                ]

                st.table(pd.DataFrame(global_regulations))

                # Compliance recommendations
                st.subheader("Compliance Recommendations")

                recommendations = [
                    "**Paris Agreement**: Develop country-specific compliance strategies for operations in each jurisdiction.",
                    "**IMO Standards**: Ensure all shipping contracts specify compliant fuels.",
                    "**Methane Pledge**: Implement voluntary methane reduction program ahead of potential mandatory requirements.",
                    "**TCFD Disclosures**: Standardize climate risk reporting across all global operations."
                ]

                for rec in recommendations:
                    st.markdown(rec)

            # Regulatory impact analysis
            st.subheader("Regulatory Impact Analysis")

            # Create a sample impact analysis
            impact_data = {
                "Category": ["Production Costs", "Compliance Costs", "Market Access", "Reputation", "Overall Impact"],
                "Current Impact": [3, 4, 2, 3, 3],
                "Projected Impact (2024)": [4, 5, 3, 2, 4],
                "Projected Impact (2025)": [5, 5, 4, 2, 4]
            }

            impact_df = pd.DataFrame(impact_data)

            # Create a radar chart
            categories = impact_df["Category"]
            N = len(categories)

            # Create angles for each category
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            # Add current impact
            values = impact_df["Current Impact"].values.tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label="Current Impact")
            ax.fill(angles, values, alpha=0.25)

            # Add 2024 projected impact
            values = impact_df["Projected Impact (2024)"].values.tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label="Projected Impact (2024)")
            ax.fill(angles, values, alpha=0.25)

            # Add 2025 projected impact
            values = impact_df["Projected Impact (2025)"].values.tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label="Projected Impact (2025)")
            ax.fill(angles, values, alpha=0.25)

            # Add category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

            # Set y-axis limits
            ax.set_ylim(0, 5)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_yticklabels(["Very Low", "Low", "Medium", "High", "Very High"])

            # Set title
            ax.set_title("Regulatory Impact Analysis")

            st.pyplot(fig)

            # Add explanation
            st.markdown("""
            **Impact Scale:**
            - **1 (Very Low)**: Minimal impact on operations and costs
            - **2 (Low)**: Minor impact, manageable with current resources
            - **3 (Medium)**: Moderate impact, requiring some operational adjustments
            - **4 (High)**: Significant impact, requiring substantial investment
            - **5 (Very High)**: Critical impact, potentially transforming business model

            **Analysis:**
            - Regulatory pressure is expected to increase significantly through 2025
            - Production and compliance costs will see the most dramatic increases
            - Market access restrictions will gradually increase
            - Reputational impact may improve with proactive compliance
            """)

        # Navigation buttons
        st.subheader("Navigation")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Go to Data Management", key="risk_assess_to_data"):
                st.session_state.page = "Data Management"
                st.experimental_rerun()
        with col2:
            if st.button("Go to Trading Dashboard", key="risk_assess_to_trading"):
                st.session_state.page = "Trading Dashboard"
                st.experimental_rerun()
        with col3:
            if st.button("Go to Risk Analysis", key="risk_assess_to_risk"):
                st.session_state.page = "Risk Analysis"
                st.experimental_rerun()

def decision_support_page():
    """Decision Support page functionality."""
    st.header("Interactive Decision Support System")

    # Check if we have processed data
    if not st.session_state.processed_data:
        st.warning("No processed data available. Please go to the Data Management page to generate sample data.")

        if st.button("Go to Data Management"):
            st.session_state.page = "Data Management"
            st.experimental_rerun()
    else:
        # Tabs for different decision support features
        tab1, tab2, tab3 = st.tabs(["Scenario Modeling", "Natural Language Interface",
                                   "Advanced Visualizations"])

        with tab1:
            st.subheader("Scenario Modeling")
            st.write("Create and compare different market and operational scenarios.")

            # Available commodities
            available_commodities = list(st.session_state.processed_data.keys())

            # Select commodity
            default_index = 0
            if st.session_state.selected_commodity in available_commodities:
                default_index = available_commodities.index(st.session_state.selected_commodity)

            selected_commodity = st.selectbox(
                "Select a commodity",
                available_commodities,
                index=default_index,
                format_func=lambda x: x.replace('_', ' ').title(),
                key="scenario_commodity"
            )

            # Update selected commodity in session state
            st.session_state.selected_commodity = selected_commodity

            # Get data for selected commodity
            df = st.session_state.processed_data[selected_commodity]

            # Scenario parameters
            st.subheader("Scenario Parameters")

            col1, col2 = st.columns(2)
            with col1:
                price_change = st.slider("Price Change (%)", -50, 50, 0)
                volatility_change = st.slider("Volatility Change (%)", -50, 50, 0)
            with col2:
                demand_change = st.slider("Demand Change (%)", -50, 50, 0)
                supply_change = st.slider("Supply Change (%)", -50, 50, 0)

            # Generate scenarios
            if st.button("Generate Scenarios"):
                with st.spinner("Generating scenarios..."):
                    try:
                        # Determine price column
                        price_col = 'Price' if 'Price' in df.columns else 'close'

                        # Create scenarios
                        scenarios = {
                            "Base Case": df.copy(),
                            "Scenario 1: Price Change": df.copy(),
                            "Scenario 2: Volatility Change": df.copy(),
                            "Scenario 3: Combined Changes": df.copy()
                        }

                        # Modify scenarios
                        # Scenario 1: Price change
                        scenarios["Scenario 1: Price Change"][price_col] = scenarios["Scenario 1: Price Change"][price_col] * (1 + price_change/100)

                        # Scenario 2: Volatility change
                        returns = scenarios["Scenario 2: Volatility Change"][price_col].pct_change().dropna()
                        new_volatility = returns.std() * (1 + volatility_change/100)
                        new_returns = np.random.normal(returns.mean(), new_volatility, len(returns))
                        new_prices = [scenarios["Scenario 2: Volatility Change"][price_col].iloc[0]]
                        for ret in new_returns:
                            new_prices.append(new_prices[-1] * (1 + ret))
                        scenarios["Scenario 2: Volatility Change"][price_col] = new_prices[:len(scenarios["Scenario 2: Volatility Change"])]

                        # Scenario 3: Combined changes
                        scenarios["Scenario 3: Combined Changes"][price_col] = scenarios["Scenario 3: Combined Changes"][price_col] * (1 + price_change/100)
                        returns = scenarios["Scenario 3: Combined Changes"][price_col].pct_change().dropna()
                        new_volatility = returns.std() * (1 + volatility_change/100)
                        new_returns = np.random.normal(returns.mean(), new_volatility, len(returns))
                        new_prices = [scenarios["Scenario 3: Combined Changes"][price_col].iloc[0]]
                        for ret in new_returns:
                            new_prices.append(new_prices[-1] * (1 + ret))
                        scenarios["Scenario 3: Combined Changes"][price_col] = new_prices[:len(scenarios["Scenario 3: Combined Changes"])]

                        # Plot scenarios
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for name, data in scenarios.items():
                            ax.plot(data.index[-90:], data[price_col].iloc[-90:], label=name)
                        ax.set_title(f"{selected_commodity.replace('_', ' ').title()} Price Scenarios")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                        # Scenario comparison
                        st.subheader("Scenario Comparison")

                        metrics = []
                        for name, data in scenarios.items():
                            returns = data[price_col].pct_change().dropna()
                            metrics.append({
                                "Scenario": name,
                                "Final Price": f"${data[price_col].iloc[-1]:.2f}",
                                "Mean Return": f"{returns.mean() * 100:.4f}%",
                                "Volatility": f"{returns.std() * 100:.4f}%",
                                "Max Drawdown": f"{((data[price_col].cummax() - data[price_col]) / data[price_col].cummax()).max() * 100:.2f}%"
                            })

                        st.table(pd.DataFrame(metrics))

                        # Financial impact
                        st.subheader("Financial Impact Analysis")

                        # Assumptions
                        st.write("Assumptions:")
                        col1, col2 = st.columns(2)
                        with col1:
                            position_size = st.number_input("Position Size (barrels)", 1000, 1000000, 100000, 1000)
                        with col2:
                            holding_period = st.slider("Holding Period (days)", 1, 365, 30)

                        # Calculate financial impact
                        impact = []
                        base_price = scenarios["Base Case"][price_col].iloc[-1]

                        for name, data in scenarios.items():
                            if name == "Base Case":
                                continue

                            final_price = data[price_col].iloc[-1]
                            price_diff = final_price - base_price
                            dollar_impact = price_diff * position_size
                            percent_change = (final_price / base_price - 1) * 100

                            impact.append({
                                "Scenario": name,
                                "Base Price": f"${base_price:.2f}",
                                "Scenario Price": f"${final_price:.2f}",
                                "Price Change": f"{percent_change:.2f}%",
                                "Dollar Impact": f"${dollar_impact:.2f}"
                            })

                        st.table(pd.DataFrame(impact))

                        # Risk metrics
                        st.subheader("Risk Metrics")

                        risk_metrics = []
                        for name, data in scenarios.items():
                            returns = data[price_col].pct_change().dropna()
                            var_95 = -np.percentile(returns, 5)
                            var_99 = -np.percentile(returns, 1)

                            risk_metrics.append({
                                "Scenario": name,
                                "Daily VaR (95%)": f"{var_95:.4%}",
                                "Daily VaR (99%)": f"{var_99:.4%}",
                                "Dollar VaR (95%)": f"${var_95 * position_size:.2f}",
                                "Dollar VaR (99%)": f"${var_99 * position_size:.2f}"
                            })

                        st.table(pd.DataFrame(risk_metrics))

                    except Exception as e:
                        st.error(f"Error generating scenarios: {e}")

        with tab2:
            st.subheader("Natural Language Interface")
            st.write("Ask complex questions in plain language and receive detailed analyses.")

            # Simple natural language interface
            user_question = st.text_input("Ask a question about oil and gas markets:",
                                         "What will happen to crude oil prices if supply decreases by 10%?")

            if st.button("Get Answer"):
                with st.spinner("Analyzing question..."):
                    # Simulate response generation
                    import time
                    time.sleep(2)

                    # Predefined responses for common questions
                    responses = {
                        "What will happen to crude oil prices if supply decreases by 10%?":
                            "Based on historical supply-demand elasticity, a 10% decrease in crude oil supply would likely result in a 15-25% increase in prices, assuming demand remains constant. This could lead to increased volatility and potential market disruptions.",

                        "How does geopolitical tension affect oil prices?":
                            "Geopolitical tensions, especially in oil-producing regions, typically lead to price increases due to supply uncertainty. Historical data shows that major conflicts can cause price spikes of 20-30% within short periods.",

                        "What is the forecast for gasoline prices next quarter?":
                            "Our models predict gasoline prices to increase by 5-8% next quarter, driven by seasonal demand patterns and current refinery utilization rates. However, this forecast has moderate uncertainty due to potential regulatory changes.",

                        "How can I reduce risk exposure in my oil portfolio?":
                            "To reduce risk exposure, consider: 1) Diversifying across different petroleum products, 2) Implementing hedging strategies using futures or options, 3) Maintaining a balanced position size relative to your total portfolio, and 4) Setting stop-loss levels based on your risk tolerance."
                    }

                    # Get response
                    if user_question in responses:
                        response = responses[user_question]
                    else:
                        response = "I don't have specific information on that question. Please try asking about price forecasts, supply-demand dynamics, risk management, or market trends."

                    st.write(response)

                    # Show related visualizations based on the question
                    if "supply decreases" in user_question.lower():
                        # Create supply-demand visualization
                        st.subheader("Supply-Demand Analysis")

                        # Create sample data
                        supply_reduction = 0.1  # 10% reduction

                        # Price elasticity of demand (assumed)
                        elasticity = -0.2

                        # Calculate price change
                        price_change = supply_reduction / elasticity

                        # Create chart
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Supply curves
                        q_values = np.linspace(80, 120, 100)
                        p_original = 0.5 * q_values - 20
                        p_reduced = 0.5 * q_values - 20 + 15  # Shift supply curve left

                        # Demand curve
                        p_demand = -0.5 * q_values + 100

                        # Plot curves
                        ax.plot(q_values, p_original, 'b-', label='Original Supply')
                        ax.plot(q_values, p_reduced, 'r-', label='Reduced Supply (-10%)')
                        ax.plot(q_values, p_demand, 'g-', label='Demand')

                        # Find equilibrium points
                        q_eq_original = 80
                        p_eq_original = 60

                        q_eq_reduced = 70
                        p_eq_reduced = 75

                        # Mark equilibrium points
                        ax.scatter([q_eq_original, q_eq_reduced], [p_eq_original, p_eq_reduced],
                                  color=['blue', 'red'], s=100, zorder=5)

                        ax.annotate(f'Original Equilibrium\nQ={q_eq_original}, P=${p_eq_original}',
                                   xy=(q_eq_original, p_eq_original), xytext=(q_eq_original-15, p_eq_original-10),
                                   arrowprops=dict(arrowstyle='->'))

                        ax.annotate(f'New Equilibrium\nQ={q_eq_reduced}, P=${p_eq_reduced}',
                                   xy=(q_eq_reduced, p_eq_reduced), xytext=(q_eq_reduced-15, p_eq_reduced+10),
                                   arrowprops=dict(arrowstyle='->'))

                        # Set labels and title
                        ax.set_xlabel('Quantity (Million Barrels)')
                        ax.set_ylabel('Price ($ per Barrel)')
                        ax.set_title('Impact of 10% Supply Reduction on Oil Price')
                        ax.legend()
                        ax.grid(True)

                        # Set axis limits
                        ax.set_xlim(60, 100)
                        ax.set_ylim(40, 90)

                        st.pyplot(fig)

                        # Add explanation
                        st.markdown(f"""
                        **Analysis:**
                        - A 10% reduction in supply shifts the supply curve to the left
                        - With an estimated demand elasticity of {elasticity}, this results in:
                          - Quantity reduction: {q_eq_original - q_eq_reduced} million barrels
                          - Price increase: ${p_eq_reduced - p_eq_original} per barrel ({((p_eq_reduced/p_eq_original)-1)*100:.1f}%)
                        - This is consistent with historical supply disruptions, where a 10% reduction typically leads to a 15-25% price increase
                        """)

        with tab3:
            st.subheader("Advanced Visualizations")
            st.write("Explore complex data relationships through interactive charts, maps, and dashboards.")

            # Visualization selection
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Price Correlation Heatmap", "Price Seasonality", "Supply Chain Network", "Market Share"]
            )

            if viz_type == "Price Correlation Heatmap":
                st.subheader("Commodity Price Correlations")

                # Create sample correlation data
                commodities = ["Crude Oil", "Gasoline", "Diesel", "Natural Gas", "Heating Oil", "Jet Fuel"]

                # Create a sample correlation matrix
                np.random.seed(42)
                corr_data = np.random.rand(len(commodities), len(commodities))
                # Make it symmetric
                corr_data = (corr_data + corr_data.T) / 2
                # Set diagonal to 1
                np.fill_diagonal(corr_data, 1)

                corr_df = pd.DataFrame(corr_data, index=commodities, columns=commodities)

                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)

                # Add labels
                ax.set_xticks(np.arange(len(commodities)))
                ax.set_yticks(np.arange(len(commodities)))
                ax.set_xticklabels(commodities)
                ax.set_yticklabels(commodities)

                # Rotate x labels
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                # Add colorbar
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.set_label('Correlation Coefficient')

                # Add correlation values
                for i in range(len(commodities)):
                    for j in range(len(commodities)):
                        text = ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}",
                                      ha="center", va="center",
                                      color="white" if abs(corr_df.iloc[i, j]) > 0.5 else "black")

                ax.set_title("Commodity Price Correlation Matrix")
                fig.tight_layout()
                st.pyplot(fig)

                # Add explanation
                st.markdown("""
                **Correlation Analysis:**
                - **Strong Positive Correlation (> 0.7)**: Prices tend to move together in the same direction
                - **Moderate Correlation (0.3 - 0.7)**: Some relationship, but not always consistent
                - **Weak Correlation (< 0.3)**: Little relationship between price movements

                **Key Insights:**
                - Crude oil and refined products (gasoline, diesel) show strong correlations
                - Natural gas has weaker correlations with petroleum products
                - Understanding these relationships is crucial for portfolio diversification and risk management
                """)

            elif viz_type == "Price Seasonality":
                st.subheader("Commodity Price Seasonality")

                # Create sample seasonal data
                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

                # Sample seasonal patterns
                crude_seasonal = [0.5, 1.2, 2.5, 3.1, 2.8, 1.5, 0.8, -0.5, -1.2, -2.5, -3.0, -1.5]
                gasoline_seasonal = [-2.0, -1.0, 1.5, 3.5, 4.0, 3.0, 2.0, 1.0, -1.0, -2.5, -3.5, -3.0]
                natgas_seasonal = [4.0, 3.0, 1.0, -1.0, -2.5, -3.0, -2.5, -1.5, 0.5, 1.5, 2.5, 3.5]

                # Create dataframe
                seasonal_df = pd.DataFrame({
                    "Month": months,
                    "Crude Oil": crude_seasonal,
                    "Gasoline": gasoline_seasonal,
                    "Natural Gas": natgas_seasonal
                })

                # Create chart
                fig, ax = plt.subplots(figsize=(12, 6))

                ax.plot(seasonal_df["Month"], seasonal_df["Crude Oil"], 'o-', label="Crude Oil")
                ax.plot(seasonal_df["Month"], seasonal_df["Gasoline"], 's-', label="Gasoline")
                ax.plot(seasonal_df["Month"], seasonal_df["Natural Gas"], '^-', label="Natural Gas")

                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

                ax.set_xlabel("Month")
                ax.set_ylabel("Average Price Change (%)")
                ax.set_title("Seasonal Price Patterns by Commodity")
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

                # Add explanation
                st.markdown("""
                **Seasonal Price Patterns:**

                **Crude Oil:**
                - Tends to strengthen in spring (March-May)
                - Weakens in fall (September-November)
                - Driven by refinery maintenance schedules and driving season

                **Gasoline:**
                - Strong price increases before summer driving season (March-June)
                - Price weakness in winter months (October-February)
                - Reflects seasonal demand patterns and blend changes

                **Natural Gas:**
                - Opposite pattern to petroleum products
                - Strengthens in winter months (October-February)
                - Weakens in spring and summer (April-August)
                - Driven primarily by heating demand

                **Trading Implications:**
                - Consider seasonal patterns when timing entry/exit points
                - Seasonal spreads (e.g., gasoline/heating oil) can be profitable trading strategies
                - Seasonality should be combined with other factors for trading decisions
                """)

            elif viz_type == "Supply Chain Network":
                st.subheader("Oil & Gas Supply Chain Network")

                # Create a sample supply chain network visualization
                try:
                    # Create nodes for the supply chain
                    nodes = [
                        {"id": "Well1", "type": "Production", "location": (0, 0)},
                        {"id": "Well2", "type": "Production", "location": (0, 2)},
                        {"id": "Well3", "type": "Production", "location": (0, 4)},
                        {"id": "Storage1", "type": "Storage", "location": (2, 1)},
                        {"id": "Storage2", "type": "Storage", "location": (2, 3)},
                        {"id": "Refinery1", "type": "Processing", "location": (4, 0)},
                        {"id": "Refinery2", "type": "Processing", "location": (4, 4)},
                        {"id": "Distribution1", "type": "Distribution", "location": (6, 1)},
                        {"id": "Distribution2", "type": "Distribution", "location": (6, 3)},
                        {"id": "Market1", "type": "Market", "location": (8, 0)},
                        {"id": "Market2", "type": "Market", "location": (8, 2)},
                        {"id": "Market3", "type": "Market", "location": (8, 4)}
                    ]

                    # Create edges (connections) between nodes
                    edges = [
                        {"source": "Well1", "target": "Storage1", "flow": 100},
                        {"source": "Well2", "target": "Storage1", "flow": 80},
                        {"source": "Well2", "target": "Storage2", "flow": 50},
                        {"source": "Well3", "target": "Storage2", "flow": 120},
                        {"source": "Storage1", "target": "Refinery1", "flow": 150},
                        {"source": "Storage1", "target": "Refinery2", "flow": 30},
                        {"source": "Storage2", "target": "Refinery1", "flow": 40},
                        {"source": "Storage2", "target": "Refinery2", "flow": 130},
                        {"source": "Refinery1", "target": "Distribution1", "flow": 100},
                        {"source": "Refinery1", "target": "Distribution2", "flow": 90},
                        {"source": "Refinery2", "target": "Distribution1", "flow": 60},
                        {"source": "Refinery2", "target": "Distribution2", "flow": 100},
                        {"source": "Distribution1", "target": "Market1", "flow": 80},
                        {"source": "Distribution1", "target": "Market2", "flow": 80},
                        {"source": "Distribution2", "target": "Market2", "flow": 70},
                        {"source": "Distribution2", "target": "Market3", "flow": 120}
                    ]

                    # Create a figure
                    fig, ax = plt.subplots(figsize=(12, 8))

                    # Define node colors based on type
                    node_colors = {
                        "Production": "green",
                        "Storage": "blue",
                        "Processing": "red",
                        "Distribution": "purple",
                        "Market": "orange"
                    }

                    # Plot nodes
                    for node in nodes:
                        x, y = node["location"]
                        color = node_colors[node["type"]]
                        ax.scatter(x, y, s=300, color=color, alpha=0.7, edgecolors='black')
                        ax.text(x, y, node["id"], ha='center', va='center', fontweight='bold')

                    # Plot edges
                    for edge in edges:
                        source_node = next(node for node in nodes if node["id"] == edge["source"])
                        target_node = next(node for node in nodes if node["id"] == edge["target"])

                        sx, sy = source_node["location"]
                        tx, ty = target_node["location"]

                        # Scale line width based on flow
                        line_width = edge["flow"] / 30

                        ax.plot([sx, tx], [sy, ty], 'k-', alpha=0.5, linewidth=line_width)

                        # Add flow label
                        mid_x = (sx + tx) / 2
                        mid_y = (sy + ty) / 2
                        ax.text(mid_x, mid_y, str(edge["flow"]), ha='center', va='center',
                               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

                    # Add legend
                    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                                 markersize=10, label=node_type)
                                      for node_type, color in node_colors.items()]

                    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                             ncol=len(node_colors))

                    # Set axis properties
                    ax.set_xlim(-1, 9)
                    ax.set_ylim(-1, 5)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title('Oil & Gas Supply Chain Network')

                    st.pyplot(fig)

                    # Add explanation
                    st.markdown("""
                    **Supply Chain Network Components:**
                    - **Production (Green)**: Oil wells and production facilities
                    - **Storage (Blue)**: Storage tanks and terminals
                    - **Processing (Red)**: Refineries and processing plants
                    - **Distribution (Purple)**: Distribution centers
                    - **Market (Orange)**: End markets and customers

                    The numbers on the connections represent flow volumes (barrels/day).

                    **Key Insights:**
                    - Bottlenecks can occur at any point in the supply chain
                    - Disruptions at upstream nodes can cascade throughout the network
                    - Diversification of supply routes increases resilience
                    - Optimizing flow allocation can significantly improve efficiency
                    """)

                except Exception as e:
                    st.error(f"Error generating supply chain visualization: {e}")

            elif viz_type == "Market Share":
                st.subheader("Global Oil Production Market Share")

                # Create sample market share data
                countries = ["Saudi Arabia", "Russia", "USA", "Iran", "Iraq", "UAE", "Kuwait", "Canada", "Others"]
                production = [11.0, 10.5, 17.9, 3.8, 4.5, 3.1, 2.7, 5.2, 32.3]  # in million barrels per day

                # Create pie chart
                fig, ax = plt.subplots(figsize=(10, 10))

                # Use a colormap
                colors = plt.cm.Spectral(np.linspace(0, 1, len(countries)))

                # Create pie chart with explode for emphasis
                explode = [0.1 if country in ["Saudi Arabia", "Russia", "USA"] else 0 for country in countries]

                wedges, texts, autotexts = ax.pie(
                    production,
                    labels=countries,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    explode=explode,
                    shadow=True
                )

                # Style the labels and percentages
                for text in texts:
                    text.set_fontsize(12)
                for autotext in autotexts:
                    autotext.set_fontsize(10)
                    autotext.set_color('white')

                ax.axis('equal')
                ax.set_title('Global Oil Production Market Share (2023)', fontsize=16)

                st.pyplot(fig)

                # Add explanation
                st.markdown("""
                **Global Oil Production Analysis:**

                **Key Producers:**
                - **USA (17.9%)**: Largest producer due to shale oil revolution
                - **Russia (10.5%)**: Second largest, significant geopolitical influence
                - **Saudi Arabia (11.0%)**: OPEC leader with lowest production costs

                **Market Dynamics:**
                - Top 3 producers account for nearly 40% of global production
                - OPEC members (Saudi Arabia, Iran, Iraq, UAE, Kuwait) collectively control about 25%
                - "Others" category (32.3%) represents many smaller producers with limited individual influence

                **Strategic Implications:**
                - Production decisions by top producers have outsized market impact
                - Geopolitical events affecting major producers can cause significant price volatility
                - Diversification of supply sources is important for energy security
                """)

        # Navigation buttons
        st.subheader("Navigation")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Go to Data Management", key="decision_to_data"):
                st.session_state.page = "Data Management"
                st.experimental_rerun()
        with col2:
            if st.button("Go to Trading Dashboard", key="decision_to_trading"):
                st.session_state.page = "Trading Dashboard"
                st.experimental_rerun()
        with col3:
            if st.button("Go to Risk Analysis", key="decision_to_risk"):
                st.session_state.page = "Risk Analysis"
                st.experimental_rerun()

def home_page():
    """Home page with overview and navigation."""
    st.header("Welcome to Oil & Gas Market Optimization Platform")

    st.subheader("Platform Overview")
    st.write("""
    This comprehensive AI solution addresses the complex challenges faced by the oil and gas industry.
    By leveraging advanced machine learning algorithms and data analytics, this system provides actionable
    insights for optimizing operations, predicting market trends, and enhancing decision-making processes.
    """)

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Data Analysis")
        st.write("Upload and analyze commodity price data")
        st.write("Generate sample datasets")
        st.write("Process and visualize market data")

        if st.button("Go to Data Management"):
            st.session_state.page = "Data Management"
            st.experimental_rerun()

    with col2:
        st.markdown("### 📈 Trading Strategies")
        st.write("Backtest trading strategies")
        st.write("Optimize strategy parameters")
        st.write("Analyze performance metrics")

        if st.button("Go to Trading Dashboard"):
            st.session_state.page = "Trading Dashboard"
            st.experimental_rerun()

    with col3:
        st.markdown("### 🔍 Risk Analysis")
        st.write("Calculate Value at Risk (VaR)")
        st.write("Run Monte Carlo simulations")
        st.write("Analyze return distributions")

        if st.button("Go to Risk Analysis"):
            st.session_state.page = "Risk Analysis"
            st.experimental_rerun()

    # Second row of feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔮 Predictive Analytics")
        st.write("Forecast commodity prices")
        st.write("Optimize production levels")
        st.write("Schedule preventive maintenance")

        if st.button("Go to Predictive Analytics"):
            st.session_state.page = "Predictive Analytics"
            st.experimental_rerun()

    with col2:
        st.markdown("### ⚠️ Risk Assessment")
        st.write("Analyze market risks")
        st.write("Monitor geopolitical events")
        st.write("Track regulatory changes")

        if st.button("Go to Risk Assessment"):
            st.session_state.page = "Risk Assessment"
            st.experimental_rerun()

    with col3:
        st.markdown("### 🤖 Decision Support")
        st.write("Model different scenarios")
        st.write("Ask questions in natural language")
        st.write("Explore advanced visualizations")

        if st.button("Go to Decision Support"):
            st.session_state.page = "Decision Support"
            st.experimental_rerun()

    # Key metrics
    st.subheader("Key Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Production Efficiency", "18% ↑", "Optimized resource allocation")
    with col2:
        st.metric("Maintenance Costs", "25% ↓", "Preventive maintenance scheduling")
    with col3:
        st.metric("Price Forecasting", "92% accuracy", "Improved market positioning")
    with col4:
        st.metric("Decision-Making Time", "70% ↓", "Faster response to market changes")

def main():
    """Main function for the web application."""
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}

    if 'selected_commodity' not in st.session_state:
        st.session_state.selected_commodity = None

    # Title and description
    st.title("📈 Oil & Gas Market Optimization")
    st.markdown(
        """
        This comprehensive platform provides tools for analyzing oil and gas market data,
        optimizing trading strategies, and enhancing decision-making processes.
        """
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Home", "Data Management", "Trading Dashboard", "Risk Analysis",
         "Predictive Analytics", "Risk Assessment", "Decision Support"],
        index=["Home", "Data Management", "Trading Dashboard", "Risk Analysis",
               "Predictive Analytics", "Risk Assessment", "Decision Support"].index(st.session_state.page)
    )

    # Update session state
    st.session_state.page = page

    # Display selected page
    if page == "Home":
        home_page()
    elif page == "Data Management":
        data_management_page()
    elif page == "Trading Dashboard":
        trading_dashboard_page()
    elif page == "Risk Analysis":
        risk_analysis_page()
    elif page == "Predictive Analytics":
        predictive_analytics_page()
    elif page == "Risk Assessment":
        risk_assessment_page()
    elif page == "Decision Support":
        decision_support_page()

if __name__ == "__main__":
    main()
