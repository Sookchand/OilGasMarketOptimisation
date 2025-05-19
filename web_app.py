#!/usr/bin/env python
"""
Web application for the Oil & Gas Market Optimization system.
This application allows users to upload their own datasets or use sample data.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
import io
import base64
from datetime import datetime

# Import the Q&A component
from qa_component import qa_interface

# Define utility functions that were previously imported
def load_processed_data(commodity, data_dir='data/processed'):
    """Load processed data for a commodity."""
    file_path = os.path.join(data_dir, f"{commodity}.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df

    return pd.DataFrame()

def save_to_parquet(df, file_path):
    """Save a dataframe to a Parquet file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_parquet(file_path)
    return True

def clean_data(df):
    """Clean data by handling missing values and outliers."""
    # Make a copy of the data
    df_cleaned = df.copy()

    # Handle missing values
    df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')

    # Handle outliers using IQR method
    for col in df_cleaned.select_dtypes(include=['number']).columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Replace outliers with bounds
        df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
        df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound

    return df_cleaned

def add_features(df):
    """Add technical indicators and features to the data."""
    # Make a copy of the data
    df_features = df.copy()

    # Calculate returns
    df_features['Returns'] = df_features['Price'].pct_change()

    # Calculate moving averages
    df_features['MA_10'] = df_features['Price'].rolling(window=10).mean()
    df_features['MA_30'] = df_features['Price'].rolling(window=30).mean()
    df_features['MA_50'] = df_features['Price'].rolling(window=50).mean()

    # Calculate volatility
    df_features['Volatility'] = df_features['Returns'].rolling(window=20).std() * np.sqrt(252)

    # Calculate RSI
    delta = df_features['Price'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df_features['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    df_features['BB_Middle'] = df_features['Price'].rolling(window=20).mean()
    df_features['BB_Upper'] = df_features['BB_Middle'] + 2 * df_features['Price'].rolling(window=20).std()
    df_features['BB_Lower'] = df_features['BB_Middle'] - 2 * df_features['Price'].rolling(window=20).std()

    # Calculate MACD
    df_features['EMA_12'] = df_features['Price'].ewm(span=12, adjust=False).mean()
    df_features['EMA_26'] = df_features['Price'].ewm(span=26, adjust=False).mean()
    df_features['MACD'] = df_features['EMA_12'] - df_features['EMA_26']
    df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean()
    df_features['MACD_Histogram'] = df_features['MACD'] - df_features['MACD_Signal']

    return df_features

# Set page config
st.set_page_config(
    page_title="Oil & Gas Market Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/features', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('results/trading', exist_ok=True)

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

def process_data(df, commodity):
    """Process data for a commodity."""
    # Clean data
    df_cleaned = clean_data(df)

    # Add features
    df_features = add_features(df_cleaned)

    # Save to processed directory
    save_to_parquet(df_features, f'data/processed/{commodity}.parquet')

    return df_features

def load_data(commodity):
    """Load processed data for a commodity."""
    return load_processed_data(commodity)

def download_link(df, filename, text):
    """Generate a download link for a dataframe."""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    """Main function for the web application."""
    # Initialize session state for navigation and data storage
    if 'page' not in st.session_state:
        st.session_state.page = "Data Management"

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}

    if 'selected_commodity' not in st.session_state:
        st.session_state.selected_commodity = None

    # Title and description
    st.title("📈 Oil & Gas Market Optimization")
    st.markdown(
        """
        This application allows you to analyze oil and gas market data, backtest trading strategies,
        and optimize portfolios. You can upload your own datasets or use sample data.
        """
    )

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Data Management", "Trading Dashboard", "Risk Analysis"],
        index=["Data Management", "Trading Dashboard", "Risk Analysis"].index(st.session_state.page)
    )

    # Update session state when page changes
    st.session_state.page = page

    # Data Management page
    if page == "Data Management":
        st.header("Data Management")

        # Commodity selection
        commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']

        # Data upload section
        st.subheader("Upload Your Data")
        st.markdown(
            """
            Upload your own CSV or Excel files for each commodity. The files should have at least
            a 'Date' column and a 'Price' column. The 'Date' column will be used as the index.
            """
        )

        # Create tabs for each commodity
        tabs = st.tabs([commodity.replace('_', ' ').title() for commodity in commodities])

        for i, commodity in enumerate(commodities):
            with tabs[i]:
                st.write(f"Upload data for {commodity.replace('_', ' ').title()}")

                # File uploader
                uploaded_file = st.file_uploader(
                    f"Choose a CSV or Excel file for {commodity.replace('_', ' ').title()}",
                    type=['csv', 'xlsx', 'xls'],
                    key=f"upload_{commodity}"
                )

                # Process uploaded file
                if uploaded_file is not None:
                    try:
                        # Determine file type
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)

                        # Check if Date column exists
                        if 'Date' not in df.columns:
                            st.error("The file must have a 'Date' column.")
                            continue

                        # Check if Price column exists
                        if 'Price' not in df.columns:
                            st.error("The file must have a 'Price' column.")
                            continue

                        # Set Date as index
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)

                        # Display data
                        st.write("Data preview:")
                        st.dataframe(df.head())

                        # Process data
                        if st.button(f"Process {commodity.replace('_', ' ').title()} Data", key=f"process_{commodity}"):
                            with st.spinner("Processing data..."):
                                df_processed = process_data(df, commodity)

                                # Store in session state
                                st.session_state.processed_data[commodity] = df_processed

                                st.success(f"Data for {commodity.replace('_', ' ').title()} processed successfully!")

                                # Display processed data
                                st.write("Processed data preview:")
                                st.dataframe(df_processed.head())

                                # Download link
                                st.markdown(
                                    download_link(
                                        df_processed,
                                        f"{commodity}_processed.csv",
                                        f"Download processed {commodity.replace('_', ' ').title()} data"
                                    ),
                                    unsafe_allow_html=True
                                )

                                # Add button to navigate to Trading Dashboard
                                if st.button(f"Analyze {commodity.replace('_', ' ').title()} Data", key=f"analyze_{commodity}"):
                                    st.session_state.selected_commodity = commodity
                                    st.session_state.page = "Trading Dashboard"
                                    st.experimental_rerun()

                    except Exception as e:
                        st.error(f"Error processing file: {e}")

                # Use sample data
                st.write("Or use sample data:")
                if st.button(f"Generate Sample Data for {commodity.replace('_', ' ').title()}", key=f"sample_{commodity}"):
                    with st.spinner("Generating sample data..."):
                        # Generate sample data
                        df_sample = generate_sample_data(commodity)

                        # Save to raw directory
                        os.makedirs('data/raw', exist_ok=True)
                        df_sample.to_csv(f'data/raw/{commodity}.csv')

                        # Process data
                        df_processed = process_data(df_sample, commodity)

                        # Store in session state
                        st.session_state.processed_data[commodity] = df_processed

                        st.success(f"Sample data for {commodity.replace('_', ' ').title()} generated and processed successfully!")

                        # Display sample data
                        st.write("Sample data preview:")
                        st.dataframe(df_sample.head())

                        # Plot sample data
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df_sample.index, df_sample['Price'])
                        ax.set_title(f"{commodity.replace('_', ' ').title()} Price")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price")
                        ax.grid(True)
                        st.pyplot(fig)

                        # Download link
                        st.markdown(
                            download_link(
                                df_processed,
                                f"{commodity}_processed.csv",
                                f"Download processed {commodity.replace('_', ' ').title()} data"
                            ),
                            unsafe_allow_html=True
                        )

                        # Add button to navigate to Trading Dashboard
                        if st.button(f"Analyze {commodity.replace('_', ' ').title()} Data", key=f"analyze_sample_{commodity}"):
                            st.session_state.selected_commodity = commodity
                            st.session_state.page = "Trading Dashboard"
                            st.experimental_rerun()

        # Generate all sample data
        st.subheader("Generate All Sample Data")
        if st.button("Generate Sample Data for All Commodities"):
            with st.spinner("Generating sample data for all commodities..."):
                for commodity in commodities:
                    # Generate sample data
                    df_sample = generate_sample_data(commodity)

                    # Save to raw directory
                    os.makedirs('data/raw', exist_ok=True)
                    df_sample.to_csv(f'data/raw/{commodity}.csv')

                    # Process data
                    df_processed = process_data(df_sample, commodity)

                    # Store in session state
                    st.session_state.processed_data[commodity] = df_processed

                st.success("Sample data for all commodities generated and processed successfully!")

                # Add button to navigate to Trading Dashboard
                if st.button("Go to Trading Dashboard"):
                    st.session_state.page = "Trading Dashboard"
                    st.experimental_rerun()

    # Trading Dashboard page
    elif page == "Trading Dashboard":
        st.header("Trading Dashboard")

        # Check if we have processed data
        if not st.session_state.processed_data:
            st.warning("No processed data available. Please go to the Data Management page to upload or generate data.")

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
                            from simple_trading_dashboard import calculate_moving_average_signals
                            results = calculate_moving_average_signals(
                                df,
                                fast_window=strategy_params['fast_window'],
                                slow_window=strategy_params['slow_window']
                            )
                        elif selected_strategy == 'rsi':
                            from simple_trading_dashboard import calculate_rsi_signals
                            results = calculate_rsi_signals(
                                df,
                                window=strategy_params['window'],
                                oversold=strategy_params['oversold'],
                                overbought=strategy_params['overbought']
                            )

                        # Calculate metrics
                        from simple_trading_dashboard import calculate_performance_metrics
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

    # Risk Analysis page
    elif page == "Risk Analysis":
        st.header("Risk Analysis")

        # Check if we have processed data
        if not st.session_state.processed_data:
            st.warning("No processed data available. Please go to the Data Management page to upload or generate data.")

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
            st.subheader("Risk Analysis Parameters")

            col1, col2 = st.columns(2)
            with col1:
                confidence_level = st.slider("Confidence Level (%)", 90, 99, 95) / 100
            with col2:
                lookback_days = st.slider("Lookback Period (days)", 30, 365, 252)

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
            returns = df['returns'].iloc[-lookback_days:].dropna()

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

            # Historical VaR
            var_percentile = 1 - confidence_level
            historical_var = -np.percentile(returns, var_percentile * 100)

            # Parametric VaR
            from scipy import stats as scipy_stats
            z_score = scipy_stats.norm.ppf(confidence_level)
            parametric_var = -(returns.mean() + z_score * returns.std())

            var_values = {
                f'{confidence_level:.0%} Historical VaR': f"{historical_var:.4%}",
                f'{confidence_level:.0%} Parametric VaR': f"{parametric_var:.4%}"
            }

            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{confidence_level:.0%} Historical VaR", var_values[f'{confidence_level:.0%} Historical VaR'])
            with col2:
                st.metric(f"{confidence_level:.0%} Parametric VaR", var_values[f'{confidence_level:.0%} Parametric VaR'])

            # Monte Carlo Simulation
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
                        time_horizon = 252  # 1 year of trading days

                        # Generate random returns
                        random_returns = np.random.normal(
                            mu,
                            sigma,
                            (time_horizon, num_simulations)
                        )

                        # Calculate cumulative returns
                        cumulative_returns = np.cumprod(1 + random_returns, axis=0)

                        # Calculate portfolio values
                        portfolio_values = initial_value * cumulative_returns

                        # Convert to DataFrame
                        dates = pd.date_range(start=pd.Timestamp.today(), periods=time_horizon, freq='B')
                        portfolio_df = pd.DataFrame(portfolio_values, index=dates)

                        # Calculate statistics
                        final_values = portfolio_values[-1, :]
                        mean_final = np.mean(final_values)
                        median_final = np.median(final_values)
                        min_final = np.min(final_values)
                        max_final = np.max(final_values)

                        # Calculate VaR
                        var_95 = initial_value - np.percentile(final_values, 5)
                        var_95_pct = var_95 / initial_value

                        # Calculate probability of loss
                        prob_loss = np.mean(final_values < initial_value)

                        # Display statistics
                        st.subheader("Simulation Statistics")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Final Value", f"${mean_final:.2f}")
                        with col2:
                            st.metric("Median Final Value", f"${median_final:.2f}")
                        with col3:
                            st.metric("95% VaR", f"{var_95_pct:.2%}")
                        with col4:
                            st.metric("Probability of Loss", f"{prob_loss:.2%}")

                        # Plot simulation
                        st.subheader("Monte Carlo Simulation")

                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Plot 20 random simulations
                        for i in range(20):
                            ax.plot(dates, portfolio_df.iloc[:, i], alpha=0.3, linewidth=1)

                        # Plot mean path
                        ax.plot(dates, portfolio_df.mean(axis=1), color='red', linewidth=2, label='Mean Path')

                        # Plot confidence intervals
                        ax.plot(dates, portfolio_df.quantile(0.05, axis=1), color='orange', linewidth=1.5, linestyle='--', label='5th Percentile')
                        ax.plot(dates, portfolio_df.quantile(0.95, axis=1), color='green', linewidth=1.5, linestyle='--', label='95th Percentile')

                        ax.set_title('Monte Carlo Simulation (1 Year)')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Portfolio Value ($)')
                        ax.legend()
                        ax.grid(True)

                        st.pyplot(fig)

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

if __name__ == "__main__":
    main()
