"""
Minimal Streamlit app for Oil & Gas Market Optimization.
This version uses only the most basic dependencies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Oil & Gas Market Optimization",
    page_icon="📈",
    layout="wide"
)

def generate_sample_data(commodity, days=1000):
    """Generate sample price data for a commodity."""
    # Set random seed for reproducibility
    np.random.seed(42 + hash(commodity) % 100)

    # Generate dates
    dates = pd.date_range(end=datetime.now(), periods=days)

    # Generate random walk for prices
    base_price = 50.0 if 'crude' in commodity else 2.0
    changes = np.random.normal(0, 0.01, days)
    prices = base_price * np.exp(np.cumsum(changes))

    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    df.set_index('Date', inplace=True)

    return df

def calculate_moving_average(prices, window):
    """Calculate moving average."""
    return prices.rolling(window=window).mean()

def calculate_rsi(prices, window=14):
    """Calculate RSI."""
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
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
    rsi = 100 - (100 / (1 + rs))

    return rsi

def main():
    """Main function."""
    # Title and description
    st.title("📈 Oil & Gas Market Optimization")
    st.markdown(
        """
        This is a simplified version of the Oil & Gas Market Optimization system.
        It demonstrates basic functionality with minimal dependencies.
        """
    )

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Data Visualization", "Trading Strategy", "Risk Analysis", "About"]
    )

    # Data Visualization page
    if page == "Data Visualization":
        st.header("Data Visualization")

        # Commodity selection
        commodity = st.selectbox(
            "Select a commodity",
            ["crude_oil", "regular_gasoline", "conventional_gasoline", "diesel"],
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Generate sample data
        data = generate_sample_data(commodity)

        # Display data
        st.subheader("Price Data")
        st.dataframe(data.head())

        # Plot data
        st.subheader("Price Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['Price'])
        ax.set_title(f"{commodity.replace('_', ' ').title()} Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        st.pyplot(fig)

    # Trading Strategy page
    elif page == "Trading Strategy":
        st.header("Trading Strategy")

        # Commodity selection
        commodity = st.selectbox(
            "Select a commodity",
            ["crude_oil", "regular_gasoline", "conventional_gasoline", "diesel"],
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Strategy selection
        strategy = st.selectbox(
            "Select a strategy",
            ["Moving Average Crossover", "RSI"]
        )

        # Generate sample data
        data = generate_sample_data(commodity)

        # Strategy parameters
        if strategy == "Moving Average Crossover":
            col1, col2 = st.columns(2)
            with col1:
                fast_ma = st.slider("Fast MA Window", 5, 50, 10)
            with col2:
                slow_ma = st.slider("Slow MA Window", 20, 200, 50)

            # Calculate moving averages
            data['Fast MA'] = calculate_moving_average(data['Price'], fast_ma)
            data['Slow MA'] = calculate_moving_average(data['Price'], slow_ma)

            # Generate signals
            data['Signal'] = 0
            data.loc[data['Fast MA'] > data['Slow MA'], 'Signal'] = 1
            data.loc[data['Fast MA'] < data['Slow MA'], 'Signal'] = -1

            # Plot strategy
            st.subheader("Moving Average Crossover Strategy")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index, data['Price'], label='Price')
            ax.plot(data.index, data['Fast MA'], label=f'{fast_ma}-day MA')
            ax.plot(data.index, data['Slow MA'], label=f'{slow_ma}-day MA')
            ax.set_title(f"{commodity.replace('_', ' ').title()} Moving Average Crossover")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        elif strategy == "RSI":
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_window = st.slider("RSI Window", 5, 30, 14)
            with col2:
                oversold = st.slider("Oversold Level", 10, 40, 30)
            with col3:
                overbought = st.slider("Overbought Level", 60, 90, 70)

            # Calculate RSI
            data['RSI'] = calculate_rsi(data['Price'], rsi_window)

            # Generate signals
            data['Signal'] = 0
            data.loc[data['RSI'] < oversold, 'Signal'] = 1  # Buy when oversold
            data.loc[data['RSI'] > overbought, 'Signal'] = -1  # Sell when overbought

            # Plot strategy
            st.subheader("RSI Strategy")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

            # Price chart
            ax1.plot(data.index, data['Price'], label='Price')
            ax1.set_title(f"{commodity.replace('_', ' ').title()} Price")
            ax1.set_ylabel("Price")
            ax1.grid(True)

            # RSI chart
            ax2.plot(data.index, data['RSI'], label='RSI')
            ax2.axhline(y=oversold, color='g', linestyle='--')
            ax2.axhline(y=overbought, color='r', linestyle='--')
            ax2.set_title("RSI")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("RSI")
            ax2.grid(True)

            plt.tight_layout()
            st.pyplot(fig)

    # Risk Analysis page
    elif page == "Risk Analysis":
        st.header("Risk Analysis")

        # Commodity selection
        commodity = st.selectbox(
            "Select a commodity",
            ["crude_oil", "regular_gasoline", "conventional_gasoline", "diesel"],
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Generate sample data
        data = generate_sample_data(commodity)

        # Calculate returns
        data['Returns'] = data['Price'].pct_change().fillna(0)

        # Risk analysis parameters
        st.subheader("Risk Parameters")

        col1, col2 = st.columns(2)
        with col1:
            confidence_level = st.slider("Confidence Level (%)", 90, 99, 95) / 100
        with col2:
            time_horizon = st.slider("Time Horizon (days)", 1, 30, 1)

        # Display return statistics
        st.subheader("Return Statistics")

        mean_return = data['Returns'].mean()
        std_return = data['Returns'].std()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Daily Return", f"{mean_return:.4%}")
        with col2:
            st.metric("Daily Volatility", f"{std_return:.4%}")
        with col3:
            st.metric("Annualized Volatility", f"{std_return * np.sqrt(252):.4%}")

        # Plot return distribution
        st.subheader("Return Distribution")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data['Returns'], bins=50, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--')
        ax.axvline(x=mean_return, color='red', linestyle='-', label=f'Mean: {mean_return:.4%}')
        ax.axvline(x=mean_return - 2*std_return, color='orange', linestyle='--', label=f'2σ Down: {mean_return - 2*std_return:.4%}')
        ax.axvline(x=mean_return + 2*std_return, color='green', linestyle='--', label=f'2σ Up: {mean_return + 2*std_return:.4%}')

        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{commodity.replace("_", " ").title()} Daily Return Distribution')
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # Calculate Value at Risk (VaR)
        st.subheader("Value at Risk (VaR)")

        # Historical VaR
        sorted_returns = sorted(data['Returns'])
        var_percentile = 1 - confidence_level
        var_index = int(len(sorted_returns) * var_percentile)
        historical_var = -sorted_returns[var_index]

        # Parametric VaR (using normal distribution approximation)
        # For 95% confidence, z-score is approximately 1.645
        # For 99% confidence, z-score is approximately 2.326
        z_score = 1.645 if confidence_level == 0.95 else 2.326 if confidence_level == 0.99 else 2.0
        parametric_var = -(mean_return + z_score * std_return)

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

        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running simulation..."):
                # Set parameters
                initial_price = data['Price'].iloc[-1]
                num_simulations = 100
                days = 252  # One trading year

                # Set random seed for reproducibility
                np.random.seed(42)

                # Generate random returns
                simulation_returns = np.random.normal(
                    mean_return,
                    std_return,
                    (days, num_simulations)
                )

                # Calculate price paths
                price_paths = np.zeros((days, num_simulations))
                price_paths[0] = initial_price

                for t in range(1, days):
                    price_paths[t] = price_paths[t-1] * (1 + simulation_returns[t])

                # Create dates for the simulation
                last_date = data.index[-1]
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

                ax.set_title(f'Monte Carlo Simulation: {commodity.replace("_", " ").title()} Price (1 Year)')
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

    # About page
    elif page == "About":
        st.header("About")
        st.markdown(
            """
            ## Oil & Gas Market Optimization System

            This is a simplified version of the Oil & Gas Market Optimization system,
            designed to work with minimal dependencies on Streamlit Cloud.

            ### Features

            - Data visualization for oil and gas commodities
            - Trading strategy simulation
            - Risk analysis with Value at Risk (VaR)
            - Monte Carlo simulations

            ### Full Version

            The full version includes:

            - Data management with custom dataset upload
            - Multiple trading strategies
            - Advanced risk metrics
            - Portfolio optimization
            - Performance metrics

            ### Contact

            For more information, visit [sookchandportfolio.netlify.app](https://sookchandportfolio.netlify.app)
            """
        )

if __name__ == "__main__":
    main()
