#!/usr/bin/env python
"""
Simplified web application for the Oil & Gas Market Optimization system.
This version is optimized for Streamlit Cloud deployment.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

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
os.makedirs('logs', exist_ok=True)

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
        and optimize portfolios. You can use sample data or upload your own datasets.
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
        
        # Generate sample data
        st.subheader("Generate Sample Data")
        
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
    
    # Trading Dashboard page
    elif page == "Trading Dashboard":
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
    
    # Risk Analysis page
    elif page == "Risk Analysis":
        st.header("Risk Analysis")
        
        # Check if we have processed data
        if not st.session_state.processed_data:
            st.warning("No processed data available. Please go to the Data Management page to generate sample data.")
            
            if st.button("Go to Data Management"):
                st.session_state.page = "Data Management"
                st.experimental_rerun()
        else:
            st.info("Risk Analysis functionality is simplified in this version. For full functionality, please use the complete application.")
            
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
