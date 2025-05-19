"""
Simplified web application for the Oil & Gas Market Optimization system.
This version uses minimal dependencies and features to ensure compatibility.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# Set page config with minimal settings
st.set_page_config(
    page_title="Oil & Gas Market Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Utility functions
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

def load_data(commodity):
    """Load processed data for a commodity."""
    file_path = f'data/processed/{commodity}.csv'
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    
    return pd.DataFrame()

def process_data(df, commodity):
    """Process data for a commodity."""
    # Clean data (simple version)
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
    
    # Add basic features
    df_features = df_cleaned.copy()
    df_features['Returns'] = df_features['Price'].pct_change()
    df_features['MA_10'] = df_features['Price'].rolling(window=10).mean()
    df_features['MA_30'] = df_features['Price'].rolling(window=30).mean()
    
    # Save to processed directory
    os.makedirs('data/processed', exist_ok=True)
    df_features.to_csv(f'data/processed/{commodity}.csv')
    
    return df_features

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
    
    # Calculate returns
    data['returns'] = data['Price'].pct_change()
    
    # Calculate strategy returns
    data['strategy_returns'] = data['signal'].shift(1) * data['returns']
    
    # Calculate cumulative returns
    data['cumulative_returns'] = (1 + data['returns']).cumprod() - 1
    data['strategy_cumulative_returns'] = (1 + data['strategy_returns']).cumprod() - 1
    
    return data

def main():
    """Main function for the web application."""
    # Title and description
    st.title("📈 Oil & Gas Market Optimization")
    st.markdown(
        """
        This application allows you to analyze oil and gas market data, backtest trading strategies,
        and optimize portfolios. You can use sample data to explore the features.
        """
    )
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Data Management", "Trading Dashboard", "About"]
    )
    
    # Data Management page
    if page == "Data Management":
        st.header("Data Management")
        
        # Commodity selection
        commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']
        
        # Generate sample data section
        st.subheader("Generate Sample Data")
        st.write("Generate sample data for testing and exploration.")
        
        # Create columns for the commodities
        cols = st.columns(len(commodities))
        
        for i, commodity in enumerate(commodities):
            with cols[i]:
                st.write(f"**{commodity.replace('_', ' ').title()}**")
                if st.button(f"Generate Data", key=f"gen_{commodity}"):
                    with st.spinner(f"Generating data for {commodity}..."):
                        # Generate sample data
                        df_sample = generate_sample_data(commodity)
                        
                        # Save to raw directory
                        os.makedirs('data/raw', exist_ok=True)
                        df_sample.to_csv(f'data/raw/{commodity}.csv')
                        
                        # Process data
                        df_processed = process_data(df_sample, commodity)
                        
                        st.success(f"Data generated!")
                        
                        # Display sample data
                        st.write("Preview:")
                        st.dataframe(df_sample.head(3))
        
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
                    process_data(df_sample, commodity)
                
                st.success("Sample data for all commodities generated successfully!")
    
    # Trading Dashboard page
    elif page == "Trading Dashboard":
        st.header("Trading Dashboard")
        
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
            st.error("No commodity data found. Please go to the Data Management page to generate sample data.")
            return
        
        # Select commodity
        selected_commodity = st.selectbox(
            "Select a commodity",
            available_commodities,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            fast_window = st.slider("Fast Window", 5, 50, 10)
        with col2:
            slow_window = st.slider("Slow Window", 20, 200, 50)
        
        # Run backtest button
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    # Load data
                    df = data_dict[selected_commodity]
                    
                    # Run strategy
                    results = calculate_moving_average_signals(
                        df,
                        fast_window=fast_window,
                        slow_window=slow_window
                    )
                    
                    # Display results
                    st.subheader("Backtest Results")
                    
                    # Calculate metrics
                    total_return = (1 + results['strategy_returns'].dropna()).prod() - 1
                    
                    # Display metrics
                    st.metric("Total Return", f"{total_return:.2%}")
                    
                    # Plot results
                    st.subheader("Performance Chart")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(results.index, results['cumulative_returns'], label='Buy & Hold')
                    ax.plot(results.index, results['strategy_cumulative_returns'], label='Strategy')
                    ax.set_ylabel('Cumulative Returns')
                    ax.legend()
                    ax.grid(True)
                    
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error running backtest: {e}")
    
    # About page
    elif page == "About":
        st.header("About")
        
        st.write("""
        ## Oil & Gas Market Optimization System
        
        This is a simplified version of the Oil & Gas Market Optimization system, designed to be more compatible with different environments.
        
        ### Features
        
        - Generate sample data for oil and gas commodities
        - Backtest moving average crossover trading strategies
        - Visualize performance metrics and charts
        
        ### Usage Tips
        
        1. Start by generating sample data in the Data Management page
        2. Go to the Trading Dashboard to backtest strategies
        3. Adjust parameters and compare results
        
        ### Full Version
        
        The full version includes:
        
        - Custom data upload
        - Multiple trading strategies
        - Risk analysis tools
        - Advanced visualization options
        - Q&A functionality
        """)

if __name__ == "__main__":
    main()
