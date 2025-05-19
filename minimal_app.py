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
        ["Data Visualization", "Trading Strategy", "About"]
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
            - Simple technical analysis
            
            ### Full Version
            
            The full version includes:
            
            - Data management with custom dataset upload
            - Multiple trading strategies
            - Risk analysis with Value at Risk (VaR)
            - Monte Carlo simulations
            - Performance metrics
            
            ### Contact
            
            For more information, visit [sookchandportfolio.netlify.app](https://sookchandportfolio.netlify.app)
            """
        )

if __name__ == "__main__":
    main()
