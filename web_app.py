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

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import project modules
from src.utils.data_utils import load_processed_data, save_to_parquet
from src.pipeline.data_cleaning import clean_data
from src.pipeline.feature_engineering import add_features

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
        ["Data Management", "Trading Dashboard", "Risk Analysis"]
    )
    
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
                
                st.success("Sample data for all commodities generated and processed successfully!")
    
    # Trading Dashboard page
    elif page == "Trading Dashboard":
        # Redirect to trading dashboard
        st.header("Trading Dashboard")
        st.markdown(
            """
            Click the button below to launch the trading dashboard.
            """
        )
        
        if st.button("Launch Trading Dashboard"):
            # Run the trading dashboard in a new process
            os.system("streamlit run simple_trading_dashboard.py &")
            st.success("Trading dashboard launched in a new tab!")
            
            # Provide a direct link
            st.markdown("[Click here if the dashboard doesn't open automatically](http://localhost:8502)")
    
    # Risk Analysis page
    elif page == "Risk Analysis":
        # Redirect to risk analysis
        st.header("Risk Analysis")
        st.markdown(
            """
            Click the button below to launch the risk analysis dashboard.
            """
        )
        
        if st.button("Launch Risk Analysis Dashboard"):
            # Run the risk analysis dashboard in a new process
            os.system("streamlit run simple_trading_dashboard.py --page 'Risk Analysis' &")
            st.success("Risk analysis dashboard launched in a new tab!")
            
            # Provide a direct link
            st.markdown("[Click here if the dashboard doesn't open automatically](http://localhost:8503)")

if __name__ == "__main__":
    main()
