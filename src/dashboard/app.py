"""
Streamlit dashboard for the Oil & Gas Market Optimization project.
This dashboard visualizes commodity data, forecasts, and market insights.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.data_utils import load_processed_data
from src.models.forecasting.arima_forecaster import ARIMAForecaster
from src.models.forecasting.xgboost_forecaster import XGBoostForecaster
from src.models.forecasting.lstm_forecaster import LSTMForecaster
from src.rag.agents.insight_qa_agent import InsightQAAgent, create_qa_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
PROCESSED_DATA_DIR = 'data/processed'
FEATURES_DATA_DIR = 'data/features'
MODELS_DIR = 'models/forecasting'
RESULTS_DIR = 'results/model_selection'
INSIGHTS_CSV = 'data/processed/insights.csv'
CHROMA_DIR = 'data/chroma'

# Set page config
st.set_page_config(
    page_title="Oil & Gas Market Optimization",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_commodity_data(commodity: str) -> pd.DataFrame:
    """
    Load processed data for a specific commodity.

    Parameters
    ----------
    commodity : str
        Name of the commodity (e.g., 'crude_oil')

    Returns
    -------
    pd.DataFrame
        Processed data for the commodity
    """
    file_path = os.path.join(PROCESSED_DATA_DIR, f"{commodity}.parquet")

    try:
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} rows for {commodity} from {file_path}")
            return df
        else:
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def load_model(commodity: str) -> Optional[Any]:
    """
    Load the best trained model for a specific commodity.

    Parameters
    ----------
    commodity : str
        Name of the commodity (e.g., 'crude_oil')

    Returns
    -------
    Optional[Any]
        Trained model or None if not found
    """
    # First, check if we have a best model selection
    best_model_file = os.path.join(RESULTS_DIR, f"{commodity}_best_model.json")

    if os.path.exists(best_model_file):
        try:
            with open(best_model_file, 'r') as f:
                best_model_info = json.load(f)

            model_type = best_model_info.get('best_model')
            logger.info(f"Best model for {commodity} is {model_type}")

            # Load the best model
            model_path = os.path.join(MODELS_DIR, f"{commodity}_{model_type}.pkl")

            if os.path.exists(model_path):
                if model_type == 'arima':
                    model = ARIMAForecaster.load(model_path)
                elif model_type == 'xgboost':
                    model = XGBoostForecaster.load(model_path)
                elif model_type == 'lstm':
                    model = LSTMForecaster.load(model_path)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    return None

                logger.info(f"Loaded {model_type} model for {commodity} from {model_path}")
                return model
            else:
                logger.warning(f"Best model file not found: {model_path}")
        except Exception as e:
            logger.error(f"Error loading best model info: {e}")

    # If no best model or error, try to load models in order of preference
    model_types = ['arima', 'xgboost', 'lstm']

    for model_type in model_types:
        model_path = os.path.join(MODELS_DIR, f"{commodity}_{model_type}.pkl")

        try:
            if os.path.exists(model_path):
                if model_type == 'arima':
                    model = ARIMAForecaster.load(model_path)
                elif model_type == 'xgboost':
                    model = XGBoostForecaster.load(model_path)
                elif model_type == 'lstm':
                    model = LSTMForecaster.load(model_path)

                logger.info(f"Loaded {model_type} model for {commodity} from {model_path}")
                return model
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {e}")

    logger.warning(f"No models found for {commodity}")
    return None

def plot_commodity_price(df: pd.DataFrame, commodity: str) -> go.Figure:
    """
    Create a Plotly figure for commodity price.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with commodity data
    commodity : str
        Name of the commodity

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = px.line(
        df,
        x=df.index,
        y=df.columns[0],  # Assuming first column is price
        title=f"{commodity.replace('_', ' ').title()} Price",
        labels={'x': 'Date', 'y': 'Price'},
        template='plotly_white'
    )

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

def plot_forecast(
    df: pd.DataFrame,
    model: ARIMAForecaster,
    commodity: str,
    forecast_days: int = 30
) -> go.Figure:
    """
    Create a Plotly figure with historical data and forecast.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with commodity data
    model : ARIMAForecaster
        Trained ARIMA model
    commodity : str
        Name of the commodity
    forecast_days : int, optional
        Number of days to forecast, by default 30

    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Get the target column
    target_column = model.target_column

    # Generate in-sample predictions
    predictions = model.predict(df)

    # Generate forecast
    forecast = model.model_fit.forecast(steps=forecast_days)

    # Create forecast index
    last_date = df.index[-1]
    forecast_index = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq='D'
    )

    # Create figure
    fig = go.Figure()

    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[target_column],
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        )
    )

    # Add fitted values
    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions,
            mode='lines',
            name='Fitted',
            line=dict(color='green', dash='dash')
        )
    )

    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        )
    )

    # Add confidence intervals if available
    if hasattr(model.model_fit, 'get_forecast'):
        forecast_obj = model.model_fit.get_forecast(steps=forecast_days)
        conf_int = forecast_obj.conf_int()

        fig.add_trace(
            go.Scatter(
                x=forecast_index,
                y=conf_int.iloc[:, 0],
                mode='lines',
                name='Lower CI',
                line=dict(width=0),
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_index,
                y=conf_int.iloc[:, 1],
                mode='lines',
                name='Upper CI',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{commodity.replace('_', ' ').title()} Forecast",
        xaxis_title="Date",
        yaxis_title=target_column,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

def plot_commodity_comparison(data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Create a Plotly figure comparing multiple commodities.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with commodity data

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()

    for commodity, df in data_dict.items():
        if not df.empty:
            # Normalize to percentage change from first date
            first_value = df.iloc[0, 0]
            normalized = (df.iloc[:, 0] / first_value - 1) * 100

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=normalized,
                    mode='lines',
                    name=commodity.replace('_', ' ').title()
                )
            )

    fig.update_layout(
        title="Commodity Price Comparison (% Change)",
        xaxis_title="Date",
        yaxis_title="% Change from Start",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

def plot_correlation_heatmap(data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Create a correlation heatmap for commodity prices.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with commodity data

    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Create a DataFrame with all commodity prices
    price_df = pd.DataFrame()

    for commodity, df in data_dict.items():
        if not df.empty:
            price_df[commodity] = df.iloc[:, 0]

    # Calculate correlation matrix
    corr_matrix = price_df.corr()

    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Commodity Price Correlation",
        labels=dict(x="Commodity", y="Commodity", color="Correlation")
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

def main():
    """Main function for the Streamlit dashboard."""
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)

    # Title and description
    st.title("🛢️ Oil & Gas Market Optimization Dashboard")
    st.markdown(
        """
        This dashboard provides insights into oil and gas commodity prices,
        forecasts, and market trends using AI-powered analytics.
        """
    )

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Market Overview", "Price Forecasting", "Commodity Analysis", "Market Intelligence", "About"]
    )

    # Available commodities
    commodities = ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel']
    available_commodities = []

    # Load data for all commodities
    data_dict = {}
    for commodity in commodities:
        df = load_commodity_data(commodity)
        if not df.empty:
            data_dict[commodity] = df
            available_commodities.append(commodity)

    if not available_commodities:
        st.error("No commodity data found. Please run the data pipeline first.")
        return

    # Market Overview page
    if page == "Market Overview":
        st.header("Market Overview")

        # Commodity comparison
        st.subheader("Commodity Price Comparison")
        comparison_fig = plot_commodity_comparison(data_dict)
        st.plotly_chart(comparison_fig, use_container_width=True)

        # Correlation heatmap
        st.subheader("Price Correlation")
        corr_fig = plot_correlation_heatmap(data_dict)
        st.plotly_chart(corr_fig, use_container_width=True)

        # Individual commodity charts
        st.subheader("Individual Commodity Prices")
        cols = st.columns(2)

        for i, (commodity, df) in enumerate(data_dict.items()):
            with cols[i % 2]:
                fig = plot_commodity_price(df, commodity)
                st.plotly_chart(fig, use_container_width=True)

    # Price Forecasting page
    elif page == "Price Forecasting":
        st.header("Price Forecasting")

        # Select commodity
        selected_commodity = st.selectbox(
            "Select a commodity",
            available_commodities,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Load model
        model = load_model(selected_commodity)

        if model is not None:
            # Forecast parameters
            forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30)

            # Display forecast
            st.subheader(f"{selected_commodity.replace('_', ' ').title()} Price Forecast")
            forecast_fig = plot_forecast(
                data_dict[selected_commodity],
                model,
                selected_commodity,
                forecast_days
            )
            st.plotly_chart(forecast_fig, use_container_width=True)

            # Model information
            st.subheader("Model Information")

            # Display model type and details based on the model class
            model_type = "Unknown"
            if isinstance(model, ARIMAForecaster):
                model_type = "ARIMA"
                st.write(f"Model: {model_type}{model.order}")
                if model.seasonal_order:
                    st.write(f"Seasonal: {model.seasonal_order}")

                # Model summary
                if hasattr(model.model_fit, 'summary'):
                    with st.expander("Model Summary"):
                        st.text(str(model.model_fit.summary()))

            elif isinstance(model, XGBoostForecaster):
                model_type = "XGBoost"
                st.write(f"Model: {model_type}")
                st.write(f"Parameters: {model.params}")

                # Feature importance
                if model.feature_importance is not None:
                    with st.expander("Feature Importance"):
                        top_features = model.feature_importance.head(10)
                        st.bar_chart(top_features.set_index('feature')['importance'])

            elif isinstance(model, LSTMForecaster):
                model_type = "LSTM"
                st.write(f"Model: {model_type}")
                st.write(f"Parameters: {model.params}")
                st.write(f"Sequence Length: {model.sequence_length}")

                # Model architecture
                if model.model is not None:
                    with st.expander("Model Architecture"):
                        stringlist = []
                        model.model.summary(print_fn=lambda x: stringlist.append(x))
                        st.text("\n".join(stringlist))
        else:
            st.error(f"No trained model found for {selected_commodity}. Please run the modeling pipeline first.")

    # Commodity Analysis page
    elif page == "Commodity Analysis":
        st.header("Commodity Analysis")

        # Select commodity
        selected_commodity = st.selectbox(
            "Select a commodity",
            available_commodities,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        df = data_dict[selected_commodity]

        # Price statistics
        st.subheader("Price Statistics")

        # Create metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current Price",
                f"${df.iloc[-1, 0]:.2f}",
                f"{(df.iloc[-1, 0] - df.iloc[-2, 0]) / df.iloc[-2, 0]:.2%}"
            )

        with col2:
            st.metric(
                "1-Month Change",
                f"{(df.iloc[-1, 0] - df.iloc[-30, 0]) / df.iloc[-30, 0]:.2%}"
            )

        with col3:
            st.metric(
                "Average (30d)",
                f"${df.iloc[-30:, 0].mean():.2f}"
            )

        with col4:
            st.metric(
                "Volatility (30d)",
                f"{df.iloc[-30:, 0].pct_change().std() * np.sqrt(252):.2%}"
            )

        # Price chart
        st.subheader("Price History")
        price_fig = plot_commodity_price(df, selected_commodity)
        st.plotly_chart(price_fig, use_container_width=True)

        # Price distribution
        st.subheader("Price Distribution")
        hist_fig = px.histogram(
            df,
            x=df.columns[0],
            nbins=50,
            title=f"{selected_commodity.replace('_', ' ').title()} Price Distribution",
            labels={df.columns[0]: 'Price'},
            template='plotly_white'
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    # Market Intelligence page
    elif page == "Market Intelligence":
        st.header("Market Intelligence")

        st.markdown(
            """
            This page provides AI-powered market intelligence using a Retrieval-Augmented Generation (RAG) system.
            Ask questions about oil and gas markets to get insights based on our knowledge base.
            """
        )

        # Check if insights file exists
        if not os.path.exists(INSIGHTS_CSV):
            st.warning(
                """
                No market insights found. Please run the RAG pipeline first:
                ```
                python -m src.pipeline.rag_pipeline
                ```
                """
            )
        else:
            try:
                # Create QA agent
                qa_agent = create_qa_agent(INSIGHTS_CSV, CHROMA_DIR)

                # Commodity filter
                commodity_options = ["All Commodities"] + available_commodities
                selected_commodity = st.selectbox(
                    "Filter by commodity",
                    commodity_options,
                    format_func=lambda x: x.replace('_', ' ').title() if x != "All Commodities" else x
                )

                # Convert "All Commodities" to None for the agent
                commodity_filter = None if selected_commodity == "All Commodities" else selected_commodity

                # Question input
                question = st.text_input("Ask a question about oil and gas markets")

                if question:
                    with st.spinner("Searching for insights..."):
                        # Get answer
                        response = qa_agent.answer_question(
                            question=question,
                            n_results=5,
                            commodity=commodity_filter
                        )

                    # Display answer
                    st.subheader("Answer")
                    st.write(response['answer'])

                    # Display sources
                    st.subheader("Sources")
                    for i, source in enumerate(response['sources']):
                        with st.expander(f"Source {i+1} - {source['commodity'].replace('_', ' ').title()} (Relevance: {source['relevance']:.2f})"):
                            st.write(source['content'])

                # Example questions
                st.subheader("Example Questions")
                example_questions = [
                    "What are the current trends in crude oil prices?",
                    "How do gasoline prices correlate with crude oil?",
                    "What factors affect diesel fuel prices?",
                    "What is the outlook for oil markets in the next quarter?"
                ]

                for question in example_questions:
                    if st.button(question):
                        st.session_state.question = question
                        st.experimental_rerun()

            except Exception as e:
                st.error(f"Error initializing QA agent: {e}")
                st.info(
                    """
                    Please make sure you have run the RAG pipeline:
                    ```
                    python -m src.pipeline.rag_pipeline
                    ```
                    """
                )

    # About page
    elif page == "About":
        st.header("About")
        st.markdown(
            """
            ## Oil & Gas Market Optimization

            This dashboard is part of the AI-Driven Market and Supply Optimization for Oil & Gas Commodities project.

            ### Features

            - **Market Overview**: Compare commodity prices and correlations
            - **Price Forecasting**: View AI-generated price forecasts
            - **Commodity Analysis**: Analyze individual commodity statistics and trends

            ### Data Sources

            - EIA (U.S. Energy Information Administration)
            - Yahoo Finance

            ### Models

            - ARIMA/SARIMA for time series forecasting
            - XGBoost for machine learning-based forecasting
            - LSTM for deep learning-based forecasting
            - Automatic model selection to choose the best performing model

            ### Contact

            For more information, please contact the project team.
            """
        )

if __name__ == "__main__":
    main()
