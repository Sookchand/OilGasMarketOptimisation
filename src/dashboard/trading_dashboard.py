"""
Trading dashboard for the Oil & Gas Market Optimization project.
This dashboard visualizes trading strategies, backtests, and risk metrics.
"""

import os
import sys
import json
import logging
import glob
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

from src.trading.strategies.trend_following import MovingAverageCrossover, MACDStrategy
from src.trading.strategies.mean_reversion import RSIStrategy, BollingerBandsStrategy
from src.trading.strategies.volatility_breakout import DonchianChannelStrategy, ATRChannelStrategy
from src.trading.execution.backtester import Backtester
from src.trading.execution.performance_metrics import calculate_returns_metrics, plot_performance
from src.risk.var_calculator import VaRCalculator
from src.risk.monte_carlo import MonteCarloSimulator
from src.utils.data_utils import load_processed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
PROCESSED_DATA_DIR = 'data/processed'
RESULTS_DIR = 'results/trading'

# Set page config
st.set_page_config(
    page_title="Oil & Gas Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_commodity_data(commodity: str) -> pd.DataFrame:
    """
    Load processed commodity data.
    
    Parameters
    ----------
    commodity : str
        Commodity name
    
    Returns
    -------
    pd.DataFrame
        Commodity data
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

def prepare_data_for_trading(
    df: pd.DataFrame,
    price_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare data for trading.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    price_col : str, optional
        Price column name, by default None (use first column)
    
    Returns
    -------
    pd.DataFrame
        Prepared data
    """
    # Make a copy of the data
    data = df.copy()
    
    # If price_col not specified, use the first column
    if price_col is None:
        price_col = data.columns[0]
    
    # Rename price column to 'close' for consistency
    data = data.rename(columns={price_col: 'close'})
    
    # Calculate OHLC if not available
    if 'open' not in data.columns:
        data['open'] = data['close'].shift(1)
    
    if 'high' not in data.columns:
        data['high'] = data['close']
    
    if 'low' not in data.columns:
        data['low'] = data['close']
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Drop NaN values
    data = data.dropna()
    
    return data

def create_strategy(
    strategy_type: str,
    **kwargs
) -> Any:
    """
    Create a trading strategy.
    
    Parameters
    ----------
    strategy_type : str
        Strategy type
    **kwargs : dict
        Strategy parameters
    
    Returns
    -------
    Any
        Trading strategy
    """
    if strategy_type == 'ma_crossover':
        return MovingAverageCrossover(**kwargs)
    elif strategy_type == 'macd':
        return MACDStrategy(**kwargs)
    elif strategy_type == 'rsi':
        return RSIStrategy(**kwargs)
    elif strategy_type == 'bollinger':
        return BollingerBandsStrategy(**kwargs)
    elif strategy_type == 'donchian':
        return DonchianChannelStrategy(**kwargs)
    elif strategy_type == 'atr':
        return ATRChannelStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

def load_backtest_results() -> List[Dict[str, Any]]:
    """
    Load all backtest results.
    
    Returns
    -------
    List[Dict[str, Any]]
        List of backtest results
    """
    results = []
    
    # Get all JSON files in results directory
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            # Add file path
            result['file_path'] = file_path
            
            # Add to results
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return results

def plot_backtest_results(results: pd.DataFrame) -> go.Figure:
    """
    Plot backtest results.
    
    Parameters
    ----------
    results : pd.DataFrame
        Backtest results
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Portfolio Value', 'Drawdown'),
        row_heights=[0.7, 0.3]
    )
    
    # Add portfolio value
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add drawdown
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='red'),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Backtest Results',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        yaxis2_title='Drawdown (%)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        height=600
    )
    
    return fig

def plot_var_analysis(returns: pd.Series) -> go.Figure:
    """
    Plot Value at Risk analysis.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Calculate VaR
    calculator = VaRCalculator(returns)
    var_95 = calculator.historical_var()
    var_99 = calculator.historical_var(confidence_level=0.99)
    
    # Calculate Expected Shortfall
    es_95 = calculator.calculate_expected_shortfall()
    
    # Create histogram of returns
    fig = px.histogram(
        returns,
        nbins=50,
        title='Return Distribution with VaR',
        labels={'value': 'Return', 'count': 'Frequency'},
        template='plotly_white'
    )
    
    # Add VaR lines
    fig.add_vline(
        x=-var_95,
        line_dash="dash",
        line_color="red",
        annotation_text=f"95% VaR: {var_95:.2%}",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=-var_99,
        line_dash="dash",
        line_color="darkred",
        annotation_text=f"99% VaR: {var_99:.2%}",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=-es_95,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"95% ES: {es_95:.2%}",
        annotation_position="top right"
    )
    
    return fig

def plot_monte_carlo_simulation(returns: pd.Series, initial_value: float = 10000.0) -> go.Figure:
    """
    Plot Monte Carlo simulation.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    initial_value : float, optional
        Initial portfolio value, by default 10000.0
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Create simulator
    simulator = MonteCarloSimulator(
        returns=returns,
        initial_value=initial_value,
        num_simulations=100,
        time_horizon=252
    )
    
    # Run simulation
    simulations = simulator.run_simulation()
    
    # Calculate statistics
    stats = simulator.calculate_statistics()
    
    # Create figure
    fig = go.Figure()
    
    # Add simulations
    for i in range(20):  # Plot 20 random simulations
        fig.add_trace(
            go.Scatter(
                x=simulations.index,
                y=simulations[i],
                mode='lines',
                opacity=0.3,
                line=dict(color='blue'),
                showlegend=False
            )
        )
    
    # Add mean path
    mean_path = simulations.mean(axis=1)
    fig.add_trace(
        go.Scatter(
            x=simulations.index,
            y=mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(color='red', width=2)
        )
    )
    
    # Add confidence intervals
    percentile_5 = simulations.quantile(0.05, axis=1)
    percentile_95 = simulations.quantile(0.95, axis=1)
    
    fig.add_trace(
        go.Scatter(
            x=simulations.index,
            y=percentile_5,
            mode='lines',
            name='5th Percentile',
            line=dict(color='green', width=1, dash='dash')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=simulations.index,
            y=percentile_95,
            mode='lines',
            name='95th Percentile',
            line=dict(color='green', width=1, dash='dash'),
            fill='tonexty'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Monte Carlo Simulation (1 Year)',
        xaxis_title='Trading Days',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        height=500
    )
    
    return fig

def trading_dashboard():
    """Main function for the trading dashboard."""
    # Create directories if they don't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Title and description
    st.title("📈 Oil & Gas Trading Dashboard")
    st.markdown(
        """
        This dashboard provides tools for backtesting trading strategies, analyzing risk,
        and optimizing portfolios for oil and gas commodities.
        """
    )
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Strategy Backtesting", "Risk Analysis", "Portfolio Optimization"]
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
            'macd': 'MACD',
            'rsi': 'RSI',
            'bollinger': 'Bollinger Bands',
            'donchian': 'Donchian Channel',
            'atr': 'ATR Channel'
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
        
        elif selected_strategy == 'macd':
            col1, col2, col3 = st.columns(3)
            with col1:
                strategy_params['fast_window'] = st.slider("Fast Window", 5, 30, 12)
            with col2:
                strategy_params['slow_window'] = st.slider("Slow Window", 15, 50, 26)
            with col3:
                strategy_params['signal_window'] = st.slider("Signal Window", 5, 20, 9)
        
        elif selected_strategy == 'rsi':
            col1, col2, col3 = st.columns(3)
            with col1:
                strategy_params['window'] = st.slider("RSI Window", 5, 30, 14)
            with col2:
                strategy_params['oversold'] = st.slider("Oversold Level", 10, 40, 30)
            with col3:
                strategy_params['overbought'] = st.slider("Overbought Level", 60, 90, 70)
        
        elif selected_strategy == 'bollinger':
            col1, col2 = st.columns(2)
            with col1:
                strategy_params['window'] = st.slider("Window", 5, 50, 20)
            with col2:
                strategy_params['num_std'] = st.slider("Standard Deviations", 1.0, 3.0, 2.0, 0.1)
        
        elif selected_strategy == 'donchian':
            strategy_params['window'] = st.slider("Window", 5, 50, 20)
        
        elif selected_strategy == 'atr':
            col1, col2 = st.columns(2)
            with col1:
                strategy_params['window'] = st.slider("Window", 5, 30, 14)
            with col2:
                strategy_params['multiplier'] = st.slider("Multiplier", 1.0, 5.0, 2.0, 0.1)
        
        # Backtest parameters
        st.subheader("Backtest Parameters")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_capital = st.number_input("Initial Capital ($)", 1000.0, 1000000.0, 10000.0, 1000.0)
        with col2:
            commission = st.number_input("Commission (%)", 0.0, 1.0, 0.1, 0.01) / 100
        with col3:
            slippage = st.number_input("Slippage (%)", 0.0, 1.0, 0.1, 0.01) / 100
        
        # Run backtest button
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    # Load and prepare data
                    df = data_dict[selected_commodity]
                    data = prepare_data_for_trading(df)
                    
                    # Create strategy
                    strategy = create_strategy(selected_strategy, **strategy_params)
                    
                    # Create backtester
                    backtester = Backtester(
                        strategy=strategy,
                        initial_capital=initial_capital,
                        commission=commission,
                        slippage=slippage
                    )
                    
                    # Run backtest
                    results = backtester.run(data, price_col='close')
                    
                    # Calculate metrics
                    metrics = backtester.calculate_metrics()
                    
                    # Display results
                    st.subheader("Backtest Results")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{metrics['total_return']:.2%}")
                    with col2:
                        st.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
                    with col3:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    with col4:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                    with col2:
                        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    with col3:
                        st.metric("Number of Trades", f"{metrics['num_trades']:.0f}")
                    with col4:
                        st.metric("Avg Trade", f"{metrics['avg_trade']:.2%}")
                    
                    # Plot results
                    fig = plot_backtest_results(results)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save results
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    results_file = os.path.join(RESULTS_DIR, f"{selected_commodity}_{selected_strategy}_{timestamp}.csv")
                    results.to_csv(results_file)
                    st.success(f"Backtest results saved to {results_file}")
                    
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
        data = prepare_data_for_trading(df)
        
        # Risk analysis parameters
        st.subheader("Risk Analysis Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            confidence_level = st.slider("Confidence Level (%)", 90, 99, 95) / 100
        with col2:
            lookback_days = st.slider("Lookback Period (days)", 30, 365, 252)
        
        # Get recent returns
        returns = data['returns'].iloc[-lookback_days:]
        
        # Value at Risk
        st.subheader("Value at Risk (VaR) Analysis")
        
        # Calculate VaR
        calculator = VaRCalculator(returns, confidence_level)
        var_historical = calculator.historical_var()
        var_parametric = calculator.parametric_var()
        var_monte_carlo = calculator.monte_carlo_var()
        
        # Display VaR
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Historical VaR", f"{var_historical:.2%}")
        with col2:
            st.metric("Parametric VaR", f"{var_parametric:.2%}")
        with col3:
            st.metric("Monte Carlo VaR", f"{var_monte_carlo:.2%}")
        
        # Plot VaR
        var_fig = plot_var_analysis(returns)
        st.plotly_chart(var_fig, use_container_width=True)
        
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
                    # Create simulator
                    simulator = MonteCarloSimulator(
                        returns=returns,
                        initial_value=initial_value,
                        num_simulations=num_simulations,
                        time_horizon=252
                    )
                    
                    # Run simulation
                    simulations = simulator.run_simulation()
                    
                    # Calculate statistics
                    stats = simulator.calculate_statistics()
                    
                    # Display statistics
                    st.subheader("Simulation Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Final Value", f"${stats['mean']:.2f}")
                    with col2:
                        st.metric("Median Final Value", f"${stats['median']:.2f}")
                    with col3:
                        st.metric("95% VaR", f"{stats['var_95']:.2%}")
                    with col4:
                        st.metric("Probability of Loss", f"{stats['probability_of_loss']:.2%}")
                    
                    # Plot simulation
                    mc_fig = plot_monte_carlo_simulation(returns, initial_value)
                    st.plotly_chart(mc_fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running Monte Carlo simulation: {e}")
    
    # Portfolio Optimization page
    elif page == "Portfolio Optimization":
        st.header("Portfolio Optimization")
        
        st.info("This page allows you to optimize a portfolio of oil and gas commodities.")
        
        # Select commodities
        selected_commodities = st.multiselect(
            "Select commodities for portfolio",
            available_commodities,
            default=available_commodities,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if len(selected_commodities) < 2:
            st.warning("Please select at least 2 commodities for portfolio optimization.")
        else:
            # Optimization parameters
            st.subheader("Optimization Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                lookback_days = st.slider("Lookback Period (days)", 30, 365, 252)
            with col2:
                risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 0.0, 0.1) / 100
            
            # Run optimization button
            if st.button("Run Portfolio Optimization"):
                with st.spinner("Optimizing portfolio..."):
                    try:
                        # Prepare returns data
                        returns_data = pd.DataFrame()
                        
                        for commodity in selected_commodities:
                            df = data_dict[commodity]
                            data = prepare_data_for_trading(df)
                            returns = data['returns'].iloc[-lookback_days:]
                            returns_data[commodity] = returns
                        
                        # Drop NaN values
                        returns_data = returns_data.dropna()
                        
                        # Calculate correlation matrix
                        correlation_matrix = returns_data.corr()
                        
                        # Display correlation heatmap
                        st.subheader("Correlation Matrix")
                        corr_fig = px.imshow(
                            correlation_matrix,
                            text_auto=True,
                            color_continuous_scale='RdBu_r',
                            title="Commodity Correlation Matrix",
                            labels=dict(x="Commodity", y="Commodity", color="Correlation")
                        )
                        st.plotly_chart(corr_fig, use_container_width=True)
                        
                        # Calculate efficient frontier
                        from src.risk.portfolio_optimizer import PortfolioOptimizer
                        
                        optimizer = PortfolioOptimizer(returns_data, risk_free_rate)
                        optimizer.optimize_sharpe_ratio()
                        optimizer.calculate_efficient_frontier()
                        
                        # Get optimal weights
                        optimal_weights = optimizer.get_optimal_weights()
                        optimal_stats = optimizer.get_portfolio_stats()
                        
                        # Display optimal portfolio
                        st.subheader("Optimal Portfolio")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Expected Return", f"{optimal_stats['return']:.2%}")
                        with col2:
                            st.metric("Volatility", f"{optimal_stats['volatility']:.2%}")
                        with col3:
                            st.metric("Sharpe Ratio", f"{optimal_stats['sharpe_ratio']:.2f}")
                        
                        # Display weights
                        st.subheader("Optimal Weights")
                        
                        # Create pie chart
                        weights_fig = px.pie(
                            values=optimal_weights.values,
                            names=optimal_weights.index.map(lambda x: x.replace('_', ' ').title()),
                            title="Optimal Portfolio Weights"
                        )
                        st.plotly_chart(weights_fig, use_container_width=True)
                        
                        # Display efficient frontier
                        st.subheader("Efficient Frontier")
                        
                        ef_fig = optimizer.plot_efficient_frontier()
                        st.pyplot(ef_fig)
                        
                    except Exception as e:
                        st.error(f"Error optimizing portfolio: {e}")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the dashboard
    trading_dashboard()
