# Oil & Gas Market Optimization: User Instructions

This document provides detailed instructions on how to use the Oil & Gas Market Optimization system to achieve optimal results. It covers data acquisition, model training, trading strategy backtesting, risk management, and dashboard usage.

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Data Acquisition](#data-acquisition)
4. [Forecasting Models](#forecasting-models)
5. [Trading Strategies](#trading-strategies)
6. [Risk Management](#risk-management)
7. [Market Intelligence](#market-intelligence)
8. [Dashboards](#dashboards)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## System Overview

The Oil & Gas Market Optimization system is a comprehensive AI-driven platform for analyzing, forecasting, and optimizing trading strategies for oil and gas commodities. It integrates multiple components:

- **Data Processing Pipeline**: Acquires, cleans, and processes commodity price data
- **Forecasting Models**: Multiple models (ARIMA, XGBoost, LSTM) with automatic model selection
- **Trading Strategies**: Implementation of trend following, mean reversion, and volatility breakout strategies
- **Risk Management**: VaR analysis, Monte Carlo simulations, and portfolio optimization
- **Market Intelligence**: RAG system for answering questions about oil and gas markets
- **Interactive Dashboards**: Streamlit dashboards for visualization and analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/oil_gas_market_optimization.git
cd oil_gas_market_optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data/{raw,processed,features,insights,chroma}
mkdir -p logs
mkdir -p results/{forecasting,backtests,model_selection,monte_carlo,trading}
```

## Data Acquisition

### Using EIA API

The system can fetch data from the U.S. Energy Information Administration (EIA) API:

1. Obtain an API key from [EIA](https://www.eia.gov/opendata/)
2. Set your API key as an environment variable:
```bash
export EIA_API_KEY="your-api-key-here"
```

3. Run the data acquisition pipeline:
```bash
python -m src.pipeline.data.data_acquisition --source eia --commodities crude_oil regular_gasoline conventional_gasoline diesel
```

### Using Yahoo Finance

Alternatively, you can use Yahoo Finance data:

```bash
python -m src.pipeline.data.data_acquisition --source yahoo --commodities crude_oil regular_gasoline conventional_gasoline diesel
```

### Manual Data Import

If you have your own data sources, place CSV files in the `data/raw/` directory with the following naming convention:
- `crude_oil.csv`
- `regular_gasoline.csv`
- `conventional_gasoline.csv`
- `diesel.csv`

Each file should have at minimum a date column and a price column.

## Forecasting Models

### Training Models

To train all forecasting models and automatically select the best one:

```bash
python -m src.pipeline.main --model-type all
```

For specific model types:

```bash
python -m src.pipeline.main --model-type arima  # ARIMA/SARIMA models
python -m src.pipeline.main --model-type xgboost  # XGBoost models
python -m src.pipeline.main --model-type lstm  # LSTM models
```

### Optimal Parameters

For best results with each model type:

#### ARIMA/SARIMA
- Use differencing to achieve stationarity
- Consider seasonal components for gasoline (seasonal period = 12 for monthly data)
- Optimize order parameters using AIC/BIC criteria

#### XGBoost
- Feature engineering is crucial (include technical indicators, lagged values)
- Use cross-validation to prevent overfitting
- Regularization parameters (alpha, lambda) should be tuned

#### LSTM
- Normalize input data using StandardScaler or MinMaxScaler
- Use sequence length of 10-30 days for daily data
- Add dropout layers (0.2-0.3) to prevent overfitting

## Trading Strategies

### Backtesting Strategies

To run backtests on all trading strategies:

```bash
python -m src.pipeline.trading_pipeline --commodities crude_oil --strategy all
```

For specific strategy types:

```bash
python -m src.pipeline.trading_pipeline --commodities crude_oil --strategy ma_crossover
python -m src.pipeline.trading_pipeline --commodities crude_oil --strategy rsi
python -m src.pipeline.trading_pipeline --commodities crude_oil --strategy bollinger
```

### Strategy Optimization

#### Moving Average Crossover
- Fast MA: 5-20 days
- Slow MA: 20-100 days
- Best for trending markets

#### RSI Strategy
- Oversold threshold: 30
- Overbought threshold: 70
- Best for range-bound markets

#### Bollinger Bands
- Window: 20 days
- Standard deviations: 2
- Best for volatile markets

#### Donchian Channel
- Window: 20 days
- Best for breakout trading

## Risk Management

### Value at Risk (VaR) Analysis

To calculate VaR for a portfolio:

```bash
python -m src.pipeline.trading_pipeline --risk-analysis --commodities crude_oil regular_gasoline
```

### Monte Carlo Simulation

For scenario analysis:

```bash
python -m src.pipeline.trading_pipeline --monte-carlo-sims 1000 --commodities crude_oil
```

### Portfolio Optimization

To optimize portfolio weights:

```bash
python -m src.pipeline.trading_pipeline --portfolio-optimization --commodities crude_oil regular_gasoline conventional_gasoline diesel
```

## Market Intelligence

### Setting Up the RAG System

1. Create market insight documents in markdown format and place them in `data/insights/`
2. Run the RAG pipeline to index the insights:

```bash
python -m src.pipeline.rag_pipeline
```

### Using the Market Analyzer Agent

To generate market analysis:

```bash
python -m src.agentic_ai.market_analyzer_agent
```

This will create market summaries and anomaly detection reports in the `data/insights/` directory.

## Dashboards

### Main Dashboard

To launch the main dashboard for forecasting and market intelligence:

```bash
python run_dashboard.py
```

This dashboard provides:
- Price forecasts for each commodity
- Model comparison and evaluation metrics
- Market intelligence Q&A interface
- Data visualization and exploration

### Trading Dashboard

To launch the trading dashboard:

```bash
python run_trading_dashboard.py
```

This dashboard provides:
- Trading strategy backtesting
- Risk analysis tools
- Portfolio optimization
- Performance metrics visualization

## Best Practices

### Data Preparation
- Ensure data is clean and free of outliers
- Handle missing values appropriately (forward fill for time series)
- Use sufficient historical data (at least 2-3 years)

### Model Selection
- Compare multiple models using cross-validation
- Consider ensemble approaches for improved accuracy
- Regularly retrain models as new data becomes available

### Trading Strategy Implementation
- Include transaction costs and slippage in backtests
- Use walk-forward validation for realistic performance assessment
- Combine multiple strategies for diversification

### Risk Management
- Set appropriate position sizing based on VaR
- Use stop-loss orders to limit downside risk
- Regularly rebalance portfolio based on optimization results

## Troubleshooting

### Common Issues

#### Data Acquisition Errors
- Check API key validity
- Ensure internet connectivity
- Verify date ranges are valid

#### Model Training Errors
- Check for missing or invalid data
- Ensure sufficient memory for LSTM models
- Verify feature engineering pipeline

#### Dashboard Errors
- Check if all required packages are installed
- Ensure data files exist in the expected locations
- Verify port availability for Streamlit

### Getting Help

If you encounter issues not covered in this guide, please:
1. Check the logs in the `logs/` directory
2. Review the documentation in the `docs/` directory
3. Submit an issue on the GitHub repository

## Advanced Usage

For advanced users, the system provides:
- API endpoints for programmatic access
- Custom strategy development
- Integration with external data sources
- Automated reporting capabilities

Refer to the API documentation for more details on these advanced features.
