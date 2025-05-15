# Oil & Gas Market Optimization - User Instructions

This document provides detailed instructions for using the Oil & Gas Market Optimization system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Data Pipeline](#data-pipeline)
4. [Trading Dashboard](#trading-dashboard)
5. [Risk Analysis](#risk-analysis)
6. [Troubleshooting](#troubleshooting)

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 10GB free disk space
- Internet connection for data acquisition

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/oil-gas-market-optimization.git
cd oil-gas-market-optimization
```

### Step 2: Create a Virtual Environment

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create Necessary Directories

```bash
mkdir -p data/raw data/processed data/features data/insights data/chroma logs results/forecasting results/backtests results/model_selection results/monte_carlo results/trading
```

## Data Pipeline

The data pipeline processes raw data into a format suitable for analysis and modeling.

### Generate Sample Data

To generate sample data for testing:

```bash
python create_basic_data.py
```

This will create synthetic price data for crude oil, regular gasoline, conventional gasoline, and diesel.

### Run Data Pipeline

To process the data and generate features:

```bash
python run_data_pipeline.py
```

This script performs the following steps:
1. Data acquisition (or uses existing data)
2. Data cleaning and preprocessing
3. Feature engineering
4. Data storage in optimized formats

### Pipeline Options

The data pipeline supports several options:

```bash
python run_data_pipeline.py --help
```

Common options include:
- `--skip-acquisition`: Skip the data acquisition step
- `--skip-cleaning`: Skip the data cleaning step
- `--skip-feature-engineering`: Skip the feature engineering step
- `--commodity`: Specify a specific commodity to process (e.g., crude_oil, regular_gasoline)

## Trading Dashboard

The trading dashboard provides an interactive interface for backtesting trading strategies and analyzing risk.

### Launch the Dashboard

```bash
python run_trading_dashboard.py
```

This will start a Streamlit app that you can access in your web browser.

### Dashboard Features

#### Strategy Backtesting

1. Select a commodity from the dropdown menu
2. Choose a trading strategy:
   - Moving Average Crossover
   - MACD
   - RSI
   - Bollinger Bands
   - Donchian Channel
   - ATR Channel
3. Adjust strategy parameters using the sliders
4. Click "Run Backtest" to execute the strategy
5. View performance metrics and charts

#### Risk Analysis

1. Select a commodity from the dropdown menu
2. View return statistics and distribution
3. Analyze Value at Risk (VaR) at different confidence levels
4. Explore Monte Carlo simulations for portfolio projections

## Risk Analysis

The system includes several risk management tools:

### Value at Risk (VaR)

The VaR calculator estimates the maximum potential loss over a specified time horizon at a given confidence level.

Methods available:
- Historical VaR
- Parametric VaR
- Monte Carlo VaR

### Monte Carlo Simulation

The Monte Carlo simulator generates thousands of possible future price paths based on historical data.

Features:
- Customizable number of simulations
- Adjustable time horizon
- Multiple simulation methods (bootstrap, normal, GBM)
- Statistical analysis of simulation results

## Troubleshooting

### Common Issues

#### Missing Data Files

If you encounter errors about missing data files:

```
Error: No commodity data found. Please run the data pipeline first.
```

Solution: Run the data generation script followed by the data pipeline:

```bash
python create_basic_data.py
python run_data_pipeline.py
```

#### Dashboard Not Starting

If the dashboard fails to start:

```
Error: Could not start Streamlit server.
```

Solution: Check that Streamlit is installed and try running:

```bash
streamlit run simple_trading_dashboard.py
```

#### Memory Issues

If you encounter memory errors during processing:

```
MemoryError: ...
```

Solution: Try processing one commodity at a time:

```bash
python run_data_pipeline.py --commodity crude_oil
```

### Getting Help

If you encounter issues not covered here, please:

1. Check the logs in the `logs/` directory
2. Look for error messages in the console output
3. Consult the documentation in the `docs/` directory
4. Open an issue on the GitHub repository

## Advanced Usage

### Custom Trading Strategies

To implement a custom trading strategy:

1. Create a new file in `src/trading/strategies/`
2. Inherit from the `BaseStrategy` class
3. Implement the `generate_signals` method
4. Register your strategy in `src/trading/strategies/__init__.py`

### Custom Risk Models

To implement a custom risk model:

1. Create a new file in `src/risk/`
2. Implement your risk calculation logic
3. Update the dashboard to include your new risk model

### API Integration

The system can be integrated with external APIs:

1. Add API credentials to `.env` file (create if it doesn't exist)
2. Use the utility functions in `src/utils/data_utils.py` to fetch data
3. Process and store the data using the existing pipeline
