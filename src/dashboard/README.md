# Oil & Gas Market Optimization Dashboard

This Streamlit dashboard provides a user-friendly interface for visualizing and interacting with the Oil & Gas Market Optimization system.

## Features

- **Market Overview**: Compare commodity prices and correlations
- **Price Forecasting**: View AI-generated price forecasts
- **Commodity Analysis**: Analyze individual commodity statistics and trends
- **Interactive Visualizations**: Explore data through interactive Plotly charts

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure the main project pipeline has been run to generate the necessary data and models:

```bash
python ../pipeline/main.py
```

## Usage

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

This will start the dashboard on http://localhost:8501 by default.

## Dashboard Pages

### Market Overview

The Market Overview page provides a high-level view of all commodities, including:

- Commodity price comparison chart (normalized to percentage change)
- Price correlation heatmap
- Individual commodity price charts

### Price Forecasting

The Price Forecasting page allows you to:

- Select a commodity to forecast
- Adjust the forecast horizon
- View the forecast with confidence intervals
- Explore model details and summary statistics

### Commodity Analysis

The Commodity Analysis page provides detailed insights for individual commodities:

- Current price and recent changes
- Price statistics (average, volatility)
- Historical price chart
- Price distribution histogram

### About

The About page provides information about the project, data sources, and models used.

## Customization

You can customize the dashboard by:

- Adding new pages in the `main()` function
- Creating additional visualization functions
- Integrating more advanced models from the project

## Troubleshooting

If you encounter issues:

1. Ensure all data files exist in the expected locations
2. Check that models have been trained and saved correctly
3. Verify that all dependencies are installed
4. Check the logs in the `logs/dashboard.log` file
