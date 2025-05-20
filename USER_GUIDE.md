# Oil & Gas Market Optimization System: User Guide

## Introduction

Welcome to the Oil & Gas Market Optimization System! This comprehensive platform provides powerful tools for analyzing oil and gas market data, optimizing trading strategies, and enhancing decision-making processes. This guide will help you navigate the system and make the most of its features.

**Live Demo**: [https://7jpnmpbuxmt9bumsmvqjpn.streamlit.app/](https://7jpnmpbuxmt9bumsmvqjpn.streamlit.app/)

## Getting Started

### Accessing the System

1. Open your web browser and navigate to the URL provided above
2. The system will load the Home page with an overview of all available modules
3. Use the sidebar navigation to move between different modules

### Navigation

The system includes eight main modules:

1. **Home**: Overview and quick access to all modules
2. **Data Management**: Generate and visualize commodity data, upload real-world data
3. **Trading Dashboard**: Backtest trading strategies
4. **Risk Analysis**: Assess market risks and run simulations
5. **Predictive Analytics**: Forecast prices and compare with real-world data
6. **Risk Assessment**: Analyze market, geopolitical, and regulatory risks
7. **Decision Support**: Model scenarios and explore visualizations
8. **Data Drift Detection**: Compare models with real-world data and detect when retraining is needed

You can navigate between modules using:
- The sidebar navigation menu
- Navigation buttons at the bottom of each page

## Module-by-Module Guide

### 1. Home

The Home page provides:
- An overview of the system's capabilities
- Quick access buttons to all modules
- Key performance metrics

**Best Practice**: Start here to get familiar with the system's capabilities before diving into specific modules.

### 2. Data Management

This module allows you to:
- Generate sample data for multiple commodities
- Upload real-world CSV data for comparison
- Visualize price trends
- Process and analyze market data
- Compare training data with real-world data

**Step-by-Step Usage for Sample Data Generation**:

1. Select the "Generate Sample Data" tab
2. Click "Generate Sample Data for All Commodities" to create data for all available commodities
3. Alternatively, select a specific commodity and click "Generate Sample Data" for that commodity only
4. View the generated data in the chart and data table

**Step-by-Step Usage for Real-World Data Upload**:

1. Select the "Upload Real-World Data" tab
2. Choose a commodity from the dropdown menu
3. Click "Browse files" to select a CSV file from your computer
   - The CSV file should have columns for "Date" and "Price"
   - You can view the expected format by clicking "Show expected CSV format"
4. Once uploaded, the data will be displayed in a chart and table
5. If you have already generated sample data for the same commodity, a comparison chart will be shown
6. Click "Analyze Data Drift" to go directly to the Data Drift Detection module

**Best Practice**:
- Always generate sample data first before uploading real-world data
- Ensure your CSV files follow the expected format (Date in YYYY-MM-DD format, Price as numeric values)
- Upload data for multiple commodities to enable comprehensive analysis

### 3. Trading Dashboard

This module enables you to:
- Backtest trading strategies on different commodities
- Optimize strategy parameters
- Analyze performance metrics
- Visualize trading signals

**Step-by-Step Usage**:

1. Select a commodity from the dropdown menu
2. Choose a trading strategy (Moving Average Crossover or RSI)
3. Adjust strategy parameters using the sliders
4. Click "Run Backtest" to execute the strategy
5. Analyze the performance metrics and charts
6. Adjust parameters and rerun to optimize the strategy

**Key Parameters**:
- **Moving Average Crossover**:
  - Short Window: Length of short-term moving average (typically 5-50 days)
  - Long Window: Length of long-term moving average (typically 20-200 days)

- **RSI Strategy**:
  - RSI Period: Length of period for RSI calculation (typically 14 days)
  - Overbought Level: Threshold for selling signals (typically 70)
  - Oversold Level: Threshold for buying signals (typically 30)

**Best Practice**: Test multiple parameter combinations to find the optimal strategy for each commodity.

### 4. Risk Analysis

This module helps you:
- Calculate return statistics
- Estimate Value at Risk (VaR)
- Run Monte Carlo simulations
- Project portfolio values

**Step-by-Step Usage**:

1. Select a commodity from the dropdown menu
2. View the return statistics in the summary table
3. Adjust the confidence level and investment amount for VaR calculation
4. Click "Calculate VaR" to see the results
5. Set parameters for Monte Carlo simulation
6. Click "Run Monte Carlo Simulation" to generate projections
7. Analyze the simulation results and distribution charts

**Key Parameters**:
- **VaR Calculation**:
  - Confidence Level: Probability threshold (typically 95% or 99%)
  - Investment Amount: Value of your position

- **Monte Carlo Simulation**:
  - Number of Simulations: More simulations provide more reliable results (typically 1,000-10,000)
  - Time Horizon: Future period to simulate (in days)
  - Initial Investment: Starting portfolio value

**Best Practice**: Use multiple risk metrics together for a comprehensive risk assessment.

### 5. Predictive Analytics

This module allows you to:
- Forecast commodity prices using multiple models
- Compare forecasts with real-world data
- Calculate forecast accuracy metrics
- Optimize production levels
- Schedule preventive maintenance
- Visualize supply chain operations

**Step-by-Step Usage for Price Forecasting**:

1. Select a commodity from the dropdown menu
2. Choose a forecasting model (ARIMA, Prophet, LSTM, XGBoost, or Ensemble)
3. Set the forecast horizon (number of days to predict)
4. Adjust the confidence level for prediction intervals
5. Click "Generate Forecast" to see predictions
6. If real-world data is available for the selected commodity, the system will:
   - Display the forecast alongside actual data
   - Calculate accuracy metrics (MAPE, RMSE)
   - Show a comparison chart
7. If no real-world data is available, you'll see a message with a button to go to the Data Upload page

**Step-by-Step Usage for Other Features**:

1. Explore the production optimization tab to determine optimal output levels
2. Use the maintenance scheduling tab to plan equipment maintenance
3. View the supply chain visualization to identify optimization opportunities

**Key Parameters**:
- **Price Forecasting**:
  - Model Selection: Different models have different strengths
    - ARIMA: Good for stable time series with clear patterns
    - Prophet: Handles seasonality and holidays well
    - LSTM: Neural network approach for complex patterns
    - XGBoost: Machine learning approach with feature engineering
    - Ensemble: Combines multiple models for improved accuracy
  - Forecast Horizon: Number of days to predict (typically 7-90 days)
  - Confidence Level: Probability for prediction intervals (80-99%)

- **Production Optimization**:
  - Price Scenario: Expected market conditions
  - Production Capacity: Maximum possible output
  - Cost Structure: Fixed and variable costs

**Best Practice**:
- Upload real-world data to validate forecast accuracy
- Compare forecasts from multiple models
- Consider the confidence intervals when making decisions
- Use the accuracy metrics to determine which model performs best for each commodity

### 6. Risk Assessment

This module helps you:
- Analyze market risks
- Monitor geopolitical events
- Track regulatory changes
- Assess portfolio diversification

**Step-by-Step Usage**:

1. Select a commodity from the dropdown menu
2. Explore the Market Risk Analysis tab to see VaR and hedging recommendations
3. View the Geopolitical Risk Monitoring tab to assess global risks
4. Check the Regulatory Compliance tab to track relevant regulations
5. Analyze the portfolio diversification recommendations

**Key Features**:
- **Market Risk Analysis**:
  - Value at Risk (VaR) calculation
  - Stress testing with different scenarios
  - Hedging recommendations
  - Portfolio diversification analysis

- **Geopolitical Risk Monitoring**:
  - Global risk map
  - Current events tracking
  - Risk alerts for major developments

**Best Practice**: Regularly review all three risk categories (market, geopolitical, regulatory) for a comprehensive risk assessment.

### 7. Decision Support

This module enables you to:
- Model different market scenarios
- Ask questions in natural language
- Explore advanced visualizations
- Optimize supply chain networks

**Step-by-Step Usage**:

1. Select a commodity from the dropdown menu
2. In the Scenario Modeling tab, adjust parameters for different scenarios
3. Click "Generate Scenarios" to see the impact analysis
4. Use the Natural Language Interface tab to ask market-related questions
5. Explore the Advanced Visualizations tab to see correlation heatmaps, seasonality charts, and more

**Key Features**:
- **Scenario Modeling**:
  - Price change scenarios
  - Volatility change scenarios
  - Combined scenarios with financial impact analysis

- **Natural Language Interface**:
  - Question answering about market dynamics
  - Supply-demand analysis visualization
  - Explanations of market relationships

**Best Practice**: Create multiple scenarios with different parameter combinations to understand the range of possible outcomes.

### 8. Data Drift Detection

This module helps you:
- Compare statistical properties between training and real-world data
- Visualize distributions and time series for both datasets
- Detect significant drift in your models
- Get recommendations for model retraining
- Identify which features have drifted the most

### 9. EIA Price Drivers Integration

This module enables you to:
- Access and analyze EIA data on crude oil price drivers
- Visualize the impact of supply, demand, and inventory factors on prices
- Enhance forecasting accuracy with price drivers data
- Understand the relative importance of different market factors
- Make more informed trading and risk management decisions

**Step-by-Step Usage**:

1. Navigate to the Predictive Analytics module
2. Select a commodity from the dropdown menu
3. Choose "Price Drivers" as the forecasting model
4. Set the forecast horizon and confidence level
5. Click "Generate Forecast" to see predictions
6. The system will automatically:
   - Display the forecast with confidence intervals
   - Show the feature importance of different price drivers
   - Provide explanations of the top price drivers
7. Use the feature importance chart to understand which factors are most influential
8. Read the explanations to gain insights into how each factor affects prices

**Key Features**:
- **Price Drivers Data**:
  - OPEC production levels
  - Non-OPEC production
  - Global consumption
  - OECD inventories
  - Supply-demand balance
  - Days of supply
  - Spare production capacity

- **Feature Importance Analysis**:
  - Relative importance of each price driver
  - Visual representation through bar charts
  - Detailed explanations of top drivers

**Best Practice**:
- Compare forecasts from the Price Drivers model with other models
- Pay special attention to the top 3-5 most important features
- Consider how geopolitical events might affect the key price drivers
- Use the insights to inform your trading and risk management strategies

**Step-by-Step Usage for Data Drift Detection**:

1. First, ensure you have both:
   - Generated sample data for at least one commodity
   - Uploaded real-world data for the same commodity
2. Navigate to the Data Drift Detection module
3. Select a commodity from the dropdown menu
4. The system will automatically:
   - Calculate and display statistical comparisons
   - Show visual comparisons (histograms, Q-Q plots)
   - Display time series comparison
   - Determine if significant drift has occurred
5. Review the drift analysis and recommendations
6. Use the navigation buttons to go to relevant modules based on the findings

**Key Features**:
- **Statistical Comparison**:
  - Mean, standard deviation, min, max, skewness, kurtosis
  - Percent change calculation for each statistic
  - Automatic detection of significant drift

- **Visual Comparison**:
  - Distribution histograms for both datasets
  - Q-Q plot for distribution comparison
  - Time series overlay with common date range highlighting

- **Drift Metrics**:
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - RMSE (Root Mean Square Error)

- **Recommendations**:
  - Specific actions based on drift analysis
  - Identification of features with the most significant drift
  - Suggestions for model retraining approaches

**Best Practice**:
- Check for data drift regularly when new data becomes available
- Pay special attention to the feature with the highest drift percentage
- Follow the recommended actions when significant drift is detected
- Consider retraining models with a combination of historical and new data

## Tips for Effective Use

### Data Flow Between Modules

The system maintains data consistency across modules:
- Data generated in the Data Management module is available to all other modules
- The selected commodity is remembered as you navigate between modules
- Analysis results are calculated in real-time based on the current data

### Optimal Workflow

For the most effective use of the system, follow this workflow:

1. **Start with Data Management**:
   - Generate sample data for all commodities
   - Upload real-world data for comparison
   - Explore the data to understand basic trends

2. **Analyze Trading Strategies**:
   - Test different strategies on each commodity
   - Identify the best-performing strategy and parameters

3. **Assess Risks**:
   - Calculate VaR for your selected strategies
   - Run Monte Carlo simulations to understand potential outcomes

4. **Generate Forecasts**:
   - Use multiple models to forecast future prices
   - Try the Price Drivers model for enhanced forecasting
   - Compare forecasts with real-world data
   - Evaluate forecast accuracy metrics
   - Consider the confidence intervals in your planning
   - Analyze the feature importance of price drivers

5. **Detect Data Drift**:
   - Compare statistical properties between training and real-world data
   - Check if significant drift has occurred
   - Review recommendations for model retraining
   - Identify which features have drifted the most

6. **Evaluate Broader Risks**:
   - Check market, geopolitical, and regulatory risks
   - Identify potential hedging opportunities

7. **Model Scenarios**:
   - Create scenarios based on your risk assessment
   - Analyze the financial impact of each scenario

### Performance Considerations

- Generating large datasets or running many simulations may take time
- Monte Carlo simulations with many iterations can be computationally intensive
- The system uses caching to improve performance for repeated analyses

## Troubleshooting

### Common Issues

1. **No Data Available**:
   - Solution: Go to the Data Management page and generate sample data

2. **Slow Performance**:
   - Solution: Reduce the number of simulations or the size of the dataset

3. **Visualization Not Showing**:
   - Solution: Try refreshing the page or regenerating the chart

4. **CSV Upload Error**:
   - Solution: Ensure your CSV file has the correct format (Date and Price columns)
   - Solution: Check that the Date column is in YYYY-MM-DD format

5. **No Data Drift Detection Available**:
   - Solution: Make sure you have both generated sample data and uploaded real-world data for the same commodity

6. **Forecast Comparison Not Showing**:
   - Solution: Upload real-world data that includes dates after the last date in your training data

### Getting Help

If you encounter issues not covered in this guide:
- Check the DOCUMENTATION.md file for technical details
- Refer to the README.md file for general information
- Contact the system administrator for further assistance

## Conclusion

The Oil & Gas Market Optimization System provides powerful tools for analyzing market data, optimizing trading strategies, and making informed decisions. By following this guide, you can leverage the full potential of the system to gain valuable insights and competitive advantages in the oil and gas industry.

Happy analyzing!
