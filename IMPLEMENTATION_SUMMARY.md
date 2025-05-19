# Oil & Gas Market Optimization System: Implementation Summary

## Project Overview

The Oil & Gas Market Optimization System is a comprehensive AI-powered platform designed to provide actionable insights for the oil and gas industry. It integrates advanced data analytics, machine learning algorithms, and interactive visualizations to help users optimize operations, predict market trends, and enhance decision-making processes.

**Live Demo**: [https://7jpnmpbuxmt9bumsmvqjpn.streamlit.app/](https://7jpnmpbuxmt9bumsmvqjpn.streamlit.app/)

## Key Features Implemented

### 1. Comprehensive Module Structure

The system is organized into seven interconnected modules:

- **Data Management**: For data generation, visualization, processing, and CSV upload
- **Trading Dashboard**: For strategy backtesting and optimization
- **Risk Analysis**: For risk assessment and simulation
- **Predictive Analytics**: For forecasting and optimization with real-world data comparison
- **Risk Assessment**: For market, geopolitical, and regulatory risk analysis
- **Decision Support**: For scenario modeling and advanced visualization
- **Data Drift Detection**: For comparing model predictions with real-world data and detecting when models need retraining

### 2. Interactive User Interface

- **Streamlit-based Web Application**: Provides an intuitive, responsive interface
- **Sidebar Navigation**: Allows easy movement between modules
- **Session State Management**: Maintains data consistency across modules
- **Interactive Controls**: Sliders, dropdowns, and buttons for parameter adjustment

### 3. Advanced Analytics

- **Trading Strategy Implementation**: Moving Average Crossover and RSI strategies
- **Risk Metrics Calculation**: Value at Risk (VaR), Monte Carlo simulation
- **Forecasting Models**: Time series forecasting with multiple models
- **Optimization Algorithms**: For production and supply chain optimization

### 4. Visualization Capabilities

- **Interactive Charts**: Price trends, trading signals, return distributions
- **Geospatial Visualization**: Global risk maps for geopolitical analysis
- **Network Diagrams**: Supply chain visualization
- **Correlation Heatmaps**: For portfolio diversification analysis

### 5. Decision Support Tools

- **Scenario Modeling**: With financial impact analysis
- **Natural Language Interface**: For market-related queries
- **Advanced Visualizations**: Correlation heatmaps, seasonality charts
- **Risk Assessment Framework**: For comprehensive risk evaluation

## Technologies Used

### Core Technologies

- **Python 3.8+**: Primary programming language
- **Streamlit 1.22.0+**: Web application framework
- **Pandas 1.5.0+**: Data manipulation and analysis
- **NumPy 1.23.0+**: Numerical computing
- **Matplotlib 3.6.0+**: Data visualization
- **Scikit-learn 1.1.0+**: Machine learning algorithms
- **SciPy 1.9.0+**: Scientific computing
- **Statsmodels 0.13.0+**: Statistical models and tests

### Implementation Approach

- **Modular Design**: Each component is implemented as a separate function
- **Object-Oriented Programming**: For strategy implementation and model development
- **Functional Programming**: For data processing and visualization
- **Event-Driven Architecture**: For user interaction handling

## Implementation Details

### 1. Data Management

**Key Implementation Features**:
- Random walk with drift for synthetic data generation
- Interactive date range selection
- Data persistence using Streamlit's session state
- Visualization with Matplotlib
- CSV file upload for real-world data
- Data validation and preprocessing
- Comparison between training and real-world data

**Code Highlights**:
```python
def generate_sample_data(commodity, start_date=None, periods=1000):
    """Generate sample data for a commodity."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=periods)

    # Parameters for different commodities
    params = {
        'crude_oil': {'volatility': 0.02, 'drift': 0.0005},
        'gasoline': {'volatility': 0.025, 'drift': 0.0004},
        'diesel': {'volatility': 0.018, 'drift': 0.0003},
        'natural_gas': {'volatility': 0.03, 'drift': 0.0002}
    }

    # Get parameters for the selected commodity
    vol = params.get(commodity, {'volatility': 0.02, 'drift': 0.0005})['volatility']
    drift = params.get(commodity, {'volatility': 0.02, 'drift': 0.0005})['drift']

    # Generate price data
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(drift, vol, periods)
    price = 100  # Starting price
    prices = [price]

    for ret in returns:
        price *= (1 + ret)
        prices.append(price)

    # Create DataFrame
    dates = pd.date_range(start=start_date, periods=periods+1, freq='D')
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    df.set_index('Date', inplace=True)

    return df
```

### 2. Trading Dashboard

**Key Implementation Features**:
- Technical indicator calculation (Moving Averages, RSI)
- Signal generation based on strategy rules
- Performance metrics calculation (returns, Sharpe ratio)
- Visualization of trading signals and equity curve

**Code Highlights**:
```python
def calculate_strategy_returns(signals, initial_capital=100000.0, transaction_cost=0.001):
    """Calculate strategy returns based on signals."""
    # Create a DataFrame for positions and portfolio value
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['price'] = signals['price']
    positions['signal'] = signals['signal']

    # Buy a position when signal is 1, sell when signal is 0
    positions['position'] = positions['signal'].diff()

    # Calculate actual position accounting for transaction costs
    positions['holdings'] = positions['signal'] * positions['price'] * initial_capital

    # Calculate transaction costs
    positions['trade_costs'] = abs(positions['position']) * positions['price'] * initial_capital * transaction_cost

    # Calculate cash position
    positions['cash'] = initial_capital - (positions['signal'].diff().fillna(0) * positions['price'] * initial_capital) - positions['trade_costs'].cumsum()

    # Calculate total portfolio value
    positions['portfolio_value'] = positions['holdings'] + positions['cash']

    # Calculate returns
    positions['returns'] = positions['portfolio_value'].pct_change()

    return positions
```

### 3. Risk Analysis

**Key Implementation Features**:
- Historical and parametric VaR calculation
- Monte Carlo simulation for portfolio projection
- Return distribution analysis
- Visualization of risk metrics and projections

**Code Highlights**:
```python
def run_monte_carlo(df, num_simulations=1000, days=252, initial_investment=100000):
    """Run Monte Carlo simulation for price projection."""
    # Calculate daily returns
    returns = df['Price'].pct_change().dropna()

    # Get mean and standard deviation of returns
    mu = returns.mean()
    sigma = returns.std()

    # Run simulations
    simulation_df = pd.DataFrame()
    last_price = df['Price'].iloc[-1]

    for i in range(num_simulations):
        # Create list to store prices
        prices = [last_price]

        # Generate random returns
        for j in range(days):
            # Generate random return
            rand_return = np.random.normal(mu, sigma)

            # Calculate new price
            new_price = prices[-1] * (1 + rand_return)

            # Add new price to list
            prices.append(new_price)

        # Add prices to DataFrame
        simulation_df[f'sim_{i}'] = prices

    # Calculate portfolio value
    portfolio_value = simulation_df * (initial_investment / last_price)

    return simulation_df, portfolio_value
```

### 4. Predictive Analytics

**Key Implementation Features**:
- Time series forecasting with multiple models (ARIMA, Prophet, LSTM, XGBoost, Ensemble)
- Real-world data comparison with forecast accuracy metrics (MAPE, RMSE)
- Confidence intervals for price predictions
- Production optimization based on market conditions
- Maintenance scheduling with failure prediction
- Supply chain visualization and optimization

**Code Highlights**:
```python
def generate_forecast(df, model_type='arima', forecast_horizon=30, confidence=0.95):
    """Generate price forecast using selected model."""
    # Get price data
    price_data = df['Price'].values

    # Create time index
    time_index = np.arange(len(price_data))

    # Split data into train and test
    train_size = int(len(price_data) * 0.8)
    train_data = price_data[:train_size]
    test_data = price_data[train_size:]

    # Fit model based on type
    if model_type == 'arima':
        # Fit ARIMA model
        model = ARIMA(train_data, order=(5,1,0))
        model_fit = model.fit()

        # Generate forecast
        forecast = model_fit.forecast(steps=forecast_horizon)

        # Calculate confidence intervals
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        std_err = np.sqrt(model_fit.cov_params().diagonal())
        forecast_err = z * std_err

        lower_bound = forecast - forecast_err
        upper_bound = forecast + forecast_err

    elif model_type == 'linear':
        # Fit linear regression model
        model = LinearRegression()
        model.fit(time_index[:train_size].reshape(-1, 1), train_data)

        # Generate forecast
        forecast_index = np.arange(len(price_data), len(price_data) + forecast_horizon)
        forecast = model.predict(forecast_index.reshape(-1, 1))

        # Calculate confidence intervals
        y_pred = model.predict(time_index[:train_size].reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(train_data, y_pred))

        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        forecast_err = z * rmse

        lower_bound = forecast - forecast_err
        upper_bound = forecast + forecast_err

    # Create forecast DataFrame
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

    forecast_df = pd.DataFrame({
        'Forecast': forecast,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound
    }, index=forecast_dates)

    return forecast_df
```

### 5. Risk Assessment

**Key Implementation Features**:
- Market risk analysis with VaR and stress testing
- Geopolitical risk monitoring with global risk map
- Regulatory compliance tracking and impact analysis
- Portfolio diversification analysis with correlation matrix

### 6. Decision Support

**Key Implementation Features**:
- Scenario modeling with parameter adjustments
- Natural language interface for market queries
- Advanced visualization techniques
- Supply chain network optimization

### 7. Data Drift Detection

**Key Implementation Features**:
- Statistical comparison between training and real-world data
- Visual comparison with histograms and Q-Q plots
- Time series comparison with common date range highlighting
- Drift metrics calculation (MAE, MAPE, RMSE)
- Automatic detection of significant drift based on configurable thresholds
- Detailed recommendations for model retraining
- Identification of features with the most significant drift

**Code Highlights**:
```python
def data_drift_detection_page():
    """Data Drift Detection page functionality."""
    # Get commodities that have both processed and real-world data
    common_commodities = [
        commodity for commodity in st.session_state.processed_data.keys()
        if commodity in st.session_state.real_world_data
    ]

    # Calculate basic statistics for both datasets
    training_stats = {
        'Mean': training_data[price_col].mean(),
        'Std Dev': training_data[price_col].std(),
        'Min': training_data[price_col].min(),
        'Max': training_data[price_col].max(),
        'Skewness': training_data[price_col].skew(),
        'Kurtosis': training_data[price_col].kurtosis()
    }

    # Calculate percent change
    percent_change = {
        key: ((real_stats[key] - training_stats[key]) / training_stats[key]) * 100
        for key in training_stats.keys()
    }

    # Detect significant drift
    drift_threshold = 10  # 10% change threshold
    significant_drift = any(abs(v) > drift_threshold for v in percent_change.values())

    # Identify which features drifted the most
    max_drift_feature = max(percent_change.items(), key=lambda x: abs(x[1]))
```

## Best Practices for Using the System

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
   - Compare forecasts with real-world data
   - Evaluate forecast accuracy metrics (MAPE, RMSE)
   - Consider the confidence intervals in your planning

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

## Conclusion

The Oil & Gas Market Optimization System represents a comprehensive solution for the oil and gas industry, providing powerful tools for data analysis, trading strategy optimization, risk assessment, and decision support. The implementation leverages modern technologies and best practices to deliver a robust, user-friendly platform that can help users gain valuable insights and competitive advantages in the market.

For detailed usage instructions, please refer to the USER_GUIDE.md file. For technical documentation, see DOCUMENTATION.md.
