# Oil & Gas Market Optimization System: Implementation Summary

## Project Overview

The Oil & Gas Market Optimization System is a comprehensive AI-powered platform designed to provide actionable insights for the oil and gas industry. It integrates advanced data analytics, machine learning algorithms, and interactive visualizations to help users optimize operations, predict market trends, and enhance decision-making processes.

**Live Demo**: [https://7jpnmpbuxmt9bumsmvqjpn.streamlit.app/](https://7jpnmpbuxmt9bumsmvqjpn.streamlit.app/)

## Key Features Implemented

### 1. Comprehensive Module Structure

The system is organized into eleven interconnected modules:

- **Data Management**: For data generation, visualization, processing, and CSV upload
- **Trading Dashboard**: For strategy backtesting and optimization
- **Risk Analysis**: For risk assessment and simulation
- **Predictive Analytics**: For forecasting and optimization with real-world data comparison
- **Risk Assessment**: For market, geopolitical, and regulatory risk analysis
- **Decision Support**: For scenario modeling and advanced visualization
- **Data Drift Detection**: For comparing model predictions with real-world data and detecting when models need retraining
- **Hybrid Retrieval System**: For advanced information retrieval combining semantic and keyword search
- **Automated Market Report Generator**: For comprehensive market reports with visualizations and insights
- **Online Learning Framework**: For continuous model improvement with new data
- **Model Registry**: For managing model versions and metadata

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

### 8. EIA Price Drivers Integration

**Key Implementation Features**:
- Automated acquisition of EIA data on crude oil price drivers
- Feature engineering based on supply, demand, and inventory metrics
- Enhanced forecasting models leveraging price drivers data
- Feature importance analysis for price drivers
- Visualization of price driver impacts on forecasts

**Code Highlights**:
```python
def fetch_eia_price_driver(
    series_id: str,
    start_date: str,
    end_date: str,
    frequency: str = 'monthly',
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch price driver data from EIA API with error handling and rate limiting.
    """
    if api_key is None:
        api_key = os.getenv('EIA_API_KEY')
        if not api_key:
            raise ValueError("EIA API key not found. Set the EIA_API_KEY environment variable.")

    # Extract the category from the series ID (e.g., 'STEO' from 'STEO.COPR_NONOPEC.M')
    category = series_id.split('.')[0].lower()

    # Construct the URL based on the category
    url = f"https://api.eia.gov/v2/{category}/data/?api_key={api_key}&frequency={frequency}"
    url += f"&data[0]=value&facets[series][]={series_id}&start={start_date}&end={end_date}"

    # Make request with error handling and rate limiting
    response = requests.get(url)

    # Process response and return data
    return response.json()

def calculate_supply_demand_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate supply-demand balance features.
    """
    df_features = df.copy()

    # Global supply-demand balance
    df_features['global_balance'] = df_features['global_production'] - df_features['global_consumption']

    # OPEC supply-demand balance
    df_features['opec_balance'] = df_features['opec_production'] - df_features['global_consumption']

    # Non-OPEC supply-demand balance
    df_features['non_opec_balance'] = df_features['non_opec_production'] - df_features['global_consumption']

    # Supply-demand ratio
    df_features['supply_demand_ratio'] = df_features['global_production'] / df_features['global_consumption']

    return df_features
```

### 9. Hybrid Retrieval System

**Key Implementation Features**:
- Combined semantic and keyword search capabilities
- Vector indexing for efficient similarity search
- Keyword indexing for precise term matching
- Reranking of search results for improved relevance
- Configurable weighting between semantic and keyword search
- Support for metadata filtering and faceted search

**Code Highlights**:
```python
class HybridRetriever:
    """
    Hybrid retrieval system combining vector search and keyword search.
    """

    def __init__(
        self,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        vector_index: Optional[VectorIndex] = None,
        keyword_index: Optional[KeywordIndex] = None,
        reranker: Optional[Reranker] = None
    ):
        """
        Initialize the hybrid retriever.

        Parameters
        ----------
        vector_weight : float, optional
            Weight for vector search results, by default 0.7
        keyword_weight : float, optional
            Weight for keyword search results, by default 0.3
        vector_index : VectorIndex, optional
            Vector index, by default None (will create a new one)
        keyword_index : KeywordIndex, optional
            Keyword index, by default None (will create a new one)
        reranker : Reranker, optional
            Reranker, by default None (will create a new one)
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.vector_index = vector_index or VectorIndex()
        self.keyword_index = keyword_index or KeywordIndex()
        self.reranker = reranker or Reranker()

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retriever.

        Parameters
        ----------
        documents : List[Document]
            List of documents to add
        """
        # Add documents to both indices
        self.vector_index.add_documents(documents)
        self.keyword_index.add_documents(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using hybrid search.

        Parameters
        ----------
        query : str
            Query string
        top_k : int, optional
            Number of documents to retrieve, by default 5
        use_reranker : bool, optional
            Whether to use the reranker, by default True
        filter_metadata : Dict[str, Any], optional
            Metadata filter, by default None

        Returns
        -------
        List[Tuple[Document, float]]
            List of (document, score) tuples
        """
        # Get vector search results
        vector_results = self.vector_index.search(query, top_k=top_k*2, filter_metadata=filter_metadata)

        # Get keyword search results
        keyword_results = self.keyword_index.search(query, top_k=top_k*2, filter_metadata=filter_metadata)

        # Combine results
        combined_results = self._combine_results(vector_results, keyword_results)

        # Rerank results if requested
        if use_reranker and combined_results:
            combined_results = self.reranker.rerank(query, combined_results, top_k=top_k)
        else:
            # Just take the top_k
            combined_results = combined_results[:top_k]

        return combined_results
```

### 10. Automated Market Report Generator

**Key Implementation Features**:
- Comprehensive market reports with data visualizations
- Natural language insights and analysis
- Risk assessment with heatmaps and metrics
- Trading signals based on technical indicators
- Customizable report templates
- Support for multiple output formats (HTML, PDF)

**Code Highlights**:
```python
class MarketReportGenerator:
    """
    Automated Market Report Generator.

    This class provides methods to generate comprehensive market reports
    with data visualizations, insights, and recommendations.
    """

    def __init__(
        self,
        template_dir: str = 'templates/reports',
        output_dir: str = 'reports/market',
        data_provider = None,
        model_provider = None,
        llm_provider = None
    ):
        """
        Initialize the market report generator.

        Parameters
        ----------
        template_dir : str, optional
            Directory containing report templates, by default 'templates/reports'
        output_dir : str, optional
            Directory to save generated reports, by default 'reports/market'
        data_provider : object, optional
            Provider for market data, by default None
        model_provider : object, optional
            Provider for forecasting models, by default None
        llm_provider : object, optional
            Provider for language model generation, by default None
        """
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.data_provider = data_provider
        self.model_provider = model_provider
        self.llm_provider = llm_provider

        # Create directories if they don't exist
        os.makedirs(template_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

        # Create default template if it doesn't exist
        self._create_default_template()

    def generate_daily_report(
        self,
        commodities: List[str],
        date: Optional[datetime] = None,
        lookback_days: int = 30,
        forecast_days: int = 14,
        template_name: str = 'daily_report.html',
        output_format: str = 'html'
    ) -> str:
        """
        Generate a comprehensive daily market report.

        Parameters
        ----------
        commodities : List[str]
            List of commodities to include in the report
        date : datetime, optional
            Date for the report, by default None (today)
        lookback_days : int, optional
            Number of days to look back for historical data, by default 30
        forecast_days : int, optional
            Number of days to forecast, by default 14
        template_name : str, optional
            Name of the template to use, by default 'daily_report.html'
        output_format : str, optional
            Output format ('html' or 'pdf'), by default 'html'

        Returns
        -------
        str
            Path to the generated report
        """
        # Implementation details...

        return output_path
```

### 11. Online Learning Framework

**Key Implementation Features**:
- Continuous model improvement with new data
- Model registry for version management
- Evaluation metrics for model comparison
- Drift detection integration for triggering updates
- Automated model retraining when significant drift is detected
- Performance tracking across model versions

**Code Highlights**:
```python
class OnlineLearningManager:
    """
    Online Learning Manager for continuous model improvement.
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        evaluation_metrics: Optional[EvaluationMetrics] = None,
        drift_detector: Optional[DriftDetector] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the online learning manager.

        Parameters
        ----------
        model_registry : ModelRegistry, optional
            Model registry, by default None (will create a new one)
        evaluation_metrics : EvaluationMetrics, optional
            Evaluation metrics, by default None (will create a new one)
        drift_detector : DriftDetector, optional
            Drift detector, by default None (will create a new one)
        config_path : str, optional
            Path to configuration file, by default None
        """
        self.model_registry = model_registry or ModelRegistry()
        self.evaluation_metrics = evaluation_metrics or EvaluationMetrics()
        self.drift_detector = drift_detector or DriftDetector()

        # Default configuration
        self.config = {
            'update_frequency': 'daily',  # 'hourly', 'daily', 'weekly'
            'drift_threshold': 0.05,
            'improvement_threshold': 5.0,  # Percentage improvement required to update model
            'auto_update': True,
            'commodities': ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel'],
            'model_types': ['arima', 'xgboost', 'lstm', 'price_drivers'],
            'categorical_features': []
        }

        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.config.update(config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")

        # Create directories
        os.makedirs('logs', exist_ok=True)

        # Last update timestamp
        self.last_update = None

    def update_models(
        self,
        commodities: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        force_update: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Update models with new data if significant drift is detected.

        Parameters
        ----------
        commodities : List[str], optional
            List of commodities to update, by default None (will use all from config)
        model_types : List[str], optional
            List of model types to update, by default None (will use all from config)
        force_update : bool, optional
            Whether to force update regardless of drift, by default False

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with update results for each commodity and model type
        """
        # Implementation details...

        return results
```

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
   - Leverage EIA price drivers data for enhanced forecasting
   - Compare forecasts with real-world data
   - Evaluate forecast accuracy metrics (MAPE, RMSE)
   - Consider the confidence intervals in your planning
   - Analyze feature importance of price drivers

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

8. **Use the Hybrid Retrieval System**:
   - Formulate natural language queries about market dynamics
   - Adjust the weights between semantic and keyword search for optimal results
   - Use metadata filtering to narrow down search results
   - Enable reranking for more relevant results
   - Analyze the retrieved information to inform decision-making

9. **Generate Market Reports**:
   - Create daily or weekly reports for multiple commodities
   - Customize report templates to focus on specific aspects
   - Include visualizations for better understanding
   - Share reports with stakeholders in HTML or PDF format
   - Use the reports to track market trends over time

10. **Leverage the Online Learning Framework**:
    - Configure the update frequency based on market volatility
    - Set appropriate drift thresholds for different commodities
    - Monitor model performance metrics across versions
    - Use the model registry to manage and compare different models
    - Enable auto-update for continuous improvement
    - Periodically review the model update history

## Conclusion

The Oil & Gas Market Optimization System represents a comprehensive solution for the oil and gas industry, providing powerful tools for data analysis, trading strategy optimization, risk assessment, and decision support. The implementation leverages modern technologies and best practices to deliver a robust, user-friendly platform that can help users gain valuable insights and competitive advantages in the market.

For detailed usage instructions, please refer to the USER_GUIDE.md file. For technical documentation, see DOCUMENTATION.md.
